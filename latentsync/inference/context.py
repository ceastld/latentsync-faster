from latentsync.configs.config import GLOBAL_CONFIG
import torch
from dataclasses import dataclass
from typing import Callable, List, Optional, Union
from functools import cached_property
from omegaconf import OmegaConf
from diffusers import AutoencoderTiny, DPMSolverMultistepScheduler
from diffusers.utils.import_utils import is_xformers_available
from latentsync.models.unet import UNet3DConditionModel
from latentsync.whisper.audio2feature import Audio2Feature

@dataclass
class LipsyncContext:
    # Basic parameters
    audio_sample_rate: int = GLOBAL_CONFIG.lipsync.audio_sample_rate
    video_fps: int = GLOBAL_CONFIG.lipsync.video_fps
    num_frames: int = GLOBAL_CONFIG.lipsync.num_frames
    audio_batch_size: int = GLOBAL_CONFIG.lipsync.audio_batch_size
    height: int = GLOBAL_CONFIG.lipsync.height
    width: int = GLOBAL_CONFIG.lipsync.width
    resolution: int = GLOBAL_CONFIG.lipsync.width
    samples_per_frame: int = GLOBAL_CONFIG.lipsync.samples_per_frame

    # Inference parameters
    num_inference_steps: int = GLOBAL_CONFIG.lipsync.num_inference_steps
    guidance_scale: float = GLOBAL_CONFIG.lipsync.guidance_scale
    eta: float = GLOBAL_CONFIG.lipsync.eta

    # Optional parameters
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None
    callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None
    callback_steps: int = 1

    # Runtime parameters
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    weight_dtype: torch.dtype = GLOBAL_CONFIG.lipsync.weight_dtype
    batch_size: int = 1
    do_classifier_free_guidance: bool = None
    num_channels_latents: int = None
    extra_step_kwargs: dict = None

    use_compile: bool = False

    def __post_init__(self):
        # Set do_classifier_free_guidance based on guidance_scale
        self.do_classifier_free_guidance = self.guidance_scale > 1.0

    def create_audio_encoder(self) -> Audio2Feature:
        """Create audio encoder for processing audio samples"""
        return Audio2Feature(
            model_path=GLOBAL_CONFIG.whisper_model_path,
            device=self.device,
            num_frames=self.num_frames,
        )

    def create_unet(self) -> UNet3DConditionModel:
        """Create UNet model for diffusion"""
        unet, _ = UNet3DConditionModel.from_pretrained(
            OmegaConf.to_container(GLOBAL_CONFIG.unet_config.model),
            GLOBAL_CONFIG.latentsync_unet_path,
            device=self.device,
        )
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        return unet.eval().to(dtype=self.weight_dtype)

    def create_unet_onnx(self) -> UNet3DConditionModel:
        """Create ONNX UNet model for diffusion with same interface as PyTorch UNet"""
        import os
        from latentsync.models.onnx_wrapper import ONNXModelWrapper
        
        # 构建ONNX模型路径 - 使用相同的名称但后缀为.onnx
        onnx_path = os.path.join(os.path.dirname(GLOBAL_CONFIG.latentsync_unet_path), 
                                "latentsync_unet.onnx")
        
        # 检查ONNX模型是否存在
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"ONNX model not found at {onnx_path}. Please export the model first.")
        
        # 创建ONNX模型包装器
        unet = ONNXModelWrapper(onnx_path, device=self.device)
        
        # 设置与PyTorch模型相同的dtype属性（仅用于欺骗编译器，不影响实际运行）
        unet.dtype = self.weight_dtype
        
        return unet

    def create_vae(self) -> AutoencoderTiny:
        """Create VAE model for encoding/decoding images"""
        vae = AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=self.weight_dtype)
        vae.config.scaling_factor = 1.0
        vae.config.shift_factor = 0
        return vae.eval().to(self.device)

    def create_scheduler(self) -> DPMSolverMultistepScheduler:
        """Create diffusion scheduler"""
        return DPMSolverMultistepScheduler.from_pretrained(
            GLOBAL_CONFIG.config_dir, algorithm_type="dpmsolver++", solver_order=1
        )

    @classmethod
    def from_dict(cls, config_dict: dict) -> "LipsyncContext":
        """Create context from dictionary"""
        return cls(**config_dict)
