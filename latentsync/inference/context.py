from latentsync.configs.config import GLOBAL_CONFIG, LipsyncConfig
import torch
from dataclasses import dataclass
from typing import Callable, List, Optional, Union
from functools import cached_property
from omegaconf import OmegaConf
from diffusers import AutoencoderTiny, AutoencoderKL, DPMSolverMultistepScheduler
from diffusers.utils.import_utils import is_xformers_available
from latentsync.models.unet import UNet3DConditionModel
from latentsync.whisper.audio2feature import Audio2Feature
from latentsync.models_v15.unet import UNet3DConditionModel as UNet3DConditionModel_v15


@dataclass
class LipsyncContext:
    config: LipsyncConfig = GLOBAL_CONFIG.lipsync
    # Basic parameters
    audio_sample_rate: int = config.audio_sample_rate
    video_fps: int = config.video_fps
    num_frames: int = config.num_frames
    audio_batch_size: int = config.audio_batch_size
    height: int = config.height
    width: int = config.width
    resolution: int = config.width
    samples_per_frame: int = config.samples_per_frame

    # Inference parameters
    num_inference_steps: int = config.num_inference_steps
    guidance_scale: float = config.guidance_scale
    eta: float = config.eta

    # Optional parameters
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None
    callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None
    callback_steps: int = 1

    # Runtime parameters
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    weight_dtype: torch.dtype = config.weight_dtype
    batch_size: int = 1
    do_classifier_free_guidance: bool = None
    num_channels_latents: int = None
    extra_step_kwargs: dict = None

    # Model selection flags
    use_compile: bool = False
    use_onnx: bool = False
    use_trt: bool = False

    def __post_init__(self):
        # Set do_classifier_free_guidance based on guidance_scale
        self.do_classifier_free_guidance = self.guidance_scale > 1.0

        # Ensure only one model type is selected
        if sum([self.use_compile, self.use_onnx, self.use_trt]) > 1:
            raise ValueError("Only one of use_compile, use_onnx, or use_trt can be True at the same time")

    def create_audio_encoder(self) -> Audio2Feature:
        """Create audio encoder for processing audio samples"""
        return Audio2Feature(
            model_path=GLOBAL_CONFIG.whisper_model_path,
            device=self.device,
            num_frames=self.num_frames,
        )

    def create_unet(self) -> UNet3DConditionModel:
        """Create UNet model for diffusion"""
        if self.use_trt:
            return self.create_unet_trt()
        elif self.use_onnx:
            return self.create_unet_onnx()
        else:
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
        onnx_path = os.path.join(os.path.dirname(GLOBAL_CONFIG.latentsync_unet_path_v15), "unet.onnx")

        # 检查ONNX模型是否存在
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"ONNX model not found at {onnx_path}. Please export the model first.")

        # 创建ONNX模型包装器
        unet = ONNXModelWrapper(onnx_path, device=self.device)

        # 设置与PyTorch模型相同的dtype属性（仅用于欺骗编译器，不影响实际运行）
        unet.dtype = self.weight_dtype

        return unet

    def create_unet_trt(self) -> UNet3DConditionModel:
        """Create TensorRT UNet model for diffusion with same interface as PyTorch UNet"""
        import os
        from latentsync.models.trt_wrapper import TRTModelWrapper

        # 构建TensorRT引擎路径 - 使用相同的名称但后缀为.engine
        engine_path = os.path.join(os.path.dirname(GLOBAL_CONFIG.latentsync_unet_path_v15), "latentsync_unet.engine")

        # 检查TensorRT引擎是否存在
        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"TensorRT engine not found at {engine_path}. Please export the model first.")

        # 创建TensorRT模型包装器
        unet = TRTModelWrapper(engine_path, device=self.device)

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
        self.num_inference_steps = 2
        return DPMSolverMultistepScheduler.from_pretrained(
            GLOBAL_CONFIG.config_dir,
            algorithm_type="dpmsolver++",
            solver_order=2,
            # use_karras_sigmas=True,
        )

    @classmethod
    def from_dict(cls, config_dict: dict) -> "LipsyncContext":
        """Create context from dictionary"""
        return cls(**config_dict)


class LipsyncContext_v15(LipsyncContext):
    def __post_init__(self):
        super().__post_init__()
        self.num_frames: int = 24

    def create_vae(self) -> AutoencoderKL:
        """Create VAE model for encoding/decoding images"""
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=self.weight_dtype)
        vae.config.scaling_factor = 0.18215
        vae.config.shift_factor = 0
        return vae.eval().to(self.device)

    def create_unet(self) -> UNet3DConditionModel_v15:
        """Create UNet model for diffusion"""
        unet, _ = UNet3DConditionModel_v15.from_pretrained(
            OmegaConf.to_container(GLOBAL_CONFIG.unet_config_v15.model),
            GLOBAL_CONFIG.latentsync_unet_path_v15,
            device=self.device,
        )
        return unet.eval().to(dtype=self.weight_dtype)
