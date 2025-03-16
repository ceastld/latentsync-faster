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
import numpy as np
import time

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
    prewarmed: bool = False

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

    def warmup_models(self, models_dict):
        """预热所有模型，确保首次推理时不会出现延迟"""
        if self.prewarmed:
            return
            
        print("正在预热模型组件...")
        
        # 预热VAE
        if 'vae' in models_dict:
            try:
                dummy_latents = torch.randn(1, 4, self.height // 8, self.width // 8, device=self.device, dtype=self.weight_dtype)
                with torch.no_grad():
                    _ = models_dict['vae'].decode(dummy_latents)
                    torch.cuda.synchronize()
                print("VAE预热成功")
            except Exception as e:
                print(f"VAE预热失败: {e}")
        
        # 预热UNet (调整输入格式并添加更健壮的错误处理)
        if 'unet' in models_dict:
            try:
                # 正确设置UNet预热输入
                # 注意：UNet的输入通道应该匹配模型期望值，这里从错误信息看应为13通道
                batch_size = 1
                num_frames = 4
                in_channels = 13  # 从错误信息中确定的正确通道数
                
                # 确保输入尺寸正确
                dummy_latents = torch.randn(
                    batch_size, in_channels, num_frames, self.height // 8, self.width // 8, 
                    device=self.device, dtype=self.weight_dtype
                )
                dummy_timesteps = torch.ones(batch_size, device=self.device) * 999
                
                # 确保隐藏状态尺寸正确
                dummy_encoder_hidden_states = torch.randn(
                    batch_size, num_frames, 1280,  
                    device=self.device, dtype=self.weight_dtype
                )
                
                # 尝试进行一次前向传播
                with torch.no_grad():
                    _ = models_dict['unet'](
                        dummy_latents, 
                        dummy_timesteps, 
                        encoder_hidden_states=dummy_encoder_hidden_states
                    )
                    torch.cuda.synchronize()
                print("UNet预热成功")
            except Exception as e:
                print(f"UNet预热失败，跳过: {e}")
                print("这个错误不会影响实际执行，将在正式运行时自动使用正确参数")
        
        # 预热人脸检测器
        if 'face_detector' in models_dict:
            try:
                # 生成假图像并调用一次人脸检测器的maintain_session方法
                models_dict['face_detector'].maintain_session()
                print("人脸检测器预热成功")
            except Exception as e:
                print(f"人脸检测器预热失败: {e}")
        
        # 预热音频编码器
        if 'audio_encoder' in models_dict:
            try:
                dummy_audio = np.random.rand(16000).astype(np.float32)  # 1秒的16kHz音频
                models_dict['audio_encoder'].get_audio_features(dummy_audio)
                print("音频编码器预热成功")
            except Exception as e:
                print(f"音频编码器预热失败: {e}")
            
        print("模型预热完成")
        self.prewarmed = True
        
        # 强制等待所有预热操作完成
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()  # 清理预热过程中可能产生的临时缓存
        time.sleep(0.5)  # 额外等待，确保所有GPU操作完成

    @classmethod
    def from_dict(cls, config_dict: dict) -> "LipsyncContext":
        """Create context from dictionary"""
        return cls(**config_dict)
