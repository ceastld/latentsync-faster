from configs.config import GLOBAL_CONFIG
import torch
from dataclasses import dataclass
from typing import Callable, List, Optional, Union


@dataclass
class LipsyncContext:
    # Basic parameters
    audio_sample_rate: int = GLOBAL_CONFIG.lipsync.audio_sample_rate
    video_fps: int = GLOBAL_CONFIG.lipsync.video_fps
    num_frames: int = GLOBAL_CONFIG.lipsync.num_frames
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


    def to_dict(self) -> dict:
        """Convert context to dictionary for easy access"""
        return {
            "audio_sample_rate": self.audio_sample_rate,
            "video_fps": self.video_fps,
            "num_frames": self.num_frames,
            "height": self.height,
            "width": self.width,
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
            "weight_dtype": self.weight_dtype,
            "eta": self.eta,
            "device": self.device,
            "batch_size": self.batch_size,
            "do_classifier_free_guidance": self.do_classifier_free_guidance,
            "num_channels_latents": self.num_channels_latents,
        }

    @classmethod
    def from_dict(cls, config_dict: dict) -> "LipsyncContext":
        """Create context from dictionary"""
        return cls(**config_dict)
