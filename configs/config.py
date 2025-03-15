from functools import cached_property
import os
import torch
from omegaconf import OmegaConf
from pathlib import Path

class Config:
    config_dir = os.path.dirname(__file__)
    checkpoint_dir = str(Path(__file__).parent.parent / "checkpoints")

    def get_config_path(self, *sub_path):
        return os.path.join(self.config_dir, *sub_path)

    @cached_property
    def unet_config(self):
        return OmegaConf.load(self.get_config_path("unet/second_stage.yaml"))

    @cached_property
    def audio_config(self):
        return OmegaConf.load(self.get_config_path("audio.yaml"))

    @cached_property
    def scheduler_config(self):
        return OmegaConf.load(self.get_config_path("scheduler_config.json"))

    @cached_property
    def whisper_model_path(self):
        return os.path.join(self.checkpoint_dir, "whisper/tiny.pt")

    @cached_property
    def latentsync_unet_path(self):
        return os.path.join(self.checkpoint_dir, "latentsync_unet.pt")
    
    @cached_property
    def lipsync(self):
        return LipsyncConfig()

def get_dtype():
    is_fp16_supported = (
        torch.cuda.is_available() and torch.cuda.get_device_capability()[0] > 7
    )
    return torch.float16 if is_fp16_supported else torch.float32

class LipsyncConfig:
    # Audio and video parameters
    audio_sample_rate = 16000
    video_fps = 25
    samples_per_frame = int(16000 / 25)
    
    # Frame parameters
    num_frames = 8 # batch size
    height = 256
    width = 256
    
    # Diffusion parameters
    num_inference_steps = 3
    guidance_scale = 1.5
    weight_dtype = get_dtype()
    eta = 0.0

GLOBAL_CONFIG = Config()

if __name__ == "__main__":
    print(GLOBAL_CONFIG.whisper_model_path)