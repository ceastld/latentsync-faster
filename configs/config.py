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

class LipsyncConfig:
    # Audio and video parameters
    audio_sample_rate = 16000
    video_fps = 25
    
    # Frame parameters
    num_frames = 8 # batch size
    height = 256
    width = 256
    
    # Diffusion parameters
    num_inference_steps = 3
    guidance_scale = 1.5
    weight_type = torch.float16
    eta = 0.0
    mask = "fix_mask"

GLOBAL_CONFIG = Config()

if __name__ == "__main__":
    print(GLOBAL_CONFIG.whisper_model_path)