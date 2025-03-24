from dataclasses import dataclass
from functools import cached_property
import os
import torch
from omegaconf import OmegaConf
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent
CHECKPOINT_DIR = str(ROOT_DIR / "checkpoints")
ASSETS_DIR = str(ROOT_DIR / "assets")
CONFIG_DIR = os.path.dirname(__file__)
OUTPUT_DIR = str(ROOT_DIR / "output")


class Config:
    config_dir = CONFIG_DIR
    checkpoint_dir = CHECKPOINT_DIR
    assets_dir = ASSETS_DIR

    def get_config_path(self, *sub_path):
        return os.path.join(self.config_dir, *sub_path)

    @cached_property
    def unet_config(self):
        return OmegaConf.load(self.get_config_path("unet/stage2_v1.yaml"))

    @cached_property
    def unet_config_v15(self):
        return OmegaConf.load(self.get_config_path("unet/stage2_v15.yaml"))

    @cached_property
    def audio_config(self):
        return OmegaConf.load(self.get_config_path("audio.yaml"))

    @cached_property
    def scheduler_config(self):
        return OmegaConf.load(self.get_config_path("scheduler_config.json"))

    @cached_property
    def whisper_model_path(self):
        return os.path.join(CHECKPOINT_DIR, "whisper/tiny.pt")

    @cached_property
    def latentsync_unet_path(self):
        return os.path.join(CHECKPOINT_DIR, "latentsync_unet.pt")

    @cached_property
    def latentsync_unet_path_v15(self):
        return os.path.join(CHECKPOINT_DIR, "v15", "latentsync_unet.pt")

    @cached_property
    def face_detector_path(self):
        return os.path.join(CHECKPOINT_DIR, "face/face_detector_fixed.onnx")

    @cached_property
    def landmark_detector_path(self):
        return os.path.join(CHECKPOINT_DIR, "face/landmark_detector_fixed.onnx")

    @cached_property
    def lipsync(self):
        return LipsyncConfig()

    @cached_property
    def lipsync_v15(self):
        return LipsyncConfig_v15()

    @cached_property
    def inference(self):
        return InferenceConfig()


def get_dtype():
    is_fp16_supported = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] > 7
    return torch.float16 if is_fp16_supported else torch.float32


class LipsyncConfig:
    # Audio and video parameters
    audio_sample_rate = 16000
    video_fps = 25
    samples_per_frame = int(16000 / 25)
    # only for parallel inference
    audio_batch_size = 20

    # Frame parameters
    num_frames = 8
    height = 256
    width = 256

    # Diffusion parameters
    num_inference_steps = 2
    guidance_scale = 1.5
    weight_dtype = get_dtype()
    eta = 0.0
    seed = 1247

class LipsyncConfig_v15(LipsyncConfig):
    num_frames = 16


@dataclass
class InferPackage:
    video_path: str
    audio_path: str
    video_out_path: str


class InferenceConfig:
    @property
    def default_audio_path(self):
        return self.obama.audio_path

    @property
    def default_video_path(self):
        return self.obama.video_path

    @cached_property
    def obama(self):
        return InferPackage(
            video_path=os.path.join(ASSETS_DIR, "obama.mp4"),
            audio_path=os.path.join(ASSETS_DIR, "cxk.mp3"),
            video_out_path=os.path.join(OUTPUT_DIR, "obama_cxk.mp4"),
        )

    @cached_property
    def obama_top(self):
        return InferPackage(
            video_path=os.path.join(ASSETS_DIR, "obama_top.mp4"),
            audio_path=os.path.join(ASSETS_DIR, "cxk.mp3"),
            video_out_path=os.path.join(OUTPUT_DIR, "obama_cxk_top.mp4"),
        )

    @cached_property
    def obama1(self):
        return InferPackage(
            video_path=os.path.join(ASSETS_DIR, "obama1.mp4"),
            audio_path=os.path.join(ASSETS_DIR, "cxk.mp3"),
            video_out_path=os.path.join(OUTPUT_DIR, "obama_cxk1.mp4"),
        )

    @property
    def demo1(self):
        return InferPackage(
            video_path=os.path.join(ASSETS_DIR, "demo1_video.mp4"),
            audio_path=os.path.join(ASSETS_DIR, "demo1_audio.wav"),
            video_out_path=os.path.join(OUTPUT_DIR, "demo1_out.mp4"),
        )


GLOBAL_CONFIG = Config()

if __name__ == "__main__":
    print(GLOBAL_CONFIG.whisper_model_path)
    print(GLOBAL_CONFIG.inference.default_audio_path)
    print(GLOBAL_CONFIG.lipsync.num_frames)
