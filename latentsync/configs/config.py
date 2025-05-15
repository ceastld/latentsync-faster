from dataclasses import dataclass
from functools import cached_property
import os
import torch
from omegaconf import OmegaConf
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent
CHECKPOINT_DIR = str(ROOT_DIR / "checkpoints")
ASSETS_DIR = str(ROOT_DIR / "assets")
OUTPUT_DIR = str(ROOT_DIR / "output")
TEST_DIR = str(ROOT_DIR / "testset")


class Config:
    checkpoint_dir = CHECKPOINT_DIR
    assets_dir = ASSETS_DIR
    output_dir = OUTPUT_DIR
    test_dir = TEST_DIR

    @cached_property
    def mask_image_path(self):
        return os.path.join(ROOT_DIR, "latentsync/utils/mask.png")

    @cached_property
    def lipsync(self):
        return LipsyncConfig()

    @cached_property
    def inference(self):
        return InferenceConfig(assets_dir=ASSETS_DIR, output_dir=OUTPUT_DIR)


def get_dtype():
    is_fp16_supported = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] > 7
    return torch.float16 if is_fp16_supported else torch.float32


class LipsyncConfig:
    # Audio and video parameters
    audio_sample_rate = 16000
    video_fps = 25
    samples_per_frame = int(16000 / 25)
    # only for parallel inference
    audio_batch_size = 16
    face_batch_size = 5
    use_gaussian_blur = True
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
    vae_type = "tiny"

    def __init__(self, checkpoint_dir: str = None) -> None:
        self.checkpoint_dir = checkpoint_dir or CHECKPOINT_DIR
        self.config_dir = os.path.dirname(__file__)

    def get_config_path(self, *sub_path):
        return os.path.join(self.config_dir, *sub_path)

    @cached_property
    def unet_config(self):
        return OmegaConf.load(self.get_config_path("unet/stage2_v1.yaml"))

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
    def face_detector_path(self):
        return os.path.join(self.checkpoint_dir, "face/face_detector_fixed.onnx")

    @cached_property
    def landmark_detector_path(self):
        return os.path.join(self.checkpoint_dir, "face/landmark_detector_fixed.onnx")


class LipsyncConfig_v15(LipsyncConfig):
    num_frames = 16
    vae_type = "kl"

    @cached_property
    def unet_config(self):
        return OmegaConf.load(self.get_config_path("unet/stage2_v15.yaml"))

    @cached_property
    def latentsync_unet_path(self):
        return os.path.join(self.checkpoint_dir, "v15", "latentsync_unet.pt")


@dataclass
class InferPackage:
    video_path: str
    audio_path: str
    video_out_path: str


class InferenceConfig:
    def __init__(self, assets_dir: str, output_dir: str):
        assert os.path.exists(assets_dir), f"Assets directory {assets_dir} does not exist"
        self.assets_dir = assets_dir
        self.output_dir = output_dir

    @property
    def default_audio_path(self):
        return self.obama.audio_path

    @property
    def default_video_path(self):
        return self.obama.video_path

    def create_demo(self, video_path: str, audio_path: str, video_out_path: str):
        return InferPackage(
            video_path=os.path.join(self.assets_dir, video_path),
            audio_path=os.path.join(self.assets_dir, audio_path),
            video_out_path=os.path.join(self.output_dir, video_out_path),
        )

    @cached_property
    def obama(self):
        return self.create_demo("obama.mp4", "cxk.mp3", "obama_cxk.mp4")

    @cached_property
    def obama_top(self):
        return self.create_demo("obama_top.mp4", "cxk.mp3", "obama_cxk_top.mp4")

    @cached_property
    def obama1(self):
        return self.create_demo("obama1.mp4", "cxk.mp3", "obama_cxk1.mp4")

    @property
    def demo1(self):
        return self.create_demo("demo1_video.mp4", "demo1_audio.wav", "demo1_out.mp4")

    @property
    def demo_large_pose(self):
        return self.create_demo("large_pose.mp4", "cxk.mp3", "large_pose_cxk.mp4")


GLOBAL_CONFIG = Config()

if __name__ == "__main__":
    config = LipsyncConfig_v15()
    print(config.audio_sample_rate)
