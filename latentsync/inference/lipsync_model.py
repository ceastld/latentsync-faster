import argparse
from dataclasses import dataclass
from functools import cached_property
import os
from typing import List, Optional, Union
import numpy as np
from omegaconf import OmegaConf
import torch
from diffusers import AutoencoderTiny, DPMSolverMultistepScheduler
from latentsync.models.unet import UNet3DConditionModel
from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
from diffusers.utils.import_utils import is_xformers_available
from latentsync.utils.util import check_ffmpeg_installed
from latentsync.whisper.audio2feature import Audio2Feature
from configs.config import GLOBAL_CONFIG


class LipsyncContext:
    audio_sample_rate: int = 16000
    video_fps: int = 25
    num_frames: int = 8
    height: int = 512
    width: int = 512
    num_inference_steps: int = 3
    guidance_scale: float = 1.5
    weight_type: str = torch.float16
    eta: float = 0.0
    mask: str = "fix_mask"
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None

    def __init__(self) -> None:
        pass

    def init_pipeline(self, pipeline: LipsyncPipeline):
        pipeline._initialize_parameters(
            self.num_frames,
            self.height,
            self.width,
            self.mask,
            self.guidance_scale,
            None,
        )
        pipeline.video_fps = self.video_fps
        self.extra_step_kwargs = pipeline.prepare_extra_step_kwargs(self.generator, self.eta)

@dataclass
class LipsyncMetadata:
    processed_frame: np.ndarray
    box: np.ndarray
    affine_matrice: np.ndarray
    original_frame: np.ndarray

class LipsyncModel:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.pipeline: LipsyncPipeline

    @cached_property
    def dtype(self):
        is_fp16_supported = (
            torch.cuda.is_available() and torch.cuda.get_device_capability()[0] > 7
        )
        return torch.float16 if is_fp16_supported else torch.float32

    @cached_property
    def pipeline(self):
        config = GLOBAL_CONFIG.unet_config

        scheduler = DPMSolverMultistepScheduler.from_pretrained(
            GLOBAL_CONFIG.config_dir, algorithm_type="dpmsolver++", solver_order=1
        )

        audio_encoder = Audio2Feature(
            model_path=GLOBAL_CONFIG.whisper_model_path,
            device="cuda",
            num_frames=config.data.num_frames,
        )
        vae = AutoencoderTiny.from_pretrained(
            "madebyollin/taesd", torch_dtype=self.dtype
        )
        vae.config.scaling_factor = 1.0
        vae.config.shift_factor = 0

        unet, _ = UNet3DConditionModel.from_pretrained(
            OmegaConf.to_container(config.model),
            GLOBAL_CONFIG.latentsync_unet_path,
            device="cpu",
        )

        unet = unet.to(dtype=self.dtype)

        # set xformers
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()

        pipeline = LipsyncPipeline(
            vae=vae,
            audio_encoder=audio_encoder,
            unet=unet,
            scheduler=scheduler,
        ).to(self.device)

        return pipeline

    def infer_setup(self, lipsync_context: LipsyncContext):
        self.pipeline.unet.eval()
        check_ffmpeg_installed()
        lipsync_context.init_pipeline(self.pipeline)
 
    def inference_batch(self, input_batch:List[LipsyncMetadata]) -> List[LipsyncMetadata]:
        pass
