import argparse
from dataclasses import dataclass
from functools import cached_property
import os
from typing import List, Optional, Union, Callable
import numpy as np
from omegaconf import OmegaConf
import torch
from diffusers import AutoencoderTiny, DPMSolverMultistepScheduler
from latentsync.models.unet import UNet3DConditionModel
from latentsync.pipelines.lipsync_pipeline import LipsyncMetadata, LipsyncPipeline
from diffusers.utils.import_utils import is_xformers_available
from latentsync.utils.util import check_ffmpeg_installed
from latentsync.whisper.audio2feature import Audio2Feature
from configs.config import GLOBAL_CONFIG


@dataclass
class LipsyncContext:
    config = GLOBAL_CONFIG.unet_config
    audio_sample_rate: int = 16000
    video_fps: int = 25
    num_frames: int = config.data.num_frames
    height: int = config.data.resolution
    width: int = config.data.resolution
    num_inference_steps: int = 3
    guidance_scale: float = 1.5
    weight_type: str = torch.float16
    eta: float = 0.0
    mask: str = "fix_mask"
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None
    callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None
    callback_steps: int = 1

    def init_pipeline(self, pipeline: LipsyncPipeline):
        self.params = pipeline._initialize_parameters(
            self.num_frames,
            self.height,
            self.width,
            self.mask,
            self.guidance_scale,
            self.callback_steps,
        )
        pipeline.video_fps = self.video_fps
        self.extra_step_kwargs = pipeline.prepare_extra_step_kwargs(
            self.generator, self.eta
        )


class LipsyncModel:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.pipeline: LipsyncPipeline
        self.lipsync_context: LipsyncContext = None

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

        pipeline: LipsyncPipeline = LipsyncPipeline(
            vae=vae,
            audio_encoder=audio_encoder,
            unet=unet,
            scheduler=scheduler,
        ).to(self.device)

        pipeline.unet.eval()

        return pipeline

    def infer_setup(self, lipsync_context: LipsyncContext):
        check_ffmpeg_installed()
        lipsync_context.init_pipeline(self.pipeline)
        self.lipsync_context = lipsync_context

    @torch.no_grad()
    def process_frame(self, frame: np.ndarray):
        return self.pipeline._preprocess_face(frame)

    @torch.no_grad()
    def process_batch(
        self,
        metadata_list: List[LipsyncMetadata],
        audio_samples: np.ndarray,
    ):
        assert metadata_list is not None
        assert self.lipsync_context is not None

        # Extract processed faces from metadata
        faces = torch.stack([metadata.face for metadata in metadata_list])

        # 2. Process audio for this batch
        whisper_feature = None
        if self.pipeline.unet.add_audio_layer:
            # Convert audio samples to tensor if needed
            if not isinstance(audio_samples, torch.Tensor):
                audio_samples = torch.from_numpy(audio_samples)

            # Process audio samples for this batch
            whisper_feature = self.pipeline.audio_encoder.samples2feat(audio_samples)

            audio_features = self.pipeline.audio_encoder.feature2chunks(
                feature_array=whisper_feature, fps=self.pipeline.video_fps
            )

            # Ensure we have enough audio features for all frames
            if len(audio_features) < len(faces):
                # Pad with last feature if needed
                last_feature = (
                    audio_features[-1]
                    if audio_features
                    else torch.zeros_like(audio_features[0])
                )
                audio_features.extend(
                    [last_feature] * (len(faces) - len(audio_features))
                )
            else:
                # Trim extra features
                audio_features = audio_features[: len(faces)]
        else:
            audio_features = None

        # 3. Run diffusion inference
        synced_faces_batch, _ = self.pipeline._run_diffusion_batch(
            faces,
            audio_features,
            self.lipsync_context.params,
            self.lipsync_context.num_inference_steps,
            self.lipsync_context.guidance_scale,
            self.lipsync_context.weight_type,
            self.lipsync_context.extra_step_kwargs,
            self.lipsync_context.generator,
            self.lipsync_context.callback,
            self.lipsync_context.callback_steps,
        )

        # 4. Update metadata with processed faces
        for i, metadata in enumerate(metadata_list):
            metadata_list[i] = LipsyncMetadata(
                face=synced_faces_batch[i],
                box=metadata.box,
                affine_matrice=metadata.affine_matrice,
                original_frame=metadata.original_frame,
            )

        # 5. Restore processed frames
        output_frames = self.pipeline.restore_video(metadata_list)

        return output_frames
