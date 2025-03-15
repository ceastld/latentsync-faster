import argparse
from functools import cached_property
import os
from typing import List, Optional, Union, Tuple
import numpy as np
from omegaconf import OmegaConf
import torch
from diffusers import AutoencoderTiny, DPMSolverMultistepScheduler
from latentsync.inference.context import LipsyncContext
from latentsync.models.unet import UNet3DConditionModel
from latentsync.pipelines import (
    LipsyncPipeline,
    LipsyncDiffusionPipeline,
)

from diffusers.utils.import_utils import is_xformers_available
from latentsync.pipelines.metadata import LipsyncMetadata
from latentsync.utils.timer import Timer
from latentsync.utils.util import check_ffmpeg_installed
from latentsync.whisper.audio2feature import Audio2Feature
from configs.config import GLOBAL_CONFIG


class ModelWrapper:
    def __init__(self, context: LipsyncContext):
        self.context = context

    @property
    def device(self):
        return self.context.device

    @property
    def dtype(self):
        return self.context.weight_dtype

    @cached_property
    def audio_encoder(self):
        return Audio2Feature(
            model_path=GLOBAL_CONFIG.whisper_model_path,
            device=self.device,
            num_frames=self.context.num_frames,
        )

    @cached_property
    def unet(self):
        unet, _ = UNet3DConditionModel.from_pretrained(
            OmegaConf.to_container(GLOBAL_CONFIG.unet_config.model),
            GLOBAL_CONFIG.latentsync_unet_path,
            device=self.device,
        )
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        return unet.eval().to(dtype=self.dtype)

    @cached_property
    def vae(self):
        vae = AutoencoderTiny.from_pretrained(
            "madebyollin/taesd", torch_dtype=self.dtype
        )
        vae.config.scaling_factor = 1.0
        vae.config.shift_factor = 0
        return vae

    @cached_property
    def scheduler(self):
        return DPMSolverMultistepScheduler.from_pretrained(
            GLOBAL_CONFIG.config_dir, algorithm_type="dpmsolver++", solver_order=1
        )

    @cached_property
    def pipeline(self):
        return LipsyncPipeline(
            vae=self.vae,
            audio_encoder=self.audio_encoder,
            unet=self.unet,
            scheduler=self.scheduler,
            lipsync_context=self.context,
        ).to(self.device)

    @cached_property
    def diffusion_pipeline(self):
        return LipsyncDiffusionPipeline(
            vae=self.vae,
            audio_encoder=self.audio_encoder,
            unet=self.unet,
            scheduler=self.scheduler,
            lipsync_context=self.context,
        ).to(self.device)


def get_lipsync_pipeline(context: LipsyncContext) -> LipsyncPipeline:
    check_ffmpeg_installed()
    wrapper = ModelWrapper(context=context)
    return wrapper.pipeline


class LipsyncModel:
    def __init__(self, context: LipsyncContext):
        self.context: LipsyncContext = context
        self.pipeline = get_lipsync_pipeline(context)

    @property
    def device(self):
        return self.context.device

    @torch.no_grad()
    def process_audio(
        self, audio_samples: np.ndarray, num_faces: int = -1
    ) -> Optional[torch.Tensor]:
        """Process audio samples and align them with the number of faces."""
        # Process audio samples for this batch
        whisper_feature = self.pipeline.audio_encoder.samples2feat(audio_samples)

        audio_features = self.pipeline.audio_encoder.feature2chunks(
            feature_array=whisper_feature, fps=self.pipeline.video_fps
        )

        # Align audio features with the number of faces
        return self.align_audio_features(audio_features, num_faces)

    @Timer()
    @torch.no_grad()
    def process_audio_with_pre(self, pre_audio_samples, audio_samples):
        samples_per_frame = self.context.samples_per_frame
        num_frames = int(np.ceil(len(audio_samples) / samples_per_frame))
        combined_samples = np.concatenate([pre_audio_samples, audio_samples])
        combined_features = self.process_audio(combined_samples)
        return combined_features[-num_frames:] if num_frames > 0 else []

    @torch.no_grad()
    def align_audio_features(
        self, audio_features: List[torch.Tensor], num_faces: int
    ) -> List[torch.Tensor]:
        """Ensure audio features match the number of faces by padding or trimming."""
        if len(audio_features) < num_faces:
            # Pad with last feature if needed
            last_feature = (
                audio_features[-1]
                if audio_features
                else torch.zeros_like(audio_features[0])
            )
            audio_features.extend([last_feature] * (num_faces - len(audio_features)))
        elif num_faces > 0:
            # Trim extra features
            audio_features = audio_features[:num_faces]

        return audio_features

    @torch.no_grad()
    def _run_diffusion(
        self, faces: torch.Tensor, audio_features: Optional[List[torch.Tensor]]
    ) -> torch.Tensor:
        synced_faces_batch, _ = self.pipeline._run_diffusion_batch(
            faces, audio_features, self.context
        )
        return synced_faces_batch

    @Timer()
    @torch.no_grad()
    def process_batch(
        self,
        metadata_list: List[LipsyncMetadata],
        audio_features: Optional[List[torch.Tensor]],
    ):
        faces = torch.stack([metadata.face for metadata in metadata_list])
        audio_features = self.align_audio_features(audio_features, len(faces))

        synced_faces_batch = self._run_diffusion(faces, audio_features)

        for i, metadata in enumerate(metadata_list):
            metadata.set_sync_face(synced_faces_batch[i])

        output_frames = self.pipeline.restore_video(metadata_list)
        return output_frames

    @cached_property
    def face_processor(self):
        return self.pipeline.face_processor

    @torch.no_grad()
    def process_video(self, video_frames: List[np.ndarray], audio_samples: np.ndarray):
        """Process a video with corresponding audio samples."""
        assert self.context is not None

        # 1. Process audio samples
        audio_features = self.process_audio(audio_samples, len(video_frames))

        # 2. Define batch size based on context
        batch_size = self.context.num_frames

        # 3. Process video frames in batches
        for i in range(0, len(video_frames), batch_size):
            # Get current batch of frames
            batch_frames = video_frames[i : i + batch_size]

            # Get corresponding audio features for this batch
            batch_audio_features = (
                audio_features[i : i + batch_size] if audio_features else None
            )

            # Preprocess frames to get metadata
            metadata_list = self.face_processor.prepare_face_batch(batch_frames)

            # Skip batch if no faces were detected
            if not metadata_list:
                for frame in batch_frames:
                    yield frame
                continue

            # Process the batch
            batch_output_frames = self.process_batch(
                metadata_list, batch_audio_features
            )

            # Add processed frames to output
            for frame in batch_output_frames:
                yield frame

            print(
                f"Processed batch {i//batch_size + 1}/{(len(video_frames) + batch_size - 1) // batch_size}"
            )
