import argparse
from dataclasses import dataclass
from functools import cached_property
import os
from typing import List, Optional, Union, Callable, Tuple
import numpy as np
from omegaconf import OmegaConf
import torch
from diffusers import AutoencoderTiny, DPMSolverMultistepScheduler
from latentsync.models.unet import UNet3DConditionModel
from latentsync.pipelines import (
    LipsyncMetadata,
    LipsyncPipeline,
    LipsyncDiffusionPipeline,
)

from diffusers.utils.import_utils import is_xformers_available
from latentsync.utils.util import check_ffmpeg_installed
from latentsync.whisper.audio2feature import Audio2Feature
from configs.config import GLOBAL_CONFIG


@dataclass
class LipsyncContext:
    audio_sample_rate: int = 16000
    video_fps: int = 25
    num_frames: int = 8
    height: int = 256
    width: int = 256
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


def get_lipsync_pipeline(dtype, device, use_compile=False) -> LipsyncPipeline:
    check_ffmpeg_installed()

    config = GLOBAL_CONFIG.unet_config

    scheduler = DPMSolverMultistepScheduler.from_pretrained(
        GLOBAL_CONFIG.config_dir, algorithm_type="dpmsolver++", solver_order=1
    )

    audio_encoder = Audio2Feature(
        model_path=GLOBAL_CONFIG.whisper_model_path,
        device="cuda",
        num_frames=config.data.num_frames,
    )
    vae = AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=dtype)
    vae.config.scaling_factor = 1.0
    vae.config.shift_factor = 0

    unet, _ = UNet3DConditionModel.from_pretrained(
        OmegaConf.to_container(config.model),
        GLOBAL_CONFIG.latentsync_unet_path,
        device="cpu",
    )

    unet = unet.to(dtype=dtype)

    # set xformers
    if is_xformers_available():
        unet.enable_xformers_memory_efficient_attention()

    pipeline: LipsyncPipeline = LipsyncPipeline(
        vae=vae,
        audio_encoder=audio_encoder,
        unet=unet,
        scheduler=scheduler,
        use_compile=use_compile,
    ).to(device)

    pipeline.unet.eval()

    return pipeline


class LipsyncModel:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.lipsync_context: LipsyncContext = None

    @cached_property
    def dtype(self):
        is_fp16_supported = (
            torch.cuda.is_available() and torch.cuda.get_device_capability()[0] > 7
        )
        return torch.float16 if is_fp16_supported else torch.float32

    @cached_property
    def pipeline(self):
        return get_lipsync_pipeline(self.dtype, self.device)

    def infer_setup(self, lipsync_context: LipsyncContext):
        lipsync_context.init_pipeline(self.pipeline)
        self.lipsync_context = lipsync_context

    @torch.no_grad()
    def _extract_faces(self, metadata_list: List[LipsyncMetadata]) -> torch.Tensor:
        """Extract processed faces from metadata and stack them into a batch."""
        return torch.stack([metadata.face for metadata in metadata_list])

    @torch.no_grad()
    def process_audio(self, audio_samples: np.ndarray, num_faces: int = -1) -> Optional[torch.Tensor]:
        """Process audio samples and align them with the number of faces."""
        if not self.pipeline.unet.add_audio_layer:
            return None

        # Convert audio samples to tensor if needed
        if not isinstance(audio_samples, torch.Tensor):
            audio_samples = torch.from_numpy(audio_samples)

        # Process audio samples for this batch
        whisper_feature = self.pipeline.audio_encoder.samples2feat(audio_samples)

        audio_features = self.pipeline.audio_encoder.feature2chunks(
            feature_array=whisper_feature, fps=self.pipeline.video_fps
        )

        # Align audio features with the number of faces
        return self._align_audio_features(audio_features, num_faces)

    @torch.no_grad()
    def _align_audio_features(self, audio_features: List[torch.Tensor], num_faces: int) -> List[torch.Tensor]:
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
        """Run the diffusion model to generate synced faces."""
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
        return synced_faces_batch

    @torch.no_grad()
    def _update_metadata(
        self, metadata_list: List[LipsyncMetadata], synced_faces_batch: torch.Tensor
    ) -> List[LipsyncMetadata]:
        """Update metadata with processed faces."""
        updated_metadata = []
        for i, metadata in enumerate(metadata_list):
            updated_metadata.append(LipsyncMetadata(
                face=synced_faces_batch[i],
                box=metadata.box,
                affine_matrice=metadata.affine_matrice,
                original_frame=metadata.original_frame,
            ))
        return updated_metadata

    @torch.no_grad()
    def process_frame(self, frame: np.ndarray):
        return self.pipeline._preprocess_face(frame)

    @torch.no_grad()
    def process_batch(
        self,
        metadata_list: List[LipsyncMetadata],
        audio_features: Optional[List[torch.Tensor]],
    ):
        """Process a batch of frames with corresponding audio samples."""
        assert metadata_list is not None
        assert self.lipsync_context is not None

        # 1. Extract processed faces from metadata
        faces = self._extract_faces(metadata_list)
        
        # 2. Process audio for this batch
        audio_features = self._align_audio_features(audio_features, len(faces))
        
        # 3. Run diffusion inference
        synced_faces_batch = self._run_diffusion(faces, audio_features)
        
        # 4. Update metadata with processed faces
        updated_metadata = self._update_metadata(metadata_list, synced_faces_batch)
        
        # 5. Restore processed frames
        output_frames = self.pipeline.restore_video(updated_metadata)

        return output_frames
    
    @torch.no_grad()
    def process_video(self, video_frames: List[np.ndarray], audio_samples: np.ndarray):
        """Process a video with corresponding audio samples."""
        assert self.lipsync_context is not None
        
        # 1. Process audio samples
        audio_features = self.process_audio(audio_samples, len(video_frames))
        
        # 2. Define batch size based on context
        batch_size = self.lipsync_context.num_frames
        
        # 3. Process video frames in batches
        for i in range(0, len(video_frames), batch_size):
            # Get current batch of frames
            batch_frames = video_frames[i:i+batch_size]
            
            # Get corresponding audio features for this batch
            batch_audio_features = audio_features[i:i+batch_size] if audio_features else None
            
            # Preprocess frames to get metadata
            metadata_list = []
            for frame in batch_frames:
                try:
                    # Process each frame to get face metadata
                    metadata = self.process_frame(frame)
                    metadata_list.append(metadata)
                except Exception as e:
                    print(f"Face preprocessing failed: {e}")
                    # If processing fails and there are other successfully processed frames, use the previous frame's result
                    if len(metadata_list) > 0:
                        metadata_list.append(metadata_list[-1])
                    # Otherwise skip this frame (this should be handled better in production)
            
            # Skip batch if no faces were detected
            if not metadata_list:
                print(f"No faces detected in batch {i//batch_size + 1}")
                continue
                
            # Process the batch
            batch_output_frames = self.process_batch(metadata_list, batch_audio_features)
            
            # Add processed frames to output
            for frame in batch_output_frames:
                yield frame
            
            print(f"Processed batch {i//batch_size + 1}/{(len(video_frames) + batch_size - 1) // batch_size}")
        