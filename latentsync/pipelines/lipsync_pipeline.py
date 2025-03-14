# Adapted from https://github.com/guoyww/AnimateDiff/blob/main/animatediff/pipelines/pipeline_animation.py

from typing import Callable, List, Optional, Tuple, Union
import time

import torch


from diffusers.models import AutoencoderKL
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.utils import logging
from latentsync.inference.context import LipsyncContext
from latentsync.pipelines.lipsync_diffusion_pipeline import LipsyncDiffusionPipeline
from latentsync.pipelines.lipsync_diffusion_pipeline import LipsyncMetadata

from ..models.unet import UNet3DConditionModel
from ..utils.util import read_audio, check_ffmpeg_installed
from ..whisper.audio2feature import Audio2Feature

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

class LipsyncPipeline(LipsyncDiffusionPipeline):
    def __init__(
        self,
        vae: AutoencoderKL,
        audio_encoder: Audio2Feature,
        unet: UNet3DConditionModel,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
        use_compile: bool = False,
    ):
        super().__init__(vae, unet, scheduler, use_compile)
        self.audio_encoder = audio_encoder
        self.register_modules(audio_encoder=audio_encoder)
        self.set_progress_bar_config(desc="Steps")

    def init_with_context(self, context: LipsyncContext):
        """Initialize pipeline parameters with context"""
        # Initialize basic parameters
        super().init_with_context(context)

        # Set video fps
        self.video_fps = context.video_fps

        # Prepare extra step kwargs
        context.extra_step_kwargs = self.prepare_extra_step_kwargs(
            context.generator, context.eta
        )

        # Set progress bar config
        self.set_progress_bar_config(desc=f"Sample frames: {context.num_frames}")

        return self

    def _prepare_audio_batch(self, whisper_feature: Optional[torch.Tensor], batch_idx: int, 
                           num_frames_in_batch: int, context: LipsyncContext) -> Optional[List[torch.Tensor]]:
        """Prepare audio features for the current batch"""
        """NOT the core function for audio_processor. Need further work for stream processing"""
        if not self.unet.add_audio_layer or whisper_feature is None:
            return None
            
        # Split audio features in batches of size num_frames
        start_idx = batch_idx * context.num_frames
        
        # Calculate feature count using actual frame count for the batch
        chunks = self.audio_encoder.feature2chunks(
            feature_array=whisper_feature, 
            fps=self.video_fps
        )
        
        # Ensure we don't exceed the range
        end_idx = min(start_idx + num_frames_in_batch, len(chunks))
        if start_idx >= len(chunks):
            return None
            
        selected_chunks = chunks[start_idx:end_idx]
        # If the actual extracted features are fewer than frames, pad by repeating the last feature
        while len(selected_chunks) < num_frames_in_batch:
            if len(selected_chunks) > 0:
                selected_chunks.append(selected_chunks[-1])
            else:
                # If no valid features, create zero-filled features
                selected_chunks.append(torch.zeros_like(chunks[0]) if len(chunks) > 0 else 
                                      torch.zeros((1, context.num_channels_latents)))
        
        return selected_chunks
        
    @torch.no_grad()
    def __call__(
        self,
        video_path: str,
        audio_path: str,
        video_out_path: str,
        video_mask_path: str = None,
        context: Optional[LipsyncContext] = None,
        **kwargs,
    ):
        """Execute lip sync inference, using stream processing for video frames"""
        # Save model training state and set to evaluation mode
        is_train = self.unet.training
        self.unet.eval()

        # Initialize context if not provided
        if context is None:
            context = LipsyncContext(**kwargs)
        
        # Initialize parameters
        check_ffmpeg_installed()
        self.init_with_context(context)
        
        # Record total start time
        total_start_time = time.time()
        
        # Prepare audio features (process all audio at once)
        audio_samples = read_audio(audio_path)

        audio_feature_start = time.time()
        whisper_feature = self.audio_encoder.audio2feat(audio_path) if self.unet.add_audio_layer else None
        audio_feature_end = time.time()
        print(f"Audio feature extraction completed, time: {audio_feature_end - audio_feature_start:.2f} seconds")
        
        # Initialize video stream reading
        video_capture, total_frames = self._init_video_stream(video_path)
        
        # Get total batch count (for audio feature segmentation)
        total_batches = (total_frames + context.num_frames - 1) // context.num_frames
        
        # Calculate total audio feature count (if any)
        total_audio_chunks = 0
        if whisper_feature is not None:
            chunks = self.audio_encoder.feature2chunks(feature_array=whisper_feature, fps=context.video_fps)
            total_audio_chunks = len(chunks)
            print(f"Total audio feature count: {total_audio_chunks}, total video frame count: {total_frames}")
        
        # Initialize result storage
        metadata_list_all = []  # 存储所有批次的metadata列表
        batch_times = []
        batch_step_times = {}  # Store time for each batch step
        
        # Start stream processing
        print(f"Starting stream processing video: {video_path}")
        print(f"Total frames: {total_frames}, each batch frame count: {context.num_frames}, expected batch count: {total_batches}")
        
        batch_idx = 0
        while True:
            batch_step_times[batch_idx] = {}
            
            # Record batch start time
            batch_start_time = time.time()
            
            # 1. Read a batch of video frames
            read_start = time.time()
            frames, is_video_last_batch = self._read_frame_batch(video_capture, context.num_frames)
            read_end = time.time()
            batch_step_times[batch_idx]["Read frames"] = read_end - read_start
            
            if len(frames) == 0:
                print("Video frames read completed, exiting processing")
                break
                
            print(f"Processing batch {batch_idx+1}/{total_batches}, frame count: {len(frames)}")
            
            # 2. Facial preprocessing (single batch processing)
            preprocess_start = time.time()
            metadata_list = self._preprocess_face_batch(frames)
            preprocess_end = time.time()
            batch_step_times[batch_idx]["Facial preprocessing"] = preprocess_end - preprocess_start
            
            if metadata_list is None:
                print(f"Batch {batch_idx+1} No valid face detected, skipping")
                batch_idx += 1
                continue
                
            # 从metadata中提取处理后的人脸图像，用于后续处理
            faces = torch.stack([metadata.face for metadata in metadata_list])
                
            # Store metadata list for final restoration
            metadata_list_all.append(metadata_list)
            
            # 3. Prepare audio features for the current batch
            audio_start = time.time()
            current_audio_features = self._prepare_audio_batch(
                whisper_feature, batch_idx, len(faces), context
            )
            audio_end = time.time()
            batch_step_times[batch_idx]["Audio feature preparation"] = audio_end - audio_start
            
            # Check if audio features have ended
            audio_last_batch = False
            if whisper_feature is not None:
                start_idx = batch_idx * context.num_frames
                if start_idx >= total_audio_chunks - len(faces):
                    audio_last_batch = True
                    
            # 4. Run diffusion inference on the current batch
            diffusion_start = time.time()
            synced_faces_batch, diffusion_step_times = self._run_diffusion_batch(
                faces, current_audio_features, context
            )
            diffusion_end = time.time()
            batch_step_times[batch_idx]["Diffusion inference"] = diffusion_end - diffusion_start
            
            # Merge diffusion step time records
            for step, step_time in diffusion_step_times.items():
                batch_step_times[batch_idx][step] = step_time
            
            # 更新metadata_list中的face字段，存储处理后的人脸
            for i, metadata in enumerate(metadata_list):
                # 创建一个新的LipsyncMetadata对象，包含原始数据和处理后的人脸
                updated_metadata = LipsyncMetadata(
                    face=synced_faces_batch[i],  # 使用处理后的人脸
                    box=metadata.box,
                    affine_matrice=metadata.affine_matrice,
                    original_frame=metadata.original_frame
                )
                metadata_list[i] = updated_metadata
            
            # Calculate and record batch processing time
            batch_end_time = time.time()
            batch_time = batch_end_time - batch_start_time
            batch_times.append(batch_time)
            
            # Output batch step times
            print(f"Batch {batch_idx+1} processing completed, total time: {batch_time:.2f} seconds")
            print("   Step times:")
            for step, step_time in batch_step_times[batch_idx].items():
                print(f"      {step}: {step_time:.2f} seconds")
            
            batch_idx += 1
            
            # Break if both video and audio are finished
            if is_video_last_batch and audio_last_batch:
                break
        
        # Close video capture
        video_capture.release()
        
        # Restore video frames and save results
        return self._restore_and_save_stream(
            metadata_list_all,
            audio_samples,
            context.video_fps,
            context.audio_sample_rate,
            video_out_path
        )
    
