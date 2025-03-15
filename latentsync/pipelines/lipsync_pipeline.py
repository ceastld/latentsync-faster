# Adapted from https://github.com/guoyww/AnimateDiff/blob/main/animatediff/pipelines/pipeline_animation.py

from typing import Callable, List, Optional, Tuple, Union
import time
import os
import shutil
import subprocess

import torch
import cv2
import numpy as np
import soundfile as sf
from tqdm import tqdm


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
from latentsync.pipelines.metadata import LipsyncMetadata

from ..models.unet import UNet3DConditionModel
from ..utils.util import read_audio, check_ffmpeg_installed, write_video
from ..whisper.audio2feature import Audio2Feature
from ..utils.timer import Timer

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
        lipsync_context: LipsyncContext,
    ):
        super().__init__(vae=vae, unet=unet, scheduler=scheduler, lipsync_context=lipsync_context)
        self.audio_encoder = audio_encoder
        self.register_modules(audio_encoder=audio_encoder)

    @Timer(name="prepare_audio_batch")
    def _prepare_audio_batch(self, whisper_feature: Optional[torch.Tensor], batch_idx: int, 
                           num_frames_in_batch: int, context: LipsyncContext) -> Optional[List[torch.Tensor]]:
        """Prepare audio features for the current batch"""
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

    @Timer(name="process_batch")
    def _process_batch(self, frames: List[np.ndarray], whisper_feature: Optional[torch.Tensor], 
                      batch_idx: int, context: LipsyncContext) -> Optional[List[LipsyncMetadata]]:
        """Process a single batch of frames"""
        # 1. Facial preprocessing
        metadata_list: Optional[List[LipsyncMetadata]] = self.face_processor.prepare_face_batch(frames)
        
        if metadata_list is None:
            print(f"Batch {batch_idx+1} No valid face detected, skipping")
            return None
            
        faces = torch.stack([metadata.face for metadata in metadata_list])
        
        # 2. Prepare audio features for the current batch
        current_audio_features = self._prepare_audio_batch(
            whisper_feature, batch_idx, len(faces), context
        )
        
        # 3. Run diffusion inference on the current batch
        synced_faces_batch, _ = self._run_diffusion_batch(
            faces, current_audio_features, context
        )
        
        # 4. 更新metadata_list中的face字段，存储处理后的人脸
        for i, metadata in enumerate(metadata_list):
            metadata.set_sync_face(synced_faces_batch[i])
            
        return metadata_list
        

    @torch.no_grad()
    def __call__(
        self,
        video_path: str,
        audio_path: str,
        video_out_path: str,
    ):
        """Execute lip sync inference, using stream processing for video frames"""
        context = self.lipsync_context
        
        # Prepare audio features (process all audio at once)
        audio_samples = read_audio(audio_path)
        whisper_feature = self.audio_encoder.audio2feat(audio_path) if self.unet.add_audio_layer else None
        
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
        
        # Start stream processing
        print(f"Starting stream processing video: {video_path}")
        print(f"Total frames: {total_frames}, each batch frame count: {context.num_frames}, expected batch count: {total_batches}")
        
        batch_idx = 0
        # 使用tqdm创建进度条
        progress_bar = tqdm(total=total_frames, desc="Processing frames batch", unit="frame")
        while True:
            # 更新进度条描述，显示当前处理的批次
            progress_bar.set_description(f"Processing batch {batch_idx+1}/{total_batches}")
            
            # Read a batch of video frames
            frames, is_video_last_batch = self._read_frame_batch(video_capture, context.num_frames)
            
            if len(frames) == 0:
                print("Video frames read completed, exiting processing")
                break
                
            # Process the batch
            metadata_list = self._process_batch(frames, whisper_feature, batch_idx, context)
            if metadata_list is not None:
                metadata_list_all.append(metadata_list)
            
            # Check if audio features have ended
            audio_last_batch = False
            if whisper_feature is not None:
                start_idx = batch_idx * context.num_frames
                if start_idx >= total_audio_chunks - len(frames):
                    audio_last_batch = True
            
            batch_idx += 1
            # 更新进度条
            progress_bar.update(len(frames))
            
            # Break if both video and audio are finished
            if is_video_last_batch and audio_last_batch:
                break
        
        # 关闭进度条
        progress_bar.close()
        
        # Close video capture
        video_capture.release()
        
        # Print timing statistics
        Timer.print_stats()
        
        # Restore video frames and save results
        return self._restore_and_save_stream(
            metadata_list_all,
            audio_samples,
            context.video_fps,
            context.audio_sample_rate,
            video_out_path
        )
    
