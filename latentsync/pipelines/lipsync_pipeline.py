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
from latentsync.pipelines.lipsync_diffusion_pipeline import LipsyncMetadata

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
        use_compile: bool = False,
    ):
        super().__init__(vae, unet, scheduler, use_compile)
        self.audio_encoder = audio_encoder
        self.register_modules(audio_encoder=audio_encoder)
        self.set_progress_bar_config(desc="Steps")

    @Timer(name="init_with_context")
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
        metadata_list: Optional[List[LipsyncMetadata]] = self._preprocess_face_batch(frames)
        
        if metadata_list is None:
            print(f"Batch {batch_idx+1} No valid face detected, skipping")
            return None
            
        # 从metadata中提取处理后的人脸图像，用于后续处理
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
            # 创建一个新的LipsyncMetadata对象，包含原始数据和处理后的人脸
            updated_metadata = LipsyncMetadata(
                face=synced_faces_batch[i],  # 使用处理后的人脸
                box=metadata.box,
                affine_matrice=metadata.affine_matrice,
                original_frame=metadata.original_frame
            )
            metadata_list[i] = updated_metadata
            
        return metadata_list
        
    @Timer(name="init_video_stream")
    def _init_video_stream(self, video_path: str) -> tuple[cv2.VideoCapture, int]:
        """Initialize video stream reading"""
        # Open video file
        video_capture = cv2.VideoCapture(video_path)
        if not video_capture.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

        # Get total frame count
        total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

        return video_capture, total_frames

    @Timer(name="read_frame_batch")
    def _read_frame_batch(self, video_capture: cv2.VideoCapture, batch_size: int) -> tuple[List[np.ndarray], bool]:
        """Read a batch of video frames"""
        frames: List[np.ndarray] = []
        is_last_batch = False

        for _ in range(batch_size):
            ret, frame = video_capture.read()
            if not ret:
                is_last_batch = True
                break

            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        return frames, is_last_batch

    @Timer(name="restore_and_save_stream")
    def _restore_and_save_stream(self, metadata_list_all: List[List[LipsyncMetadata]],
                               audio_samples: torch.Tensor, video_fps: int,
                               audio_sample_rate: int, video_out_path: str) -> Optional[str]:
        """Restore processed frames to original video and save"""
        # Check if there are metadata
        if not metadata_list_all:
            print("No successfully processed frames, cannot restore video")
            return None

        print(f"Restoring video: Processed {len(metadata_list_all)} batches of data")

        # 将所有批次的metadata列表合并为一个扁平列表
        flat_metadata_list = []
        for batch_metadata in metadata_list_all:
            flat_metadata_list.extend(batch_metadata)

        # 使用restore_video方法恢复视频
        print(f"Restoring {len(flat_metadata_list)} frames of video...")
        synced_video_frames = self.restore_video(flat_metadata_list)

        # Process audio
        audio_samples_remain_length = int(len(synced_video_frames) / video_fps * audio_sample_rate)
        if audio_samples_remain_length > len(audio_samples):
            print(f"Warning: Calculated audio length ({audio_samples_remain_length}) exceeds available audio samples ({len(audio_samples)}), audio will be truncated")
            audio_samples_remain_length = len(audio_samples)

        audio_samples = audio_samples[:audio_samples_remain_length].cpu().numpy()
        print(f"Processed video length: {len(synced_video_frames)/video_fps:.2f} seconds, audio samples: {len(audio_samples)}")

        # Save results
        temp_dir = "temp"
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)

        write_video(os.path.join(temp_dir, "video.mp4"), synced_video_frames, fps=video_fps)
        sf.write(os.path.join(temp_dir, "audio.wav"), audio_samples, audio_sample_rate)

        command = f"ffmpeg -y -loglevel error -nostdin -i {os.path.join(temp_dir, 'video.mp4')} -i {os.path.join(temp_dir, 'audio.wav')} -c:v libx264 -c:a aac -q:v 0 -q:a 0 {video_out_path}"
        subprocess.run(command, shell=True)

        return video_out_path

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
    
