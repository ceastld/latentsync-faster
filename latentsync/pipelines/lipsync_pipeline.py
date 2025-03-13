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


from latentsync.pipelines.lipsync_diffusion_pipeline import InitializedParams
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

    def _initialize_parameters(self, num_frames, height, width, mask, guidance_scale, callback_steps):
        self.set_progress_bar_config(desc=f"Sample frames: {num_frames}")
        return super()._initialize_parameters(num_frames, height, width, mask, guidance_scale, callback_steps)
    
    def _prepare_audio_batch(self, whisper_feature: Optional[torch.Tensor], batch_idx: int, 
                           num_frames_in_batch: int, params: InitializedParams) -> Optional[List[torch.Tensor]]:
        """Prepare audio features for the current batch"""
        """NOT the core function for audio_processor. Need further work for stream processing"""
        if not self.unet.add_audio_layer or whisper_feature is None:
            return None
            
        # Split audio features in batches of size num_frames
        start_idx = batch_idx * params.num_frames
        
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
                                      torch.zeros((1, params.num_channels_latents)))
        
        return selected_chunks
        
    @torch.no_grad()
    def __call__(
        self,
        video_path: str,
        audio_path: str,
        video_out_path: str,
        video_mask_path: str = None,
        num_frames: int = 8,
        video_fps: int = 25,
        audio_sample_rate: int = 16000,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 3,
        guidance_scale: float = 1.5,
        weight_dtype: Optional[torch.dtype] = torch.float16,
        eta: float = 0.0,
        mask: str = "fix_mask",
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        **kwargs,
    ):
        """Execute lip sync inference, using stream processing for video frames"""
        # Save model training state and set to evaluation mode
        is_train = self.unet.training
        self.unet.eval()

        # Initialize parameters
        check_ffmpeg_installed()
        params = self._initialize_parameters(num_frames, height, width, mask, guidance_scale, callback_steps)
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        self.video_fps = video_fps
        
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
        total_batches = (total_frames + num_frames - 1) // num_frames
        
        # Calculate total audio feature count (if any)
        total_audio_chunks = 0
        if whisper_feature is not None:
            chunks = self.audio_encoder.feature2chunks(feature_array=whisper_feature, fps=video_fps)
            total_audio_chunks = len(chunks)
            print(f"Total audio feature count: {total_audio_chunks}, total video frame count: {total_frames}")
        
        # Initialize result storage
        metadata_list_all = []  # 存储所有批次的metadata列表
        batch_times = []
        batch_step_times = {}  # Store time for each batch step
        
        # Start stream processing
        print(f"Starting stream processing video: {video_path}")
        print(f"Total frames: {total_frames}, each batch frame count: {num_frames}, expected batch count: {total_batches}")
        
        batch_idx = 0
        while True:
            batch_step_times[batch_idx] = {}
            
            # Record batch start time
            batch_start_time = time.time()
            
            # 1. Read a batch of video frames
            read_start = time.time()
            frames, is_video_last_batch = self._read_frame_batch(video_capture, num_frames)
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
                whisper_feature, batch_idx, len(faces), params
            )
            audio_end = time.time()
            batch_step_times[batch_idx]["Audio feature preparation"] = audio_end - audio_start
            
            # Check if audio features have ended
            audio_last_batch = False
            if whisper_feature is not None:
                start_idx = batch_idx * params.num_frames
                if start_idx >= total_audio_chunks - len(faces):
                    audio_last_batch = True
                    
            # 4. Run diffusion inference on the current batch
            diffusion_start = time.time()
            synced_faces_batch, diffusion_step_times = self._run_diffusion_batch(
                faces, current_audio_features, params, num_inference_steps,
                guidance_scale, weight_dtype, extra_step_kwargs, generator, callback, callback_steps
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
            # print(f"Batch {batch_idx+1} processing completed, total time: {batch_time:.2f} seconds")
            # print("   Step times:")
            # for step, step_time in batch_step_times[batch_idx].items():
            #     print(f"  - {step}: {step_time:.2f} seconds")
            
            batch_idx += 1
            
            # If video or audio reaches the last batch, exit loop
            if is_video_last_batch or audio_last_batch:
                if is_video_last_batch:
                    print("Video frames processing completed, exiting processing")
                if audio_last_batch:
                    print("Audio feature processing completed, exiting processing")
                break
                
        # Close video stream
        video_capture.release()
        
        if len(metadata_list_all) == 0:
            print("No valid frames generated, check if video contains recognizable faces")
            return None
        
        # Output average batch processing time
        if batch_times:
            avg_batch_time = sum(batch_times) / len(batch_times)
            print(f"Average each batch processing time: {avg_batch_time:.2f} seconds")
            print(f"Batch processing time details: {[f'{t:.2f}s' for t in batch_times]}")
            
        # 4. Video restoration and saving
        print("Starting video restoration and saving...")
        restore_start_time = time.time()
        output_video = self._restore_and_save_stream(
            metadata_list_all,
            audio_samples, video_fps, audio_sample_rate, video_out_path
        )
        restore_end_time = time.time()
        print(f"Video restoration and saving completed, time: {restore_end_time - restore_start_time:.2f} seconds")
        
        # Calculate total time
        total_end_time = time.time()
        total_time = total_end_time - total_start_time
        print(f"Entire processing process completed, total time: {total_time:.2f} seconds")
        
        # Restore model state
        if is_train:
            self.unet.train()
            
        return output_video
    
