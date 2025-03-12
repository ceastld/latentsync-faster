# Adapted from https://github.com/guoyww/AnimateDiff/blob/main/animatediff/pipelines/pipeline_animation.py

from dataclasses import dataclass
import inspect
import os
import shutil
from typing import Callable, Dict, List, Optional, Tuple, Union
import subprocess
import time

import numpy as np
import torch
import torchvision
import cv2

from diffusers.utils import is_accelerate_available
from packaging import version

from diffusers.configuration_utils import FrozenDict
from diffusers.models import AutoencoderKL
from diffusers import DiffusionPipeline
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.utils import deprecate, logging

from einops import rearrange

from ..models.unet import UNet3DConditionModel
from ..utils.image_processor import ImageProcessor
from ..utils.util import read_video, read_audio, write_video, check_ffmpeg_installed
from ..whisper.audio2feature import Audio2Feature
import tqdm
import soundfile as sf

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

@dataclass
class InitializedParams:
    batch_size: int
    device: str
    height: int
    width: int
    do_classifier_free_guidance: bool
    num_frames: int
    num_channels_latents: int

@dataclass
class LipsyncMetadata:
    face: torch.Tensor  # 处理后的人脸图像
    box: np.ndarray  # 人脸边界框
    affine_matrice: np.ndarray  # 仿射变换矩阵
    original_frame: np.ndarray  # 原始视频帧

class LipsyncPipeline(DiffusionPipeline):
    _optional_components = []

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
        super().__init__()

        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate("clip_sample not set", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

        is_unet_version_less_0_9_0 = hasattr(unet.config, "_diffusers_version") and version.parse(
            version.parse(unet.config._diffusers_version).base_version
        ) < version.parse("0.9.0.dev0")
        is_unet_sample_size_less_64 = hasattr(unet.config, "sample_size") and unet.config.sample_size < 64
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            deprecation_message = (
                "The configuration file of the unet has set the default `sample_size` to smaller than"
                " 64 which seems highly unlikely. If your checkpoint is a fine-tuned version of any of the"
                " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
                " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
                " in the config might lead to incorrect results in future versions. If you have downloaded this"
                " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
                " the `unet/config.json` file"
            )
            deprecate("sample_size<64", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(unet.config)
            new_config["sample_size"] = 64
            unet._internal_dict = FrozenDict(new_config)

        self.unet  = unet # for code compilation
        self.audio_encoder = audio_encoder # for code compilation
        self.vae = vae # for code compilation
        self.scheduler = scheduler # for code compilation
        if use_compile:
            unet = torch.compile(unet, mode="reduce-overhead", fullgraph=True)
            vae = torch.compile(vae, mode="reduce-overhead", fullgraph=True)

        self.register_modules(
            vae=vae,
            audio_encoder=audio_encoder,
            unet=unet,
            scheduler=scheduler,
        )

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

        self.set_progress_bar_config(desc="Steps")

    def enable_vae_slicing(self):
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        self.vae.disable_slicing()

    def enable_sequential_cpu_offload(self, gpu_id=0):
        if is_accelerate_available():
            from accelerate import cpu_offload
        else:
            raise ImportError("Please install accelerate via `pip install accelerate`")

        device = torch.device(f"cuda:{gpu_id}")

        for cpu_offloaded_model in [self.unet, self.text_encoder, self.vae]:
            if cpu_offloaded_model is not None:
                cpu_offload(cpu_offloaded_model, device)

    @property
    def _execution_device(self):
        if self.device != torch.device("meta") or not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def decode_latents(self, latents):
        latents = latents / self.vae.config.scaling_factor + self.vae.config.shift_factor
        latents = rearrange(latents, "b c f h w -> (b f) c h w")
        decoded_latents = self.vae.decode(latents).sample
        return decoded_latents

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(self, height, width, callback_steps):
        assert height == width, "Height and width must be equal"

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

    def prepare_latents(self, batch_size, num_frames, num_channels_latents, height, width, dtype, device, generator):
        shape = (
            batch_size,
            num_channels_latents,
            1,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        rand_device = "cpu" if device.type == "mps" else device
        latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype).to(device)
        latents = latents.repeat(1, 1, num_frames, 1, 1)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def prepare_mask_latents(
        self, mask, masked_image, height, width, dtype, device, generator, do_classifier_free_guidance
    ):
        # resize the mask to latents shape as we concatenate the mask to the latents
        # we do that before converting to dtype to avoid breaking in case we're using cpu_offload
        # and half precision
        mask = torch.nn.functional.interpolate(
            mask, size=(height // self.vae_scale_factor, width // self.vae_scale_factor)
        )
        masked_image = masked_image.to(device=device, dtype=dtype)

        # encode the mask image into latents space so we can concatenate it to the latents
        # masked_image_latents = self.vae.encode(masked_image).latent_dist.sample(generator=generator)
        masked_image_latents = self.vae.encode(masked_image).latents
        masked_image_latents = (masked_image_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor

        # aligning device to prevent device errors when concating it with the latent model input
        masked_image_latents = masked_image_latents.to(device=device, dtype=dtype)
        mask = mask.to(device=device, dtype=dtype)

        # assume batch size = 1
        mask = rearrange(mask, "f c h w -> 1 c f h w")
        masked_image_latents = rearrange(masked_image_latents, "f c h w -> 1 c f h w")

        mask = torch.cat([mask] * 2) if do_classifier_free_guidance else mask
        masked_image_latents = (
            torch.cat([masked_image_latents] * 2) if do_classifier_free_guidance else masked_image_latents
        )
        return mask, masked_image_latents

    def prepare_image_latents(self, images, device, dtype, generator, do_classifier_free_guidance):
        images = images.to(device=device, dtype=dtype)
        # image_latents = self.vae.encode(images).latent_dist.sample(generator=generator)
        image_latents = self.vae.encode(images).latents
        image_latents = (image_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        image_latents = rearrange(image_latents, "f c h w -> 1 c f h w")
        image_latents = torch.cat([image_latents] * 2) if do_classifier_free_guidance else image_latents

        return image_latents

    def set_progress_bar_config(self, **kwargs):
        if not hasattr(self, "_progress_bar_config"):
            self._progress_bar_config = {}
        self._progress_bar_config.update(kwargs)

    @staticmethod
    def paste_surrounding_pixels_back(decoded_latents, pixel_values, masks, device, weight_dtype):
        # Paste the surrounding pixels back, because we only want to change the mouth region
        pixel_values = pixel_values.to(device=device, dtype=weight_dtype)
        masks = masks.to(device=device, dtype=weight_dtype)
        combined_pixel_values = decoded_latents * masks + pixel_values * (1 - masks)
        return combined_pixel_values

    @staticmethod
    def pixel_values_to_images(pixel_values: torch.Tensor):
        pixel_values = rearrange(pixel_values, "f c h w -> f h w c")
        pixel_values = (pixel_values / 2 + 0.5).clamp(0, 1)
        images = (pixel_values * 255).to(torch.uint8)
        images = images.cpu().numpy()
        return images

    def affine_transform_video(self, video_path):
        """使用LipsyncMetadata处理视频帧"""
        video_frames = read_video(video_path, use_decord=False)
        metadata_list = []
        
        print(f"Affine transforming {len(video_frames)} faces...")
        for frame in tqdm.tqdm(video_frames):
            try:
                face, box, affine_matrix = self.image_processor.affine_transform(frame)
                
                # 创建LipsyncMetadata对象存储处理结果
                metadata = LipsyncMetadata(
                    face=face,
                    box=box,
                    affine_matrice=affine_matrix,
                    original_frame=frame
                )
                metadata_list.append(metadata)
            except Exception as e:
                print(f"Face preprocessing failed: {e}")
                # 如果处理失败且有其他成功处理的帧，使用前一帧的结果
                if len(metadata_list) > 0:
                    metadata_list.append(metadata_list[-1])
                # 否则跳过此帧
        
        if len(metadata_list) == 0:
            return None
        
        return metadata_list

    def restore_video(self, metadata_list):
        """使用LipsyncMetadata恢复视频帧"""
        out_frames = []
        print(f"Restoring {len(metadata_list)} faces...")
        for metadata in tqdm.tqdm(metadata_list):
            metadata: LipsyncMetadata
            x1, y1, x2, y2 = metadata.box
            height = int(y2 - y1)
            width = int(x2 - x1)
            
            # 获取处理后的人脸
            face = metadata.face
            
            # 调整人脸大小
            face = torchvision.transforms.functional.resize(face, size=(height, width), antialias=True)
            face = rearrange(face, "c h w -> h w c")
            face = (face / 2 + 0.5).clamp(0, 1)
            face = (face * 255).to(torch.uint8).cpu().numpy()
            
            # 使用restorer将人脸放回原始帧
            out_frame = self.image_processor.restorer.restore_img(
                metadata.original_frame, 
                face, 
                metadata.affine_matrice
            )
            out_frames.append(out_frame)
            
        return np.stack(out_frames, axis=0)

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
    
    def _init_video_stream(self, video_path: str) -> tuple[cv2.VideoCapture, int]:
        """Initialize video stream reading"""
        # Open video file
        video_capture = cv2.VideoCapture(video_path)
        if not video_capture.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
            
        # Get total frame count
        total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        
        return video_capture, total_frames
    
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
    
    def _preprocess_face(self, frame: np.ndarray) -> Optional[LipsyncMetadata]:
        face, box, affine_matrix = self.image_processor.affine_transform(frame)
        # 创建LipsyncMetadata对象存储处理结果
        return LipsyncMetadata(
            face=face,
            box=box,
            affine_matrice=affine_matrix,
            original_frame=frame
        )
        
    def _preprocess_face_batch(self, frames: List[np.ndarray]) -> Optional[List[LipsyncMetadata]]:
        """Process a batch of frames for facial preprocessing, using the same face alignment logic as the original method"""
        """Core function for face_processor"""
        if len(frames) == 0:
            return None
            
        metadata_list: List[LipsyncMetadata] = []
        
        # Use the same preprocessing logic as original code
        for frame in frames:
            try:
                metadata = self._preprocess_face(frame)
                metadata_list.append(metadata)
            except Exception as e:
                print(f"Face preprocessing failed: {e}")
                # If processing fails and there are other successfully processed frames, use the result of the previous frame
                if len(metadata_list) > 0:
                    metadata_list.append(metadata_list[-1])
                # Otherwise skip this frame
        
        if len(metadata_list) == 0:
            return None
            
        return metadata_list
        
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
        
    def _run_diffusion_batch(self, faces: torch.Tensor, audio_features: Optional[List[torch.Tensor]], 
                          params: InitializedParams, num_inference_steps: int,
                          guidance_scale: float, weight_dtype: torch.dtype, 
                          extra_step_kwargs: dict, generator: Optional[Union[torch.Generator, List[torch.Generator]]], 
                          callback: Optional[Callable[[int, int, torch.FloatTensor], None]], 
                          callback_steps: Optional[int]) -> tuple[torch.Tensor, dict]:
        """Run diffusion inference on a single batch"""
        """Core function for diffusion_processor"""
        step_times: dict = {}  # Record time for each step
        
        # 1. Prepare latent variables
        latents_start = time.time()
        batch_size = 1  # Single batch processing
        num_frames = len(faces)
        num_channels_latents = params.num_channels_latents
        height = params.height
        width = params.width
        device = params.device
        do_classifier_free_guidance = params.do_classifier_free_guidance
        
        latents = self.prepare_latents(
            batch_size,
            num_frames,
            num_channels_latents,
            height,
            width,
            weight_dtype,
            device,
            generator,
        )
        latents_end = time.time()
        step_times["Prepare latent variables"] = latents_end - latents_start
        
        # 2. Set timesteps
        timesteps_start = time.time()
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        timesteps_end = time.time()
        step_times["Set timesteps"] = timesteps_end - timesteps_start
        
        # 3. Prepare audio embeddings
        audio_start = time.time()
        if self.unet.add_audio_layer and audio_features is not None:
            audio_embeds = torch.stack(audio_features)
            audio_embeds = audio_embeds.to(device, dtype=weight_dtype)
            if do_classifier_free_guidance:
                null_audio_embeds = torch.zeros_like(audio_embeds)
                audio_embeds = torch.cat([null_audio_embeds, audio_embeds])
        else:
            audio_embeds = None
        audio_end = time.time()
        step_times["Prepare audio embeddings"] = audio_end - audio_start
        
        # 4. Prepare face masks
        mask_prep_start = time.time()
        pixel_values, masked_pixel_values, masks = self.image_processor.prepare_masks_and_masked_images(
            faces, affine_transform=False
        )
        mask_prep_end = time.time()
        step_times["Prepare face masks"] = mask_prep_end - mask_prep_start
        
        # 5. Prepare mask latent variables
        mask_latents_start = time.time()
        mask_latents, masked_image_latents = self.prepare_mask_latents(
            masks,
            masked_pixel_values,
            height,
            width,
            weight_dtype,
            device,
            generator,
            do_classifier_free_guidance,
        )
        mask_latents_end = time.time()
        step_times["Prepare mask latent variables"] = mask_latents_end - mask_latents_start
        
        # 6. Prepare image latent variables
        image_latents_start = time.time()
        image_latents = self.prepare_image_latents(
            pixel_values,
            device,
            weight_dtype,
            generator,
            do_classifier_free_guidance,
        )
        image_latents_end = time.time()
        step_times["Prepare image latent variables"] = image_latents_end - image_latents_start
        
        # 7. Perform denoising process
        denoising_start = time.time()
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        
        denoising_step_times = []
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for j, t in enumerate(timesteps):
                step_start = time.time()
                
                # Prepare model input
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                latent_model_input = torch.cat(
                    [latent_model_input, mask_latents, masked_image_latents, image_latents], dim=1
                )
                
                # Predict noise residual
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=audio_embeds).sample
                
                # Apply guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_audio = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_audio - noise_pred_uncond)
                
                # Calculate previous noise sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
                
                # Update progress bar and callback
                if j == len(timesteps) - 1 or ((j + 1) > num_warmup_steps and (j + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and j % callback_steps == 0:
                        callback(j, t, latents)
                
                step_end = time.time()
                denoising_step_times.append(step_end - step_start)
        
        denoising_end = time.time()
        step_times["Denoising process"] = denoising_end - denoising_start
        step_times["Average denoising per step"] = sum(denoising_step_times) / len(denoising_step_times) if denoising_step_times else 0
        
        # 8. Decode and post-process
        decode_start = time.time()
        decoded_latents = self.decode_latents(latents)
        decoded_latents = self.paste_surrounding_pixels_back(
            decoded_latents, pixel_values, 1 - masks, device, weight_dtype
        )
        decode_end = time.time()
        step_times["Decode and post-process"] = decode_end - decode_start
        
        return decoded_latents, step_times
        
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
        
    def _initialize_parameters(self, num_frames, height, width, mask, guidance_scale, callback_steps):
        """Initialize and validate parameters needed for inference"""
        # Simplified version, may need to check other conditions in actual use
        num_frames = 8  # Hardcoded value from original code
        batch_size = 1
        device = self._execution_device
        self.image_processor = ImageProcessor(height, mask=mask, device="cuda")
        self.set_progress_bar_config(desc=f"Sample frames: {num_frames}")
        
        # Set height and width
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        
        # Check inputs
        self.check_inputs(height, width, callback_steps)
        
        # Determine whether to use classifier-free guidance
        do_classifier_free_guidance = guidance_scale > 1.0
        
        # Return prepared parameters using InitializedParams dataclass
        num_channels_latents = self.vae.config.latent_channels
        
        return InitializedParams(
            batch_size=batch_size,
            device=device,
            height=height,
            width=width,
            do_classifier_free_guidance=do_classifier_free_guidance,
            num_frames=num_frames,
            num_channels_latents=num_channels_latents,
        )
