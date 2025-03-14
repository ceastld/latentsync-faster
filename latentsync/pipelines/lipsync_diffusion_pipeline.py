from dataclasses import dataclass
from latentsync.inference.context import LipsyncContext
from latentsync.models.unet import UNet3DConditionModel
from latentsync.utils.image_processor import ImageProcessor
from latentsync.utils.util import read_video, write_video
import cv2
import numpy as np
import soundfile as sf
import torch
import torchvision
import tqdm
from diffusers import DiffusionPipeline
from diffusers.configuration_utils import FrozenDict
from diffusers.models import AutoencoderKL
from diffusers.schedulers import DDIMScheduler, DPMSolverMultistepScheduler, EulerAncestralDiscreteScheduler, EulerDiscreteScheduler, LMSDiscreteScheduler, PNDMScheduler
from diffusers.utils import deprecate, is_accelerate_available
from einops import rearrange
from packaging import version
import inspect
import os
import shutil
import subprocess
import time
from typing import Callable, List, Optional, Union


@dataclass
class LipsyncMetadata:
    face: torch.Tensor  # 处理后的人脸图像
    box: np.ndarray  # 人脸边界框
    affine_matrice: np.ndarray  # 仿射变换矩阵
    original_frame: np.ndarray  # 原始视频帧


class LipsyncDiffusionPipeline(DiffusionPipeline):
    _optional_components = []
    def __init__(
        self,
        vae: AutoencoderKL,
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
        # self.audio_encoder = audio_encoder # for code compilation
        self.vae = vae # for code compilation
        self.scheduler = scheduler # for code compilation
        if use_compile:
            vae = torch.compile(vae, mode="reduce-overhead", fullgraph=True)
            unet = torch.compile(unet, mode="reduce-overhead", fullgraph=True)

        self.register_modules(
            vae=vae,
            unet=unet,
            scheduler=scheduler,
        )

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

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
        t0 = time.time()
        latents = latents / self.vae.config.scaling_factor + self.vae.config.shift_factor
        t1 = time.time()
        latents = rearrange(latents, "b c f h w -> (b f) c h w")
        t2 = time.time()
        decoded_latents = self.vae.decode(latents).sample
        t3 = time.time()
        
        print(f"[Decode latents] Scaling: {(t1-t0)*1000:.2f}ms, Rearrange: {(t2-t1)*1000:.2f}ms, VAE decode: {(t3-t2)*1000:.2f}ms")
        
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

    def prepare_latents(self, context: LipsyncContext, num_frames: int):
        """Prepare latent variables for diffusion"""
        shape = (
            context.batch_size,
            context.num_channels_latents,
            1,
            context.height // self.vae_scale_factor,
            context.width // self.vae_scale_factor,
        )
        rand_device = "cpu" if context.device.type == "mps" else context.device
        latents = torch.randn(shape, generator=context.generator, device=rand_device, dtype=context.weight_dtype).to(context.device)
        latents = latents.repeat(1, 1, num_frames, 1, 1)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def prepare_mask_latents(self, context: LipsyncContext, mask: torch.Tensor, masked_image: torch.Tensor):
        """Prepare mask latent variables"""
        # resize the mask to latents shape as we concatenate the mask to the latents
        # we do that before converting to dtype to avoid breaking in case we're using cpu_offload
        # and half precision
        mask = torch.nn.functional.interpolate(
            mask, size=(context.height // self.vae_scale_factor, context.width // self.vae_scale_factor)
        )
        masked_image = masked_image.to(device=context.device, dtype=context.weight_dtype)

        # encode the mask image into latents space so we can concatenate it to the latents
        masked_image_latents = self.vae.encode(masked_image).latents
        masked_image_latents = (masked_image_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor

        # aligning device to prevent device errors when concating it with the latent model input
        masked_image_latents = masked_image_latents.to(device=context.device, dtype=context.weight_dtype)
        mask = mask.to(device=context.device, dtype=context.weight_dtype)

        # assume batch size = 1
        mask = rearrange(mask, "f c h w -> 1 c f h w")
        masked_image_latents = rearrange(masked_image_latents, "f c h w -> 1 c f h w")

        mask = torch.cat([mask] * 2) if context.do_classifier_free_guidance else mask
        masked_image_latents = (
            torch.cat([masked_image_latents] * 2) if context.do_classifier_free_guidance else masked_image_latents
        )
        return mask, masked_image_latents

    def prepare_image_latents(self, context: LipsyncContext, images: torch.Tensor):
        """Prepare image latent variables"""
        images = images.to(device=context.device, dtype=context.weight_dtype)
        image_latents = self.vae.encode(images).latents
        image_latents = (image_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        image_latents = rearrange(image_latents, "f c h w -> 1 c f h w")
        image_latents = torch.cat([image_latents] * 2) if context.do_classifier_free_guidance else image_latents

        return image_latents

    def set_progress_bar_config(self, **kwargs):
        if not hasattr(self, "_progress_bar_config"):
            self._progress_bar_config = {}
        self._progress_bar_config.update(kwargs)

    @staticmethod
    def paste_surrounding_pixels_back(decoded_latents, pixel_values, masks, device, weight_dtype):
        # Paste the surrounding pixels back, because we only want to change the mouth region
        t0 = time.time()
        pixel_values = pixel_values.to(device=device, dtype=weight_dtype)
        t1 = time.time()
        masks = masks.to(device=device, dtype=weight_dtype)
        t2 = time.time()
        combined_pixel_values = decoded_latents * masks + pixel_values * (1 - masks)
        t3 = time.time()
        
        print(f"[Paste pixels] Move pixel values to device: {(t1-t0)*1000:.2f}ms, Move masks to device: {(t2-t1)*1000:.2f}ms, Combine pixels: {(t3-t2)*1000:.2f}ms")
        
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


    def _run_diffusion_batch(self, faces: torch.Tensor, audio_features: Optional[List[torch.Tensor]],
                          context: LipsyncContext) -> tuple[torch.Tensor, dict]:
        """Run diffusion inference on a single batch"""
        """Core function for diffusion_processor"""
        step_times: dict = {}  # Record time for each step

        # 1. Prepare latent variables
        latents_start = time.time()
        num_frames = len(faces)

        latents = self.prepare_latents(context, num_frames)
        latents_end = time.time()
        step_times["Prepare latent variables"] = latents_end - latents_start

        # 2. Set timesteps
        timesteps_start = time.time()
        self.scheduler.set_timesteps(context.num_inference_steps, device=context.device)
        timesteps = self.scheduler.timesteps
        timesteps_end = time.time()
        step_times["Set timesteps"] = timesteps_end - timesteps_start

        # 3. Prepare audio embeddings
        audio_start = time.time()
        if self.unet.add_audio_layer and audio_features is not None:
            audio_embeds = torch.stack(audio_features)
            audio_embeds = audio_embeds.to(context.device, dtype=context.weight_dtype)
            if context.do_classifier_free_guidance:
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
            context,
            masks,
            masked_pixel_values,
        )
        mask_latents_end = time.time()
        step_times["Prepare mask latent variables"] = mask_latents_end - mask_latents_start

        # 6. Prepare image latent variables
        image_latents_start = time.time()
        image_latents = self.prepare_image_latents(
            context,
            pixel_values,
        )
        image_latents_end = time.time()
        step_times["Prepare image latent variables"] = image_latents_end - image_latents_start

        # 7. Perform denoising process
        denoising_start = time.time()
        num_warmup_steps = len(timesteps) - context.num_inference_steps * self.scheduler.order

        denoising_step_times = []
        with self.progress_bar(total=context.num_inference_steps) as progress_bar:
            for j, t in enumerate(timesteps):
                step_start = time.time()

                # Prepare model input
                latent_model_input = torch.cat([latents] * 2) if context.do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                latent_model_input = torch.cat(
                    [latent_model_input, mask_latents, masked_image_latents, image_latents], dim=1
                )

                # Predict noise residual
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=audio_embeds).sample

                # Apply guidance
                if context.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_audio = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + context.guidance_scale * (noise_pred_audio - noise_pred_uncond)

                # Calculate previous noise sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **context.extra_step_kwargs).prev_sample

                # Update progress bar and callback
                if j == len(timesteps) - 1 or ((j + 1) > num_warmup_steps and (j + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if context.callback is not None and j % context.callback_steps == 0:
                        context.callback(j, t, latents)

                step_end = time.time()
                denoising_step_times.append(step_end - step_start)

        denoising_end = time.time()
        step_times["Denoising process"] = denoising_end - denoising_start
        step_times["Average denoising per step"] = sum(denoising_step_times) / len(denoising_step_times) if denoising_step_times else 0

        # 8. Decode latents
        decode_start = time.time()
        decoded_latents = self.decode_latents(latents)
        decode_end = time.time()
        decode_time = decode_end - decode_start
        step_times["Decode latents"] = decode_time
        print(f"[Timing] Total decode latents time: {decode_time*1000:.2f}ms")

        # 9. Post-process (paste surrounding pixels back)
        postprocess_start = time.time()
        decoded_latents = self.paste_surrounding_pixels_back(
            decoded_latents, pixel_values, 1 - masks, context.device, context.weight_dtype
        )
        postprocess_end = time.time()
        postprocess_time = postprocess_end - postprocess_start
        step_times["Paste surrounding pixels"] = postprocess_time
        print(f"[Timing] Total paste surrounding pixels time: {postprocess_time*1000:.2f}ms")

        # Total decode and post-process time (for backward compatibility)
        total_time = decode_time + postprocess_time
        step_times["Decode and post-process"] = total_time
        print(f"[Timing] Combined decode and post-process time: {total_time*1000:.2f}ms")

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

    def init_with_context(self, context: LipsyncContext):
        """Initialize and validate parameters needed for inference"""
        # Get device
        device = self._execution_device
        
        # Initialize image processor
        self.image_processor = ImageProcessor(context.height, mask=context.mask, device=device)

        # Set height and width
        context.height = context.height or self.unet.config.sample_size * self.vae_scale_factor
        context.width = context.width or self.unet.config.sample_size * self.vae_scale_factor

        # Check inputs
        self.check_inputs(context.height, context.width, context.callback_steps)

        # Update context with device and num_channels_latents
        context.device = device
        context.num_channels_latents = self.vae.config.latent_channels
        
        return self