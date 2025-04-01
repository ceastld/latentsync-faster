from __future__ import annotations
from functools import cached_property
from latentsync.inference.context import LipsyncContext, LipsyncContext_v15
from latentsync.models.unet import UNet3DConditionModel
from latentsync.pipelines.metadata import LipsyncMetadata
from latentsync.utils.face_processor import FaceProcessor
from latentsync.utils.affine_transform import AlignRestore
from latentsync.utils.image_processor import ImageProcessor
from latentsync.utils.util import read_video, write_video
import cv2
import numpy as np
import soundfile as sf
import torch
import tqdm
from diffusers import DiffusionPipeline
from diffusers.configuration_utils import FrozenDict
from diffusers.models import AutoencoderKL, AutoencoderTiny
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
from latentsync.utils.timer import Timer


class LipsyncDiffusionPipeline(DiffusionPipeline):
    _optional_components = []
    def __init__(
        self,
        vae: Union[AutoencoderTiny, AutoencoderKL],
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
        self.lipsync_context = lipsync_context

        vae = torch.compile(vae, mode="reduce-overhead", fullgraph=True)
        if lipsync_context.use_compile:
            unet = torch.compile(unet, mode="reduce-overhead", fullgraph=True)

        self.register_modules(
            vae=vae,
            unet=unet,
            scheduler=scheduler,
            lipsync_context=lipsync_context,
        )

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

        self.init_with_context(lipsync_context)

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

    # @Timer()
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

    # @Timer()
    def prepare_latents(self, context: LipsyncContext, num_frames: int):
        """Prepare latent variables for diffusion"""
        shape = (
            context.batch_size,
            context.num_channels_latents,
            1,
            context.height // self.vae_scale_factor,
            context.width // self.vae_scale_factor,
        )
        # rand_device = "cpu" if context.device.type == "mps" else context.device
        latents = torch.randn(shape, generator=context.generator, device=context.device, dtype=context.weight_dtype).to(context.device)
        latents = latents.repeat(1, 1, num_frames, 1, 1)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    @Timer()
    def prepare_mask_latents(self, context: LipsyncContext, mask: torch.Tensor, masked_image: torch.Tensor):
        """Prepare mask latent variables"""
        # resize the mask to latents shape as we concatenate the mask to the latents
        # we do that before converting to dtype to avoid breaking in case we're using cpu_offload
        # and half precision
        mask = torch.nn.functional.interpolate(
            mask, size=(context.height // self.vae_scale_factor, context.width // self.vae_scale_factor)
        )

        # encode the mask image into latents space so we can concatenate it to the latents
        masked_image_latents = self.get_vae_latents(masked_image)

        # assume batch size = 1
        mask = rearrange(mask, "f c h w -> 1 c f h w")
        masked_image_latents = rearrange(masked_image_latents, "f c h w -> 1 c f h w")

        mask = torch.cat([mask] * 2) if context.do_classifier_free_guidance else mask
        masked_image_latents = (
            torch.cat([masked_image_latents] * 2) if context.do_classifier_free_guidance else masked_image_latents
        )
        return mask, masked_image_latents
    
    @Timer()
    def get_vae_latents(self, images):
        if self.lipsync_context.vae_type == "kl":
            masked_image_latents = self.vae.encode(images).latent_dist.sample(generator=self.lipsync_context.generator)
            masked_image_latents = (masked_image_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor
            return masked_image_latents # AutoencoderKL
        elif self.lipsync_context.vae_type == "tiny":
            return self.vae.encode(images).latents # AutoencoderTiny
        else:
            raise ValueError(f"Unsupported VAE type: {self.lipsync_context.vae_type}")

    # @Timer()
    def prepare_image_latents(self, context: LipsyncContext, images: torch.Tensor):
        """Prepare image latent variables"""
        images = images.to(device=context.device, dtype=context.weight_dtype)
        image_latents = self.get_vae_latents(images)
        image_latents = rearrange(image_latents, "f c h w -> 1 c f h w")
        image_latents = torch.cat([image_latents] * 2) if context.do_classifier_free_guidance else image_latents
        return image_latents

    @staticmethod
    def pixel_values_to_images(pixel_values: torch.Tensor):
        pixel_values = rearrange(pixel_values, "f c h w -> f h w c")
        pixel_values = (pixel_values / 2 + 0.5).clamp(0, 1)
        images = (pixel_values * 255).to(torch.uint8)
        images = images.cpu().numpy()
        return images
    
    @cached_property
    def restorer(self):
        return AlignRestore()

    def restore_video(self, metadata_list: List[LipsyncMetadata]):
        """使用LipsyncMetadata恢复视频帧"""
        out_frames = []
        
        for metadata in tqdm.tqdm(metadata_list, desc="Restoring video"):
            out_frame = self.restorer.restore_img(
                metadata.original_frame,
                metadata.sync_face,
                metadata.affine_matrix
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
    
    # @Timer()
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

    @cached_property
    def face_processor(self):
        return self.lipsync_context.create_face_processor()

    @Timer()
    def _denoising_step(self, latents, t, audio_embeds, mask_latents, masked_image_latents, image_latents, context: LipsyncContext):
        """Execute a single denoising step"""
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
        
        return latents

    # @Timer()
    def _run_diffusion_batch(self, faces: torch.Tensor, audio_features: Optional[List[torch.Tensor]],
                          context: LipsyncContext) -> tuple[torch.Tensor, dict]:
        """Run diffusion inference on a single batch"""
        """Core function for diffusion_processor"""
        
        # 1. Prepare latent variables
        num_frames = len(faces)
        faces = faces.to(context.device)
        latents = self.prepare_latents(context, num_frames)

        # 2. Set timesteps
        self.scheduler.set_timesteps(context.num_inference_steps, device=context.device)
        timesteps = self.scheduler.timesteps

        # 3. Prepare audio embeddings
        if self.unet.add_audio_layer and audio_features is not None:
            audio_embeds = torch.stack(audio_features)
            audio_embeds = audio_embeds.to(context.device, dtype=context.weight_dtype)
            if context.do_classifier_free_guidance:
                null_audio_embeds = torch.zeros_like(audio_embeds)
                audio_embeds = torch.cat([null_audio_embeds, audio_embeds])
        else:
            audio_embeds = None

        # 4. Prepare face masks
        pixel_values, masked_pixel_values, masks = self.image_processor.prepare_masks_and_masked_images(faces)
        pixel_values = pixel_values.to(context.device, dtype=context.weight_dtype)
        masked_pixel_values = masked_pixel_values.to(context.device, dtype=context.weight_dtype)
        masks = masks.to(context.device, dtype=context.weight_dtype)
        
        # 5. Prepare mask latent variables
        mask_latents, masked_image_latents = self.prepare_mask_latents(
            context,
            masks,
            masked_pixel_values,
        )

        # 6. Prepare image latent variables
        image_latents = self.prepare_image_latents(
            context,
            pixel_values,
        )

        # 7. Perform denoising process
        num_warmup_steps = len(timesteps) - context.num_inference_steps * self.scheduler.order

        # with self.progress_bar(total=context.num_inference_steps) as progress_bar:
        for j, t in enumerate(timesteps):
            # Execute denoising step
            latents = self._denoising_step(
                latents, t, audio_embeds, mask_latents, 
                masked_image_latents, image_latents, context
            )

            # Update progress bar and callback
            if j == len(timesteps) - 1 or ((j + 1) > num_warmup_steps and (j + 1) % self.scheduler.order == 0):
                # progress_bar.update()
                if context.callback is not None and j % context.callback_steps == 0:
                    context.callback(j, t, latents)

        # 8. Decode latents
        decoded_latents = self.decode_latents(latents)

        # 9. Post-process (paste surrounding pixels back)
        decoded_latents = decoded_latents * (1 - masks) + pixel_values * masks

        return decoded_latents, {}
    
    # @Timer()
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
    
    @cached_property
    def image_processor(self):
        return ImageProcessor(resolution=self.lipsync_context.resolution, device=self.lipsync_context.device)

    def init_with_context(self, context: LipsyncContext):
        """Initialize and validate parameters needed for inference"""
        # Get device
        device = self._execution_device

        # Set height and width
        context.height = context.height or self.unet.config.sample_size * self.vae_scale_factor
        context.width = context.width or self.unet.config.sample_size * self.vae_scale_factor

        # Check inputs
        self.check_inputs(context.height, context.width, context.callback_steps)

        # Update context with device and num_channels_latents
        context.device = device
        context.num_channels_latents = self.vae.config.latent_channels
        
        # Set video fps
        self.video_fps = context.video_fps

        # Prepare extra step kwargs
        context.extra_step_kwargs = self.prepare_extra_step_kwargs(
            context.generator, context.eta
        )

        return self