from accelerate.utils import set_seed as acc_seed
from latentsync.configs.config import CHECKPOINT_DIR, LipsyncConfig, LipsyncConfig_v15
import torch
from dataclasses import dataclass
from typing import Callable, List, Optional, Union
from omegaconf import OmegaConf
from diffusers import AutoencoderTiny, AutoencoderKL, DPMSolverMultistepScheduler
from diffusers.utils.import_utils import is_xformers_available
from latentsync.models.unet import UNet3DConditionModel
from latentsync.utils.face_processor import FaceProcessor
from latentsync.whisper.audio2feature import Audio2Feature
from latentsync.models_v15.unet import UNet3DConditionModel as UNet3DConditionModel_v15


def set_seed(seed: int):
    if seed != -1:
        acc_seed(seed)
    else:
        torch.seed()
        print(f"Initial seed: {torch.initial_seed()}")


class LipsyncContext:
    def __init__(
        self,
        # Basic parameters
        audio_sample_rate: int = None,
        video_fps: int = None,
        num_frames: int = None,
        audio_batch_size: int = None,
        height: int = None,
        width: int = None,
        resolution: int = None,
        samples_per_frame: int = None,
        # Inference parameters
        num_inference_steps: int = None,
        guidance_scale: float = None,
        eta: float = None,
        # Optional parameters
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        # Runtime parameters
        device: str = None,
        weight_dtype: torch.dtype = None,
        batch_size: int = 1,
        do_classifier_free_guidance: bool = None,
        num_channels_latents: int = None,
        extra_step_kwargs: dict = None,
        # Model selection flags
        use_compile: bool = False,
        use_onnx: bool = False,
        use_trt: bool = False,
        seed: int = None,
        # VAE selection
        vae_type: Optional[str] = None,
        checkpoint_dir: str = None,
    ):
        checkpoint_dir = checkpoint_dir or CHECKPOINT_DIR
        self.config = config = self.get_config(checkpoint_dir)

        # Basic parameters
        self.audio_sample_rate = audio_sample_rate or config.audio_sample_rate
        self.video_fps = video_fps or config.video_fps
        self.num_frames = num_frames or config.num_frames
        self.audio_batch_size = audio_batch_size or config.audio_batch_size
        self.height = height or config.height
        self.width = width or config.width
        self.resolution = resolution or self.width
        self.samples_per_frame = samples_per_frame or config.samples_per_frame

        # Inference parameters
        self.num_inference_steps = num_inference_steps or config.num_inference_steps
        self.guidance_scale = guidance_scale or config.guidance_scale
        self.eta = eta or config.eta

        # Optional parameters
        self.generator = generator
        self.callback = callback
        self.callback_steps = callback_steps

        # Runtime parameters
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.weight_dtype = weight_dtype or config.weight_dtype
        self.batch_size = batch_size
        self.do_classifier_free_guidance = do_classifier_free_guidance
        self.num_channels_latents = num_channels_latents
        self.extra_step_kwargs = extra_step_kwargs

        # Model selection flags
        self.use_compile = use_compile
        self.use_onnx = use_onnx
        self.use_trt = use_trt
        self.seed = seed or config.seed

        # VAE selection
        self.vae_type = (vae_type or config.vae_type).lower()
        if self.vae_type not in ["tiny", "kl"]:
            raise ValueError("vae_type must be either 'tiny' or 'kl'")

        # Post initialization
        self._post_init()

    def get_config(self, checkpoint_dir: str) -> LipsyncConfig:
        return LipsyncConfig(checkpoint_dir)

    def _post_init(self):
        # Set do_classifier_free_guidance based on guidance_scale
        self.do_classifier_free_guidance = self.guidance_scale > 1.0

        # Ensure only one model type is selected
        if sum([self.use_compile, self.use_onnx, self.use_trt]) > 1:
            raise ValueError("Only one of use_compile, use_onnx, or use_trt can be True at the same time")

        if self.seed is not None:
            set_seed(self.seed)

    def create_audio_encoder(self) -> Audio2Feature:
        """Create audio encoder for processing audio samples"""
        return Audio2Feature(
            model_path=self.config.whisper_model_path,
            device=self.device,
            num_frames=self.num_frames,
        )

    def create_unet(self) -> UNet3DConditionModel:
        """Create UNet model for diffusion"""
        if self.use_trt:
            return self.create_unet_trt()
        elif self.use_onnx:
            return self.create_unet_onnx()
        else:
            unet, _ = UNet3DConditionModel.from_pretrained(
                OmegaConf.to_container(self.config.unet_config.model),
                self.config.latentsync_unet_path,
                device=self.device,
            )
            if is_xformers_available():
                unet.enable_xformers_memory_efficient_attention()
            return unet.eval().to(dtype=self.weight_dtype)

    def create_unet_onnx(self) -> UNet3DConditionModel:
        """Create ONNX UNet model for diffusion with same interface as PyTorch UNet"""
        import os
        from latentsync.models.onnx_wrapper import ONNXModelWrapper

        # Build ONNX model path - use same name but with .onnx suffix
        onnx_path = os.path.join(os.path.dirname(self.config.latentsync_unet_path_v15), "unet.onnx")

        # Check if ONNX model exists
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"ONNX model not found at {onnx_path}. Please export the model first.")

        # Create ONNX model wrapper
        unet = ONNXModelWrapper(onnx_path, device=self.device)

        # Set same dtype attribute as PyTorch model (only for compiler, doesn't affect runtime)
        unet.dtype = self.weight_dtype

        return unet

    def create_unet_trt(self) -> UNet3DConditionModel:
        """Create TensorRT UNet model for diffusion with same interface as PyTorch UNet"""
        import os
        from latentsync.models.trt_wrapper import TRTModelWrapper

        # Build TensorRT engine path - use same name but with .engine suffix
        engine_path = os.path.join(os.path.dirname(self.config.latentsync_unet_path_v15), "latentsync_unet.engine")

        # Check if TensorRT engine exists
        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"TensorRT engine not found at {engine_path}. Please export the model first.")

        # Create TensorRT model wrapper
        unet = TRTModelWrapper(engine_path, device=self.device)

        # Set same dtype attribute as PyTorch model (only for compiler, doesn't affect runtime)
        unet.dtype = self.weight_dtype

        return unet

    def create_vae(self) -> Union[AutoencoderTiny, AutoencoderKL]:
        """Create VAE model for encoding/decoding images based on selected type"""
        if self.vae_type == "tiny":
            return self.create_vae_tiny()
        else:  # kl
            return self.create_vae_kl()

    def create_vae_tiny(self) -> AutoencoderTiny:
        """Create VAE model for encoding/decoding images"""
        vae = AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=self.weight_dtype)
        vae.config.scaling_factor = 1.0
        vae.config.shift_factor = 0
        return vae.eval().to(self.device)

    def create_vae_kl(self) -> AutoencoderKL:
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=self.weight_dtype)
        vae.config.scaling_factor = 0.18215
        vae.config.shift_factor = 0
        return vae.eval().to(self.device)

    def create_scheduler(self) -> DPMSolverMultistepScheduler:
        """Create diffusion scheduler"""
        return DPMSolverMultistepScheduler.from_pretrained(
            self.config.config_dir,
            algorithm_type="dpmsolver++",
            solver_order=2,
            # use_karras_sigmas=True,
        )

    def create_face_processor(self) -> FaceProcessor:
        return FaceProcessor(
            resolution=self.resolution,
            device=self.device,
            config=self.config,
        )

    @classmethod
    def from_dict(cls, config_dict: dict) -> "LipsyncContext":
        """Create context from dictionary"""
        return cls(**config_dict)

    @staticmethod
    def from_version(version: str, **kwargs) -> "LipsyncContext":
        """
        version: str in ["v15", "v10"]
        """
        version = version or "v15"
        if version == "v15":
            return LipsyncContext_v15(**kwargs)
        elif version == "v10":
            return LipsyncContext(**kwargs)
        else:
            raise ValueError(f"Invalid version: {version}")


class LipsyncContext_v15(LipsyncContext):
    def get_config(self, checkpoint_dir: str) -> LipsyncConfig_v15:
        return LipsyncConfig_v15(checkpoint_dir)

    def create_unet(self) -> UNet3DConditionModel_v15:
        """Create UNet model for diffusion"""
        unet, _ = UNet3DConditionModel_v15.from_pretrained(
            OmegaConf.to_container(self.config.unet_config.model),
            self.config.latentsync_unet_path,
            device=self.device,
        )
        return unet.eval().to(dtype=self.weight_dtype)
