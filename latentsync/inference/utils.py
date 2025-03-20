from accelerate.utils import set_seed as acc_seed
import numpy as np
import torch
from latentsync.inference.context import LipsyncContext
from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
from latentsync.pipelines.lipsync_diffusion_pipeline import LipsyncDiffusionPipeline
from latentsync.whisper.whisper.audio import load_audio


def create_diffusion_pipeline(context: LipsyncContext) -> LipsyncDiffusionPipeline:
    """Create lipsync diffusion pipeline with all components
    
    Args:
        context: Lipsync context
    
    Returns:
        LipsyncDiffusionPipeline: The lipsync diffusion pipeline
    """
    return LipsyncDiffusionPipeline(
        vae=context.create_vae(),
        unet=context.create_unet(),
        scheduler=context.create_scheduler(),
        lipsync_context=context,
    ).to(context.device)


def create_pipeline(context: LipsyncContext) -> LipsyncPipeline:
    """Create lipsync pipeline with all components
    
    Args:
        context: Lipsync context
    
    Returns:
        LipsyncPipeline: The lipsync pipeline
    """
    return LipsyncPipeline(
        vae=context.create_vae(),
        audio_encoder=context.create_audio_encoder(),
        unet=context.create_unet(),
        scheduler=context.create_scheduler(),
        lipsync_context=context,
    ).to(context.device)


def set_seed(seed: int):
    if seed != -1:
        acc_seed(seed)
    else:
        torch.seed()
        print(f"Initial seed: {torch.initial_seed()}")


def load_audio_clips(audio_path: str, samples_per_frame: int):
    audio_samples = load_audio(audio_path)
    padding_size = samples_per_frame - len(audio_samples) % samples_per_frame
    audio_samples = np.pad(
        audio_samples,
        (0, padding_size),
        mode="constant",
    )
    audio_clips = audio_samples.reshape(-1, samples_per_frame)
    return audio_clips