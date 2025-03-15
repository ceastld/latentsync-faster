from accelerate.utils import set_seed as acc_seed
import torch
from latentsync.inference.context import LipsyncContext
from latentsync.pipelines import LipsyncDiffusionPipeline
from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline


def create_diffusion_pipeline(context: LipsyncContext) -> LipsyncDiffusionPipeline:
    """Create lipsync diffusion pipeline with all components"""
    return LipsyncDiffusionPipeline(
        vae=context.create_vae(),
        unet=context.create_unet(),
        scheduler=context.create_scheduler(),
        lipsync_context=context,
    ).to(context.device)


def create_pipeline(context: LipsyncContext) -> LipsyncPipeline:
    """Create lipsync pipeline with all components"""
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