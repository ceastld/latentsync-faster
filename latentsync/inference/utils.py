from typing import List
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


def preprocess_audio(audio: np.ndarray, samples_per_frame: int):
    if len(audio) % samples_per_frame != 0:
        padding_size = samples_per_frame - len(audio) % samples_per_frame
        audio = np.pad(audio, (0, padding_size), mode="constant")
    return audio.reshape(-1, samples_per_frame)


def load_audio_clips(audio_path: str, samples_per_frame: int):
    audio_samples = load_audio(audio_path)
    audio_clips = preprocess_audio(audio_samples, samples_per_frame)
    return audio_clips


def align_audio_features(audio_features: List[torch.Tensor], num_faces: int) -> List[torch.Tensor]:
    """Ensure audio features match the number of faces by padding or trimming."""
    if len(audio_features) < num_faces:
        # Pad with last feature if needed
        last_feature = audio_features[-1] if audio_features else torch.zeros_like(audio_features[0])
        audio_features.extend([last_feature] * (num_faces - len(audio_features)))
    elif num_faces > 0:
        # Trim extra features
        audio_features = audio_features[:num_faces]

    return audio_features
