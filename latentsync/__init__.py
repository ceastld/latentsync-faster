"""
LatentSync: Audio Conditioned Latent Diffusion Models for Lip Sync

This package provides tools for lip-syncing videos using latent diffusion models.
"""

__version__ = "0.1.0" 

from latentsync.inference.latentsync import LatentSync
from latentsync.inference.lipsync_model import LipsyncModel
from latentsync.inference.audio_infer import AudioProcessor
from latentsync.inference.context import LipsyncContext, LipsyncContext_v15
from latentsync.configs.config import GLOBAL_CONFIG
from latentsync.utils.timer import Timer