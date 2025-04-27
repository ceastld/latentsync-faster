"""
LatentSync: Audio Conditioned Latent Diffusion Models for Lip Sync

This package provides tools for lip-syncing videos using latent diffusion models.

Requirements:
    - Audio: 16000Hz sample rate, numpy array (float32)
    - Video: 25fps, RGB images, numpy array (uint8)

Example:
    ```python
    import asyncio
    import cv2
    import numpy as np
    from latentsync import LatentSync
    
    async def example():
        model = LatentSync(version="v15")
        
        # Push frames (RGB format)
        frame = cv2.imread("input.jpg")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        model.push_frame(frame)
        
        # Push audio (16000Hz)
        audio_data = load_audio("input.mp3")  # Your audio loading function
        model.push_audio(audio_data)
        
        model.add_end_task()
        
        # Stream results
        frames = []
        async for frame in model.result_stream():
            frames.append(frame)
        
        save_frames_to_video(frames, "output.mp4", audio_path="input.mp3")
    
    asyncio.run(example())
    ```
"""

__version__ = "0.1.1" 

from latentsync.inference.utils import create_pipeline
from latentsync.inference.latentsync import LatentSync
from latentsync.pipelines.metadata import LipsyncMetadata
from latentsync.inference.lipsync_model import LipsyncModel
from latentsync.utils.face_processor import FaceProcessor
from latentsync.inference.audio_infer import AudioProcessor
from latentsync.inference.context import LipsyncContext, LipsyncContext_v15
from latentsync.configs.config import GLOBAL_CONFIG
from latentsync.utils.timer import Timer
from latentsync.inference._datas import AudioFrame