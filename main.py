import asyncio
import logging
import cv2
import torch
import time
import os
from typing import List, Tuple
from tqdm import tqdm
from latentsync import GLOBAL_CONFIG, LatentSync, Timer
from latentsync.inference.utils import load_audio_clips
from latentsync.utils.video import LazyVideoWriter, get_total_frames, save_frames_to_video

# Import performance testing functions
from performance_test import speed_test_with_timing

# Fix ONNX Runtime thread affinity issues in Slurm environment
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["ONNX_NUM_THREADS"] = "1"
os.environ["OMP_WAIT_POLICY"] = "PASSIVE"

logging.basicConfig(level=logging.INFO, format="[%(asctime)s.%(msecs)03d][%(levelname)s][%(name)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)


async def speed_test(model: LatentSync, max_frames: int = 240) -> None:
    """Test model speed and record frame timing information"""
    example = GLOBAL_CONFIG.inference.obama
    model.push_video_stream(example.video_path, example.audio_path, max_frames=max_frames, max_input_fps=26)
    
    # Use the timing function with the result stream
    await model.save_to_video(example.video_out_path, total_frames=max_frames)
    # await speed_test_with_timing(model.result_stream())


async def main():
    model = LatentSync(enable_progress=True, vae_type="kl", use_vad=True)
    await model.model_warmup()
    await speed_test(model)

if __name__ == "__main__":
    # Timer.enable()
    torch.backends.cuda.matmul.allow_tf32 = True
    asyncio.run(main())
    # audio_clips = load_audio_clips("assets/cxk.mp3", 625)
    # print(audio_clips[0])
    # Timer.summary()
    # asyncio.run(test())
