import asyncio
import logging
import cv2
from tqdm import tqdm
from latentsync import GLOBAL_CONFIG, LatentSync, Timer
from latentsync.inference.utils import load_audio_clips
from latentsync.utils.video import LazyVideoWriter, get_total_frames, save_frames_to_video

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s.%(msecs)03d][%(levelname)s][%(name)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

async def test_obama(model: LatentSync):
    example = GLOBAL_CONFIG.inference.obama
    model.push_video_stream(example.video_path, example.audio_path)
    total_frames = get_total_frames(example.video_path)
    await model.save_to_video(example.video_out_path, total_frames=total_frames)
    
async def main():
    model = LatentSync(enable_progress=True, vae_type="tiny")
    await test_obama(model)

if __name__ == "__main__":
    # Timer.enable()
    asyncio.run(main())
    # audio_clips = load_audio_clips("assets/cxk.mp3", 625)
    # print(audio_clips[0])
    # Timer.summary()
    # asyncio.run(test())
