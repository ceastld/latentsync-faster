import asyncio
import logging
import cv2
from tqdm import tqdm
from latentsync import GLOBAL_CONFIG, LatentSync, Timer
from latentsync.inference.utils import load_audio_clips
from latentsync.utils.video import save_frames_to_video


logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s.%(msecs)03d][%(levelname)s][%(name)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


async def main(max_frames: int = None):
    model = LatentSync(version="v15")
    example = GLOBAL_CONFIG.inference.obama
    model.push_video_stream(example.video_path, example.audio_path, max_frames, fps=25)
    results = []

    pbar = None
    async for result in model.result_stream():
        results.append(result)
        if pbar is None:
            pbar = tqdm(desc="Processing")
        pbar.update(1)
    pbar.close()

    save_frames_to_video(results, example.video_out_path, audio_path=example.audio_path)
    print(f"Saved to {example.video_out_path}")

async def test():
    model = LatentSync(version="v15")
    audio_path = "assets/cxk.mp3"
    save_path = "output/gx_cxk.mp4"
    model.push_img_and_audio("gx.jpg", audio_path)
    results = await model.get_all_results()
    save_frames_to_video(results, save_path, audio_path=audio_path)
    print(f"Saved to {save_path}")

if __name__ == "__main__":
    # Timer.enable()
    asyncio.run(main())
    # Timer.summary()
    # asyncio.run(test())
