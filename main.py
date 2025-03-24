import asyncio
import logging
from tqdm import tqdm
from latentsync import GLOBAL_CONFIG, LatentSync, Timer
from latentsync.utils.video import save_frames_to_video


logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(levelname)s][%(name)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S,%f'
)
logger = logging.getLogger(__name__)


async def main(max_frames: int = None):
    model = LatentSync()
    example = GLOBAL_CONFIG.inference.obama
    model.auto_push_data(example.video_path, example.audio_path, max_frames, fps=25)
    results = []

    pbar = None
    async for result in model.result_stream():
        results.append(result)
        if pbar is None:
            pbar = tqdm(desc="Processing")
        pbar.update(1)
    pbar.close()

    save_frames_to_video(results, example.video_out_path, audio_path=example.audio_path)


if __name__ == "__main__":
    # Timer.enable()
    asyncio.run(main())
    Timer.summary()
