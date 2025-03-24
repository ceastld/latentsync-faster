import asyncio
from latentsync.configs.config import GLOBAL_CONFIG
from latentsync.inference.latentsync import LatentSync
from latentsync.utils.timer import Timer


async def main(max_frames: int = None):
    model = LatentSync(
        # disable_progress=False,
        video_fps=25,
    )
    example = GLOBAL_CONFIG.inference.obama
    await model.test(
        video_path=example.video_path,
        audio_path=example.audio_path,
        save_path=example.video_out_path,
        max_frames=max_frames,
    )

if __name__ == "__main__":
    # Timer.enable()
    asyncio.run(main())
    Timer.summary()
