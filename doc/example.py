import asyncio
from latentsync import *
from latentsync.utils.video import save_frames_to_video

async def main(max_frames: int = None):
    model = LatentSync(version="v15", enable_progress=False)
    example = GLOBAL_CONFIG.inference.obama
    model.push_video_stream(example.video_path, example.audio_path, max_frames, fps=25)
    results = await model.get_all_results()
    save_frames_to_video(results, example.video_out_path, audio_path=example.audio_path)
    print(f"Saved to {example.video_out_path}")


if __name__ == "__main__":
    asyncio.run(main())

