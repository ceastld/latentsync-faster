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

async def test():
    model = LatentSync()
    audio_path = "assets/cxk.mp3"
    save_path = "output/gx_cxk.mp4"
    model.push_img_and_audio("gx.jpg", audio_path)
    results = await model.get_all_results()
    save_frames_to_video(results, save_path, audio_path=audio_path)
    print(f"Saved to {save_path}")

async def test_obama(model: LatentSync):
    example = GLOBAL_CONFIG.inference.obama1
    model.push_video_stream(example.video_path, example.audio_path, fps=25)
    results = await model.get_all_results()
    logger.info(f"Results: {len(results)}")
    save_frames_to_video(results, example.video_out_path, audio_path=example.audio_path, save_images=True)
    print(f"Saved to {example.video_out_path}")
    
async def main():
    model = LatentSync(enable_progress=False)
    # print(model.context.num_inference_steps)
    await test_obama(model)
    model.setup_model()
    await test_obama(model)

if __name__ == "__main__":
    # Timer.enable()
    asyncio.run(main())
    # audio_clips = load_audio_clips("assets/cxk.mp3", 625)
    # print(audio_clips[0])
    # Timer.summary()
    # asyncio.run(test())
