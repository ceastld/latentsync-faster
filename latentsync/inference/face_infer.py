import asyncio
import cv2
import numpy as np
from tqdm import tqdm
from latentsync.configs.config import GLOBAL_CONFIG
from latentsync.inference.context import LipsyncContext
from latentsync.utils.face_processor import FaceProcessor
from latentsync.utils.timer import Timer
from latentsync.utils.video import VideoReader
from latentsync.inference.multi_infer import MultiThreadInference

class FaceInference(MultiThreadInference):
    def __init__(self, context: LipsyncContext, num_workers=1, worker_timeout=60):
        super().__init__(num_workers, worker_timeout)
        self.context = context

    def get_model(self):
        return self.context.create_face_processor()
    
    def infer_task(self, model: FaceProcessor, image: np.ndarray):
        # Apply Gaussian blur with kernel size 3x3 before face detection
        image = cv2.GaussianBlur(image, (3, 3), 0)
        return model.prepare_face(image)
    
    def push_frame(self, frame: np.ndarray):
        self.add_one_task(frame)
    
async def auto_push_face(video_path: str, infer: FaceInference):
    reader = VideoReader(video_path)
    for frame in reader:
        infer.push_frame(frame)
        # await asyncio.sleep(1/100)
    reader.release()
    infer.add_end_task()

async def wait_for_results(infer: FaceInference):
    results = []
    pbar = tqdm(desc="Processing Face")
    async for result in infer.result_stream():
        results.append(result)
        pbar.update(1)
    pbar.close()
    return results

async def main():
    context = LipsyncContext()
    infer = FaceInference(context, num_workers=1)
    infer.start_workers()
    infer.wait_worker_loaded()
    task = asyncio.create_task(auto_push_face(GLOBAL_CONFIG.inference.default_video_path, infer))
    results = await wait_for_results(infer)
    await task
    # print(results)

if __name__ == "__main__":  
    Timer.enable()
    asyncio.run(main())
    Timer.summary()