import asyncio
import numpy as np
from tqdm import tqdm
from latentsync.configs.config import GLOBAL_CONFIG
from latentsync.inference.context import LipsyncContext
from latentsync.utils.face_processor import FaceProcessor
from latentsync.utils.timer import Timer
from latentsync.utils.video import VideoReader
from latentsync.inference.multi_infer import MultiProcessInference

class FaceInference(MultiProcessInference):
    def __init__(self, context: LipsyncContext, num_workers=1, worker_timeout=60):
        super().__init__(num_workers, worker_timeout)
        self.context = context

    def get_model(self):
        return FaceProcessor(self.context.resolution, self.context.device)
    
    def worker(self):
        Timer.enable()
        return super().worker()
    
    def infer_task(self, model: FaceProcessor, image: np.ndarray):
        return model.prepare_face(image)
    
    def push_frame(self, frame: np.ndarray):
        self.add_one_task(frame)
    
async def auto_push_face(video_path: str, infer: FaceInference):
    with VideoReader(video_path) as video_reader:
        for frame in video_reader:
            infer.push_frame(frame)
            # await asyncio.sleep(0.04)
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
    infer = FaceInference(context)
    infer.start_workers()
    await auto_push_face(GLOBAL_CONFIG.inference.default_video_path, infer)
    results = await wait_for_results(infer)
    # print(results)

if __name__ == "__main__":  
    Timer.enable()
    asyncio.run(main())
    Timer.summary()