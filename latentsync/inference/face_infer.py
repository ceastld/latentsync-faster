import asyncio
import cv2
import numpy as np
from tqdm import tqdm
from latentsync.configs.config import GLOBAL_CONFIG
from latentsync.inference.context import LipsyncContext
from latentsync.pipelines.metadata import LipsyncMetadata
from latentsync.utils.face_processor import FaceProcessor
from latentsync.utils.timer import Timer
from latentsync.utils.video import VideoReader
from latentsync.inference.multi_infer import MultiThreadInference
from typing import Dict, Any, Tuple, List
from latentsync.inference.buffer_infer import BufferInference

class FaceInference(MultiThreadInference[np.ndarray, LipsyncMetadata]):
    def __init__(self, context: LipsyncContext, num_workers=1, worker_timeout=60):
        super().__init__(num_workers, worker_timeout)
        self.context = context

    def get_model(self):
        return self.context.face_processor
    
    def infer_task(self, model: FaceProcessor, image: np.ndarray) -> LipsyncMetadata:
        # Apply Gaussian blur with kernel size 3x3 before face detection
        if self.context.use_gaussian_blur:
            image = cv2.GaussianBlur(image, (3, 3), 0)
        try:
            return model.prepare_face(image)
        except Exception:
            return LipsyncMetadata(original_frame=image)
    
    def push_frame(self, frame: np.ndarray):
        self.add_one_task(frame)
    
class FaceBatchInference(BufferInference[np.ndarray, LipsyncMetadata]):
    def __init__(self, context: LipsyncContext, num_workers=1, worker_timeout=60):
        super().__init__(num_workers, worker_timeout)
        self.context = context
        self.batch_count = 0  # Track how many batches have been processed
        
    def get_batch_size(self) -> int:
        """Get dynamic batch size: 1, 3, 5, 5, 5, ..."""
        if self.batch_count == 0:
            return 1
        elif self.batch_count == 1:
            return 3
        else:
            return 5
    
    def get_model(self):
        return self.context.face_processor
    
    def push_frame(self, frame: np.ndarray):
        """Push a video frame to the inference buffer"""
        self.push_data(frame)
        
    def push_frame_batch(self, frames: List[np.ndarray]):
        """Push a batch of video frames to the inference buffer"""
        self.push_data_batch(frames)
    
    def infer_task(self, model: FaceProcessor, data: List[np.ndarray]):
        """Process a batch of frames using face processor with interpolation"""
        # Increment batch count when processing a batch
        self.batch_count += 1
        
        # Use interpolation method to process faces efficiently
        if self.context.use_gaussian_blur:
            data = [cv2.GaussianBlur(frame, (3, 3), 0) for frame in data]
            
        metadata_list = model.prepare_face_batch_with_interpolation(data)
        
        if metadata_list is None:
            # If face processing failed, return empty metadata
            return [LipsyncMetadata(original_frame=frame) for frame in data]
        
        return metadata_list

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