from latentsync.inference.context import LipsyncContext
from latentsync.inference.lipsync_model import LipsyncModel
from latentsync.inference.multi_infer import MultiProcessInference, MultiThreadInference, InferenceTask
from latentsync.pipelines.metadata import LipsyncMetadata
import torch
from typing import List, Union, Dict, Any, Optional, TypeVar

from latentsync.utils.affine_transform import AlignRestore
from latentsync.utils.timer import Timer
from latentsync.inference.buffer_infer import BufferInference


class LipsyncInference(MultiThreadInference[LipsyncMetadata, LipsyncMetadata]):
    def __init__(self, context: LipsyncContext, num_workers=1, worker_timeout=60):
        super().__init__(num_workers, worker_timeout)
        self.context = context

    def get_model(self):
        return LipsyncModel(self.context)

    def push_data(self, data: LipsyncMetadata):
        self.add_one_task(data)

    def worker(self):
        self.data_buffer: List[LipsyncMetadata] = []
        self.result_start_idx = 0
        super().worker()

    @Timer("lipsync_diffusion")
    def process_task(self, model: LipsyncModel, task: InferenceTask[LipsyncMetadata]) -> None:
        data_buffer = self.data_buffer
        if len(data_buffer) == 0:
            self.result_start_idx = task.idx
        data_buffer.append(task.data)
        if len(data_buffer) >= self.context.num_frames:
            results = model.process_metadata_batch(data_buffer)
            for i, result in enumerate(results):
                self._set_result(self.result_start_idx + i, result)
            self.data_buffer = []

class LipsyncBatchInference(BufferInference[LipsyncMetadata, LipsyncMetadata]):
    def __init__(self, context: LipsyncContext, num_workers=1, worker_timeout=60):
        super().__init__(num_workers, worker_timeout)
        self.context = context
        self.batch_count = 0  # Track how many batches have been processed
    
    def get_batch_size(self) -> int:
        """Get dynamic batch size: 4, 8, 16, 16, 16, ..."""
        if self.batch_count == 0:
            return 16
        elif self.batch_count == 1:
            return 12
        else:
            return 12
    
    def get_model(self):
        return LipsyncModel(self.context)
    
    def push_data(self, data: Union[LipsyncMetadata, List[LipsyncMetadata]]):
        if isinstance(data, list):
            self.push_data_batch(data)
        else:
            super().push_data(data)

    def infer_task(self, model: LipsyncModel, data: List[LipsyncMetadata]):
        assert len(data) <= self.context.num_frames, f"data length should <= num_frames: {self.context.num_frames}"
        
        # Increment batch count when processing a batch
        self.batch_count += 1
        
        # if any face in data is None, return data with None face
        if any(metadata.face is None for metadata in data):
            self.logger.info("No face, return data with None face")
            for metadata in data:
                metadata.face = None
            return data
        if any(metadata.audio_feature is None for metadata in data):
            self.logger.info("No audio feature, return data with None audio_feature")
            for metadata in data:
                metadata.audio_feature = None
            return data
        return model.process_metadata_batch(data)

class LipsyncRestore(MultiThreadInference[LipsyncMetadata, LipsyncMetadata]):
    def __init__(self, context: LipsyncContext, num_workers=1, worker_timeout=60):
        super().__init__(num_workers, worker_timeout)
        self.context = context

    def get_model(self):
        return AlignRestore()
    
    def push_data(self, data: Union[LipsyncMetadata, List[LipsyncMetadata]]):
        self.add_tasks
        if isinstance(data, list):
            for d in data:
                self.add_one_task(d)
        else:
            self.add_one_task(data)

    def infer_task(self, model: AlignRestore, data: LipsyncMetadata) -> LipsyncMetadata:
        # if sync_face is None, return original_frame
        if data.sync_face is not None:
            data.restored_frame = model.restore_img(data.original_frame, data.sync_face, data.affine_matrix)
        return data