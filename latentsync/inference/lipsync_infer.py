from latentsync.inference.context import LipsyncContext
from latentsync.inference.lipsync_model import LipsyncModel
from latentsync.inference.multi_infer import MultiProcessInference, MultiThreadInference, InferenceTask
from latentsync.pipelines.metadata import LipsyncMetadata
import torch
from typing import List, Union

from latentsync.utils.affine_transform import AlignRestore
from latentsync.utils.timer import Timer
from latentsync.inference.buffer_infer import BufferInference


class LipsyncInference(MultiThreadInference):
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
    def process_task(self, model: LipsyncModel, task: InferenceTask) -> None:
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
        super().__init__(context.num_frames, num_workers, worker_timeout)
        self.context = context
    
    def get_model(self):
        return LipsyncModel(self.context)
    
    def push_data(self, data: Union[LipsyncMetadata, List[LipsyncMetadata]]):
        if isinstance(data, list):
            self.push_data_batch(data)
        else:
            super().push_data(data)

    def infer_task(self, model: LipsyncModel, data: List[LipsyncMetadata]):
        assert len(data) <= self.context.num_frames, f"data length should <= num_frames: {self.context.num_frames}"
        # if any face in data is None, return data with None face
        if any(metadata.face is None for metadata in data):
            for metadata in data:
                metadata.face = None
            return data
        return model.process_metadata_batch(data)

class LipsyncRestore(MultiThreadInference):
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

    def infer_task(self, model: AlignRestore, data: LipsyncMetadata):
        # if sync_face is None, return original_frame
        if data.sync_face is None:
            return data.original_frame
        return model.restore_img(data.original_frame, data.sync_face, data.affine_matrix)
