from latentsync.inference.context import LipsyncContext
from latentsync.inference.lipsync_model import LipsyncModel
from latentsync.inference.multi_infer import MultiThreadInference
from latentsync.pipelines.metadata import LipsyncMetadata
import torch
from typing import List

from latentsync.utils.affine_transform import AlignRestore


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

    def process_task(self, model: LipsyncModel, idx, data: LipsyncMetadata):
        data_buffer = self.data_buffer
        if len(data_buffer) == 0:
            self.result_start_idx = idx
        data_buffer.append(data)
        if len(data_buffer) >= self.context.num_frames:
            audio_features = [
                torch.from_numpy(data.audio_feature).to(self.context.device).to(self.context.weight_dtype)
                for data in data_buffer
                if data.audio_feature is not None
            ]
            results = model.process_batch(data_buffer, audio_features)
            for i, result in enumerate(results):
                self._set_result(self.result_start_idx + i, result)
            self.data_buffer = []

class LipsyncRestore(MultiThreadInference):
    def __init__(self, context: LipsyncContext, num_workers=1, worker_timeout=60):
        super().__init__(num_workers, worker_timeout)
        self.context = context

    def get_model(self):
        return AlignRestore()
    
    def push_data(self, data: LipsyncMetadata):
        self.add_one_task(data)

    def infer_task(self, model: AlignRestore, data: LipsyncMetadata):
        return model.restore_img(data.original_frame, data.sync_face, data.affine_matrix)
