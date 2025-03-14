import numpy as np
import torch
from .multi_infer import MultiProcessInference


class AudioInfer(MultiProcessInference):
    def __init__(self, num_workers=1, worker_timeout=60):
        super().__init__(num_workers, worker_timeout)

    def get_model(self):
        pass
    
    def worker(self):
        pass

    def push_audio(self, audio: np.ndarray):
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()
        
        