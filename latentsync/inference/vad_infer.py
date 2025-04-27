from typing import List
from latentsync.inference._datas import AudioFrame
from latentsync.utils.vad import SileroVAD
from .buffer_infer import BufferInference
from .multi_infer import MultiThreadInference


class VadInference(MultiThreadInference[AudioFrame, AudioFrame]):
    def __init__(self):
        super().__init__()

    def get_model(self):
        return SileroVAD()

    def infer_task(self, model: SileroVAD, input_data: AudioFrame) -> AudioFrame:
        input_data.is_speech = model.detect(input_data.audio_samples)
        return input_data
