import numpy as np


from dataclasses import dataclass


@dataclass
class AudioVideoFrame:
    audio_samples: np.ndarray
    video_frame: np.ndarray


@dataclass
class AudioFrame:
    audio_samples: np.ndarray
    is_speech: bool = True  # Flag to indicate if the audio contains speech


@dataclass
class VideoFrame:
    frame: np.ndarray


class DataSegmentEnd:
    pass