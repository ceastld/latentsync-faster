from typing import List
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

    @staticmethod
    def from_numpy(data: np.ndarray, spf: int) -> 'AudioFrame' | List['AudioFrame']:
        """Convert a numpy array to an AudioFrame.

        Args:
            data: np.ndarray, audio samples

        Returns:
            AudioFrame: An AudioFrame containing the audio samples
        """
        audio_frames = AudioFrame.split_audio(data, spf)
        if len(audio_frames) == 1:
            return audio_frames[0]
        else:
            return audio_frames

    @staticmethod
    def split_audio(data: np.ndarray, spf: int) -> List['AudioFrame']:
        """Split audio data into multiple AudioFrame objects.

        Args:
            data: np.ndarray, audio samples
            spf: int, samples per frame

        Returns:
            List[AudioFrame]: List of AudioFrame objects
        """
        if len(data) % spf != 0:
            data = np.pad(data, (0, spf - len(data) % spf), mode='constant')
        audio_clips = [AudioFrame(audio_samples=data[i:i+spf]) for i in range(0, len(data), spf)]
        return audio_clips

@dataclass
class VideoFrame:
    frame: np.ndarray


class DataSegmentEnd:
    pass