from abc import ABC, abstractmethod
from typing import AsyncGenerator, TypeVar, Union, List, Iterator, Any
import numpy as np

from latentsync.inference._datas import AudioVideoFrame

T = TypeVar("T")


class VideoGenerator(ABC):
    """Abstract base class for video generation.
    
    Defines the interface for classes that generate video with synchronized audio.
    """
    
    @abstractmethod
    def push_frame(self, frame: Union[np.ndarray, List[np.ndarray]]) -> None:
        """Push one or more frames to the processing pipeline.
        
        Args:
            frame: Single frame or list of frames in RGB format.
        """
        pass
    
    @abstractmethod
    def push_audio(self, audio: np.ndarray) -> None:
        """Push audio data to the processing pipeline.
        
        Args:
            audio: Audio data to process.
        """
        pass
    
    @abstractmethod
    def add_end_task(self) -> None:
        """Add an end task to the processing pipeline.
        
        This signals that no more frames or audio will be added.
        """
        pass
    
    @abstractmethod
    def result_stream(self) -> AsyncGenerator[AudioVideoFrame, None]:
        """Get the stream of processed results.
        
        Returns:
            Iterator of processed frames.
        """
        pass
