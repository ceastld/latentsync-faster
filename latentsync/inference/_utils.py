import asyncio
from collections import deque
import numpy as np
import torch
from tqdm import tqdm

from latentsync.inference._datas import AudioVideoFrame
from latentsync.utils.video import LazyVideoWriter
from ._types import T

from typing import AsyncGenerator, Deque, Generic, List, Tuple


class SileroVAD:
    """Voice Activity Detection using Silero VAD model.

    This class provides methods to detect speech in audio segments.

    Args:
        threshold (float, optional): Speech detection threshold. Defaults to 0.5.
        sampling_rate (int, optional): Expected audio sampling rate. Defaults to 16000.
    """

    def __init__(self, threshold: float = 0.5, sampling_rate: int = 16000):
        self.threshold = threshold
        self.sampling_rate = sampling_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_model()

    def _load_model(self):
        """Load the Silero VAD model."""
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
        )
        self.model = model.to(self.device)
        self.get_speech_timestamps = utils[0]

    def is_speech(self, audio: np.ndarray) -> bool:
        """Detect if audio segment contains speech.

        Args:
            audio (np.ndarray): Audio samples with shape (T,)

        Returns:
            bool: True if speech is detected, False otherwise
        """
        # Convert numpy array to torch tensor
        audio_tensor = torch.tensor(audio, dtype=torch.float32, device=self.device)

        # Get speech timestamps
        speech_timestamps = self.get_speech_timestamps(audio_tensor, self.model, threshold=self.threshold, sampling_rate=self.sampling_rate)

        # If there are any speech timestamps, return True
        return len(speech_timestamps) > 0

    def get_speech_segments(self, audio: np.ndarray) -> List[Tuple[int, int]]:
        """Get timestamps of speech segments in audio.

        Args:
            audio (np.ndarray): Audio samples with shape (T,)

        Returns:
            List[Tuple[int, int]]: List of (start, end) sample indices for speech segments
        """
        # Convert numpy array to torch tensor
        audio_tensor = torch.tensor(audio, dtype=torch.float32, device=self.device)

        # Get speech timestamps
        speech_timestamps = self.get_speech_timestamps(audio_tensor, self.model, threshold=self.threshold, sampling_rate=self.sampling_rate)

        # Convert timestamps to (start, end) tuples
        return [(segment["start"], segment["end"]) for segment in speech_timestamps]


class FPSController(Generic[T]):
    """Controls the output speed of a data stream based on a target FPS.

    This class buffers incoming data and outputs it at a controlled rate defined by the FPS.

    Args:
        fps (float): Target frames per second
        max_buffer_size (int, optional): Maximum size of the buffer. Defaults to None (unlimited).
        immediate_output_count (int, optional): Number of initial items to output immediately
            without FPS control. Defaults to 0.
    """

    def __init__(self, fps: float, max_buffer_size: int = None, immediate_output_count: int = 0):
        self.fps = fps
        self.frame_interval = 1.0 / fps
        self.buffer: Deque[T] = deque() if max_buffer_size is None else deque(maxlen=max_buffer_size)
        self.last_frame_time = 0
        self._stop_event = asyncio.Event()
        self.immediate_output_count = immediate_output_count
        self.output_count = 0

    def push_data(self, data: T):
        """Add data to the buffer.

        Args:
            data (T): Data to add to the buffer
        """
        self.buffer.append(data)

    def push_data_batch(self, data_batch: List[T]):
        """Add a batch of data to the buffer.

        Args:
            data_batch (List[T]): List of data to add to the buffer
        """
        for data in data_batch:
            self.push_data(data)

    def stop(self):
        """Signal the stream to stop."""
        self._stop_event.set()

    async def stream(self):
        """Stream data at the controlled FPS rate.

        Initial items up to immediate_output_count will be yielded immediately without FPS control.
        After that, items are yielded according to the specified FPS rate.

        Yields:
            T: Data from the buffer at the controlled rate
        """
        self.last_frame_time = asyncio.get_event_loop().time()

        while not self._stop_event.is_set():
            if not self.buffer:
                # If buffer is empty, wait a bit and check again
                await asyncio.sleep(0.01)
                continue

            # Check if we should output immediately (for initial items)
            if self.output_count < self.immediate_output_count:
                yield self.buffer.popleft()
                self.output_count += 1
                continue

            current_time = asyncio.get_event_loop().time()
            elapsed = current_time - self.last_frame_time

            # If enough time has passed according to FPS, yield the next item
            if elapsed >= self.frame_interval:
                yield self.buffer.popleft()
                self.last_frame_time = current_time
                self.output_count += 1
            else:
                # Wait until it's time for the next frame
                await asyncio.sleep(max(0, self.frame_interval - elapsed))

        # Drain remaining items in buffer when stopped
        while self.buffer:
            yield self.buffer.popleft()


def log_exception(func):
    """Decorator to catch and log exceptions.

    Usage:
        @log_exception
        def some_function():
            # function code

    Args:
        func: The function to be decorated

    Returns:
        The wrapped function that logs exceptions
    """
    import functools
    import traceback

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"Exception in {func.__name__}: {str(e)}")
            print(traceback.format_exc())
            raise  # Re-raise the exception after logging

    return wrapper


@log_exception
async def save_async_frames(
    frames_iter: AsyncGenerator[AudioVideoFrame, None],
    video_path: str,
    video_fps: int,
    audio_sr: int,
    save_images: bool = False,
    disable_progress: bool = False,
    total_frames: int = None,
):
    """Save the processed results to a video file.

    Args:
        video_path (str): Path to save the output video.
        save_images (bool, optional): Whether to save individual frames as images. Defaults to False.
        total_frames (int, optional): Total number of frames for progress bar. Defaults to None.

    Returns:
        int: Number of frames processed.
    """
    count = 0

    with tqdm(desc="Saving video", total=total_frames, disable=disable_progress) as pbar:
        with LazyVideoWriter(
            video_path,
            fps=video_fps,
            audio_sr=audio_sr,
            save_images=save_images,
        ) as writer:
            async for result in frames_iter:
                writer.write(result.video_frame)
                writer.write_audio_frame(result.audio_samples)
                # if count == 0:
                #     pbar.reset()
                count += 1
                pbar.update(1)

    print(f"Processed {count} frames")
    print(f"Saved to {video_path}")
    return count
