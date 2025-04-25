import argparse
import asyncio
from dataclasses import dataclass
from functools import cached_property
import os
import time
from typing import List, Set, Union, TypeVar, Generic, Deque, Any
from collections import deque

import cv2
import numpy as np
from tqdm import tqdm
from latentsync.configs.config import GLOBAL_CONFIG
from latentsync.inference.lipsync_infer import LipsyncBatchInference, LipsyncRestore
from latentsync.inference.audio_infer import AudioBatchInference
from latentsync.inference.context import LipsyncContext, LipsyncContext_v15
from latentsync.inference.face_infer import FaceInference
from latentsync.inference.multi_infer import MultiThreadInference
from latentsync.inference.utils import load_audio_clips
from latentsync.pipelines.metadata import LipsyncMetadata
from latentsync.utils.affine_transform import AlignRestore
from latentsync.utils.timer import Timer
from latentsync.utils.video import cycle_video_stream, VideoReader, save_frames_to_video

T = TypeVar('T')

class DataSegmentEnd:
    pass

@dataclass
class VideoFrame:
    frame: np.ndarray

@dataclass
class AudioFrame:
    audio_samples: np.ndarray
    
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


class LatentSyncInference:
    def __init__(self, context: LipsyncContext, worker_timeout=60, enable_progress=False, fps=25, immediate_frames=25):
        self.context = context
        self.enable_progress = enable_progress
        self.fps = fps
        self.immediate_frames = immediate_frames

        self.loop = asyncio.get_event_loop()
        self.tasks: Set[asyncio.Task] = set()
        self.metadata_queue: asyncio.Queue[LipsyncMetadata] = asyncio.Queue()
        self.audio_feature_queue: asyncio.Queue[np.ndarray] = asyncio.Queue()

        # Add FPS controllers with immediate output parameter
        self.frame_controller = FPSController[Union[VideoFrame, DataSegmentEnd]](
            fps=fps, 
            immediate_output_count=immediate_frames
        )
        self.audio_controller = FPSController[Union[AudioFrame, DataSegmentEnd]](
            fps=fps, 
            immediate_output_count=immediate_frames
        )

        self.lipsync_model = LipsyncBatchInference(context=context, worker_timeout=worker_timeout)
        self.lipsync_model.start_workers()
        self.lipsync_model.wait_worker_loaded()
        self.face_model = FaceInference(context=context, worker_timeout=worker_timeout)
        self.face_model.start_workers()
        self.audio_model = AudioBatchInference(context=context, worker_timeout=worker_timeout)
        self.audio_model.start_workers()
        self.lipsync_restore = LipsyncRestore(context=context, worker_timeout=worker_timeout)
        self.lipsync_restore.start_workers()
        self.stopped = False
        self.wait_loaded()
        
        # Start controller streams
        self.create_task(self._stream_frames())
        self.create_task(self._stream_audio())

    @property
    def workers(self):
        for k, v in self.__dict__.items():
            if isinstance(v, MultiThreadInference):
                yield v

    def wait_loaded(self):
        for worker in self.workers:
            worker.wait_worker_loaded()

    def push_frame(self, frame: np.ndarray):
        """Push a frame to the frame controller.
        
        Args:
            frame: np.ndarray, shape: (H, W, 3), dtype: uint8
        """
        self.frame_controller.push_data(VideoFrame(frame=frame))
        
    def push_frames(self, frames: List[np.ndarray]):
        """Push multiple frames to the frame controller.
        
        Args:
            frames: List of frames, each with shape: (H, W, 3), dtype: uint8
        """
        video_frames = [VideoFrame(frame=f) for f in frames]
        self.frame_controller.push_data_batch(video_frames)

    def push_audio(self, audio: np.ndarray):
        """Push audio data to the audio controller.
        
        Args:
            audio: np.ndarray, shape: (T,), dtype: float32
        """
        self.audio_controller.push_data(AudioFrame(audio_samples=audio))
        
    def push_audio_batch(self, audio_batch: List[np.ndarray]):
        """Push multiple audio segments to the audio controller.
        
        Args:
            audio_batch: List of audio segments
        """
        audio_frames = [AudioFrame(audio_samples=a) for a in audio_batch]
        self.audio_controller.push_data_batch(audio_frames)
        
    async def _stream_frames(self):
        """Stream frames from the frame controller to the face model at the controlled rate."""
        async for data in self.frame_controller.stream():
            if isinstance(data, DataSegmentEnd):
                self.face_model.add_end_task()
                break
            else:
                self.face_model.push_frame(data.frame)
            
    async def _stream_audio(self):
        """Stream audio from the audio controller to the audio model at the controlled rate."""
        async for data in self.audio_controller.stream():
            if isinstance(data, DataSegmentEnd):
                self.audio_model.add_end_task()
                break
            else:
                self.audio_model.push_audio(data.audio_samples)

    def stop_workers(self):
        if self.stopped:
            return
        # Stop FPS controllers
        if hasattr(self, 'frame_controller'):
            self.frame_controller.stop()
        if hasattr(self, 'audio_controller'):
            self.audio_controller.stop()
        for task in self.tasks:
            task.cancel()
        for worker in self.workers:
            worker.stop_workers()
        self.stopped = True

    def create_task(self, coro):
        task = self.loop.create_task(coro)
        self.tasks.add(task)

    def is_alive(self):
        return (
            any(worker.is_alive() for worker in self.workers)
            or not self.metadata_queue.empty()
            or not self.audio_feature_queue.empty()
        )

    def start_processing(self):
        self.create_task(self.process_frame())
        self.create_task(self.process_audio())
        self.create_task(self.push_data_to_lipsync())
        self.create_task(self.push_data_to_lipsync_restore())

    async def process_frame(self):
        pbar = None
        async for data in self.face_model.result_stream():
            await self.metadata_queue.put(data)
            if pbar is None:
                pbar = tqdm(desc="Processing face", disable=not self.enable_progress)
            pbar.update(1)
        self.metadata_queue.put_nowait(None)
        pbar.close()

    async def process_audio(self):
        pbar = None
        async for data_list in self.audio_model.result_stream():
            for data in data_list:
                await self.audio_feature_queue.put(data)
            if pbar is None:
                pbar = tqdm(desc="Processing audio", disable=not self.enable_progress)
            pbar.update(len(data_list))
        self.audio_feature_queue.put_nowait(None)
        pbar.close()

    async def push_data_to_lipsync(self):
        while self.is_alive():
            metadata = await self.metadata_queue.get()
            if metadata is None:
                break
            audio_feature = await self.audio_feature_queue.get()
            if audio_feature is None:
                break
            metadata.audio_feature = audio_feature
            self.lipsync_model.push_data(metadata)
        self.lipsync_model.add_end_task()

    async def push_data_to_lipsync_restore(self):
        pbar = None
        async for metadata_list in self.lipsync_model.result_stream():
            self.lipsync_restore.push_data(metadata_list)
            if pbar is None:
                pbar = tqdm(desc="Pushing data to lipsync restore", disable=not self.enable_progress)
            pbar.update(len(metadata_list))
        self.lipsync_restore.add_end_task()
        pbar.close()

    def result_stream(self):
        return self.lipsync_restore.result_stream()

    def add_end_task(self):
        """Signal the end of data stream for both frame and audio controllers."""
        self.frame_controller.push_data(DataSegmentEnd())
        self.audio_controller.push_data(DataSegmentEnd())


class LatentSync:
    """A class for lip-syncing videos using latent diffusion models.

    This class provides methods for processing frames and audio data manually.
    It supports both synchronous and asynchronous operations for frame and audio processing.

    Args:
        version (str, optional): Model version to use. Defaults to None.
        enable_progress (bool, optional): Whether to enable progress bars. Defaults to False.
        video_fps (int, optional): Target FPS for video processing. Defaults to 25.
        worker_timeout (int, optional): Timeout for worker processes in seconds. Defaults to 60.
        num_frames (int, optional): Maximum number of frames to process. Defaults to None.
        immediate_frames (int, optional): Number of initial frames to output immediately without FPS control. Defaults to 0.

    Examples:
        Manual frame and audio processing:
        ```python
        model = LatentSync(version="v15")

        # Push frames
        frame = cv2.imread("input.jpg")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        model.push_frames(frame)  # Single frame
        model.push_frames([frame] * 10)  # Multiple frames

        # Push audio
        audio_data = load_audio("input.mp3")  # Your audio loading function
        model.push_audio(audio_data)

        # Mark end of input
        model.model.add_end_task()

        # Process results as they come in
        frames = []
        async for frame in model.result_stream():
            frames.append(frame)
            # You can process each frame as it's generated
            # process_frame(frame)
        ```
    """

    def __init__(
        self,
        version=None,
        enable_progress=False,
        video_fps: int = 25,
        worker_timeout: int = 3600,
        num_frames: int = None,
        checkpoint_dir: str = None,
        immediate_frames: int = 0,
        **kwargs,
    ):
        self.context = LipsyncContext.from_version(version, num_frames=num_frames, checkpoint_dir=checkpoint_dir, **kwargs)
        self.enable_progress = enable_progress
        self.video_fps = video_fps
        self.worker_timeout = worker_timeout
        self.immediate_frames = immediate_frames
        self.model = None
        self.setup_model()

    def setup_model(self):
        if isinstance(self.model, LatentSyncInference):
            self.model.stop_workers()
            
        self.model = LatentSyncInference(
            context=self.context,
            enable_progress=self.enable_progress,
            worker_timeout=self.worker_timeout,
            fps=self.video_fps,
            immediate_frames=self.immediate_frames,
        )
        self.model.start_processing()

    def stop_workers(self):
        """Stop all worker processes."""
        self.model.stop_workers()

    def add_end_task(self):
        """Add an end task to the processing pipeline."""
        self.model.add_end_task()

    def push_frames(self, frame: Union[np.ndarray, List[np.ndarray]]):
        """Push one or more frames to the processing pipeline.

        Args:
            frame (Union[np.ndarray, List[np.ndarray]]): Single frame or list of frames.
                Each frame should be a numpy array in RGB format.

        Raises:
            ValueError: If frame type is not supported.
        """
        if isinstance(frame, np.ndarray):
            self.model.push_frame(frame)
        elif isinstance(frame, list):
            self.model.push_frames(frame)
        else:
            raise ValueError(f"Invalid frame type: {type(frame)}")

    def push_audio(self, audio: np.ndarray):
        """Push audio data to the processing pipeline.

        Args:
            audio (np.ndarray): Audio data as numpy array with sample rate 16000.
                The audio will be automatically padded to match the frame rate.
        """
        spf = self.context.samples_per_frame
        if len(audio) % spf != 0:
            audio = np.pad(audio, (0, spf - len(audio) % spf), mode="constant")
            
        # If audio is a long sequence, split it into chunks corresponding to frames
        if len(audio) > spf:
            audio_chunks = [audio[i:i+spf] for i in range(0, len(audio), spf)]
            self.model.push_audio_batch(audio_chunks)
        else:
            self.model.push_audio(audio)

    def push_video_stream(self, video_path, audio_path, max_frames: int = None):
        """Push a video stream with audio for processing.

        Args:
            video_path (str): Path to the input video file.
            audio_path (str): Path to the input audio file.
            max_frames (int, optional): Maximum number of frames to process. Defaults to None.
            fps (int, optional): Target FPS for video processing. Defaults to 30.
        """
        assert os.path.exists(video_path), f"Video file {video_path} does not exist"
        assert os.path.exists(audio_path), f"Audio file {audio_path} does not exist"
        self.model.create_task(self._push_video_streaming(video_path, audio_path, max_frames))

    async def _push_video_streaming(self, video_path, audio_path, max_frames: int = None):
        audio_clips = load_audio_clips(audio_path, self.context.samples_per_frame)
        
        # No need to manually control FPS here, just push data as fast as possible
        # The FPS controllers will handle the output rate
        for i, frame in enumerate(cycle_video_stream(video_path, max_frames=max_frames)):
            self.push_frames(frame)
            self.push_audio(audio_clips[i % len(audio_clips)])
            
            # Small sleep to prevent overwhelming the CPU
            await asyncio.sleep(0.001)

        self.model.add_end_task()

    def result_stream(self):
        """Get an async iterator for streaming results as they are generated.

        Returns:
            AsyncIterator: An async iterator that yields processed frames.

        Example:
            ```python
            # Process results as they come in
            frames = []
            async for frame in model.result_stream():
                frames.append(frame)
                # You can process each frame as it's generated
                # process_frame(frame)
            ```
        """
        return self.model.result_stream()

    async def get_all_results(self, total: int = None, disable_progress: bool = False):
        """Get all processed results as a list.

        Args:
            total (int, optional): Total number of expected results for progress bar.
                Defaults to None.
            disable_progress (bool, optional): Whether to disable progress bar.
                Defaults to False.

        Returns:
            List: List of all processed frames.

        Note:
            This method will wait for all processing to complete before returning.
            For streaming results as they come in, use result_stream() instead.

        Example:
            ```python
            # Get all results at once (not recommended for large datasets)
            results = await model.get_all_results()

            # Get results with progress bar (not recommended for large datasets)
            results = await model.get_all_results(total=100)
            ```
        """
        pbar = None
        output_frames = []
        async for data in self.model.result_stream():
            output_frames.append(data)
            if pbar is None:
                pbar = tqdm(desc="results", total=total, disable=disable_progress)
            pbar.update(1)
        if pbar is not None:
            pbar.close()
        return output_frames

    def push_img_and_audio(self, image_path: str, audio_path: str):
        """Push a single image with corresponding audio.
        
        Args:
            image_path (str): Path to the image file
            audio_path (str): Path to the audio file
        """
        assert os.path.exists(image_path), f"Image file {image_path} does not exist"
        assert os.path.exists(audio_path), f"Audio file {audio_path} does not exist"
        audio_clips = load_audio_clips(audio_path, self.context.samples_per_frame)
        frame = cv2.imread(image_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.push_frames([frame] * len(audio_clips))
        self.push_audio(audio_clips)
        self.model.add_end_task()
