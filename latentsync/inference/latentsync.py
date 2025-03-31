import argparse
import asyncio
from functools import cached_property
from typing import List, Set, Union

import cv2
import numpy as np
from tqdm import tqdm
from latentsync.configs.config import GLOBAL_CONFIG
from latentsync.inference.lipsync_infer import LipsyncBatchInference, LipsyncInference, LipsyncRestore
from latentsync.inference.audio_infer import AudioBatchInference, AudioInference
from latentsync.inference.context import LipsyncContext, LipsyncContext_v15
from latentsync.inference.face_infer import FaceInference
from latentsync.inference.multi_infer import MultiThreadInference
from latentsync.inference.utils import load_audio_clips
from latentsync.pipelines.metadata import LipsyncMetadata
from latentsync.utils.affine_transform import AlignRestore
from latentsync.utils.timer import Timer
from latentsync.utils.video import cycle_video_stream, VideoReader, save_frames_to_video


class LatentSyncInference:
    def __init__(self, context: LipsyncContext, worker_timeout=60, enable_progress=False):
        self.context = context
        self.enable_progress = enable_progress

        self.loop = asyncio.get_event_loop()
        self.tasks: Set[asyncio.Task] = set()
        self.metadata_queue: asyncio.Queue[LipsyncMetadata] = asyncio.Queue()
        self.audio_feature_queue: asyncio.Queue[np.ndarray] = asyncio.Queue()

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

    @property
    def workers(self):
        for k, v in self.__dict__.items():
            if isinstance(v, MultiThreadInference):
                yield v

    def wait_loaded(self):
        for worker in self.workers:
            worker.wait_worker_loaded()

    def push_face(self, frame: np.ndarray):
        """frame: np.ndarray, shape: (H, W, 3), dtype: uint8"""
        self.face_model.push_frame(frame)

    def push_audio(self, audio: np.ndarray):
        """audio: np.ndarray, shape: (T,), dtype: float32"""
        self.audio_model.push_audio(audio)

    def stop_workers(self):
        if self.stopped:
            return
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
        self.face_model.add_end_task()
        self.audio_model.add_end_task()

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

    def __init__(self, version=None, enable_progress=False, video_fps: int = 25, worker_timeout: int = 60, num_frames: int = None):
        self.context = LipsyncContext.from_version(version, num_frames=num_frames)
        self.enable_progress = enable_progress
        self.video_fps = video_fps
        self.model = LatentSyncInference(
            context=self.context,
            enable_progress=enable_progress,
            worker_timeout=worker_timeout,
        )
        self.model.start_processing()

    def stop_workers(self):
        """Stop all worker processes."""
        self.model.stop_workers()

    def push_frames(self, frame: Union[np.ndarray, List[np.ndarray]]):
        """Push one or more frames to the processing pipeline.

        Args:
            frame (Union[np.ndarray, List[np.ndarray]]): Single frame or list of frames.
                Each frame should be a numpy array in RGB format.

        Raises:
            ValueError: If frame type is not supported.

        Example:
            ```python
            # Single frame
            frame = cv2.imread("input.jpg")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            model.push_frames(frame)
            
            # Multiple frames
            model.push_frames([frame] * 10)
            ```
        """
        if isinstance(frame, np.ndarray):
            self.model.push_face(frame)
        elif isinstance(frame, list):
            for f in frame:
                self.model.push_face(f)
        else:
            raise ValueError(f"Invalid frame type: {type(frame)}")
        
    def push_img_and_audio(self, image_path: str, audio_path: str):
        audio_clips = load_audio_clips(audio_path, self.context.samples_per_frame)
        frame = cv2.imread(image_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.push_frames([frame]*len(audio_clips))
        self.push_audio(audio_clips)
        self.model.add_end_task()

    def push_audio(self, audio: np.ndarray):
        """Push audio data to the processing pipeline.

        Args:
            audio (np.ndarray): Audio data as numpy array with sample rate 16000.
                The audio will be automatically padded to match the frame rate.

        Note:
            The audio data should have a sample rate of 16000 Hz.
            The length will be automatically padded to match the frame rate.

        Example:
            ```python
            # Load and push audio data
            audio_data = load_audio("input.mp3")  # Your audio loading function
            model.push_audio(audio_data)
            ```
        """
        spf = self.context.samples_per_frame
        if len(audio) % spf != 0:
            audio = np.pad(audio, (0, spf - len(audio) % spf), mode="constant")
        self.model.push_audio(audio)

    def push_video_stream(self, video_path, audio_path, max_frames: int = None, fps: int = 30):
        """Push a video stream with audio for processing.

        Args:
            video_path (str): Path to the input video file.
            audio_path (str): Path to the input audio file.
            max_frames (int, optional): Maximum number of frames to process. Defaults to None.
            fps (int, optional): Target FPS for video processing. Defaults to 30.

        Note:
            This method creates an asynchronous task for video streaming.
            The video will be processed frame by frame at the specified FPS.
        """
        self.model.create_task(self._push_video_streaming(video_path, audio_path, max_frames, fps))

    async def _push_video_streaming(self, video_path, audio_path, max_frames: int = None, fps: int = 30):
        audio_clips = load_audio_clips(audio_path, self.context.samples_per_frame)
        frame_interval = 1 / fps  # Target frame interval for 30fps
        last_frame_time = asyncio.get_event_loop().time()
        # 240 frames
        for i, frame in enumerate(cycle_video_stream(video_path, max_frames=max_frames)):
            current_time = asyncio.get_event_loop().time()
            elapsed = current_time - last_frame_time

            # Calculate sleep time to maintain 30fps
            sleep_time = max(0, frame_interval - elapsed)
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

            self.push_frames(frame)
            self.push_audio(audio_clips[i % len(audio_clips)])
            last_frame_time = asyncio.get_event_loop().time()

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
