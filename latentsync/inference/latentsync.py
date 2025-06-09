import asyncio
import os
import time
from typing import AsyncGenerator, AsyncIterator, List, Set, Union, TypeVar

import cv2
import numpy as np
from tqdm import tqdm
from latentsync.configs.config import GLOBAL_CONFIG
from latentsync.inference._datas import AudioFrame, DataSegmentEnd, AudioVideoFrame, VideoFrame
from latentsync.inference._types import VideoGenerator
from latentsync.inference._utils import FPSController, save_async_frames
from latentsync.inference.lipsync_infer import LipsyncBatchInference, LipsyncRestore
from latentsync.inference.audio_infer import AudioBatchInference
from latentsync.inference.context import LipsyncContext
from latentsync.inference.face_infer import FaceBatchInference
from latentsync.inference.multi_infer import MultiThreadInference
from latentsync.inference.utils import load_audio_clips
from latentsync.pipelines.metadata import AudioMetadata, LipsyncMetadata
from latentsync.utils.vad import SileroVAD
from latentsync.utils.video import cycle_video_stream, LazyVideoWriter


class LatentSyncInference:
    def __init__(self, context: LipsyncContext, worker_timeout=60, enable_progress=False, max_input_fps=25, immediate_frames=25):
        self.context = context
        self.enable_progress = enable_progress
        self.max_input_fps = max_input_fps
        self.immediate_frames = immediate_frames

        self.loop = asyncio.get_event_loop()
        self.tasks: Set[asyncio.Task] = set()
        # Use bounded queues for backpressure control
        self.metadata_queue: asyncio.Queue[LipsyncMetadata] = asyncio.Queue(maxsize=1000)
        self.audio_data_queue: asyncio.Queue[AudioMetadata] = asyncio.Queue(maxsize=1000)

        # Add FPS controllers with immediate output parameter
        self.frame_controller = FPSController[Union[VideoFrame, DataSegmentEnd]](fps=max_input_fps, immediate_output_count=immediate_frames)
        self.audio_controller = FPSController[Union[AudioFrame, DataSegmentEnd]](fps=max_input_fps, immediate_output_count=immediate_frames)
        
        # Add output FPS controller for result stream
        self.output_controller = FPSController[Union[AudioVideoFrame, DataSegmentEnd]](fps=context.video_fps)

        self.lipsync_model = LipsyncBatchInference(context=context, worker_timeout=worker_timeout)
        self.lipsync_model.start_workers()
        self.lipsync_model.wait_worker_loaded()
        self.face_model = FaceBatchInference(context=context, worker_timeout=worker_timeout)
        self.face_model.start_workers()
        self.audio_model = AudioBatchInference(context=context, worker_timeout=worker_timeout, use_vad=context.use_vad)
        self.audio_model.start_workers()
        self.lipsync_restore = LipsyncRestore(context=context, worker_timeout=worker_timeout)
        self.lipsync_restore.start_workers()
        self.stopped = False
        self.wait_loaded()

        # Start controller streams
        self.create_task(self._stream_frames())
        self.create_task(self._stream_audio())
        self.create_task(self._stream_output())
        self.create_task(self._monitor_queues())  # Add queue monitoring

    @property
    def workers(self):
        for k, v in self.__dict__.items():
            if isinstance(v, MultiThreadInference):
                yield v

    def wait_loaded(self):
        for worker in self.workers:
            worker.wait_worker_loaded()

    def push_frame(self, frame: Union[np.ndarray, List[np.ndarray]]):
        """Push one or more frames to the frame controller.

        Args:
            frame: Single frame (np.ndarray with shape: (H, W, 3), dtype: uint8) or
                   list of frames
        """
        if isinstance(frame, list):
            video_frames = [VideoFrame(frame=f) for f in frame]
            self.frame_controller.push_data_batch(video_frames)
        else:
            self.frame_controller.push_data(VideoFrame(frame=frame))

    def push_audio(self, audio: Union[AudioFrame, List[AudioFrame]]):
        """Push audio data to the audio controller.

        Args:
            audio: Single AudioFrame object or list of AudioFrame objects
        """
        if isinstance(audio, list):
            self.audio_controller.push_data_batch(audio)
        else:
            self.audio_controller.push_data(audio)

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
                # AudioModel 需要接收 np.ndarray 数据
                self.audio_model.push_audio(data.audio_samples)

    def stop_workers(self):
        if self.stopped:
            return
        # Stop FPS controllers
        if hasattr(self, "frame_controller"):
            self.frame_controller.stop()
        if hasattr(self, "audio_controller"):
            self.audio_controller.stop()
        if hasattr(self, "output_controller"):
            self.output_controller.stop()
        for task in self.tasks:
            task.cancel()
        for worker in self.workers:
            worker.stop_workers()
        self.stopped = True

    def create_task(self, coro):
        task = self.loop.create_task(coro)
        self.tasks.add(task)

    def is_alive(self):
        return any(worker.is_alive() for worker in self.workers) or not self.metadata_queue.empty() or not self.audio_data_queue.empty()

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
        async for audio_data in self.audio_model.result_stream():
            await self.audio_data_queue.put(audio_data)
            if pbar is None:
                pbar = tqdm(desc="Processing audio", disable=not self.enable_progress)
            pbar.update(1)
        self.audio_data_queue.put_nowait(None)
        pbar.close()

    async def push_data_to_lipsync(self):
        while self.is_alive():
            metadata = await self.metadata_queue.get()
            if metadata is None:
                break
            audio_data = await self.audio_data_queue.get()
            if audio_data is None:
                break
            metadata.audio_feature = audio_data.audio_feature
            metadata.audio_samples = audio_data.audio_samples
            self.lipsync_model.push_data(metadata)
        self.lipsync_model.add_end_task()

    async def push_data_to_lipsync_restore(self):
        pbar = None
        async for metadata in self.lipsync_model.result_stream():
            self.lipsync_restore.push_data(metadata)
            if pbar is None:
                pbar = tqdm(desc="Pushing data to lipsync restore", disable=not self.enable_progress)
            pbar.update(1)
        self.lipsync_restore.add_end_task()
        pbar.close()

    async def _stream_output(self):
        """Stream results from the lipsync restoration process to the output controller."""
        async for result in self.lipsync_restore.result_stream():
            output_frame = AudioVideoFrame(
                audio_samples=result.audio_samples,
                video_frame=result.lipsync_frame,
            )
            self.output_controller.push_data(output_frame)
        # Signal end of output stream
        self.output_controller.push_data(DataSegmentEnd())

    async def result_stream(self) -> AsyncGenerator[AudioVideoFrame, None]:
        """Stream results from the output controller at controlled FPS.

        Yields:
            AudioVideoFrame: A result containing processed video frame and audio sample.
        """
        async for data in self.output_controller.stream():
            if isinstance(data, DataSegmentEnd):
                break
            yield data

    def add_end_task(self):
        """Signal the end of data stream for both frame and audio controllers."""
        self.frame_controller.push_data(DataSegmentEnd())
        self.audio_controller.push_data(DataSegmentEnd())

    def add_video_end_task(self):
        """Signal the end of data stream for video controller."""
        self.frame_controller.push_data(DataSegmentEnd())

    def add_audio_end_task(self):
        """Signal the end of data stream for audio controller."""
        self.audio_controller.push_data(DataSegmentEnd())

    def add_output_end_task(self):
        """Signal the end of data stream for output controller."""
        self.output_controller.push_data(DataSegmentEnd())

    async def _monitor_queues(self):
        """Monitor queue sizes and log performance metrics."""
        while not self.stopped:
            try:
                await asyncio.sleep(2.0)  # Check every 2 seconds
                if self.enable_progress:
                    metadata_size = self.metadata_queue.qsize()
                    audio_size = self.audio_data_queue.qsize()
                    if metadata_size > 50 or audio_size > 50:
                        print(f"Queue sizes - Face: {metadata_size}, Audio: {audio_size}")
            except Exception:
                break


class LatentSync(VideoGenerator):
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
    """

    def __init__(
        self,
        version=None,
        enable_progress=False,
        max_input_fps: int = 25,
        worker_timeout: int = 3600,
        num_frames: int = None,
        checkpoint_dir: str = None,
        immediate_frames: int = 25,
        use_vad: bool = True,
        **kwargs,
    ):
        self.context = LipsyncContext.from_version(version, num_frames=num_frames, checkpoint_dir=checkpoint_dir, use_vad=use_vad, **kwargs)
        self.enable_progress = enable_progress
        self.max_input_fps = max_input_fps
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
            max_input_fps=self.max_input_fps,
            immediate_frames=self.immediate_frames,
        )
        self.model.start_processing()

    def stop_workers(self):
        """Stop all worker processes."""
        self.model.stop_workers()

    def add_end_task(self):
        """Add an end task to the processing pipeline."""
        self.model.add_end_task()

    def add_video_end_task(self):
        """Signal the end of data stream for video controller."""
        self.model.add_video_end_task()

    def add_audio_end_task(self):
        """Signal the end of data stream for audio controller."""
        self.model.add_audio_end_task()

    def add_output_end_task(self):
        """Signal the end of data stream for output controller."""
        self.model.add_output_end_task()

    def push_frame(self, frame: Union[np.ndarray, List[np.ndarray]]):
        """Push one or more frames to the processing pipeline.

        Args:
            frame: Single frame (np.ndarray with shape: (H, W, 3), dtype: uint8) or
                   list of frames in RGB format.

        Raises:
            ValueError: If frame type is not supported.
        """
        self.model.push_frame(frame)

    def push_audio(self, audio: np.ndarray):
        """Push audio data to the processing pipeline.

        Args:
            audio: np.ndarray with shape: (T,), dtype: float32, sample rate 16000.
        """
        self.model.push_audio(AudioFrame.from_numpy(audio, self.context.samples_per_frame))

    def push_video_stream(self, video_path, audio_path, max_frames: int = None, max_input_fps: int = 30):
        """Push a video stream with audio for processing.

        Args:
            video_path (str): Path to the input video file.
            audio_path (str): Path to the input audio file.
            max_frames (int, optional): Maximum number of frames to process. Defaults to None.
            fps (int, optional): Target FPS for video processing. Defaults to 30.
        """
        assert os.path.exists(video_path), f"Video file {video_path} does not exist"
        assert os.path.exists(audio_path), f"Audio file {audio_path} does not exist"
        self.model.create_task(self._push_video_streaming(video_path, audio_path, max_frames, max_input_fps))

    async def _push_video_streaming(self, video_path, audio_path, max_frames: int = None, max_input_fps: int = 40):
        audio_clips = load_audio_clips(audio_path, self.context.samples_per_frame)

        # No need to manually control FPS here, just push data as fast as possible
        # The FPS controllers will handle the output rate
        for i, frame in enumerate(cycle_video_stream(video_path, max_frames=max_frames)):
            self.push_frame(frame)
            # Create AudioFrame from numpy array
            self.push_audio(audio_clips[i % len(audio_clips)])

            # Small sleep to prevent overwhelming the CPU
            await asyncio.sleep(1 / max_input_fps)

        self.model.add_end_task()

    def result_stream(self) -> AsyncGenerator[AudioVideoFrame, None]:
        return self.model.result_stream()

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
        self.push_frame([frame] * len(audio_clips))

        # Convert and push audio frames
        audio_frames = [AudioFrame.from_numpy(clip) for clip in audio_clips]
        self.model.push_audio(audio_frames)

        self.model.add_end_task()

    async def save_to_video(self, video_path, save_images=False, total_frames=None):
        await save_async_frames(
            self.result_stream(),
            video_path,
            video_fps=self.context.video_fps,
            audio_sr=self.context.audio_sample_rate,
            save_images=save_images,
            disable_progress=not self.enable_progress,
            total_frames=total_frames,
        )

    async def inference(self, video_path, audio_path, output_path):
        # Get total frames from input video
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        self.push_video_stream(video_path, audio_path)
        await self.save_to_video(output_path, total_frames=total_frames)

    async def model_warmup(self):
        demo = GLOBAL_CONFIG.inference.obama1
        start_time = time.time()
        self.push_video_stream(demo.video_path, demo.audio_path, max_frames=1)
        async for frame in self.result_stream():
            pass
        end_time = time.time()
        print(f"Model warmup time: {end_time - start_time} seconds")
        self.setup_model()


class AvatarGenerator:
    def __init__(self, latent_sync: LatentSync = None, vad: SileroVAD = None, use_vad: bool = True):
        self.latent_sync = latent_sync or LatentSync(use_vad=use_vad)
        self.vad = vad or SileroVAD()
        self._av_queue = asyncio.Queue[AudioVideoFrame]()
        self._video_queue = asyncio.Queue[VideoFrame]()
        self._audio_queue = asyncio.Queue[AudioFrame]()

    async def start(self):
        async def aaa():
            pass

        pass

    async def push_audio(self, audio: AudioFrame):
        audio.is_speech = self.vad.detect(audio.audio_samples)
        await self._audio_queue.put(audio)

    async def push_frame(self, frame: VideoFrame):
        await self._video_queue.put(frame)

    async def __aiter__(self) -> AsyncIterator[AudioVideoFrame | DataSegmentEnd]:
        return self.stream_impl()

    async def stream_impl(self):

        pass

    async def close(self):
        pass
