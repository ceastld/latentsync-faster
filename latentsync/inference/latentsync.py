import asyncio
from dataclasses import dataclass
import os
from typing import AsyncGenerator, List, Set, Union, TypeVar, Generic, Deque, Tuple
from collections import deque

import cv2
import numpy as np
import torch
from tqdm import tqdm
from latentsync.inference.lipsync_infer import LipsyncBatchInference, LipsyncRestore
from latentsync.inference.audio_infer import AudioBatchInference
from latentsync.inference.context import LipsyncContext
from latentsync.inference.face_infer import FaceInference
from latentsync.inference.multi_infer import MultiThreadInference
from latentsync.inference.utils import load_audio_clips
from latentsync.pipelines.metadata import AudioMetadata, LipsyncMetadata
from latentsync.utils.video import cycle_video_stream, LazyVideoWriter

T = TypeVar("T")


class DataSegmentEnd:
    pass


@dataclass
class VideoFrame:
    frame: np.ndarray


@dataclass
class AudioFrame:
    audio_samples: np.ndarray
    is_speech: bool = True  # Flag to indicate if the audio contains speech


@dataclass
class LipsyncResult:
    audio_samples: np.ndarray
    video_frame: np.ndarray


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


class LatentSyncInference:
    def __init__(self, context: LipsyncContext, worker_timeout=60, enable_progress=False, max_input_fps=25, immediate_frames=25):
        self.context = context
        self.enable_progress = enable_progress
        self.max_input_fps = max_input_fps
        self.immediate_frames = immediate_frames

        self.loop = asyncio.get_event_loop()
        self.tasks: Set[asyncio.Task] = set()
        self.metadata_queue: asyncio.Queue[LipsyncMetadata] = asyncio.Queue()
        self.audio_data_queue: asyncio.Queue[AudioMetadata] = asyncio.Queue()

        # Add FPS controllers with immediate output parameter
        self.frame_controller = FPSController[Union[VideoFrame, DataSegmentEnd]](fps=max_input_fps, immediate_output_count=immediate_frames)
        self.audio_controller = FPSController[Union[AudioFrame, DataSegmentEnd]](fps=max_input_fps, immediate_output_count=immediate_frames)

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
        if hasattr(self, "frame_controller"):
            self.frame_controller.stop()
        if hasattr(self, "audio_controller"):
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

    async def result_stream(self) -> AsyncGenerator[LipsyncResult, None]:
        """Stream results from the lipsync restoration process.
        
        Yields:
            LipsyncResult: A result containing processed video frame and audio sample.
        """
        async for result in self.lipsync_restore.result_stream():
            yield LipsyncResult(
                audio_samples=result.audio_samples,
                video_frame=result.lipsync_frame,
            )

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
        max_input_fps: int = 25,
        worker_timeout: int = 3600,
        num_frames: int = None,
        checkpoint_dir: str = None,
        immediate_frames: int = 0,
        **kwargs,
    ):
        self.context = LipsyncContext.from_version(version, num_frames=num_frames, checkpoint_dir=checkpoint_dir, **kwargs)
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
            audio_chunks = [audio[i : i + spf] for i in range(0, len(audio), spf)]
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
        results = []
        async for data in self.model.result_stream():
            results.append(data)
            if pbar is None:
                pbar = tqdm(desc="results", total=total, disable=disable_progress)
            pbar.update(1)
        if pbar is not None:
            pbar.close()
        return results

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

    async def save_to_video(self, video_path, save_images=False, total_frames=None):
        """Save the processed results to a video file.

        Args:
            video_path (str): Path to save the output video.
            save_images (bool, optional): Whether to save individual frames as images. Defaults to False.
            total_frames (int, optional): Total number of frames for progress bar. Defaults to None.

        Returns:
            int: Number of frames processed.
        """
        count = 0
        import logging
        from tqdm import tqdm

        logger = logging.getLogger(__name__)

        pbar = None
        if self.enable_progress:
            pbar = tqdm(desc="Saving video", total=total_frames)

        with LazyVideoWriter(
            video_path,
            fps=self.context.video_fps,
            audio_sr=self.context.audio_sample_rate,
            save_images=save_images,
        ) as writer:
            async for result in self.result_stream():
                writer.write(result.video_frame)
                writer.write_audio_frame(result.audio_samples)
                count += 1
                if pbar:
                    pbar.update(1)

        if pbar:
            pbar.close()

        logger.info(f"Processed {count} frames")
        print(f"Saved to {video_path}")
        return count


class VadLatentSync:
    """Voice Activity Detection wrapper for LatentSync.

    This class wraps LatentSync and provides additional functionality for skipping
    silent segments using Voice Activity Detection.

    Args:
        version (str, optional): Model version to use. Defaults to None.
        enable_progress (bool, optional): Whether to enable progress bars. Defaults to False.
        video_fps (int, optional): Target FPS for video processing. Defaults to 25.
        worker_timeout (int, optional): Timeout for worker processes in seconds. Defaults to 60.
        num_frames (int, optional): Maximum number of frames to process. Defaults to None.
        checkpoint_dir (str, optional): Directory for model checkpoints. Defaults to None.
        immediate_frames (int, optional): Number of initial frames to output immediately without FPS control. Defaults to 0.
        vad_threshold (float, optional): Threshold for speech detection. Defaults to 0.5.
        sampling_rate (int, optional): Expected audio sampling rate. Defaults to 16000.
        **kwargs: Additional arguments passed to LatentSync.
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
        vad_threshold: float = 0.5,
        sampling_rate: int = 16000,
        **kwargs,
    ):
        # Initialize LatentSync
        self.latent_sync = LatentSync(
            version=version,
            enable_progress=enable_progress,
            video_fps=video_fps,
            worker_timeout=worker_timeout,
            num_frames=num_frames,
            checkpoint_dir=checkpoint_dir,
            immediate_frames=immediate_frames,
            **kwargs,
        )

        # Initialize Silero VAD
        self.vad = SileroVAD(threshold=vad_threshold, sampling_rate=sampling_rate)
        self.vad_threshold = vad_threshold
        self.sampling_rate = sampling_rate

        # Buffer for frames and processing state
        self.frame_buffer = []
        self.result_queue = asyncio.Queue()
        self.processing_task = None
        self.is_processing = False

        # Set up processing task
        self.loop = asyncio.get_event_loop()
        self.start_processing_task()

    def start_processing_task(self):
        """Start the background task for processing results."""
        if self.processing_task is None or self.processing_task.done():
            self.processing_task = self.loop.create_task(self._process_results())

    async def _process_results(self):
        """Background task that processes results from LatentSync."""
        async for result in self.latent_sync.result_stream():
            await self.result_queue.put(result)

    def push_frames(self, frames: Union[np.ndarray, List[np.ndarray]]):
        """Buffer frames for processing.

        Args:
            frames (Union[np.ndarray, List[np.ndarray]]): Frame or list of frames to process.
        """
        if isinstance(frames, np.ndarray):
            self.frame_buffer.append(frames)
        elif isinstance(frames, list):
            self.frame_buffer.extend(frames)
        else:
            raise ValueError(f"Invalid frame type: {type(frames)}")

    def push_audio(self, audio: np.ndarray):
        """Push audio and process buffered frames based on VAD result.

        Args:
            audio (np.ndarray): Audio samples to process.
        """
        # Detect if the audio contains speech
        has_speech = self.vad.is_speech(audio)

        # If audio contains speech, process the buffered frames
        if has_speech and self.frame_buffer:
            self.latent_sync.push_frames(self.frame_buffer)
            self.latent_sync.push_audio(audio)
            self.is_processing = True

        # Clear frame buffer regardless of speech detection result
        self.frame_buffer = []

    def push_video_stream(self, video_path, audio_path, max_frames: int = None):
        """Push a video stream with audio and apply VAD processing.

        Args:
            video_path (str): Path to the input video file.
            audio_path (str): Path to the input audio file.
            max_frames (int, optional): Maximum number of frames to process. Defaults to None.
        """
        assert os.path.exists(video_path), f"Video file {video_path} does not exist"
        assert os.path.exists(audio_path), f"Audio file {audio_path} does not exist"

        # Load audio clips
        audio_clips = load_audio_clips(audio_path, self.latent_sync.context.samples_per_frame)

        # Process video frames with VAD
        speech_segments = []

        # First, analyze all audio to find speech segments
        for i, audio in enumerate(audio_clips):
            if self.vad.is_speech(audio):
                speech_segments.append(i)

        # Now process only frames with speech
        if not speech_segments:
            # No speech detected, return
            self.latent_sync.add_end_task()
            return

        # Group consecutive speech segments
        grouped_segments = []
        current_group = [speech_segments[0]]

        for i in range(1, len(speech_segments)):
            if speech_segments[i] == speech_segments[i - 1] + 1:
                # Continuous segment
                current_group.append(speech_segments[i])
            else:
                # New segment
                grouped_segments.append(current_group)
                current_group = [speech_segments[i]]

        # Add the last group
        if current_group:
            grouped_segments.append(current_group)

        # Process each group of speech segments
        for group in grouped_segments:
            # Get frames and audio for this group
            group_frames = []
            group_audio = []

            for idx in group:
                if idx < len(audio_clips):
                    try:
                        # Get frame for this index
                        frame_idx = min(idx, max_frames - 1) if max_frames else idx
                        frame = None
                        for i, f in enumerate(cycle_video_stream(video_path, max_frames=frame_idx + 1)):
                            if i == frame_idx:
                                frame = f
                                break

                        if frame is not None:
                            group_frames.append(frame)
                            group_audio.append(audio_clips[idx])
                    except Exception as e:
                        print(f"Error accessing frame {idx}: {e}")

            # Process this group
            if group_frames and group_audio:
                self.latent_sync.push_frames(group_frames)
                self.latent_sync.push_audio(np.concatenate(group_audio))
                self.is_processing = True

        self.latent_sync.add_end_task()

    def push_img_and_audio(self, image_path: str, audio_path: str):
        """Push a single image with corresponding audio.

        Args:
            image_path (str): Path to the image file
            audio_path (str): Path to the audio file
        """
        assert os.path.exists(image_path), f"Image file {image_path} does not exist"
        assert os.path.exists(audio_path), f"Audio file {audio_path} does not exist"

        # Load image
        frame = cv2.imread(image_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Load audio clips
        audio_clips = load_audio_clips(audio_path, self.latent_sync.context.samples_per_frame)

        # Detect speech in each audio clip
        speech_indices = []
        for i, audio in enumerate(audio_clips):
            if self.vad.is_speech(audio):
                speech_indices.append(i)

        # Only process frames with speech
        if speech_indices:
            speech_audio = np.concatenate([audio_clips[i] for i in speech_indices])
            speech_frames = [frame] * len(speech_indices)

            self.latent_sync.push_frames(speech_frames)
            self.latent_sync.push_audio(speech_audio)
            self.is_processing = True

        self.latent_sync.add_end_task()

    def close(self):
        """Stop all worker processes."""
        if self.processing_task:
            self.processing_task.cancel()
        self.latent_sync.stop_workers()

    def add_end_task(self):
        """Add an end task to the processing pipeline."""
        if self.frame_buffer:
            # Process any remaining frames if needed
            # This is a simple approach - for real applications,
            # you might want to check if these frames contain speech
            self.latent_sync.push_frames(self.frame_buffer)
            self.frame_buffer = []
        self.latent_sync.add_end_task()

    async def result_stream(self):
        """Get an async iterator for streaming results as they are generated.

        This stream will only yield results for audio segments that contained speech.

        Yields:
            Frame data from processed speech segments.
        """
        self.start_processing_task()  # Ensure processing task is running

        while self.is_processing or not self.result_queue.empty():
            try:
                result = await self.result_queue.get()
                yield result
            except asyncio.CancelledError:
                break

    async def get_all_results(self, total: int = None, disable_progress: bool = False):
        """Get all processed results as a list."""
        results = []
        pbar = None

        if total and not disable_progress:
            pbar = tqdm(desc="results", total=total)

        async for result in self.result_stream():
            results.append(result)
            if pbar:
                pbar.update(1)

        if pbar:
            pbar.close()

        return results
