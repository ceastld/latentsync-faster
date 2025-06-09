import asyncio
from collections import deque
import numpy as np
import torch
from tqdm import tqdm

from latentsync.inference._datas import AudioVideoFrame
from latentsync.utils.video import LazyVideoWriter
from ._types import T

from typing import AsyncGenerator, Deque, Generic, List, Tuple

class FPSController(Generic[T]):
    """Controls the output speed of a data stream based on a target FPS.

    This class buffers incoming data and outputs it at a controlled rate defined by the FPS.
    Ensures output FPS is between target_fps and target_fps+1 (e.g., 25~26fps).

    Args:
        fps (float): Target frames per second
        max_buffer_size (int, optional): Maximum size of the buffer. Defaults to None (unlimited).
        immediate_output_count (int, optional): Number of initial items to output immediately
            without FPS control. Defaults to 0.
    """

    def __init__(self, fps: float, max_buffer_size: int = None, immediate_output_count: int = 0):
        self.fps = fps
        self.frame_interval = 1.0 / fps
        self.min_frame_interval = 1.0 / (fps + 1)  # Maximum allowed FPS is target_fps + 1
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
        After that, items are yielded at a rate between target_fps and target_fps+1.
        Optimized to ensure early frames maintain >= target_fps.

        Yields:
            T: Data from the buffer at the controlled rate
        """
        start_time = None  # Will be set when first controlled frame is output
        last_output_time = asyncio.get_event_loop().time()
        grace_period_frames = 3  # First 3 frames get preferential treatment

        while not self._stop_event.is_set():
            if not self.buffer:
                # If buffer is empty, wait a bit and check again
                await asyncio.sleep(0.001)
                continue

            # Check if we should output immediately (for initial items)
            if self.output_count < self.immediate_output_count:
                yield self.buffer.popleft()
                self.output_count += 1
                last_output_time = asyncio.get_event_loop().time()
                continue

            current_time = asyncio.get_event_loop().time()
            
            # Initialize start_time when we begin FPS-controlled output
            if start_time is None:
                start_time = current_time
            
            elapsed = current_time - start_time
            time_since_last = current_time - last_output_time
            
            # Calculate expected frame count based on elapsed time and target FPS
            # Start counting from when FPS control begins
            expected_frame_count = elapsed * self.fps
            actual_frame_count = self.output_count - self.immediate_output_count
            
            # Check if enough time has passed since last output (prevent output > target_fps+1)
            min_interval_ok = time_since_last >= self.min_frame_interval
            
            # Check if we need to output to maintain >= target_fps
            need_to_catch_up = actual_frame_count < expected_frame_count
            
            # Special handling for early frames to ensure good startup performance
            is_early_frame = self.output_count < (self.immediate_output_count + grace_period_frames)
            
            # Output conditions:
            should_output = False
            wait_time = 0
            
            if is_early_frame and min_interval_ok:
                # For early frames, output as soon as min interval is met
                should_output = True
            elif need_to_catch_up and min_interval_ok:
                # We're behind schedule and min interval has passed
                should_output = True
            elif not need_to_catch_up:
                # We're on schedule or ahead, calculate precise wait time
                next_frame_time = start_time + (actual_frame_count + 1) / self.fps
                wait_time = next_frame_time - current_time
                
                if wait_time <= 0.001 and min_interval_ok:
                    # Time to output next frame
                    should_output = True
                elif wait_time > 0.001:
                    # Wait for the right time, but also respect min interval
                    min_wait_time = self.min_frame_interval - time_since_last
                    actual_wait_time = max(wait_time, min_wait_time) if min_wait_time > 0 else wait_time
                    # Reduce wait time for early frames
                    if is_early_frame:
                        actual_wait_time = min(actual_wait_time, self.min_frame_interval)
                    await asyncio.sleep(min(actual_wait_time, self.frame_interval * 0.8))
                    continue
            else:
                # Need to catch up but haven't met minimum interval - wait a bit
                remaining_min_interval = self.min_frame_interval - time_since_last
                if remaining_min_interval > 0:
                    # Shorter wait for early frames
                    wait_duration = min(remaining_min_interval, 0.01)
                    if is_early_frame:
                        wait_duration = min(wait_duration, 0.005)  # Even shorter for early frames
                    await asyncio.sleep(wait_duration)
                continue
            
            if should_output and self.buffer:
                yield self.buffer.popleft()
                self.output_count += 1
                last_output_time = asyncio.get_event_loop().time()

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
    """Save the processed results to a video file with optimized async writing.

    Args:
        video_path (str): Path to save the output video.
        save_images (bool, optional): Whether to save individual frames as images. Defaults to False.
        total_frames (int, optional): Total number of frames for progress bar. Defaults to None.

    Returns:
        int: Number of frames processed.
    """
    count = 0
    pbar = None
    write_queue = asyncio.Queue(maxsize=50)  # Buffer up to 50 frames
    writer_task = None
    
    async def background_writer():
        """Background task to write frames to video file."""
        nonlocal writer
        while True:
            try:
                item = await write_queue.get()
                if item is None:  # Sentinel to stop
                    break
                frame, audio = item
                writer.write(frame)
                writer.write_audio_frame(audio)
                write_queue.task_done()
                # Small yield to prevent blocking
                await asyncio.sleep(0)
            except Exception as e:
                print(f"Error in background writer: {e}")
                break

    with LazyVideoWriter(
        video_path,
        fps=video_fps,
        audio_sr=audio_sr,
        save_images=save_images,
    ) as writer:
        # Start background writer task
        writer_task = asyncio.create_task(background_writer())
        
        try:
            async for result in frames_iter:
                if pbar is None:
                    pbar = tqdm(desc="Saving video", total=total_frames, disable=disable_progress)
                
                # Add to write queue (this will block if queue is full, providing backpressure)
                await write_queue.put((result.video_frame, result.audio_samples))
                count += 1
                pbar.update(1)
                
                # Yield occasionally to keep the stream flowing
                if count % 5 == 0:
                    await asyncio.sleep(0)
        
        finally:
            # Signal background writer to stop
            await write_queue.put(None)
            if writer_task:
                await writer_task

    if pbar is not None:
        pbar.close()

    print(f"Processed {count} frames")
    print(f"Saved to {video_path}")
    return count
