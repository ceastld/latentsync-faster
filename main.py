import asyncio
import logging
import cv2
import torch
import time
from typing import List
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS'] 
matplotlib.rcParams['axes.unicode_minus'] = False
from tqdm import tqdm
from latentsync import GLOBAL_CONFIG, LatentSync, Timer
from latentsync.inference.utils import load_audio_clips
from latentsync.utils.video import LazyVideoWriter, get_total_frames, save_frames_to_video

logging.basicConfig(level=logging.INFO, format="[%(asctime)s.%(msecs)03d][%(levelname)s][%(name)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)


def plot_frame_analysis(frame_times: List[float], frame_indices: List[int], delay: float = 0.0) -> None:
    """Plot frame timing analysis including time and FPS curves"""
    if len(frame_times) < 2:
        logger.warning("Need at least 2 frames to calculate timing analysis")
        return
    
    # Keep frame_times in seconds (no conversion needed)
    frame_times_s = frame_times
    
    # Calculate time differences and smoothed FPS
    time_diffs = []
    fps_values = []
    window_size = 20  # Use 20 frames for smoothing
    
    # Calculate FPS using sliding window approach
    for i in range(1, len(frame_times)):
        diff = frame_times[i] - frame_times[i-1]
        time_diffs.append(diff)  # Keep in seconds
        
        # Calculate smoothed FPS using sliding window
        if i < window_size:
            # For early frames, use cumulative average
            window_frames = i + 1
            window_time = frame_times[i]
        else:
            # Use sliding window of recent frames
            window_frames = window_size
            window_time = frame_times[i] - frame_times[i - window_size]
        
        # Calculate FPS for this window
        smoothed_fps = window_frames / window_time if window_time > 0 else 0
        fps_values.append(smoothed_fps)
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Processing Time (s) vs Frame Index
    ax1.plot(frame_times_s, frame_indices, 'b-', marker='o', markersize=4, linewidth=2)
    ax1.set_xlabel('Processing Time (s)')
    ax1.set_ylabel('Frame Index')
    ax1.set_title('Processing Progress: Time vs Frame Index (excluding initial delay)')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, max(frame_indices) if frame_indices else 1)
    
    # Plot 2: Processing Time (s) vs Speed (FPS)
    frame_times_fps = frame_times[1:] if len(frame_times) > 1 else []
    if fps_values and frame_times_fps:
        ax2.plot(frame_times_fps, fps_values, 'r-', marker='s', markersize=4, linewidth=2)
        ax2.set_xlabel('Processing Time (s)')
        ax2.set_ylabel('Speed (FPS)')
        ax2.set_title(f'Smoothed Processing Speed Over Time (window size: {window_size})')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, max(frame_times_fps))
        
        # Add final smoothed FPS line
        final_fps = fps_values[-1] if fps_values else 0
        ax2.axhline(y=final_fps, color='g', linestyle='--', alpha=0.7, 
                   label=f'Final Smoothed FPS: {final_fps:.2f}')
        ax2.legend()
    
    plt.tight_layout()
    plt.savefig('frame_analysis.png', dpi=300, bbox_inches='tight')
    plt.close(fig)  # Close the figure to free memory
    logger.info("Frame analysis plot saved to frame_analysis.png")
    
    # Print statistics
    if fps_values:
        total_frames = len(frame_times)
        processing_time = frame_times[-1]  # Time from first frame to last frame
        total_time = delay + processing_time  # Including initial delay
        overall_fps = total_frames / processing_time if processing_time > 0 else 0
        
        logger.info(f"Frame analysis statistics:")
        logger.info(f"  Initial delay: {delay:.3f} s")
        logger.info(f"  Total frames: {total_frames}")
        logger.info(f"  Processing time: {processing_time:.3f} s (from first to last frame)")
        logger.info(f"  Total time: {total_time:.3f} s (including delay)")
        logger.info(f"  Processing FPS: {overall_fps:.2f} (global fps)")
        logger.info(f"  Overall FPS: {total_frames / total_time:.2f} (including delay)")
        logger.info(f"  Final smoothed FPS: {fps_values[-1]:.2f} (window size: {window_size})")
        logger.info(f"  Average frame interval: {sum(time_diffs) / len(time_diffs):.3f} s")


async def speed_test_with_timing(async_iter, max_frames: int = None) -> None:
    """Test async iterator speed and record frame timing information"""
    frame_times: List[float] = []
    frame_indices: List[int] = []
    frame_count = 0
    
    logger.info("Starting frame timing analysis...")
    start_time = time.time()
    first_frame_time = None
    
    async for frame in async_iter:
        current_time = time.time()
        
        # Record first frame time for delay calculation
        if first_frame_time is None:
            first_frame_time = current_time
            initial_delay = first_frame_time - start_time
            logger.info(f"Initial delay (before first frame): {initial_delay:.3f}s")
        
        # Calculate time relative to first frame output
        processing_time = current_time - first_frame_time
        frame_times.append(processing_time)
        frame_indices.append(frame_count)
        
        logger.info(f"Frame {frame_count}: processing time {processing_time:.3f}s")
        frame_count += 1
        
        # Check max_frames limit if specified
        if max_frames is not None and frame_count >= max_frames:
            break
    
    logger.info(f"Processed {frame_count} frames")
    
    # Plot the analysis with delay information
    if frame_times and first_frame_time is not None:
        delay = first_frame_time - start_time
        plot_frame_analysis(frame_times, frame_indices, delay)


async def speed_test(model: LatentSync, max_frames: int = 240) -> None:
    """Test model speed and record frame timing information"""
    example = GLOBAL_CONFIG.inference.obama
    model.push_video_stream(example.video_path, example.audio_path, max_frames=max_frames)
    # await model.save_to_video(example.video_out_path, total_frames=max_frames)
    
    # Use the new timing function with the result stream
    await speed_test_with_timing(model.result_stream())
    # async for frame in model.result_stream():
    #     pass


async def main():
    model = LatentSync(enable_progress=True, vae_type="kl", use_vad=True)
    await model.model_warmup()
    await speed_test(model)

if __name__ == "__main__":
    # Timer.enable()
    torch.backends.cuda.matmul.allow_tf32 = True
    asyncio.run(main())
    # audio_clips = load_audio_clips("assets/cxk.mp3", 625)
    # print(audio_clips[0])
    # Timer.summary()
    # asyncio.run(test())
