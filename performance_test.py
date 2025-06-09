import asyncio
import logging
import time
from typing import List, Tuple
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS'] 
matplotlib.rcParams['axes.unicode_minus'] = False

# GPU monitoring
try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    print("GPUtil not available. GPU monitoring disabled. Install with: pip install GPUtil")
    GPU_AVAILABLE = False

logger = logging.getLogger(__name__)


def get_gpu_utilization() -> Tuple[float, float, float]:
    """Get current GPU utilization, memory usage, and temperature.
    
    Returns:
        tuple: (utilization_percent, memory_percent, temperature_celsius)
    """
    if not GPU_AVAILABLE:
        return 0.0, 0.0, 0.0
    
    try:
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]  # Use first GPU
            return gpu.load * 100, gpu.memoryUtil * 100, gpu.temperature
        else:
            return 0.0, 0.0, 0.0
    except Exception as e:
        # Fallback to nvidia-ml-py if GPUtil fails
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            
            # Get utilization
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            
            # Get memory info
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            mem_percent = (mem_info.used / mem_info.total) * 100
            
            # Get temperature
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            
            return util.gpu, mem_percent, temp
        except:
            return 0.0, 0.0, 0.0


def plot_frame_analysis(frame_times: List[float], frame_indices: List[int], delay: float = 0.0, 
                        gpu_data: List[Tuple[float, float, float, float]] = None) -> None:
    """Plot frame timing analysis including time and FPS curves with detailed stable-state analysis and GPU metrics
    
    Args:
        frame_times: List of frame processing times in seconds
        frame_indices: List of frame indices
        delay: Initial delay before first frame
        gpu_data: List of (timestamp, gpu_util, memory_util, temperature) tuples
    """
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
    
    # Find stable region (exclude initial high FPS anomalies)
    stable_start_idx = min(50, len(fps_values) // 3)  # Start analysis after 50 frames or 1/3 of frames
    stable_fps_values = fps_values[stable_start_idx:] if len(fps_values) > stable_start_idx else fps_values
    stable_frame_times = frame_times_s[stable_start_idx+1:] if len(frame_times_s) > stable_start_idx+1 else frame_times_s[1:]
    
    # Create subplots with GPU metrics
    has_gpu_data = gpu_data is not None and len(gpu_data) > 0
    if has_gpu_data:
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))  # 2x3 layout for GPU data
    else:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))  # 2x2 layout without GPU
    
    # Plot 1: Processing Time (s) vs Frame Index
    ax1 = axes[0, 0]
    ax1.plot(frame_times_s, frame_indices, 'b-', marker='o', markersize=4, linewidth=2)
    ax1.set_xlabel('Processing Time (s)')
    ax1.set_ylabel('Frame Index')
    ax1.set_title('Processing Progress: Time vs Frame Index')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, max(frame_indices) if frame_indices else 1)
    
    # Plot 2: Full FPS curve
    ax2 = axes[0, 1]
    frame_times_fps = frame_times[1:] if len(frame_times) > 1 else []
    if fps_values and frame_times_fps:
        ax2.plot(frame_times_fps, fps_values, 'r-', marker='s', markersize=3, linewidth=2)
        ax2.set_xlabel('Processing Time (s)')
        ax2.set_ylabel('Speed (FPS)')
        ax2.set_title(f'Full FPS Curve (window size: {window_size})')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, max(frame_times_fps))
        
        # Add target FPS line
        ax2.axhline(y=25, color='g', linestyle='--', alpha=0.7, label='Target 25 FPS')
        ax2.axhline(y=26, color='orange', linestyle='--', alpha=0.7, label='Max 26 FPS')
        ax2.legend()
    
    # Plot 3: GPU Utilization (if available)
    if has_gpu_data:
        ax3 = axes[0, 2]
        gpu_times = [item[0] for item in gpu_data]
        gpu_utils = [item[1] for item in gpu_data]
        memory_utils = [item[2] for item in gpu_data]
        
        ax3_twin = ax3.twinx()
        line1 = ax3.plot(gpu_times, gpu_utils, 'g-', linewidth=2, label='GPU Utilization')
        line2 = ax3_twin.plot(gpu_times, memory_utils, 'purple', linewidth=2, label='Memory Utilization')
        
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('GPU Utilization (%)', color='g')
        ax3_twin.set_ylabel('Memory Utilization (%)', color='purple')
        ax3.set_title('GPU & Memory Utilization')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 100)
        ax3_twin.set_ylim(0, 100)
        
        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax3.legend(lines, labels, loc='upper left')
    
    # Plot 4: Stable-state FPS analysis (zoomed in)
    ax4 = axes[1, 0]
    if stable_fps_values and stable_frame_times:
        ax4.plot(stable_frame_times, stable_fps_values, 'g-', marker='o', markersize=3, linewidth=2)
        ax4.set_xlabel('Processing Time (s)')
        ax4.set_ylabel('Speed (FPS)')
        ax4.set_title(f'Stable-State FPS (from frame {stable_start_idx+1})')
        ax4.grid(True, alpha=0.3)
        
        # Add target FPS lines
        ax4.axhline(y=25, color='r', linestyle='--', alpha=0.7, label='Target 25 FPS')
        ax4.axhline(y=26, color='orange', linestyle='--', alpha=0.7, label='Max 26 FPS')
        
        # Calculate and show stable FPS statistics
        stable_mean_fps = sum(stable_fps_values) / len(stable_fps_values)
        stable_min_fps = min(stable_fps_values)
        stable_max_fps = max(stable_fps_values)
        
        ax4.axhline(y=stable_mean_fps, color='blue', linestyle='-', alpha=0.8, 
                   label=f'Stable Mean: {stable_mean_fps:.2f} FPS')
        ax4.legend()
        
        # Set Y-axis limits to focus on stable range
        fps_range = stable_max_fps - stable_min_fps
        ax4.set_ylim(max(0, stable_min_fps - fps_range*0.1), stable_max_fps + fps_range*0.1)
    
    # Plot 5: Frame interval distribution
    ax5 = axes[1, 1]
    if time_diffs:
        stable_time_diffs = time_diffs[stable_start_idx:] if len(time_diffs) > stable_start_idx else time_diffs
        ax5.hist(stable_time_diffs, bins=30, alpha=0.7, color='purple', edgecolor='black')
        ax5.set_xlabel('Frame Interval (s)')
        ax5.set_ylabel('Frequency')
        ax5.set_title('Stable-State Frame Interval Distribution')
        ax5.grid(True, alpha=0.3)
        
        # Add target interval lines
        target_interval = 1/25  # 40ms for 25 FPS
        max_interval = 1/24     # ~41.67ms (minimum acceptable)
        ax5.axvline(x=target_interval, color='r', linestyle='--', alpha=0.7, label=f'Target: {target_interval*1000:.1f}ms')
        ax5.axvline(x=max_interval, color='orange', linestyle='--', alpha=0.7, label=f'Min acceptable: {max_interval*1000:.1f}ms')
        ax5.legend()
    
    # Plot 6: GPU Temperature (if available)
    if has_gpu_data:
        ax6 = axes[1, 2]
        gpu_temps = [item[3] for item in gpu_data]
        ax6.plot(gpu_times, gpu_temps, 'red', linewidth=2, marker='o', markersize=3)
        ax6.set_xlabel('Time (s)')
        ax6.set_ylabel('Temperature (°C)')
        ax6.set_title('GPU Temperature')
        ax6.grid(True, alpha=0.3)
        
        # Add temperature warning lines
        ax6.axhline(y=80, color='orange', linestyle='--', alpha=0.7, label='Warning: 80°C')
        ax6.axhline(y=90, color='red', linestyle='--', alpha=0.7, label='Critical: 90°C')
        ax6.legend()
    
    plt.tight_layout()
    plt.savefig('frame_analysis.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info("Frame analysis plot saved to frame_analysis.png")
    
    # Enhanced statistics with stable-state and GPU analysis
    if fps_values:
        total_frames = len(frame_times)
        processing_time = frame_times[-1]
        total_time = delay + processing_time
        overall_fps = total_frames / processing_time if processing_time > 0 else 0
        
        logger.info(f"=== FRAME ANALYSIS STATISTICS ===")
        logger.info(f"Initial delay: {delay:.3f} s")
        logger.info(f"Total frames: {total_frames}")
        logger.info(f"Processing time: {processing_time:.3f} s")
        logger.info(f"Overall FPS: {overall_fps:.2f}")
        
        # Stable-state analysis
        if stable_fps_values:
            stable_mean_fps = sum(stable_fps_values) / len(stable_fps_values)
            stable_min_fps = min(stable_fps_values)
            stable_max_fps = max(stable_fps_values)
            stable_std = (sum((x - stable_mean_fps)**2 for x in stable_fps_values) / len(stable_fps_values))**0.5
            
            # Count frames meeting target FPS
            frames_above_25 = sum(1 for fps in stable_fps_values if fps >= 25.0)
            frames_in_range = sum(1 for fps in stable_fps_values if 25.0 <= fps <= 26.0)
            
            logger.info(f"=== STABLE-STATE ANALYSIS (from frame {stable_start_idx+1}) ===")
            logger.info(f"Stable frames analyzed: {len(stable_fps_values)}")
            logger.info(f"Stable mean FPS: {stable_mean_fps:.3f}")
            logger.info(f"Stable FPS range: {stable_min_fps:.3f} - {stable_max_fps:.3f}")
            logger.info(f"Stable FPS std dev: {stable_std:.3f}")
            logger.info(f"Frames >= 25 FPS: {frames_above_25}/{len(stable_fps_values)} ({frames_above_25/len(stable_fps_values)*100:.1f}%)")
            logger.info(f"Frames in 25-26 FPS range: {frames_in_range}/{len(stable_fps_values)} ({frames_in_range/len(stable_fps_values)*100:.1f}%)")
            
            # Recent performance (last 50 frames)
            recent_fps = stable_fps_values[-50:] if len(stable_fps_values) >= 50 else stable_fps_values
            recent_mean = sum(recent_fps) / len(recent_fps)
            logger.info(f"Recent FPS (last {len(recent_fps)} stable frames): {recent_mean:.3f}")
        
        # GPU statistics
        if has_gpu_data:
            gpu_utils = [item[1] for item in gpu_data]
            memory_utils = [item[2] for item in gpu_data]
            gpu_temps = [item[3] for item in gpu_data]
            
            avg_gpu_util = sum(gpu_utils) / len(gpu_utils)
            avg_memory_util = sum(memory_utils) / len(memory_utils)
            avg_temp = sum(gpu_temps) / len(gpu_temps)
            max_temp = max(gpu_temps)
            
            logger.info(f"=== GPU ANALYSIS ===")
            logger.info(f"Average GPU utilization: {avg_gpu_util:.1f}%")
            logger.info(f"Average memory utilization: {avg_memory_util:.1f}%")
            logger.info(f"Average temperature: {avg_temp:.1f}°C")
            logger.info(f"Max temperature: {max_temp:.1f}°C")
        
        if time_diffs:
            avg_interval = sum(time_diffs) / len(time_diffs)
            logger.info(f"Average frame interval: {avg_interval:.3f} s ({avg_interval*1000:.1f} ms)")
            
            stable_time_diffs = time_diffs[stable_start_idx:] if len(time_diffs) > stable_start_idx else time_diffs
            if stable_time_diffs:
                stable_avg_interval = sum(stable_time_diffs) / len(stable_time_diffs)
                logger.info(f"Stable average interval: {stable_avg_interval:.3f} s ({stable_avg_interval*1000:.1f} ms)")
                
                # Count intervals meeting target
                target_intervals = sum(1 for interval in stable_time_diffs if interval <= 1/25)
                logger.info(f"Intervals <= 40ms: {target_intervals}/{len(stable_time_diffs)} ({target_intervals/len(stable_time_diffs)*100:.1f}%)")


async def speed_test_with_timing(async_iter, max_frames: int = None) -> None:
    """Test async iterator speed and record frame timing information with GPU monitoring"""
    frame_times: List[float] = []
    frame_indices: List[int] = []
    gpu_data: List[Tuple[float, float, float, float]] = []  # (timestamp, gpu_util, memory_util, temperature)
    frame_count = 0
    
    logger.info("Starting frame timing analysis with GPU monitoring...")
    start_time = time.time()
    first_frame_time = None
    
    # Background GPU monitoring task
    async def gpu_monitor():
        """Continuously monitor GPU utilization"""
        while True:
            try:
                current_time = time.time()
                if first_frame_time is not None:  # Only start monitoring after first frame
                    processing_time = current_time - first_frame_time
                    gpu_util, memory_util, temp = get_gpu_utilization()
                    gpu_data.append((processing_time, gpu_util, memory_util, temp))
                await asyncio.sleep(0.1)  # Sample every 100ms
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"GPU monitoring error: {e}")
                await asyncio.sleep(1.0)  # Wait longer on error
    
    # Start GPU monitoring task
    gpu_task = asyncio.create_task(gpu_monitor()) if GPU_AVAILABLE else None
    
    try:
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
    
    finally:
        # Stop GPU monitoring
        if gpu_task:
            gpu_task.cancel()
            try:
                await gpu_task
            except asyncio.CancelledError:
                pass
    
    logger.info(f"Processed {frame_count} frames")
    logger.info(f"Collected {len(gpu_data)} GPU samples")
    
    # Plot the analysis with delay and GPU information
    if frame_times and first_frame_time is not None:
        delay = first_frame_time - start_time
        plot_frame_analysis(frame_times, frame_indices, delay, gpu_data if gpu_data else None) 