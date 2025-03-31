# LatentSync Usage Guide

## Basic Requirements

### Audio Requirements
- Sample rate: 16000Hz
- Format: numpy array (float32)
- Length: automatically padded to match frame rate

### Video Requirements
- Frame rate: 25fps
- Format: RGB images (h,w,3)
- Input: numpy array (uint8)

## Usage Example

```python
import asyncio
import cv2
import numpy as np
from latentsync import LatentSync

async def example():
    # Initialize model
    model = LatentSync(version="v15")
    
    # Push frames
    frame = cv2.imread("input.jpg")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    model.push_frames(frame)  # Single frame
    model.push_frames([frame] * 10)  # Multiple frames
    
    # Push audio (16000Hz sample rate)
    audio_data = load_audio("input.mp3")  # Ensure audio is 16000Hz
    model.push_audio(audio_data)
    
    # Mark end of input
    model.model.add_end_task()
    
    # Stream results
    frames = []
    async for frame in model.result_stream():
        frames.append(frame)
        # Process each frame as it's generated
        # process_frame(frame)
    
    # Save results
    from latentsync.utils.video import save_frames_to_video
    save_frames_to_video(frames, "output.mp4", audio_path="input.mp3")

# Run example
asyncio.run(example())
```

## Important Notes

1. Audio Processing
   - Ensure audio data is 16000Hz sample rate
   - Audio data will be automatically padded to match frame rate
   - Recommended to use `librosa` or `soundfile` for audio loading

2. Video Processing
   - Input video will be automatically adjusted to 25fps
   - Ensure input images are in RGB format
   - Use `cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)` for color space conversion

3. Performance Optimization
   - Use `result_stream()` for streaming processing
   - Avoid using `get_all_results()` for large datasets
   - Process frames in real-time without waiting for all results 