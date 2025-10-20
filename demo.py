import os
import asyncio
import gradio as gr
import numpy as np
import soundfile as sf
import subprocess
import tempfile
import torch
import logging
from latentsync import LatentSync, GLOBAL_CONFIG
from latentsync.inference.utils import load_audio_clips

# Fix ONNX Runtime thread affinity issues in Slurm environment
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["ONNX_NUM_THREADS"] = "1"
os.environ["OMP_WAIT_POLICY"] = "PASSIVE"

# Setup logging
logging.basicConfig(level=logging.INFO, format="[%(asctime)s.%(msecs)03d][%(levelname)s][%(name)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

# Global model instance
model = None

def convert_video_to_25fps(input_video_path: str, output_video_path: str) -> str:
    """
    Convert input video to 25fps using ffmpeg
    """
    try:
        cmd = [
            'ffmpeg', '-i', input_video_path,
            '-r', '25',  # Set frame rate to 25fps
            '-c:v', 'libx264',  # Use H.264 codec
            '-preset', 'fast',  # Fast encoding
            '-y',  # Overwrite output file
            output_video_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return output_video_path
    except subprocess.CalledProcessError as e:
        raise gr.Error(f"视频转换失败: {e.stderr}")
    except Exception as e:
        raise gr.Error(f"视频转换出错: {str(e)}")

async def initialize_model():
    """Initialize the LatentSync model"""
    global model
    if model is None:
        try:
            model = LatentSync(enable_progress=True, vae_type="kl", use_vad=True)
            await model.model_warmup()
            logger.info("Model initialized successfully")
        except Exception as e:
            logger.error(f"Model initialization failed: {e}")
            raise gr.Error(f"模型初始化失败: {str(e)}")
    return model

def process_video(video_path, audio_input, seed=1247):
    """
    Process the video and audio files to create lip-synced video using LatentSync
    """
    # Create output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)
    os.makedirs("temp", exist_ok=True)
    
    # Convert video to 25fps first
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    temp_video_path = os.path.join("temp", f"{video_name}_25fps.mp4")
    
    try:
        # Convert input video to 25fps
        convert_video_to_25fps(video_path, temp_video_path)
    except Exception as e:
        raise gr.Error(f"视频帧率转换失败: {str(e)}")
    
    # Handle audio input (Gradio returns a tuple of (sample_rate, numpy_array) for audio)
    if isinstance(audio_input, tuple):
        sample_rate, audio_data = audio_input
        # Save audio data to a temporary file
        temp_audio_path = os.path.join("temp", "temp_audio.wav")
        sf.write(temp_audio_path, audio_data, sample_rate)
        audio_path = temp_audio_path
    else:
        audio_path = audio_input
    
    # Generate output path
    audio_name = "input_audio"
    output_path = f"output/{video_name}_{audio_name}_synced.mp4"
    
    try:
        # Run the processing using asyncio
        return asyncio.run(process_video_async(temp_video_path, audio_path, output_path))
    except Exception as e:
        raise gr.Error(f"处理失败: {str(e)}")
    finally:
        # Clean up temporary files
        if isinstance(audio_input, tuple) and os.path.exists(temp_audio_path):
            try:
                os.remove(temp_audio_path)
            except:
                pass
        # Clean up temporary video file
        if os.path.exists(temp_video_path):
            try:
                os.remove(temp_video_path)
            except:
                pass

async def process_video_async(video_path, audio_path, output_path):
    """Async processing function using LatentSync"""
    try:
        # Initialize model if needed
        model = await initialize_model()
        
        # Set random seed
        torch.manual_seed(1247)
        
        # Push video stream to model
        model.push_video_stream(video_path, audio_path, max_frames=240, max_input_fps=26)
        
        # Save to video
        await model.save_to_video(output_path, total_frames=240)
        
        logger.info(f"Processing completed: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Async processing failed: {e}")
        raise gr.Error(f"异步处理失败: {str(e)}")

# Create Gradio interface
demo = gr.Interface(
    fn=process_video,
    inputs=[
        gr.Video(label="Input Video"),
        gr.Audio(label="Input Audio"),
        gr.Slider(minimum=0, maximum=9999, value=1247, step=1, label="Random Seed")
    ],
    outputs=gr.Video(label="Lip-synced Video"),
    title="Lip Sync Demo",
    description="Upload a video and audio file to create a lip-synced video. Video will be automatically converted to 25fps for optimal processing.",
    examples=[
        ["assets/obama.mp4", "assets/cxk.mp3", 1247],
        ["assets/demo1_video.mp4", "assets/demo1_audio.wav", 1247],
        ["assets/demo2_video.mp4", "assets/demo2_audio.wav", 1247],
        ["assets/demo3_video.mp4", "assets/demo3_audio.wav", 1247],
    ],
    cache_examples=False
)

if __name__ == "__main__":
    # Enable TensorFloat-32 for better performance
    torch.backends.cuda.matmul.allow_tf32 = True
    
    # Launch Gradio interface
    demo.launch(
        share=False,
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True,
        quiet=False
    ) 