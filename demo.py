import os
import gradio as gr
import numpy as np
import soundfile as sf
import subprocess
import tempfile
from latentsync.inference.context import LipsyncContext
from latentsync.inference.utils import create_pipeline

# Global variables for lazy initialization
context = None
pipeline = None

def initialize_pipeline():
    """Initialize the pipeline lazily to avoid network issues during import"""
    global context, pipeline
    if context is None or pipeline is None:
        try:
            context = LipsyncContext()
            pipeline = create_pipeline(context)
        except Exception as e:
            raise gr.Error(f"初始化失败，请检查网络连接或模型文件: {str(e)}")
    return context, pipeline

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

def process_video(video_path, audio_input, seed=1247):
    """
    Process the video and audio files to create lip-synced video
    """
    # Initialize pipeline lazily
    try:
        context, pipeline = initialize_pipeline()
    except Exception as e:
        return None, f"初始化失败: {str(e)}"
    
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
    audio_name = "input_audio"  # Use a fixed name since we're handling numpy array
    output_path = f"output/{video_name}_{audio_name}_synced.mp4"
    
    try:
        # Run the pipeline with the 25fps video
        pipeline(
            video_path=temp_video_path,  # Use the converted 25fps video
            audio_path=audio_path,
            video_out_path=output_path,
        )
        return output_path
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
        # You can add example inputs here if you have any
    ],
    cache_examples=False
)

if __name__ == "__main__":
    # Set environment variables to avoid network issues
    os.environ["HF_HUB_OFFLINE"] = "1"  # Use offline mode for HuggingFace
    os.environ["TRANSFORMERS_OFFLINE"] = "1"  # Use offline mode for transformers
    
    # Launch with specific server settings to avoid network issues
    demo.launch(
        share=False,  # Disable share to avoid network issues
        server_name="127.0.0.1",  # Use localhost
        server_port=7860,
        show_error=True,
        quiet=False
    ) 