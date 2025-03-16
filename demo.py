import os
import gradio as gr
import numpy as np
import soundfile as sf
from latentsync.inference.context import LipsyncContext
from latentsync.inference.utils import create_pipeline, set_seed

# Initialize the lip sync context and pipeline
context = LipsyncContext()
pipeline = create_pipeline(context)

def process_video(video_path, audio_input, seed=1247):
    """
    Process the video and audio files to create lip-synced video
    """
    # Create output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)
    os.makedirs("temp", exist_ok=True)
    
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
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    audio_name = "input_audio"  # Use a fixed name since we're handling numpy array
    output_path = f"output/{video_name}_{audio_name}_synced.mp4"
    
    # Set random seed
    set_seed(seed)
    
    try:
        # Run the pipeline
        pipeline(
            video_path=video_path,
            audio_path=audio_path,
            video_out_path=output_path,
        )
        return output_path
    except Exception as e:
        raise gr.Error(f"处理失败: {str(e)}")
    finally:
        # Clean up temporary audio file
        if isinstance(audio_input, tuple) and os.path.exists(temp_audio_path):
            try:
                os.remove(temp_audio_path)
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
    description="Upload a video and audio file to create a lip-synced video.",
    examples=[
        # You can add example inputs here if you have any
    ],
    cache_examples=False
)

if __name__ == "__main__":
    demo.launch(share=True) 