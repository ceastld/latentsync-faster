import numpy as np
import cv2
from latentsync.inference.context import LipsyncContext
from latentsync.inference.lipsync_model import LipsyncModel
from latentsync.pipelines.lipsync_diffusion_pipeline import LipsyncMetadata
import torch
from tqdm import tqdm
from latentsync.utils.util import read_audio
from latentsync.utils.video import save_frames_to_video, video_stream
from latentsync.whisper.whisper.audio import load_audio


def run_inference(video_path: str, audio_path: str, output_path: str):
    # Initialize model and context
    model = LipsyncModel(device="cuda")
    context = LipsyncContext()
    model.infer_setup(context)

    batch_size = context.num_frames
    samples_per_frame = context.samples_per_frame
    
    # Load video
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Load audio
    audio_samples = load_audio(audio_path)

    # Calculate number of audio clips needed
    num_audio_clips = int(
        np.ceil(audio_samples.shape[0] / samples_per_frame / batch_size) * batch_size
    )
    
    # Pad audio samples if necessary
    audio_samples = np.pad(
        audio_samples,
        (0, int(num_audio_clips * samples_per_frame - audio_samples.shape[0])),
        mode="constant",
    )

    # Limit total frames to available audio
    total_frames = min(total_frames, num_audio_clips)

    # Process video frames
    processed_frames = []
    frame_idx = 0
    pbar = tqdm(total=total_frames, desc="Processing frames")
    
    # Keep track of last processed audio for context
    last_audio_samples = None

    while frame_idx < total_frames:
        # Process batch_size frames
        batch_metadata = []
        frames_in_batch = 0
        
        for _ in range(batch_size):
            if frame_idx + frames_in_batch >= total_frames:
                break

            ret, frame = cap.read()
            if not ret:
                break
                
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            metadata = model.process_frame(frame)
            batch_metadata.append(metadata)
            frames_in_batch += 1

        if not batch_metadata:
            break
        
        # Calculate audio sample indices for current batch
        start_sample = int(frame_idx * samples_per_frame)
        end_sample = int((frame_idx + frames_in_batch) * samples_per_frame)
        
        # Process audio features for current batch
        batch_audio_features = []
        
        if last_audio_samples is None:
            current_audio_samples = audio_samples[start_sample:end_sample]
            batch_audio_features = model.process_audio(current_audio_samples)
        else:
            current_audio_samples = audio_samples[start_sample:end_sample]
            batch_audio_features = model.process_audio_with_pre(
                last_audio_samples, 
                current_audio_samples
            )
        
        # Save current audio samples as context for next batch
        last_audio_samples = audio_samples[start_sample:end_sample]
        
        # Run batch inference
        output_frames = model.process_batch(
            metadata_list=batch_metadata,
            audio_features=batch_audio_features,
        )

        processed_frames.extend(output_frames)
        frame_idx += frames_in_batch
        pbar.update(frames_in_batch)

    pbar.close()
    cap.release()

    # Save output video
    if processed_frames:
        save_frames_to_video(processed_frames, output_path, audio_path, fps)


if __name__ == "__main__":
    # Example usage
    video_path = "assets/obama.mp4"
    audio_path = "assets/cxk.mp3"
    output_path = "video_out+audio_context8_batch8.mp4"

    run_inference(video_path, audio_path, output_path)
