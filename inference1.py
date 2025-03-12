import numpy as np
import cv2
from latentsync.inference.lipsync_model import LipsyncModel, LipsyncContext
from latentsync.pipelines.lipsync_pipeline import LipsyncMetadata
import torch
from tqdm import tqdm

from latentsync.utils.util import read_audio
from latentsync.utils.video import save_frames_to_video


def run_inference(video_path: str, audio_path: str, output_path: str):
    # Initialize model and context
    model = LipsyncModel(device="cuda")
    context = LipsyncContext()
    model.infer_setup(context)

    batch_size = context.num_frames
    # Load video
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    audio_samples = read_audio(audio_path)
    num_audio_clips = (
        np.ceil(audio_samples.shape[0] / (16000 / 25) / batch_size) * batch_size
    )
    audio_samples = np.pad(
        audio_samples,
        (int(num_audio_clips * 16000 / 25 - audio_samples.shape[0])),
        mode="constant",
    )
    total_frames = min(total_frames, num_audio_clips)

    metadata_list = []
    processed_frames = []

    frame_idx = 0
    pbar = tqdm(total=total_frames, desc="Processing frames")
    while frame_idx < total_frames:
        # Process batch_size frames
        batch_metadata = []
        frames_in_batch = 0
        for _ in range(batch_size):
            ret, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if not ret:
                break

            metadata = model.process_frame(frame)

            batch_metadata.append(metadata)
            frames_in_batch += 1

        if not batch_metadata:
            break

        # Process the batch with audio
        start_audio_idx = int(frame_idx * 16000 / fps)
        end_audio_idx = int((frame_idx + batch_size) * 16000 / fps)
        audio_segment = audio_samples[start_audio_idx:end_audio_idx]

        # Run batch inference
        output_frames = model.process_batch(
            metadata_list=batch_metadata,
            audio_samples=audio_segment,
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
    output_path = "video_out1.mp4"

    run_inference(video_path, audio_path, output_path)
