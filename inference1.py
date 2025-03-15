import numpy as np
from latentsync.inference.audio_infer import AudioProcessor
from latentsync.inference.context import LipsyncContext
from latentsync.inference.lipsync_model import LipsyncModel
from tqdm import tqdm
from latentsync.utils.image_processor import FaceProcessor
from latentsync.utils.timer import Timer
from latentsync.utils.video import save_frames_to_video, VideoReader
from latentsync.whisper.whisper.audio import load_audio


def run_inference(video_path: str, audio_path: str, output_path: str):
    # Initialize model and context
    context = LipsyncContext()

    lipsync_model = LipsyncModel(context)

    batch_size = context.num_frames
    audio_unit = context.samples_per_frame

    # Load video
    video_reader = VideoReader(video_path)
    fps = video_reader.fps
    total_frames = video_reader.total_frames

    # Load audio
    audio_samples = load_audio(audio_path)

    # Calculate number of audio clips needed
    num_audio_clips = int(np.ceil(audio_samples.shape[0] / audio_unit / batch_size) * batch_size)

    # Pad audio samples if necessary
    audio_samples = np.pad(
        audio_samples,
        (0, int(num_audio_clips * audio_unit - audio_samples.shape[0])),
        mode="constant",
    )

    # Limit total frames to available audio
    total_frames = min(total_frames, num_audio_clips)

    # Process video frames
    processed_frames = []
    frame_idx = 0

    # Keep track of last processed audio for context
    last_audio_samples = None

    face_processor = FaceProcessor(resolution=context.resolution, device=context.device)
    audio_processor = AudioProcessor(context)

    pbar = tqdm(total=total_frames, desc="Processing frames")

    while frame_idx < total_frames:
        batch_metadata = []

        frames_batch = video_reader.read_batch(batch_size)

        batch_metadata = face_processor.prepare_face_batch(frames_batch)

        if not batch_metadata:
            break

        current_audio_samples = audio_samples[
            int(frame_idx * audio_unit) : int((frame_idx + len(frames_batch)) * audio_unit)
        ]
        batch_audio_features = audio_processor.process_audio_with_pre(last_audio_samples, current_audio_samples)
        last_audio_samples = current_audio_samples

        output_frames = lipsync_model.process_batch(
            metadata_list=batch_metadata,
            audio_features=batch_audio_features,
        )

        processed_frames.extend(output_frames)
        frame_idx += len(frames_batch)
        pbar.update(len(frames_batch))

    pbar.close()
    video_reader.release()

    # Save output video
    if processed_frames:
        save_frames_to_video(processed_frames, output_path, audio_path, fps)


if __name__ == "__main__":
    # Example usage
    video_path = "assets/obama.mp4"
    audio_path = "assets/cxk.mp3"
    output_path = "video_out+audio_context8_batch8.mp4"

    # Timer.enable()
    run_inference(video_path, audio_path, output_path)
    Timer.print_stats()
