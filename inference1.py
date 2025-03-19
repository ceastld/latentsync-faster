from typing import List
from latentsync.configs.config import GLOBAL_CONFIG
from latentsync.inference.audio_infer import AudioProcessor
from latentsync.inference.context import LipsyncContext
from latentsync.inference.lipsync_model import LipsyncModel
from tqdm import tqdm
from latentsync.inference.utils import load_audio_clips
from latentsync.pipelines.metadata import LipsyncMetadata
from latentsync.utils.face_processor import FaceProcessor
from latentsync.utils.timer import Timer
from latentsync.utils.video import save_frames_to_video, VideoReader
import argparse


def run_inference(video_path: str, audio_path: str, output_path: str, use_onnx: bool = False):
    # Initialize model and context
    context = LipsyncContext(use_compile=False)

    lipsync_model = LipsyncModel(context, use_onnx=use_onnx)

    batch_size = context.num_frames

    # Load video
    video_reader = VideoReader(video_path)
    fps = video_reader.fps
    total_frames = video_reader.total_frames

    # Load audio
    audio_clips = load_audio_clips(audio_path, context.samples_per_frame)

    # Limit total frames to available audio
    total_frames = min(total_frames, len(audio_clips))

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
        current_audio_samples = audio_clips[frame_idx : frame_idx + len(frames_batch)].flatten()
        batch_audio_features = audio_processor.process_audio_with_pre(last_audio_samples, current_audio_samples)
        last_audio_samples = current_audio_samples
        output_metadata: List[LipsyncMetadata] = lipsync_model.process_batch(
            metadata_list=batch_metadata,
            audio_features=batch_audio_features,
        )
        # for data in output_metadata:
        #     processed_frames.append(data.restore_face())
        output_frames = lipsync_model.restore_batch(output_metadata)
        processed_frames.extend(output_frames)
        frame_idx += len(frames_batch)
        pbar.update(len(frames_batch))

    pbar.close()
    video_reader.release()

    # Save output video
    if processed_frames:
        save_frames_to_video(processed_frames, output_path, audio_path, fps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LatentSync 视频唇形同步")
    parser.add_argument("--onnx", action="store_true", help="使用ONNX模型加速")
    args = parser.parse_args()
    model_type = "ONNX" if args.onnx else "PyTorch"
    print(f"使用{model_type}模型进行推理...")

    Timer.enable()
    demo = GLOBAL_CONFIG.inference.demo1
    run_inference(demo.video_path, demo.audio_path, demo.video_out_path, use_onnx=args.onnx)
    Timer.summary()
    print(f"输出视频保存到: {demo.video_out_path}")
