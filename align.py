import numpy as np
from latentsync.utils.affine_transform import transformation_from_points, AlignRestore
from latentsync.utils.video import VideoReader, save_frames_to_video
from latentsync import *

context = LipsyncContext()
face_processor = FaceProcessor(context.resolution, context.device)


def process_video(video_path: str):
    reader = VideoReader(video_path)
    for frame in reader:
        data: LipsyncMetadata = face_processor.prepare_face(frame)
        yield data.face


def main():
    demo = GLOBAL_CONFIG.inference.obama_top
    video_path = demo.video_path
    frames = process_video(video_path)
    output_path = "output/obama_aligned.mp4"
    save_frames_to_video(frames, output_path)
    print(f"Saved aligned video to {output_path}")


if __name__ == "__main__":
    main()
