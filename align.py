import numpy as np
from latentsync.inference.context import LipsyncContext
from latentsync.pipelines.metadata import LipsyncMetadata
from latentsync.utils.affine_transform import transformation_from_points, AlignRestore
from latentsync.utils.face_processor import FaceProcessor
from latentsync.utils.timer import Timer
from latentsync.utils.video import VideoReader, save_frames_to_video
from torchvision.transforms import ToTensor

context = LipsyncContext()
processor = FaceProcessor(context.resolution, context.device)


def process_video(video_path: str):
    reader = VideoReader(video_path)
    for frame in reader:
        data: LipsyncMetadata = processor.prepare_face(frame)
        yield data.face.numpy().transpose(1, 2, 0).astype(np.uint8)


def main():
    video_path = "assets/obama1.mp4"
    frames = process_video(video_path)
    output_path = "output/obama1_aligned.mp4"
    save_frames_to_video(frames, output_path)
    print(f"Saved aligned video to {output_path}")


if __name__ == "__main__":
    main()
