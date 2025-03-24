from tqdm import tqdm
from latentsync.configs.config import GLOBAL_CONFIG
from latentsync.face_detection import FaceLandmarkDetector
from latentsync.pipelines.metadata import LipsyncMetadata
from latentsync.utils.affine_transform import AlignRestore, laplacianSmooth
import cv2
import numpy as np
import torch
from einops import rearrange
from typing import List, Optional, Tuple
from latentsync.utils.timer import Timer
from latentsync.utils.video import VideoReader


class FaceProcessor:
    """Class for face detection and affine transformation operations"""

    def __init__(self, resolution: int = 512, device: str = "cpu"):
        self.resolution = resolution
        self.device = device
        self.smoother = laplacianSmooth()
        self.restorer = AlignRestore()
        self.face_detector = FaceLandmarkDetector(device=self.device)

    @torch.no_grad()
    def affine_transform(self, image: np.ndarray) -> Tuple[torch.Tensor, list, np.ndarray]:
        """
        Detect face landmarks and apply affine transformation to align the face

        Args:
            image: Input image tensor

        Returns:
            Tuple containing:
            - Transformed face image tensor
            - Bounding box coordinates [x1, y1, x2, y2]
            - Affine transformation matrix
        """

        detected_faces = self.face_detector.get_landmarks(image)
        if detected_faces is None:
            raise RuntimeError("Face not detected")
        lm68 = detected_faces[0]

        # Step 2: Smooth landmarks

        points = self.smoother.smooth(lm68)
        lmk3_ = np.zeros((3, 2))
        lmk3_[0] = points[17:22].mean(0)
        lmk3_[1] = points[22:27].mean(0)
        lmk3_[2] = points[27:36].mean(0)

        # Step 3: Calculate and apply affine transformation

        face, affine_matrix = self.restorer.align_warp_face(
            image.copy(), lmks3=lmk3_, smooth=True, border_mode="constant"
        )

        box = [0, 0, face.shape[1], face.shape[0]]  # x1, y1, x2, y2
        # face = torch.from_numpy(face)
        return face, box, affine_matrix

    @Timer()
    @torch.no_grad()
    def prepare_face(self, frame: np.ndarray) -> LipsyncMetadata:
        face, box, affine_matrix = self.affine_transform(frame)
        lipsync_metadata = LipsyncMetadata(
            face=face, box=box, affine_matrix=affine_matrix, original_frame=frame, sync_face=None
        )
        return lipsync_metadata

    @Timer()
    @torch.no_grad()
    def prepare_face_batch(self, frames: List[np.ndarray]) -> Optional[List[LipsyncMetadata]]:
        """Process a batch of frames for facial preprocessing, using the same face alignment logic as the original method"""
        """Core function for face_processor"""
        if len(frames) == 0:
            return None

        metadata_list: List[LipsyncMetadata] = []

        # 对第一帧进行预热,避免后续处理时的延迟
        # with Timer("prepare_face_batch"):
        #     if len(metadata_list) == 0:
        #         try:
        #             # 使用随机小图片进行预热
        #             warmup_img = np.random.randint(0, 255, (16, 16, 3), dtype=np.uint8)
        #             _ = self.prepare_face(warmup_img)
        #         except Exception as e:
        #             print(f"预热失败: {e}")
        #             pass
        for frame in frames:
            try:
                metadata = self.prepare_face(frame)
                metadata_list.append(metadata)
            except Exception as e:
                print(f"Face preprocessing failed: {e}")
                # If processing fails and there are other successfully processed frames, use the result of the previous frame
                if len(metadata_list) > 0:
                    metadata_list.append(metadata_list[-1])
                # Otherwise skip this frame

        if len(metadata_list) == 0:
            return None

        return metadata_list


def test_face_processor():
    face_processor = FaceProcessor(resolution=256, device="cuda")
    reader = VideoReader(GLOBAL_CONFIG.inference.demo1.video_path)
    for frame in tqdm(reader, desc="Processing frames", total=reader.total_frames):
        face_processor.prepare_face(frame)


def test_face_alignment():
    import face_alignment

    fan = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device="cuda")
    reader = VideoReader(GLOBAL_CONFIG.inference.default_video_path)
    for frame in tqdm(reader, desc="Processing frames", total=reader.total_frames):
        landmarks = fan.get_landmarks(frame)
        print(landmarks)


if __name__ == "__main__":
    Timer.enable()
    test_face_processor()
    Timer.summary()
