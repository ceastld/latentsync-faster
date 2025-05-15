from tqdm import tqdm
from latentsync.configs.config import GLOBAL_CONFIG, LipsyncConfig
from latentsync.pipelines.metadata import LipsyncMetadata
from latentsync.pipelines.metadata import DetectedFace
from latentsync.utils.affine_transform import AlignRestore, laplacianSmooth
import cv2
import numpy as np
import torch
from einops import rearrange
from typing import List, Optional, Tuple
from latentsync.utils.timer import Timer
from latentsync.utils.video import VideoReader
import insightface


class InsightLandmarkDetector:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.face_analyzer = insightface.app.FaceAnalysis(
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            allowed_modules=["detection", "landmark_3d_68", "pose"],  # 只启用需要的模块
        )
        self.face_analyzer.prepare(ctx_id=0, det_size=(224, 224))

    def detect(self, image: np.ndarray) -> Optional[DetectedFace]:
        faces = self.face_analyzer.get(image)
        if not faces:
            return None
        face = max(faces, key=lambda face: face.det_score)
        return DetectedFace(
            bbox=face.bbox,
            landmark_3d_68=face.landmark_3d_68,
            det_score=face.det_score,
            pose=face.pose,
        )


class FaceProcessor:
    """Class for face detection and affine transformation operations"""

    def __init__(self, resolution: int = 512, device: str = "cpu"):
        self.resolution = resolution
        self.device = device
        self.smoother = laplacianSmooth()
        self.restorer = AlignRestore()
        self.face_detector = InsightLandmarkDetector(device=self.device)

    @torch.no_grad()
    def affine_transform(self, image: np.ndarray) -> Tuple[torch.Tensor, np.ndarray, DetectedFace]:
        """
        Detect face landmarks and apply affine transformation to align the face

        Args:
            image: Input image tensor

        Returns:
            Tuple containing:
            - Transformed face image tensor
            - Affine transformation matrix
        """
        detected_face = self.face_detector.detect(image)
        if detected_face is None:
            raise RuntimeError("Face not detected")
        lm68 = detected_face.landmark_2d_68

        # Step 2: Smooth landmarks
        points = self.smoother.smooth(lm68)
        lmk3_ = np.zeros((3, 2))
        lmk3_[0] = points[17:22].mean(0)
        lmk3_[1] = points[22:27].mean(0)
        lmk3_[2] = points[27:36].mean(0)

        # Step 3: Calculate and apply affine transformation
        face_image, affine_matrix = self.restorer.align_warp_face(
            image.copy(), lmks3=lmk3_, smooth=True, border_mode="constant"
        )

        return face_image, affine_matrix, detected_face

    @torch.no_grad()
    def prepare_face(self, frame: np.ndarray) -> LipsyncMetadata:
        face, affine_matrix, detected_face = self.affine_transform(frame)
        lipsync_metadata = LipsyncMetadata(
            face=face,
            detected_face=detected_face,
            affine_matrix=affine_matrix,
            original_frame=frame,
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
    
    def prepare_face_batch_with_interpolation(self, frames: List[np.ndarray]) -> Optional[List[LipsyncMetadata]]:
        """Process a batch of frames for facial preprocessing, using interpolation for every other frame
        to reduce computation while maintaining smooth transitions.
        
        This method processes frames at positions 0, 2, 4... (or 0, 2, 4, 6, 7 if odd length) directly,
        and interpolates landmarks for frames at positions 1, 3, 5...
        """
        if len(frames) == 0:
            return None
            
        metadata_list: List[LipsyncMetadata] = []
        processed_frames_indices = []
        landmarks_map = {}
        
        # Determine which frames to process directly
        if len(frames) % 2 == 0:  # Even number of frames
            # Process frames at indices 0, 2, 4, 6, ...
            processed_frames_indices = list(range(0, len(frames), 2))
        else:  # Odd number of frames
            # Process frames at indices 0, 2, 4, 6, ..., len(frames)-1
            processed_frames_indices = list(range(0, len(frames)-1, 2)) + [len(frames)-1]
            
        # First, process all the frames that need direct processing
        for idx in processed_frames_indices:
            try:
                frame = frames[idx]
                # Use the existing face detection and alignment
                face, affine_matrix, detected_face = self.affine_transform(frame)
                
                # Store the landmarks for later interpolation
                lm68 = detected_face.landmark_2d_68
                points = self.smoother.smooth(lm68)
                lmk3 = np.zeros((3, 2))
                lmk3[0] = points[17:22].mean(0)
                lmk3[1] = points[22:27].mean(0)
                lmk3[2] = points[27:36].mean(0)
                
                landmarks_map[idx] = lmk3
                
                # Create metadata for this frame
                metadata = LipsyncMetadata(
                    face=face,
                    detected_face=detected_face,
                    affine_matrix=affine_matrix,
                    original_frame=frame,
                )
                
                # Store in the list at the correct position
                while len(metadata_list) <= idx:
                    metadata_list.append(None)
                metadata_list[idx] = metadata
                
            except Exception as e:
                print(f"Face preprocessing failed for frame {idx}: {e}")
                # If we already have processed frames, use the last successful one
                if landmarks_map:
                    # Find the closest processed frame
                    closest_idx = min(landmarks_map.keys(), key=lambda k: abs(k - idx))
                    landmarks_map[idx] = landmarks_map[closest_idx]
                    
                    # If we have metadata for the closest frame, use it
                    if len(metadata_list) > closest_idx and metadata_list[closest_idx] is not None:
                        while len(metadata_list) <= idx:
                            metadata_list.append(None)
                        metadata_list[idx] = metadata_list[closest_idx]
                
        # Now process frames that need interpolation
        for idx in range(len(frames)):
            if idx in processed_frames_indices or idx not in range(len(frames)):
                continue  # Skip already processed frames
                
            try:
                frame = frames[idx]
                
                # Find the nearest processed frames before and after this frame
                prev_idx = max([i for i in processed_frames_indices if i < idx], default=None)
                next_idx = min([i for i in processed_frames_indices if i > idx], default=None)
                
                # If we can't interpolate, skip
                if prev_idx is None or next_idx is None:
                    continue
                    
                # Interpolate landmarks
                prev_lmk3 = landmarks_map[prev_idx]
                next_lmk3 = landmarks_map[next_idx]
                
                # Simple linear interpolation: a1 = (a0 + a2) / 2
                alpha = (idx - prev_idx) / (next_idx - prev_idx)
                interpolated_lmk3 = prev_lmk3 * (1 - alpha) + next_lmk3 * alpha
                
                # Apply affine transformation using the interpolated landmarks
                face_image, affine_matrix = self.restorer.align_warp_face(
                    frame.copy(), lmks3=interpolated_lmk3, smooth=True, border_mode="constant"
                )
                
                # Since we don't have detected face for interpolated frames, use the one from previous
                detected_face = metadata_list[prev_idx].detected_face
                
                # Create metadata for this interpolated frame
                metadata = LipsyncMetadata(
                    face=face_image,
                    detected_face=detected_face,
                    affine_matrix=affine_matrix,
                    original_frame=frame,
                )
                
                # Store in the list at the correct position
                while len(metadata_list) <= idx:
                    metadata_list.append(None)
                metadata_list[idx] = metadata
                
            except Exception as e:
                print(f"Interpolation failed for frame {idx}: {e}")
                # Use the previous metadata if available
                if prev_idx is not None and len(metadata_list) > prev_idx and metadata_list[prev_idx] is not None:
                    while len(metadata_list) <= idx:
                        metadata_list.append(None)
                    metadata_list[idx] = metadata_list[prev_idx]
        
        # Remove any None entries and ensure list is complete
        metadata_list = [m for m in metadata_list if m is not None]
        
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
