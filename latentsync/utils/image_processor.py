from torchvision import transforms
import cv2
from einops import rearrange
import torch
import numpy as np
from typing import List, Optional, Union, Tuple

from latentsync.pipelines.metadata import LipsyncMetadata
from latentsync.utils.timer import Timer
from .affine_transform import AlignRestore, laplacianSmooth
from PIL import Image
import torchvision.transforms.functional as TF
from ..face_detection import FaceLandmarkDetector

"""
If you are enlarging the image, you should prefer to use INTER_LINEAR or INTER_CUBIC interpolation. If you are shrinking the image, you should prefer to use INTER_AREA interpolation.
https://stackoverflow.com/questions/23853632/which-kind-of-interpolation-best-for-resizing-image
"""


def load_fixed_mask(resolution: int) -> torch.Tensor:
    mask_image = Image.open("latentsync/utils/mask.png").convert("RGB")
    mask_image = mask_image.resize((resolution, resolution), Image.Resampling.LANCZOS)
    mask_image = TF.to_tensor(mask_image)
    return mask_image


class FaceProcessor:
    """Class for face detection and affine transformation operations"""

    def __init__(self, resolution: int = 512, device: str = "cpu"):
        self.resolution = resolution
        self.device = device
        self.smoother = laplacianSmooth()
        self.restorer = AlignRestore()
        self.face_detector = FaceLandmarkDetector(
            device=self.device,
            detector_type="onnx",  # Default to ONNX detector
        )

    @torch.no_grad()
    def affine_transform(
        self, image: np.ndarray
    ) -> Tuple[torch.Tensor, list, np.ndarray]:
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
        # Step 1: Detect facial landmarks
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

        # Step 4: Resize image
        face = cv2.resize(
            face, (self.resolution, self.resolution), interpolation=cv2.INTER_LANCZOS4
        )
        face = rearrange(torch.from_numpy(face), "h w c -> c h w")

        return face, box, affine_matrix
    
    @torch.no_grad()
    def prepare_face(self, frame: np.ndarray):
        face, box, affine_matrix = self.affine_transform(frame)
        lipsync_metadata = LipsyncMetadata(
            face=face,
            box=box,
            affine_matrix=affine_matrix,
            original_frame=frame,
            sync_face=None
        )
        return lipsync_metadata

    @torch.no_grad()
    def prepare_face_batch(self, frames: List[np.ndarray]) -> Optional[List[LipsyncMetadata]]:
        """Process a batch of frames for facial preprocessing, using the same face alignment logic as the original method"""
        """Core function for face_processor"""
        if len(frames) == 0:
            return None

        metadata_list: List[LipsyncMetadata] = []

        # Use the same preprocessing logic as original code
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

class ImageProcessor:
    def __init__(
        self,
        resolution: int = 512,
        device: str = "cpu",
        mask_image=None,
    ):
        self.resolution = resolution
        self.resize = transforms.Resize(
            (resolution, resolution),
            interpolation=transforms.InterpolationMode.BILINEAR,
            antialias=True,
        )
        self.normalize = transforms.Normalize([0.5], [0.5], inplace=True)

        if mask_image is None:
            self.mask_image = load_fixed_mask(resolution)
        else:
            self.mask_image = mask_image

        self.device = device

    def preprocess_image(self, image: torch.Tensor):
        image = self.resize(image)
        pixel_values = self.normalize(image / 255.0)
        masked_pixel_values = pixel_values * self.mask_image
        return pixel_values, masked_pixel_values, self.mask_image[0:1]

    @Timer()
    def prepare_masks_and_masked_images(self, images: Union[torch.Tensor, np.ndarray]):
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images)
        if images.shape[3] == 3:
            images = rearrange(images, "b h w c -> b c h w")

        results = [self.preprocess_image(image) for image in images]

        pixel_values_list, masked_pixel_values_list, masks_list = list(zip(*results))
        return (
            torch.stack(pixel_values_list),
            torch.stack(masked_pixel_values_list),
            torch.stack(masks_list),
        )

    def process_images(self, images: Union[torch.Tensor, np.ndarray]):
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images)
        if images.shape[3] == 3:
            images = rearrange(images, "b h w c -> b c h w")
        images = self.resize(images)
        pixel_values = self.normalize(images / 255.0)
        return pixel_values
