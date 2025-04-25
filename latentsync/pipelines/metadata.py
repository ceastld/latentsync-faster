from functools import cached_property
import numpy as np
import torch
import torchvision
from einops import rearrange
from latentsync.utils.image_processor import load_fixed_mask
from latentsync.utils.timer import Timer
from PIL import Image
from dataclasses import dataclass


@dataclass
class DetectedFace:
    bbox: np.ndarray
    landmark_3d_68: np.ndarray
    det_score: float
    pose: np.ndarray

    @property
    def landmark_2d_68(self):
        return self.landmark_3d_68[:, :2]

@dataclass
class AudioMetadata:
    audio_samples: np.ndarray
    audio_feature: np.ndarray

@dataclass
class LipsyncMetadata:
    original_frame: np.ndarray
    face: np.ndarray = None
    detected_face: DetectedFace = None
    affine_matrix: np.ndarray = None
    sync_face: np.ndarray = None
    audio_feature: np.ndarray = None
    audio_samples: np.ndarray = None
    restored_frame: np.ndarray = None

    def __post_init__(self):
        self.face_shape = None
        if self.face is not None:
            self.face_shape = self.face.shape[:2]

    @property
    def face_tensor(self):
        assert self.face is not None
        if isinstance(self.face, torch.Tensor):
            return self.face
        return torch.from_numpy(self.face)

    @property
    def audio_feature_tensor(self):
        assert self.audio_feature is not None
        if isinstance(self.audio_feature, torch.Tensor):
            return self.audio_feature
        return torch.from_numpy(self.audio_feature)

    @cached_property
    def mask_image(self):
        return 1-load_fixed_mask(256).to('cuda')

    @torch.no_grad()
    def set_sync_face(self, face: torch.Tensor):
        height, width = self.face_shape
        # face [-1, 1] -> [0, 255]
        face = (face / 2 + 0.5).clamp(0, 1)
        # face = face * (1 - self.mask_image) + (face * 0.5 + 0.5) * self.mask_image
        face = torchvision.transforms.functional.resize(face, size=(height, width), antialias=True)
        face = rearrange(face, "c h w -> h w c")
        # face = (face / 2 + 0.5).clamp(0, 1)
        face = (face * 255).to(torch.uint8)
        face = face.cpu().numpy()
        self.sync_face = face

    @property
    def lipsync_frame(self):
        if self.restored_frame is None:
            return self.original_frame
        return self.restored_frame
