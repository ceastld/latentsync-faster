import numpy as np
import torch
import torchvision
from einops import rearrange


from dataclasses import dataclass


@dataclass
class LipsyncMetadata:
    face: np.ndarray
    box: np.ndarray
    affine_matrix: np.ndarray
    original_frame: np.ndarray
    sync_face: np.ndarray = None
    audio_feature: np.ndarray = None

    def set_sync_face(self, face: torch.Tensor):
        x1, y1, x2, y2 = self.box
        height = int(y2 - y1)
        width = int(x2 - x1)
        # 调整人脸大小
        face = torchvision.transforms.functional.resize(face, size=(height, width), antialias=True)
        face = rearrange(face, "c h w -> h w c")
        face = (face / 2 + 0.5).clamp(0, 1)
        face = (face * 255).to(torch.uint8).cpu().numpy()
        self.sync_face = face