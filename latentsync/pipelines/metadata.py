import numpy as np
import torch
import torchvision
from einops import rearrange
from latentsync.utils.timer import Timer

from dataclasses import dataclass


@dataclass
class LipsyncMetadata:
    face: np.ndarray
    box: np.ndarray
    affine_matrix: np.ndarray
    original_frame: np.ndarray
    sync_face: np.ndarray = None
    audio_feature: np.ndarray = None
    
    # @Timer()
    def set_sync_face(self, face: torch.Tensor):
        face1 = face.clone().detach()
        x1, y1, x2, y2 = self.box
        height = int(y2 - y1)
        width = int(x2 - x1)
            
        face1 = torchvision.transforms.functional.resize(face1, size=(height, width), antialias=True)
        face1 = rearrange(face1, "c h w -> h w c")
        face1 = (face1 / 2 + 0.5).clamp(0, 1)
        face1 = (face1 * 255).to(torch.uint8)
        # face1 = face1.cpu().numpy()
        # return face1
        self.sync_face = face1