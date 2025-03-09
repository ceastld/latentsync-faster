# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from torchvision import transforms
import cv2
from einops import rearrange
import torch
import numpy as np
from typing import Union
from .affine_transform import AlignRestore, laplacianSmooth
import face_alignment
from PIL import Image
import torchvision.transforms.functional as TF
import time
import os
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


class ImageProcessor:
    def __init__(self, resolution: int = 512, mask: str = "fix_mask", device: str = "cpu", mask_image=None, enable_timing=False):
        self.resolution = resolution
        self.resize = transforms.Resize(
            (resolution, resolution), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True
        )
        self.normalize = transforms.Normalize([0.5], [0.5], inplace=True)
        self.mask = mask

        if mask == "fix_mask":
            self.smoother = laplacianSmooth()
            self.restorer = AlignRestore()

            if mask_image is None:
                self.mask_image = load_fixed_mask(resolution)
            else:
                self.mask_image = mask_image

            self.face_detector = FaceLandmarkDetector(
                device=device,
                detector_type="onnx",  # 默认使用ONNX检测器
                enable_timing=enable_timing
            )
        elif mask in ["mouth", "face", "eye"]:
            print(f"警告: mask类型 '{mask}' 依赖MediaPipe，该功能已被移除。正在使用默认的'fix_mask'替代。")
            self.mask = "fix_mask"
            self.smoother = laplacianSmooth()
            self.restorer = AlignRestore()
            if mask_image is None:
                self.mask_image = load_fixed_mask(resolution)
            else:
                self.mask_image = mask_image
                
            self.face_detector = FaceLandmarkDetector(
                device=device,
                detector_type="onnx",
                enable_timing=enable_timing
            )

    def detect_facial_landmarks(self, image: np.ndarray):
        raise NotImplementedError("MediaPipe功能已被移除，此方法不再可用。请使用'fix_mask'模式或ONNX检测器。")

    def preprocess_one_masked_image(self, image: torch.Tensor) -> np.ndarray:
        image = self.resize(image)

        if self.mask in ["mouth", "face"]:
            print(f"警告: mask类型 '{self.mask}' 依赖MediaPipe，该功能已被移除。使用默认的全局mask。")
            mask = torch.ones((1, self.resolution, self.resolution))
        elif self.mask == "half":
            mask = torch.ones((self.resolution, self.resolution))
            height = mask.shape[0]
            mask[height // 2 :, :] = 0
            mask = mask.unsqueeze(0)
        elif self.mask == "eye":
            mask = torch.ones((self.resolution, self.resolution))
            landmark_coordinates = self.detect_facial_landmarks(image)
            y = landmark_coordinates[195][1]
            mask[y:, :] = 0
            mask = mask.unsqueeze(0)
        else:
            raise ValueError("Invalid mask type")

        image = image.to(dtype=torch.float32)
        pixel_values = self.normalize(image / 255.0)
        masked_pixel_values = pixel_values * mask
        mask = 1 - mask

        return pixel_values, masked_pixel_values, mask

    def affine_transform(self, image: torch.Tensor) -> np.ndarray:
        step_times = {}
        total_start = time.time()
        
        # 步骤1: 检测面部关键点
        landmarks_start = time.time()
        detected_faces = self.face_detector.get_landmarks(image)
        if detected_faces is None:
            raise RuntimeError("Face not detected")
        lm68 = detected_faces[0]
        landmarks_end = time.time()
        step_times['面部关键点检测'] = landmarks_end - landmarks_start
        
        # 步骤2: 关键点平滑
        smooth_start = time.time()
        points = self.smoother.smooth(lm68)
        lmk3_ = np.zeros((3, 2))
        lmk3_[0] = points[17:22].mean(0)
        lmk3_[1] = points[22:27].mean(0)
        lmk3_[2] = points[27:36].mean(0)
        smooth_end = time.time()
        step_times['关键点平滑'] = smooth_end - smooth_start
        
        # 步骤3: 计算并应用仿射变换
        transform_start = time.time()
        face, affine_matrix = self.restorer.align_warp_face(
            image.copy(), lmks3=lmk3_, smooth=True, border_mode="constant"
        )
        transform_end = time.time()
        step_times['仿射变换'] = transform_end - transform_start
        
        box = [0, 0, face.shape[1], face.shape[0]]  # x1, y1, x2, y2
        
        # 步骤4: 调整图像大小
        resize_start = time.time()
        face = cv2.resize(face, (self.resolution, self.resolution), interpolation=cv2.INTER_LANCZOS4)
        face = rearrange(torch.from_numpy(face), "h w c -> c h w")
        resize_end = time.time()
        step_times['调整图像大小'] = resize_end - resize_start
        
        total_end = time.time()
        step_times['总计'] = total_end - total_start
        
        # 打印每个步骤的时间
        for step, t in step_times.items():
            print(f"  - {step}: {t:.4f}秒")
            
        return face, box, affine_matrix

    def preprocess_fixed_mask_image(self, image: torch.Tensor, affine_transform=False):
        if affine_transform:
            image, _, _ = self.affine_transform(image)
        else:
            image = self.resize(image)
        pixel_values = self.normalize(image / 255.0)
        masked_pixel_values = pixel_values * self.mask_image
        return pixel_values, masked_pixel_values, self.mask_image[0:1]

    def prepare_masks_and_masked_images(self, images: Union[torch.Tensor, np.ndarray], affine_transform=False):
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images)
        if images.shape[3] == 3:
            images = rearrange(images, "b h w c -> b c h w")
        if self.mask == "fix_mask":
            results = [self.preprocess_fixed_mask_image(image, affine_transform=affine_transform) for image in images]
        else:
            results = [self.preprocess_one_masked_image(image) for image in images]

        pixel_values_list, masked_pixel_values_list, masks_list = list(zip(*results))
        return torch.stack(pixel_values_list), torch.stack(masked_pixel_values_list), torch.stack(masks_list)

    def process_images(self, images: Union[torch.Tensor, np.ndarray]):
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images)
        if images.shape[3] == 3:
            images = rearrange(images, "b h w c -> b c h w")
        images = self.resize(images)
        pixel_values = self.normalize(images / 255.0)
        return pixel_values

    def close(self):
        """关闭并释放资源"""
        if hasattr(self, 'face_detector'):
            self.face_detector.close()


def mediapipe_lm478_to_face_alignment_lm68(lm478, return_2d=True):
    """此功能已废弃，仅保留函数签名以兼容旧代码"""
    print("警告: mediapipe_lm478_to_face_alignment_lm68函数依赖MediaPipe，该功能已被移除。返回空结果。")
    if return_2d:
        return np.zeros((68, 2))
    else:
        return np.zeros((68, 3))


landmark_points_68 = [
    162,
    234,
    93,
    58,
    172,
    136,
    149,
    148,
    152,
    377,
    378,
    365,
    397,
    288,
    323,
    454,
    389,
    71,
    63,
    105,
    66,
    107,
    336,
    296,
    334,
    293,
    301,
    168,
    197,
    5,
    4,
    75,
    97,
    2,
    326,
    305,
    33,
    160,
    158,
    133,
    153,
    144,
    362,
    385,
    387,
    263,
    373,
    380,
    61,
    39,
    37,
    0,
    267,
    269,
    291,
    405,
    314,
    17,
    84,
    181,
    78,
    82,
    13,
    312,
    308,
    317,
    14,
    87,
]


# Refer to https://storage.googleapis.com/mediapipe-assets/documentation/mediapipe_face_landmark_fullsize.png
mouth_surround_landmarks = [
    164,
    165,
    167,
    92,
    186,
    57,
    43,
    106,
    182,
    83,
    18,
    313,
    406,
    335,
    273,
    287,
    410,
    322,
    391,
    393,
]

face_surround_landmarks = [
    152,
    377,
    400,
    378,
    379,
    365,
    397,
    288,
    435,
    433,
    411,
    425,
    423,
    327,
    326,
    94,
    97,
    98,
    203,
    205,
    187,
    213,
    215,
    58,
    172,
    136,
    150,
    149,
    176,
    148,
]
