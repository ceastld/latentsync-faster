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
import mediapipe as mp
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

        if mask in ["mouth", "face", "eye"]:
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True)  # Process single image
        if mask == "fix_mask":
            self.face_mesh = None
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

    def detect_facial_landmarks(self, image: np.ndarray):
        height, width, _ = image.shape
        results = self.face_mesh.process(image)
        if not results.multi_face_landmarks:  # Face not detected
            raise RuntimeError("Face not detected")
        face_landmarks = results.multi_face_landmarks[0]  # Only use the first face in the image
        landmark_coordinates = [
            (int(landmark.x * width), int(landmark.y * height)) for landmark in face_landmarks.landmark
        ]  # x means width, y means height
        return landmark_coordinates

    def preprocess_one_masked_image(self, image: torch.Tensor) -> np.ndarray:
        image = self.resize(image)

        if self.mask == "mouth" or self.mask == "face":
            landmark_coordinates = self.detect_facial_landmarks(image)
            if self.mask == "mouth":
                surround_landmarks = mouth_surround_landmarks
            else:
                surround_landmarks = face_surround_landmarks

            points = [landmark_coordinates[landmark] for landmark in surround_landmarks]
            points = np.array(points)
            mask = np.ones((self.resolution, self.resolution))
            mask = cv2.fillPoly(mask, pts=[points], color=(0, 0, 0))
            mask = torch.from_numpy(mask)
            mask = mask.unsqueeze(0)
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
        if self.face_mesh is not None:
            self.face_mesh.close()


def mediapipe_lm478_to_face_alignment_lm68(lm478, return_2d=True):
    """将MediaPipe的478点关键点转换为face_alignment的68点格式"""
    # 定义MediaPipe到68点的映射（简化）
    mapping = {
        # 轮廓点 (0-16)
        0: 127, 1: 234, 2: 93, 3: 132, 4: 58, 5: 172, 6: 136, 7: 150, 8: 176, 9: 148, 10: 152, 11: 377, 12: 400, 13: 378, 14: 379, 15: 365, 16: 397,
        # 眉毛点 (17-26)
        17: 70, 18: 63, 19: 105, 20: 66, 21: 107, 22: 336, 23: 296, 24: 334, 25: 293, 26: 300,
        # 鼻子点 (27-35)
        27: 6, 28: 168, 29: 197, 30: 195, 31: 5, 32: 4, 33: 98, 34: 97, 35: 2,
        # 眼睛点 (36-47)
        36: 33, 37: 160, 38: 158, 39: 133, 40: 153, 41: 144, 42: 362, 43: 385, 44: 387, 45: 263, 46: 373, 47: 380,
        # 嘴巴点 (48-67)
        48: 61, 49: 40, 50: 39, 51: 37, 52: 0, 53: 267, 54: 269, 55: 270, 56: 409, 57: 291, 58: 375, 59: 321, 60: 405, 61: 314, 62: 17, 63: 84, 64: 181, 65: 91, 66: 146, 67: 61
    }

    # lm478[..., 0] *= W
    # lm478[..., 1] *= H

    lm68 = np.zeros((68, 3 if not return_2d else 2))
    for i, j in mapping.items():
        if j < len(lm478):
            lm68[i, :2] = lm478[j, :2]
            if not return_2d and len(lm478[j]) > 2:
                lm68[i, 2] = lm478[j, 2]

    return lm68


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
