# Adapted from https://github.com/guanjz20/StyleSync/blob/main/utils.py

import numpy as np
import cv2
import time
import torch
import torch.nn.functional as F


def transformation_from_points(points1, points0, smooth=True, p_bias=None):
    points2 = torch.tensor(points0, dtype=torch.float32)
    points1 = torch.tensor(points1, dtype=torch.float32)
    c1 = torch.mean(points1, dim=0)
    c2 = torch.mean(points2, dim=0)
    points1 -= c1
    points2 -= c2
    s1 = torch.std(points1)
    s2 = torch.std(points2)
    points1 /= s1
    points2 /= s2
    U, S, V = torch.svd(torch.mm(points1.T, points2))
    R = torch.mm(U, V.T).T
    sR = (s2 / s1) * R
    T = c2.reshape(2, 1) - (s2 / s1) * torch.mm(R, c1.reshape(2, 1))
    M = torch.cat((sR, T), dim=1)
    if smooth:
        bias = points2[2] - points1[2]
        if p_bias is None:
            p_bias = bias
        else:
            bias = p_bias * 0.2 + bias * 0.8
        p_bias = bias
        M[:, 2] = M[:, 2] + bias
    return M.cpu().numpy(), p_bias


class AlignRestore(object):
    def __init__(self, align_points=3, device='cuda' if torch.cuda.is_available() else 'cpu'):
        if align_points == 3:
            self.upscale_factor = 1
            ratio = 2.8
            self.crop_ratio = (ratio, ratio)
            self.face_template = np.array([[19 - 2, 30 - 10], [56 + 2, 30 - 10], [37.5, 45 - 5]])
            self.face_template = self.face_template * ratio
            self.face_size = (int(75 * self.crop_ratio[0]), int(100 * self.crop_ratio[1]))
            self.p_bias = None
            self.device = device

    def process(self, img, lmk_align=None, smooth=True, align_points=3):
        aligned_face, affine_matrix = self.align_warp_face(img, lmk_align, smooth)
        restored_img = self.restore_img(img, aligned_face, affine_matrix)
        cv2.imwrite("restored.jpg", restored_img)
        cv2.imwrite("aligned.jpg", aligned_face)
        return aligned_face, restored_img

    def align_warp_face(self, img, lmks3, smooth=True, border_mode="constant"):
        # 计算仿射变换矩阵
        affine_matrix, self.p_bias = transformation_from_points(lmks3, self.face_template, smooth, self.p_bias)
        
        if border_mode == "constant":
            border_mode = cv2.BORDER_CONSTANT
        elif border_mode == "reflect101":
            border_mode = cv2.BORDER_REFLECT101
        elif border_mode == "reflect":
            border_mode = cv2.BORDER_REFLECT

        # 应用仿射变换
        cropped_face = cv2.warpAffine(
            img,
            affine_matrix,
            self.face_size,
            flags=cv2.INTER_LANCZOS4,
            borderMode=border_mode,
            borderValue=[127, 127, 127],
        )
        
        return cropped_face, affine_matrix

    def align_warp_face2(self, img, landmark, border_mode="constant"):
        affine_matrix = cv2.estimateAffinePartial2D(landmark, self.face_template)[0]
        if border_mode == "constant":
            border_mode = cv2.BORDER_CONSTANT
        elif border_mode == "reflect101":
            border_mode = cv2.BORDER_REFLECT101
        elif border_mode == "reflect":
            border_mode = cv2.BORDER_REFLECT
        cropped_face = cv2.warpAffine(
            img, affine_matrix, self.face_size, borderMode=border_mode, borderValue=(135, 133, 132)
        )
        return cropped_face, affine_matrix

    def restore_img(self, input_img, face, affine_matrix):
        # Convert inputs to PyTorch tensors
        input_img_t = torch.from_numpy(input_img).to(self.device).float()
        face_t = torch.from_numpy(face).to(self.device).float()
        
        h, w = input_img.shape[:2]
        h_up, w_up = int(h * self.upscale_factor), int(w * self.upscale_factor)
        
        # Only upsample if necessary
        if self.upscale_factor > 1:
            upsample_img_t = F.interpolate(
                input_img_t.permute(2, 0, 1).unsqueeze(0),
                size=(h_up, w_up),
                mode='bilinear',
                align_corners=False
            ).squeeze(0).permute(1, 2, 0)
            extra_offset = 0.5 * self.upscale_factor
        else:
            upsample_img_t = input_img_t
            extra_offset = 0
            
        # Convert affine matrix to tensor and compute inverse
        affine_matrix_t = torch.from_numpy(affine_matrix).to(self.device).float()
        inverse_affine = torch.from_numpy(cv2.invertAffineTransform(affine_matrix)).to(self.device).float()
        inverse_affine *= self.upscale_factor
        inverse_affine[:, 2] += extra_offset
        
        # Create transformation grid
        theta = inverse_affine.reshape(1, 2, 3)
        grid = F.affine_grid(theta, (1, 3, h_up, w_up), align_corners=False)
        
        # Warp face using grid sample
        face_t = face_t.permute(2, 0, 1).unsqueeze(0)
        inv_restored = F.grid_sample(face_t, grid, align_corners=False, mode='bilinear')[0].permute(1, 2, 0)
        
        # Create and transform mask
        mask = torch.ones((self.face_size[1], self.face_size[0]), device=self.device)
        mask_t = mask.unsqueeze(0).unsqueeze(0)
        inv_mask = F.grid_sample(mask_t, grid, align_corners=False, mode='bilinear')[0, 0]
        
        # Erosion using max pooling approximation
        pool_size = int(2 * self.upscale_factor)
        inv_mask_erosion = -F.max_pool2d(-inv_mask.unsqueeze(0).unsqueeze(0), 
                                        kernel_size=pool_size, 
                                        stride=1, 
                                        padding=pool_size//2)[0, 0]
        
        # Compute blending mask
        total_face_area = inv_mask_erosion.sum()
        w_edge = int(total_face_area.sqrt().item()) // 20
        blur_size = w_edge * 2
        if blur_size % 2 == 0:
            blur_size += 1
            
        # Gaussian blur using separable convolution
        sigma = blur_size / 3
        kernel_size = blur_size
        kernel = torch.arange(kernel_size, device=self.device) - (kernel_size - 1) / 2
        kernel = torch.exp(-kernel**2 / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        
        # Ensure proper padding for maintaining size
        padding = kernel_size // 2
        inv_mask_center = inv_mask_erosion.unsqueeze(0).unsqueeze(0)
        inv_soft_mask = F.conv2d(inv_mask_center, 
                                kernel.view(1, 1, -1, 1), 
                                padding=(padding, 0))
        inv_soft_mask = F.conv2d(inv_soft_mask, 
                                kernel.view(1, 1, 1, -1), 
                                padding=(0, padding))
        
        # Ensure mask has correct dimensions
        inv_soft_mask = inv_soft_mask[0, 0]
        
        # Add channel dimension and ensure shapes match
        inv_soft_mask = inv_soft_mask.unsqueeze(-1).expand(-1, -1, 3)
        
        # Ensure all tensors have the same shape
        if inv_soft_mask.shape != inv_restored.shape:
            inv_soft_mask = F.interpolate(
                inv_soft_mask.permute(2, 0, 1).unsqueeze(0),
                size=inv_restored.shape[:2],
                mode='bilinear',
                align_corners=False
            ).squeeze(0).permute(1, 2, 0)
        
        # Final blending with shape checking
        result = inv_soft_mask * inv_restored + (1 - inv_soft_mask) * upsample_img_t
        
        # Convert back to numpy and ensure proper type
        result = result.clamp(0, 255).cpu().numpy().astype(np.uint8)
        return result


class laplacianSmooth:
    def __init__(self, smoothAlpha=0.3):
        self.smoothAlpha = smoothAlpha
        self.pts_last = None

    def smooth(self, pts_cur):
        
        if self.pts_last is None:
            self.pts_last = pts_cur.copy()
            return pts_cur.copy()
            
        # 计算边界
        x1 = min(pts_cur[:, 0])
        x2 = max(pts_cur[:, 0])
        y1 = min(pts_cur[:, 1])
        y2 = max(pts_cur[:, 1])
        width = x2 - x1
        
        # 平滑算法
        pts_update = []
        for i in range(len(pts_cur)):
            x_new, y_new = pts_cur[i]
            x_old, y_old = self.pts_last[i]
            tmp = (x_new - x_old) ** 2 + (y_new - y_old) ** 2
            w = np.exp(-tmp / (width * self.smoothAlpha))
            x = x_old * w + x_new * (1 - w)
            y = y_old * w + y_new * (1 - w)
            pts_update.append([x, y])
        pts_update = np.array(pts_update)
        
        # 更新上一帧数据
        self.pts_last = pts_update.copy()

        return pts_update
