# Adapted from https://github.com/guanjz20/StyleSync/blob/main/utils.py

import numpy as np
import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from latentsync.utils.timer import Timer


def transformation_from_points(points1, points0, smooth=True, p_bias=None):
    points2 = np.array(points0)
    points2 = points2.astype(np.float64)
    points1 = points1.astype(np.float64)
    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2
    s1 = np.std(points1)
    s2 = np.std(points2)
    points1 /= s1
    points2 /= s2
    U, S, Vt = np.linalg.svd(np.matmul(points1.T, points2))
    R = (np.matmul(U, Vt)).T
    sR = (s2 / s1) * R
    T = c2.reshape(2, 1) - (s2 / s1) * np.matmul(R, c1.reshape(2, 1))
    M = np.concatenate((sR, T), axis=1)
    if smooth:
        bias = points2[2] - points1[2]
        if p_bias is None:
            p_bias = bias
        else:
            bias = p_bias * 0.2 + bias * 0.8
        p_bias = bias
        M[:, 2] = M[:, 2] + bias
    return M, p_bias


class AlignRestore(object):
    def __init__(self, align_points=3, use_gpu=None, use_fp16=True):
        """
        初始化AlignRestore类。
        
        参数:
            align_points: 对齐点的数量
            use_gpu: 是否使用GPU。None表示自动检测。
            use_fp16: 是否使用半精度浮点数（FP16）。仅在使用GPU时有效。
        """
        if align_points == 3:
            self.upscale_factor = 1
            ratio = 2.8
            self.crop_ratio = (ratio, ratio)
            self.face_template = np.array([[19 - 2, 30 - 10], [56 + 2, 30 - 10], [37.5, 45 - 5]])
            self.face_template = self.face_template * ratio
            self.face_size = (int(75 * self.crop_ratio[0]), int(100 * self.crop_ratio[1]))
            self.p_bias = None
        
        # 检查GPU可用性
        self.has_cuda = torch.cuda.is_available()
        
        # 如果use_gpu是None，则自动检测；否则使用指定值
        if use_gpu is None:
            self.use_gpu = self.has_cuda
        else:
            self.use_gpu = use_gpu and self.has_cuda
        
        # self.use_gpu = False
            
        # 只有在使用GPU时，FP16才可用
        self.use_fp16 = use_fp16 and self.use_gpu
            
        # 初始化GPU辅助函数
        if self.use_gpu:
            self._init_gpu_helpers()

    def _init_gpu_helpers(self):
        """初始化GPU相关的辅助函数和变量"""
        # 预先初始化一些常用的GPU操作，以减少第一次调用restore_img时的延迟
        
        # 创建一个小的测试张量并上传到GPU
        test_tensor = torch.ones((3, 64, 64), device='cuda')
        if self.use_fp16:
            test_tensor = test_tensor.half()
            
        # 预热warp_affine_tensor操作
        test_matrix = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device='cuda')
        if self.use_fp16:
            test_matrix = test_matrix.half()
        _ = self.warp_affine_tensor(test_tensor, test_matrix.cpu().numpy(), (64, 64))
        
        # 预热erode_tensor操作
        test_mask = torch.ones((64, 64), device='cuda')
        if self.use_fp16:
            test_mask = test_mask.half()
        _ = self.erode_tensor(test_mask, 3)
        
        # 预热box_filter_tensor操作
        _ = self.box_filter_tensor(test_mask, 3)
        
        # 清理GPU内存
        torch.cuda.empty_cache()

    def to_tensor(self, img, permute=True):
        """将NumPy图像转换为PyTorch张量"""
        if not self.use_gpu:
            return img
        
        if isinstance(img, np.ndarray):
            tensor = torch.from_numpy(img).float()
        else:
            tensor = img

        if permute:
            tensor = tensor.permute(2, 0, 1)
            
        # 移动到GPU并可选地转换为FP16
        tensor = tensor.cuda()
        if self.use_fp16:
            tensor = tensor.half()
                
        return tensor
    
    def to_numpy(self, tensor, permute=True):
        """将PyTorch张量转换回NumPy数组"""
        if not isinstance(tensor, torch.Tensor):
            return tensor
            
        # 移回CPU并转为float32
        tensor = tensor.cpu().float()
        
        if permute and tensor.dim() == 3:
            # 从[C, H, W]转回[H, W, C]
            tensor = tensor.permute(1, 2, 0)
            
        return tensor.numpy()

    def warp_affine_tensor(self, src_tensor, matrix, size):
        """用PyTorch实现仿射变换，行为与OpenCV的warpAffine一致
        
        参数:
            src_tensor: 源图像张量，格式为 [C, H, W] 或 [B, C, H, W]
            matrix: 2x3 仿射变换矩阵，与OpenCV的warpAffine使用的格式相同
            size: 输出图像的大小，格式为 (height, width)
            
        返回:
            变换后的图像张量，与输入格式相同
        """
        
        # 以下是使用PyTorch的实现
        # 确保src_tensor是4D: [B, C, H, W]
        input_dim = src_tensor.dim()
        if input_dim == 2:
            src_tensor = src_tensor.unsqueeze(0).unsqueeze(0)
        elif input_dim == 3:
            src_tensor = src_tensor.unsqueeze(0)
            
        # 获取批次大小和通道数
        batch_size, num_channels, height, width = src_tensor.shape
        
        # 目标尺寸
        dst_h, dst_w = size

        # 获取输入张量的数据类型，后续创建的所有张量需要保持一致的类型
        dtype = src_tensor.dtype
        device = src_tensor.device
        
        # center_x = dst_w / 2 
        # center_y = dst_h / 2 


        
        # 分析仿射矩阵以检测旋转类型
        # 提取旋转部分（2x2子矩阵）
        rotation_matrix = matrix[:2, :2]
            
        # 检测矩阵是否包含旋转成分
        has_rotation = not (abs(rotation_matrix[0, 1]) < 1e-6 and abs(rotation_matrix[1, 0]) < 1e-6)
        
        # 创建目标坐标网格 - 注意使用与src_tensor相同的数据类型
        y_range = torch.arange(dst_h, device=device, dtype=dtype)
        x_range = torch.arange(dst_w, device=device, dtype=dtype)
        y, x = torch.meshgrid(y_range, x_range, indexing='ij')

        # 将变换矩阵转换为与src_tensor相同的数据类型
        matrix = torch.tensor(matrix, device=device, dtype=dtype)

        # 计算缩放因子（基于旋转矩阵的平方和）
        scale = torch.sqrt(matrix[0, 0] * matrix[0, 0] + matrix[1, 0] * matrix[1, 0])

        cos_theta = matrix[0, 0] / scale
        sin_theta = matrix[0, 1] / scale

        t_x = matrix[0, 2] * cos_theta - matrix[1, 2] * sin_theta
        t_y = matrix[0, 2] * sin_theta + matrix[1, 2] * cos_theta

        src_x = cos_theta / scale * x - sin_theta / scale * y - t_x / scale
        src_y = sin_theta / scale * x + cos_theta / scale * y - t_y / scale      
        
        # 处理精度问题，特别是对于旋转变换
        if has_rotation:
            src_x = src_x.clone()  # 创建副本以避免原地修改
            src_y = src_y.clone()
        
        # 调整到PyTorch的grid_sample期望的[-1,1]范围
        # 注意先将常数转换为与src_tensor相同的数据类型
        width_f = torch.tensor(width - 1, device=device, dtype=dtype)
        height_f = torch.tensor(height - 1, device=device, dtype=dtype)
        src_x_norm = 2.0 * (src_x / width_f) - 1.0
        src_y_norm = 2.0 * (src_y / height_f) - 1.0
        
        # 创建采样网格
        grid = torch.stack([src_x_norm, src_y_norm], dim=-1)
        
        # 扩展为批次维度
        grid = grid.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        
        # 使用grid_sample进行采样，设置align_corners=True以匹配OpenCV行为
        output = F.grid_sample(
            src_tensor, 
            grid, 
            mode='bilinear', 
            padding_mode='zeros', 
            align_corners=True  # 设置为True使像素中心对齐，更接近OpenCV
        )
        
        # 恢复原始维度
        if input_dim == 2:
            output = output.squeeze(0).squeeze(0)
        elif input_dim == 3:
            output = output.squeeze(0)
            
        return output

    def erode_tensor(self, tensor, kernel_size):
        """使用PyTorch的最大池化实现侵蚀操作"""
        # 确保输入是4D: [B, C, H, W]
        input_dim = tensor.dim()
        if input_dim == 2:
            tensor = tensor.unsqueeze(0).unsqueeze(0)
        elif input_dim == 3:
            tensor = tensor.unsqueeze(0)
            
        # 反转并执行最大池化（模拟侵蚀操作）
        inverted = 1.0 - tensor
        padding = kernel_size // 2
        pooled = F.max_pool2d(inverted, kernel_size=kernel_size, stride=1, padding=padding)
        eroded = 1.0 - pooled
        
        # 恢复原始维度
        if input_dim == 2:
            eroded = eroded.squeeze(0).squeeze(0)
        elif input_dim == 3:
            eroded = eroded.squeeze(0)
            
        return eroded

    def box_filter_tensor(self, tensor, kernel_size):
        """使用PyTorch的平均池化实现Box Filter"""
        # 确保输入是4D: [B, C, H, W]
        input_dim = tensor.dim()
        if input_dim == 2:
            tensor = tensor.unsqueeze(0).unsqueeze(0)
        elif input_dim == 3:
            tensor = tensor.unsqueeze(0)
            
        # 执行平均池化
        padding = kernel_size // 2
        filtered = F.avg_pool2d(tensor, kernel_size=kernel_size, stride=1, padding=padding)
        
        # 恢复原始维度
        if input_dim == 2:
            filtered = filtered.squeeze(0).squeeze(0)
        elif input_dim == 3:
            filtered = filtered.squeeze(0)
            
        return filtered

    def process(self, img, lmk_align=None, smooth=True, align_points=3):
        aligned_face, affine_matrix = self.align_warp_face(img, lmk_align, smooth)
        restored_img = self.restore_img(img, aligned_face, affine_matrix)
        return aligned_face, restored_img
    
    # @Timer()
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
        
        # img_t = img
        cropped_face = cv2.warpAffine(
            img,
            affine_matrix,
            self.face_size,
            # flags=cv2.INTER_LANCZOS4,
            # borderMode=border_mode,
            # borderValue=[127, 127, 127],
        )

        # cropped_face = cropped_face.permute(2, 1, 0).to(torch.uint8).cpu().numpy()
        
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
    
    # @Timer()
    def restore_img(self, input_img, face, affine_matrix):
        """
        还原图像的主函数，根据配置选择CPU或GPU版本的实现
        """
        if self.use_gpu:
            return self._restore_img_gpu(input_img, face, affine_matrix)
        else:
            return self._restore_img_cpu(input_img, face, affine_matrix)
        
    def _restore_img_cpu(self, input_img, face, affine_matrix):
        """使用CPU实现的还原图像函数"""
        # 步骤1: 初始化和调整大小
        h, w, _ = input_img.shape
        h_up, w_up = int(h * self.upscale_factor), int(w * self.upscale_factor)
        upsample_img = cv2.resize(input_img, (w_up, h_up), interpolation=cv2.INTER_LANCZOS4)
        
        # 步骤2: 创建反向仿射变换
        inverse_affine = cv2.invertAffineTransform(affine_matrix)
        inverse_affine *= self.upscale_factor
        if self.upscale_factor > 1:
            extra_offset = 0.5 * self.upscale_factor
        else:
            extra_offset = 0
        inverse_affine[:, 2] += extra_offset
        
        # 步骤3: 应用反向仿射变换
        inv_restored = cv2.warpAffine(face, inverse_affine, (w_up, h_up), flags=cv2.INTER_LANCZOS4)
        # 可视化 inv_restored
        cv2.imwrite("debug_inv_restored.png", cv2.cvtColor(inv_restored, cv2.COLOR_RGB2BGR))
        
        # 步骤4: 创建和变换蒙版
        mask = np.ones((self.face_size[1], self.face_size[0]), dtype=np.float32)
        inv_mask = cv2.warpAffine(mask, inverse_affine, (w_up, h_up))
        # 直接使用较小的固定大小内核进行侵蚀，避免计算动态大小
        kernel_size = max(2, int(self.upscale_factor * 2))
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        inv_mask_erosion = cv2.erode(inv_mask, kernel)
        
        # 步骤5: 计算面部区域和边缘 - 优化版
        # 使用广播操作扩展维度
        inv_mask_erosion_3d = inv_mask_erosion[:, :, np.newaxis]
        pasted_face = inv_mask_erosion_3d * inv_restored
        
        # 计算边缘大小 - 使用固定值代替动态计算
        # 对于512x512图像，w_edge约为5-6，对于1024x1024图像约为10-12
        # 我们使用固定大小，但根据图像尺寸调整
        w_edge = max(5, min(12, h_up // 100))
        erosion_radius = w_edge * 2
        
        # 创建简单结构元素用于侵蚀
        erode_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (erosion_radius, erosion_radius))
        inv_mask_center = cv2.erode(inv_mask_erosion, erode_kernel)
        
        # 使用更快的模糊方法：先用小核心做模糊，然后再用较大核心
        # 为速度优化，使用方框滤波代替高斯滤波
        blur_size = w_edge * 2
        if blur_size % 2 == 0:
            blur_size += 1
        inv_soft_mask = cv2.boxFilter(inv_mask_center, -1, (blur_size, blur_size))
        inv_soft_mask = np.expand_dims(inv_soft_mask, axis=2)
        
        # 步骤6: 最终图像合成 - 优化版
        # 直接使用numpy的混合操作
        result = inv_soft_mask * pasted_face + (1 - inv_soft_mask) * upsample_img
        
        # 检查数据类型以避免溢出
        if np.max(result) > 255:
            result = result.astype(np.uint16)
        else:
            result = result.astype(np.uint8)
        
        upsample_img = result
        
        return upsample_img
    
    # @Timer()
    def _restore_img_gpu(self, input_img, face, affine_matrix):
        """使用GPU实现的还原图像函数，支持FP16"""
        # 缓存输入图像，以便在ROI优化中使用
        self.last_input_img = input_img.copy()
        
        # 步骤1: 初始化和调整大小
        h, w, _ = input_img.shape
        h_up, w_up = h, w  # 由于upscale_factor==1，直接使用原始尺寸
        
        # 步骤2: 创建反向仿射变换
        inverse_affine = cv2.invertAffineTransform(affine_matrix)
        # 由于upscale_factor==1，不需要乘以缩放因子和额外偏移量
        
        # 计算处理区域边界
        # 获取脸部模板的四个角点
        face_h, face_w = self.face_size[1], self.face_size[0]
        corners = np.array([
            [0, 0],
            [face_w, 0],
            [face_w, face_h],
            [0, face_h]
        ], dtype=np.float32)
        
        # 使用逆仿射变换将角点映射回原图
        transformed_corners = cv2.transform(corners.reshape(1, -1, 2), inverse_affine).reshape(-1, 2)
        
        # 计算边界框
        x_min = max(0, int(np.floor(np.min(transformed_corners[:, 0]))))
        y_min = max(0, int(np.floor(np.min(transformed_corners[:, 1]))))
        x_max = min(w_up, int(np.ceil(np.max(transformed_corners[:, 0]))))
        y_max = min(h_up, int(np.ceil(np.max(transformed_corners[:, 1]))))
        
        # 向外扩展边界5个像素
        padding = 5
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(w_up, x_max + padding)
        y_max = min(h_up, y_max + padding)
        
        # 计算ROI尺寸
        roi_w = x_max - x_min
        roi_h = y_max - y_min
        
        # 如果ROI很小或接近整个图像，就不进行ROI优化
        if roi_w < 20 or roi_h < 20 or (roi_w > w_up * 0.9 and roi_h > h_up * 0.9):
            # 使用原始方法处理整个图像
            input_img_t = self.to_tensor(input_img)
            return self._restore_img_gpu_full(face, inverse_affine, input_img_t, h_up, w_up)
        
        # 否则，只处理ROI区域 - 优化：先裁剪，再转换为张量
        # 获取ROI区域
        roi_input_img = input_img[y_min:y_max, x_min:x_max]
        roi_input_img_t = self.to_tensor(roi_input_img)
        
        # 创建修改后的反向仿射变换矩阵，考虑ROI偏移
        roi_inverse_affine = inverse_affine.copy()
        roi_inverse_affine[:, 2] -= [x_min, y_min]
        
        return self._restore_img_gpu_roi_optimized(face, roi_inverse_affine, roi_input_img_t, input_img.shape, h_up, w_up, x_min, y_min, x_max, y_max)

    def _restore_img_gpu_roi_optimized(self, face, roi_inverse_affine, roi_input_img_t, full_img_shape, h_up, w_up, x_min, y_min, x_max, y_max):
        """优化版本的ROI区域处理实现，仅接收ROI区域的张量"""
        # 获取ROI尺寸
        roi_h = y_max - y_min
        roi_w = x_max - x_min
        
        # 步骤3: 仅对ROI区域应用反向仿射变换
        face_t = self.to_tensor(face)
        inv_restored_t = self.warp_affine_tensor(face_t, roi_inverse_affine, (roi_h, roi_w))

        # 步骤4: 创建和变换蒙版，仅对ROI区域
        mask = np.ones((self.face_size[1], self.face_size[0]), dtype=np.float32)
        mask_t = self.to_tensor(mask, permute=False)
        
        inv_mask_t = self.warp_affine_tensor(mask_t.unsqueeze(0), roi_inverse_affine, (roi_h, roi_w))
        inv_mask_t = inv_mask_t.squeeze(0)
        
        # 针对upscale_factor==1优化侵蚀核心大小
        kernel_size = 2  # 固定为2，因为upscale_factor==1
        inv_mask_erosion_t = self.erode_tensor(inv_mask_t, kernel_size)
        
        # 步骤5: 计算面部区域和边缘，仅对ROI区域
        inv_mask_erosion_3d = inv_mask_erosion_t.unsqueeze(0)
        
        c, h, w = inv_restored_t.shape
        if inv_mask_erosion_3d.shape[1] != h or inv_mask_erosion_3d.shape[2] != w:
            inv_mask_erosion_3d = F.interpolate(
                inv_mask_erosion_3d.unsqueeze(1), 
                size=(h, w), 
                mode='bilinear', 
                align_corners=False
            ).squeeze(1)
        
        # 直接使用ROI区域的张量
        roi_upsample_img_t = roi_input_img_t
        
        pasted_face_t = inv_mask_erosion_3d.unsqueeze(1) * inv_restored_t.unsqueeze(0)
        pasted_face_t = pasted_face_t.squeeze(0)
        
        # 针对upscale_factor==1优化边缘计算
        w_edge = 5  # 固定值，因为h_up==h且upscale_factor==1
        erosion_radius = w_edge * 2
        
        inv_mask_center_t = self.erode_tensor(inv_mask_erosion_t, erosion_radius)
        
        blur_size = w_edge * 2
        if blur_size % 2 == 0:
            blur_size += 1
        inv_soft_mask_t = self.box_filter_tensor(inv_mask_center_t, blur_size)
        
        if inv_soft_mask_t.shape[0] != h or inv_soft_mask_t.shape[1] != w:
            inv_soft_mask_t = F.interpolate(
                inv_soft_mask_t.unsqueeze(0).unsqueeze(0), 
                size=(h, w), 
                mode='bilinear', 
                align_corners=False
            ).squeeze(0).squeeze(0)
            
        inv_soft_mask_3c_t = inv_soft_mask_t.unsqueeze(0).unsqueeze(0)
        inv_soft_mask_3c_t = inv_soft_mask_3c_t.expand(-1, c, -1, -1)
        inv_soft_mask_3c_t = inv_soft_mask_3c_t.squeeze(0)
        
        # 步骤6: 仅在ROI区域合成
        roi_result_t = inv_soft_mask_3c_t * pasted_face_t + (1 - inv_soft_mask_3c_t) * roi_upsample_img_t
        
        # 处理ROI结果
        roi_result = self.to_numpy(roi_result_t)
        
        if np.max(roi_result) > 255:
            roi_result = roi_result.astype(np.uint16)
        else:
            roi_result = roi_result.astype(np.uint8)
        
        # 获取原始图像的副本，而不是创建全零数组
        # 我们需要先获取input_img，它是通过函数参数full_img_shape获取的
        full_h, full_w, _ = full_img_shape
        
        # 这里处理两种情况:
        # 1. 如果调用者传递了完整的input_img对象，则使用它
        # 2. 否则，我们需要重新构建输入图像
        if hasattr(self, 'last_input_img') and self.last_input_img.shape == full_img_shape:
            # 使用缓存的原始图像
            result = self.last_input_img.copy()
        else:
            # 如果没有原始图像缓存，则处理仅ROI区域的情况
            # 创建一个与原始图像相同大小的图像
            result = np.zeros(full_img_shape, dtype=np.uint8)
            
            # 可以在这里添加日志提示这种情况
            print("Warning: Using zero-initialized image as original image is not available")
            
        # 将ROI区域放回原图
        result[y_min:y_max, x_min:x_max] = roi_result
        
        # 清理GPU内存
        if self.use_gpu:
            torch.cuda.empty_cache()
        
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

    