import cv2
import numpy as np
from typing import List, Tuple

def resize_to_720p(frame: np.ndarray) -> np.ndarray:
    """将图像调整为720p大小，保持宽高比，不足部分用黑色填充
    
    Args:
        frame: 输入图像，RGB格式
        
    Returns:
        调整后的图像，RGB格式
    """
    target_height = 720
    target_width = 1280  # 16:9比例
    
    # 获取原始尺寸
    h, w = frame.shape[:2]
    
    # 计算缩放比例
    scale = min(target_height / h, target_width / w)
    
    # 计算新的尺寸
    new_h = int(h * scale)
    new_w = int(w * scale)
    
    # 缩放图像
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    
    # 创建黑色背景
    result = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    
    # 计算居中位置
    y_offset = (target_height - new_h) // 2
    x_offset = (target_width - new_w) // 2
    
    # 将缩放后的图像放在黑色背景中央
    result[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return result

def simulate_compression(frame: np.ndarray) -> np.ndarray:
    """降低分辨率再放大，模拟压缩失真
    
    Args:
        frame: 输入图像，RGB格式
        
    Returns:
        处理后的图像，RGB格式
    """
    h, w = frame.shape[:2]
    scale_factor = 0.5
    small_h = int(h * scale_factor)
    small_w = int(w * scale_factor)
    frame = cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
    frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)
    return frame

def add_slight_blur(frame: np.ndarray) -> np.ndarray:
    """添加轻微模糊
    
    Args:
        frame: 输入图像，RGB格式
        
    Returns:
        处理后的图像，RGB格式
    """
    kernel_size = np.random.choice([3, 5])
    frame = cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)
    return frame

def degrade_image_quality(frame: np.ndarray) -> np.ndarray:
    """降低图像质量并添加模糊
    
    Args:
        frame: 输入图像，RGB格式
        
    Returns:
        处理后的图像，RGB格式
    """
    # 1. 降低分辨率再放大，模拟压缩失真
    frame = simulate_compression(frame)
    
    # 2. 添加轻微模糊
    frame = add_slight_blur(frame)
    
    # 确保像素值在有效范围内
    frame = np.clip(frame, 0, 255).astype(np.uint8)
    
    return frame

def process_frame(frame: np.ndarray) -> np.ndarray:
    """处理单帧图像，包括质量降低和尺寸调整
    
    Args:
        frame: 输入图像，RGB格式
        
    Returns:
        处理后的图像，RGB格式
    """
    # 降低图像质量并添加模糊
    frame = degrade_image_quality(frame)
    # 调整到720p
    frame = resize_to_720p(frame)
    return frame

def process_frames(frames: List[np.ndarray]) -> List[np.ndarray]:
    """批量处理图像帧
    
    Args:
        frames: 输入图像帧列表，每个元素为RGB格式
        
    Returns:
        处理后的图像帧列表
    """
    return [process_frame(frame) for frame in frames] 