import os
import numpy as np
import cv2
import torch
import onnxruntime as ort
from typing import List, Optional, Tuple, Union

from latentsync.utils.timer import Timer
from latentsync.configs.config import GLOBAL_CONFIG

from .utils.box_utils import hard_nms

class FaceLandmarkDetector:
    """统一的人脸关键点检测接口"""
    
    def __init__(self, 
                 device: str = "cuda", 
                 face_detector_path: Optional[str] = None,
                 landmark_detector_path: Optional[str] = None):
        """
        初始化人脸关键点检测器
        
        Args:
            device: 设备，'cpu'或'cuda'
            face_detector_path: 人脸检测器模型路径，如果为None则使用默认路径
            landmark_detector_path: 关键点检测器模型路径，如果为None则使用默认路径
        """
        self.device = device
        
        # 设置默认模型路径
        if face_detector_path is None:
            face_detector_path = GLOBAL_CONFIG.face_detector_path
        if landmark_detector_path is None:
            landmark_detector_path = GLOBAL_CONFIG.landmark_detector_path
        
        self.face_detector_path = face_detector_path
        self.landmark_detector_path = landmark_detector_path
        
        # 初始化ONNX检测器
        self._init_onnx_detector()
    
    def _init_onnx_detector(self):
        """初始化ONNX检测器，支持GPU加速"""
        # 检查文件是否存在
        if not os.path.exists(self.face_detector_path):
            raise FileNotFoundError(f"人脸检测模型未找到: {self.face_detector_path}")
        if not os.path.exists(self.landmark_detector_path):
            raise FileNotFoundError(f"关键点检测模型未找到: {self.landmark_detector_path}")
        
        # 配置ONNX运行时选项，启用GPU支持
        providers = []
        if "cuda" in str(self.device).lower():
            # 检查CUDA是否可用
            if 'CUDAExecutionProvider' in ort.get_available_providers():
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                print("使用CUDA加速ONNX推理")
            else:
                print("警告: CUDA不可用，回退到CPU")
                providers = ['CPUExecutionProvider']
                self.device = "cpu"
        else:
            providers = ['CPUExecutionProvider']
        
        # 创建会话选项
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # 初始化ONNX运行时会话
        self.face_detector = ort.InferenceSession(
            self.face_detector_path, 
            sess_options=session_options,
            providers=providers
        )
        self.face_detector_input_name = self.face_detector.get_inputs()[0].name
        
        self.landmark_detector = ort.InferenceSession(
            self.landmark_detector_path,
            sess_options=session_options,
            providers=providers
        )
        self.landmark_detector_input_name = self.landmark_detector.get_inputs()[0].name
    
    def detect_face(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        检测图像中的人脸
        
        Args:
            image: 输入图像
            
        Returns:
            Optional[np.ndarray]: 人脸边界框，如果未检测到则返回None
        """
        return self._detect_face_onnx(image)
    
    def _detect_face_onnx(self, image: np.ndarray) -> Optional[np.ndarray]:
        """使用ONNX模型检测人脸"""
        orig_size = image.shape
        
        # 预处理图像
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (320, 240))
        img_mean = np.array([127, 127, 127])
        image = (image - img_mean) / 128
        image = np.transpose(image, [2, 0, 1])
        image = np.expand_dims(image, axis=0)
        image = image.astype(np.float32)
        
        # 运行检测器
        # 关键耗时点：首次运行70ms，后续运行10ms
        # with Timer("face_detector"):
        confidences, boxes = self.face_detector.run(None, {self.face_detector_input_name: image})
        
        # 获取边界框
        boxes, labels, probs = self._predict_boxes(orig_size[1], orig_size[0], confidences, boxes, 0.7)
        
        if len(boxes) == 0:
            return None
        
        # 只取概率最高的人脸
        max_idx = np.argmax(probs)
        box = boxes[max_idx]
        
        return box
    
    def _predict_boxes(self, width, height, confidences, boxes, prob_threshold=0.7, iou_threshold=0.3, top_k=-1):
        """预测人脸边界框"""
        boxes = boxes[0]
        confidences = confidences[0]
        picked_box_probs = []
        picked_labels = []
        for class_index in range(1, confidences.shape[1]):
            probs = confidences[:, class_index]
            mask = probs > prob_threshold
            probs = probs[mask]
            if probs.shape[0] == 0:
                continue
            subset_boxes = boxes[mask, :]
            box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)
            box_probs = hard_nms(box_probs,
                               iou_threshold=iou_threshold,
                               top_k=top_k)
            picked_box_probs.append(box_probs)
            picked_labels.extend([class_index] * box_probs.shape[0])
        if not picked_box_probs:
            return np.array([]), np.array([]), np.array([])
        picked_box_probs = np.concatenate(picked_box_probs)
        picked_box_probs[:, 0] *= width
        picked_box_probs[:, 1] *= height
        picked_box_probs[:, 2] *= width
        picked_box_probs[:, 3] *= height
        return picked_box_probs[:, :4].astype(np.int32), np.array(picked_labels), picked_box_probs[:, 4]
    
    # @Timer()
    def get_landmarks(self, image: Union[np.ndarray, torch.Tensor]) -> Optional[List[np.ndarray]]:
        """
        获取图像中的面部关键点
        
        Args:
            image: 输入图像，可以是NumPy数组或Tensor
            
        Returns:
            Optional[List[np.ndarray]]: 包含每个检测到的人脸的关键点数组的列表，如果未检测到人脸则返回None
        """
        # 转换输入图像格式
        if isinstance(image, torch.Tensor):
            if image.dim() == 4 and image.shape[0] == 1:
                # 如果是batch中的第一个图像
                image = image[0]
            # 将PyTorch张量转换为NumPy数组
            if image.dim() == 3 and image.shape[0] == 3:
                # CHW格式，转换为HWC
                image = image.permute(1, 2, 0).cpu().numpy()
            else:
                image = image.cpu().numpy()
            
            # 将值范围从[-1, 1]或[0, 1]转换为[0, 255]
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
        
        return self._get_landmarks_onnx(image)
    
    def _get_landmarks_onnx(self, image: np.ndarray) -> Optional[List[np.ndarray]]:
        """使用ONNX模型获取关键点"""
        # 检测人脸
        box = self._detect_face_onnx(image)
        if box is None:
            print("未检测到人脸")
            return None
        
        # 预处理关键点输入
        face_input, bbox_info = self._preprocess_landmark_input(image, box)
        
        # 运行关键点检测
        face_input = face_input.astype(np.float32)
        landmark_outputs = self.landmark_detector.run(None, {self.landmark_detector_input_name: face_input})
        
        # 处理关键点输出
        landmarks = self._process_landmark_output(landmark_outputs[0], bbox_info)
        
        return [landmarks]
    
    def _preprocess_landmark_input(self, image: np.ndarray, box: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """预处理关键点检测的输入"""
        x1, y1, x2, y2 = box
        # 扩大边界框确保包含整个脸
        w = x2 - x1
        h = y2 - y1
        size = int(max([w, h]) * 1.1)
        cx = x1 + w//2
        cy = y1 + h//2
        x1 = cx - size//2
        x2 = x1 + size
        y1 = cy - size//2
        y2 = y1 + size
        
        # 确保边界框在图像范围内
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image.shape[1], x2)
        y2 = min(image.shape[0], y2)
        
        # 裁剪和调整大小
        face = image[y1:y2, x1:x2]
        face = cv2.resize(face, (56, 56))
        
        # 转换为模型输入格式
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = face.astype(np.float32)
        face = face / 255.0
        face = (face - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        face = np.transpose(face, (2, 0, 1))
        face = np.expand_dims(face, axis=0)
        
        return face, (x1, y1, x2-x1, y2-y1)
    
    def _process_landmark_output(self, landmarks: np.ndarray, bbox_info: Tuple[int, int, int, int]) -> np.ndarray:
        """处理关键点输出"""
        x, y, w, h = bbox_info
        landmarks = landmarks.reshape(-1, 2)
        
        # 将关键点从56x56的归一化坐标转换回原始图像坐标
        landmarks[:, 0] = landmarks[:, 0] * w + x
        landmarks[:, 1] = landmarks[:, 1] * h + y
        
        return landmarks
    
    def __del__(self):
        """析构函数，释放资源"""
        # 显式释放ONNX会话资源
        if hasattr(self, 'face_detector'):
            del self.face_detector
        if hasattr(self, 'landmark_detector'):
            del self.landmark_detector
    