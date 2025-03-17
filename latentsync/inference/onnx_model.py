"""
ONNX模型加载和运行模块，提供与PyTorch UNet模型兼容的接口。
"""

import os
import torch
import numpy as np
import onnxruntime as ort
from typing import Optional, Union, Tuple, Dict, Any, List
from dataclasses import dataclass

from latentsync.configs.config import GLOBAL_CONFIG
from latentsync.models.unet import UNet3DConditionOutput

@dataclass
class ONNXConfig:
    """ONNX模型配置"""
    # 模型路径
    model_path: str
    # 模型精度
    fp16: bool = False
    # 线程数
    num_threads: int = 0
    # 内存优化等级
    optimization_level: int = 1
    # 是否启用GPU
    use_gpu: bool = True
    # 日志详细级别
    log_severity_level: int = 3  # 0-4, 0最详细
    # 执行提供商
    providers: Optional[List[Union[str, Tuple[str, Dict[str, Any]]]]] = None


class ONNXUNet:
    """ONNX UNet模型，提供与PyTorch UNet模型兼容的接口"""
    
    def __init__(self, config: ONNXConfig):
        """
        初始化ONNX UNet模型
        
        Args:
            config: ONNX模型配置
        """
        # if config is None:
        config = ONNXConfig(model_path="checkpoints/latentsync_unet.onnx", use_gpu=True, fp16=True)
        self.config = config
        self.model_path = config.model_path
        self.fp16 = config.fp16
        
        # 检查模型文件是否存在
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"ONNX模型文件不存在: {self.model_path}")
            
        # 创建会话选项
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = config.optimization_level
        
        if config.num_threads > 0:
            session_options.intra_op_num_threads = config.num_threads
            
        # 设置日志级别
        session_options.log_severity_level = config.log_severity_level
        
        # 默认执行提供商设置
        if config.providers is None:
            if config.use_gpu and 'CUDAExecutionProvider' in ort.get_available_providers():
                providers = [
                    ('CUDAExecutionProvider', {
                        'device_id': 0,
                        'arena_extend_strategy': 'kNextPowerOfTwo',
                        'gpu_mem_limit': 4 * 1024 * 1024 * 1024,
                        'cudnn_conv_algo_search': 'EXHAUSTIVE',
                    }),
                    'CPUExecutionProvider'
                ]
            else:
                providers = ['CPUExecutionProvider']
        else:
            providers = config.providers
        
        # 创建ONNX运行时会话
        print(f"正在加载ONNX模型: {self.model_path}")
        self.session = ort.InferenceSession(
            self.model_path,
            sess_options=session_options,
            providers=providers
        )
        
        # 获取模型的输入和输出名称
        self.input_names = [input.name for input in self.session.get_inputs()]
        self.output_names = [output.name for output in self.session.get_outputs()]
        
        print(f"ONNX模型加载完成，输入: {self.input_names}，输出: {self.output_names}")
        
        # 获取输入形状信息
        self.input_shapes = {input.name: input.shape for input in self.session.get_inputs()}
        print(f"输入形状: {self.input_shapes}")
        
        # 附加属性，模拟UNet模型
        self.add_audio_layer = False  # 默认不使用音频层，可以根据需要修改
        self.config_dict = {"sample_size": 64}  # 模拟UNet配置
        
    def __call__(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[UNet3DConditionOutput, Tuple]:
        """
        运行ONNX UNet模型推理，保持与PyTorch UNet模型相同的接口
        
        Args:
            sample: 输入样本，形状为 [batch, channel, frames, height, width]
            timestep: 时间步
            encoder_hidden_states: 编码器隐藏状态
            class_labels: 类别标签（可选）
            attention_mask: 注意力掩码（可选）
            down_block_additional_residuals: 下采样块额外的残差（用于ControlNet，可选）
            mid_block_additional_residual: 中间块额外的残差（用于ControlNet，可选）
            return_dict: 是否返回字典（可选，默认为True）
            
        Returns:
            UNet3DConditionOutput或元组: 与PyTorch UNet模型相同的输出格式
        """
        # 转换输入为NumPy数组
        inputs = {}
        
        # 处理sample输入
        if isinstance(sample, torch.Tensor):
            sample_np = sample.cpu().numpy()
            if self.fp16:
                sample_np = sample_np.astype(np.float16)
            else:
                sample_np = sample_np.astype(np.float32)
            inputs["sample"] = sample_np
        
        # 处理timestep输入
        if isinstance(timestep, torch.Tensor):
            timestep_np = timestep.cpu().numpy()
        elif isinstance(timestep, (int, float)):
            timestep_np = np.array([timestep], dtype=np.int64)
        else:
            timestep_np = timestep
        inputs["timestep"] = timestep_np
        
        # 处理encoder_hidden_states输入
        if isinstance(encoder_hidden_states, torch.Tensor):
            encoder_hidden_states_np = encoder_hidden_states.cpu().numpy()
            if self.fp16:
                encoder_hidden_states_np = encoder_hidden_states_np.astype(np.float16)
            else:
                encoder_hidden_states_np = encoder_hidden_states_np.astype(np.float32)
            inputs["encoder_hidden_states"] = encoder_hidden_states_np
        
        # 运行推理
        outputs = self.session.run(None, inputs)
        
        # 转换输出为PyTorch张量
        output_tensor = torch.from_numpy(outputs[0])
        
        # 如果需要移动到特定设备
        if sample.device.type != "cpu":
            output_tensor = output_tensor.to(sample.device)
        
        # 返回与PyTorch UNet模型相同的输出格式
        if not return_dict:
            return (output_tensor,)
        return UNet3DConditionOutput(sample=output_tensor)
    
    def eval(self):
        """模拟PyTorch模型的eval方法"""
        return self
    
    def to(self, device=None, dtype=None):
        """模拟PyTorch模型的to方法"""
        return self
    
    @property
    def config(self):
        """模拟UNet模型的config属性"""
        return self.config_dict


def create_onnx_unet(model_path: str, use_gpu: bool = True, fp16: bool = False) -> ONNXUNet:
    """
    创建ONNX UNet模型实例，与LipsyncContext.create_unet()方法的使用方式类似
    
    Args:
        model_path: ONNX模型文件路径
        use_gpu: 是否使用GPU加速
        fp16: 是否使用FP16精度
        
    Returns:
        ONNXUNet: ONNX UNet模型实例
    """
    config = ONNXConfig(
        model_path=model_path,
        fp16=fp16,
        use_gpu=use_gpu,
    )
    return ONNXUNet(config)


def patch_context_for_onnx(context, onnx_model_path: str, use_gpu: bool = True, fp16: bool = False):
    """
    修补LipsyncContext，替换create_unet方法以使用ONNX模型
    
    Args:
        context: LipsyncContext实例
        onnx_model_path: ONNX模型文件路径
        use_gpu: 是否使用GPU加速
        fp16: 是否使用FP16精度
        
    Returns:
        LipsyncContext: 修补后的LipsyncContext实例
    """
    # 保存原始方法
    original_create_unet = context.create_unet
    
    # 替换create_unet方法
    def create_onnx_unet_wrapper():
        return create_onnx_unet(onnx_model_path, use_gpu, fp16)
    
    # 替换方法
    context.create_unet = create_onnx_unet_wrapper
    
    return context 