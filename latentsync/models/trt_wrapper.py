import torch
import tensorrt as trt
import numpy as np
from typing import Optional, Union, List, Tuple
import pycuda.autoinit
import pycuda.driver as cuda

class TRTModelWrapper(torch.nn.Module):
    """包装TensorRT模型，使其接口与UNet3DConditionModel保持一致"""
    
    def __init__(self, model_path: str, device: str = "cuda"):
        super().__init__()
        self.model_path = model_path
        self.device = device
        
        # 添加虚拟属性以欺骗编译器
        self.dtype = torch.float32
        self._parameters = {}
        self._modules = {}
        self._buffers = {}
        self.add_audio_layer = True
        
        # 创建TensorRT运行时，使用WARNING级别减少日志输出
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        
        # 加载TensorRT引擎
        with open(model_path, 'rb') as f:
            self.engine_bytes = f.read()
        self.engine = self.runtime.deserialize_cuda_engine(self.engine_bytes)
        
        # 创建执行上下文
        self.context = self.engine.create_execution_context()
        
        # 创建CUDA流
        self.stream = cuda.Stream()
        
        # 获取模型输入输出信息
        self.input_names = []
        self.output_names = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)
        
        # 配置属性以模拟UNet3DConditionModel
        self.config = type('Config', (), {
            'sample_size': 64,
            'cross_attention_dim': 384,
            'in_channels': 13,
            'out_channels': 4,
        })()
        
        # 预分配CUDA内存
        self.cuda_buffers = {}
        for name in self.input_names + self.output_names:
            shape = self.engine.get_tensor_shape(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            # 确保size是Python内置的int类型
            size = int(np.prod(shape) * np.dtype(dtype).itemsize)
            self.cuda_buffers[name] = cuda.mem_alloc(size)
    
    def forward(
        self, 
        sample: torch.Tensor, 
        timestep: Union[torch.Tensor, float, int], 
        encoder_hidden_states: torch.Tensor,
        return_dict: bool = True,
        **kwargs
    ):
        """模拟UNet3DConditionModel的forward方法"""
        # 转换输入为numpy数组，确保timestep是1维张量
        if not isinstance(timestep, torch.Tensor):
            timestep = torch.tensor([timestep], dtype=torch.int64)
        elif timestep.ndim == 0:
            timestep = timestep.view(1)
            
        # 准备输入数据
        inputs = {
            "sample": sample.cpu().numpy(),
            "timestep": timestep.cpu().numpy(),
            "encoder_hidden_states": encoder_hidden_states.cpu().numpy()
        }
        
        # 分配输出缓冲区
        outputs = {}
        for name in self.output_names:
            shape = self.engine.get_tensor_shape(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            outputs[name] = np.empty(shape, dtype=dtype)
        
        # 分配输入缓冲区
        input_buffers = {}
        for name in self.input_names:
            shape = self.engine.get_tensor_shape(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            input_buffers[name] = inputs[name].astype(dtype)
        
        # 设置输入输出缓冲区
        for name in self.input_names:
            self.context.set_input_shape(name, input_buffers[name].shape)
            self.context.set_tensor_address(name, int(self.cuda_buffers[name]))
        for name in self.output_names:
            self.context.set_tensor_address(name, int(self.cuda_buffers[name]))
        
        # 复制输入数据到GPU
        for name in self.input_names:
            cuda.memcpy_htod_async(self.cuda_buffers[name], input_buffers[name].ravel(), self.stream)
        
        # 同步流，确保数据已经复制到GPU
        self.stream.synchronize()
        
        # 执行推理
        self.context.execute_async_v2(list(self.cuda_buffers.values()), self.stream.handle)
        
        # 同步流，确保推理完成
        self.stream.synchronize()
        
        # 复制输出数据到CPU
        output_tensor = None
        for name in self.output_names:
            cuda.memcpy_dtoh_async(outputs[name].ravel(), self.cuda_buffers[name], self.stream)
            output_tensor = torch.from_numpy(outputs[name]).to(sample.device)
        
        # 同步流，确保数据已经复制回CPU
        self.stream.synchronize()
        
        # 模拟UNet3DConditionOutput格式
        if return_dict:
            from dataclasses import dataclass
            
            @dataclass
            class UNet3DConditionOutput:
                sample: torch.FloatTensor
                
            return UNet3DConditionOutput(sample=output_tensor)
        else:
            return output_tensor
            
    def to(self, *args, **kwargs):
        """模拟to方法，但实际上TensorRT模型设备已在初始化时确定"""
        if 'dtype' in kwargs:
            self.dtype = kwargs['dtype']
        return self
        
    def eval(self):
        """模拟eval方法，TensorRT模型始终处于推理模式"""
        return self
        
    def train(self, mode=True):
        """模拟train方法，TensorRT模型始终处于推理模式"""
        return self
        
    def enable_xformers_memory_efficient_attention(self):
        """模拟启用xformers功能的方法，对TensorRT模型无效"""
        pass 

    def __getattr__(self, name):
        """处理所有未定义的属性访问"""
        if name.startswith('_'):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        
        if callable(getattr(torch.nn.Module, name, None)):
            def noop(*args, **kwargs):
                return self
            return noop
            
        return None 