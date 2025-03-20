import torch
import tensorrt as trt
import numpy as np
from typing import Optional, Union, List, Tuple
import pycuda.autoinit
import pycuda.driver as cuda
import os

class TRTModelWrapper(torch.nn.Module):
    """包装TensorRT模型，使其接口与UNet3DConditionModel保持一致"""
    
    def __init__(self, model_path: str, device: str = "cuda"):
        super().__init__()
        self.model_path = model_path
        
        # 检查模型文件是否存在
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"找不到TRT模型文件: {model_path}")
            
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
        
        # 创建执行上下文，使用用户管理的内存分配策略
        self.context = self.engine.create_execution_context_without_device_memory()
        
        # 创建CUDA流
        self.stream = cuda.Stream()
        
        # 获取模型输入输出信息并创建绑定
        self.bindings = []
        self.input_names = []
        self.output_names = []
        self.input_buffers = {}
        self.output_buffers = {}
        
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            size = int(np.prod(shape) * np.dtype(dtype).itemsize)
            
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
                # 为输入创建设备内存
                self.input_buffers[name] = cuda.mem_alloc(size)
                self.bindings.append(int(self.input_buffers[name]))
            else:
                self.output_names.append(name)
                # 为输出创建设备内存
                self.output_buffers[name] = cuda.mem_alloc(size)
                self.bindings.append(int(self.output_buffers[name]))
        
        print(f"模型输入: {self.input_names}")
        print(f"模型输出: {self.output_names}")
        
        # 配置属性以模拟UNet3DConditionModel
        self.config = type('Config', (), {
            'sample_size': 64,
            'cross_attention_dim': 384,
            'in_channels': 13,
            'out_channels': 4,
        })()
        
        # 分配工作空间内存
        workspace_size = self.engine.get_device_memory_size_for_profile_v2(0)  # 使用profile 0
        # 确保内存对齐到256字节
        workspace_size = (workspace_size + 255) & ~255
        self.workspace_buffer = cuda.mem_alloc(workspace_size)
        print(f"为workspace分配了{workspace_size/1024/1024:.2f}MB内存")
        
        # 设置设备内存
        self.context.set_device_memory(self.workspace_buffer, workspace_size)
        
        # 设置所有输入输出张量的地址
        for name in self.input_names:
            self.context.set_tensor_address(name, int(self.input_buffers[name]))
        for name in self.output_names:
            self.context.set_tensor_address(name, int(self.output_buffers[name]))
    
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
        
        # 设置输入形状
        for name in self.input_names:
            self.context.set_input_shape(name, inputs[name].shape)
        
        # 复制输入数据到GPU
        for name in self.input_names:
            cuda.memcpy_htod_async(self.input_buffers[name], inputs[name].ravel(), self.stream)
        
        # 同步流，确保数据已经复制到GPU
        self.stream.synchronize()
        
        # 执行推理
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        
        # 同步流，确保推理完成
        self.stream.synchronize()
        
        # 准备输出缓冲区
        outputs = {}
        for name in self.output_names:
            shape = self.engine.get_tensor_shape(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            outputs[name] = np.empty(shape, dtype=dtype)
        
        # 复制输出数据到CPU
        output_tensor = None
        for name in self.output_names:
            cuda.memcpy_dtoh_async(outputs[name].ravel(), self.output_buffers[name], self.stream)
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

def test_trt_model(model_path: str):
    """测试TRT模型是否能正常运行"""
    try:
        print("开始测试TRT模型...")
        
        # 创建模型实例
        model = TRTModelWrapper(model_path)
        
        # 创建测试输入
        batch_size = 1
        sample = torch.randn(2, 13, 8, 32, 32).cuda().half()  # 示例输入尺寸
        timestep = torch.tensor([999]).cuda()  # 示例时间步
        encoder_hidden_states = torch.randn(16, 50, 384).cuda().half()  # 示例编码器隐藏状态
        
        print("准备测试输入...")
        print(f"sample shape: {sample.shape}")
        print(f"timestep shape: {timestep.shape}")
        print(f"encoder_hidden_states shape: {encoder_hidden_states.shape}")
        
        # 运行推理
        print("开始推理...")
        with torch.no_grad():
            output = model(sample, timestep, encoder_hidden_states)
        
        print("推理完成!")
        print(f"输出类型: {type(output)}")
        if isinstance(output, dict):
            print(f"输出键: {output.keys()}")
            for k, v in output.items():
                print(f"{k} shape: {v.shape}")
        else:
            print(f"输出shape: {output.shape}")
            
        return True
        
    except Exception as e:
        print(f"测试失败: {str(e)}")
        return False

if __name__ == "__main__":
    # 测试模型
    model_path = "checkpoints/latentsync_unet.engine"  # 替换为实际的模型路径
    success = test_trt_model(model_path)
    if success:
        print("模型测试成功!")
    else:
        print("模型测试失败!") 