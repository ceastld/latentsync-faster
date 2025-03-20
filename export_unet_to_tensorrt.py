import argparse
import os
import tensorrt as trt
from typing import Dict, List, Tuple

def check_tensorrt_version():
    """检查 TensorRT 版本是否为 10.8.0.43"""
    current_version = trt.__version__
    required_version = "10.8.0.43"
    if current_version != required_version:
        raise RuntimeError(
            f"TensorRT 版本不匹配。当前版本: {current_version}, 需要版本: {required_version}\n"
            f"请确保使用正确的 TensorRT 版本。"
        )
    print(f"TensorRT 版本检查通过: {current_version}")

def parse_args():
    parser = argparse.ArgumentParser(description="将ONNX模型转换为TensorRT格式")
    parser.add_argument(
        "--onnx_path",
        type=str,
        default="checkpoints/unet.onnx",
        help="ONNX模型文件路径"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="checkpoints/latentsync_unet.engine",
        help="TensorRT引擎输出路径"
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="是否启用FP16精度"
    )
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=2,
        help="最大批次大小"
    )
    parser.add_argument(
        "--max_workspace_size",
        type=int,
        default=2 << 30,  # 2GB
        help="最大工作空间大小（字节）"
    )
    return parser.parse_args()

def build_engine(
    onnx_path: str,
    output_path: str,
    fp16: bool = False,
    max_batch_size: int = 2,
    max_workspace_size: int = 2 << 30
) -> None:
    """
    构建TensorRT引擎
    """
    # 检查 TensorRT 版本
    check_tensorrt_version()
    
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    
    # 解析ONNX模型
    print(f"正在解析ONNX模型: {onnx_path}")
    with open(onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            raise RuntimeError("ONNX模型解析失败")
    
    # 配置构建器
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, max_workspace_size)
    
    # 设置FP16精度
    if fp16 and builder.platform_has_fast_fp16:
        print("启用FP16精度")
        config.set_flag(trt.BuilderFlag.FP16)
    
    # 设置输入类型
    for i in range(network.num_inputs):
        input_tensor = network.get_input(i)
        if input_tensor.name == "timestep":
            input_tensor.dtype = trt.int64
            print(f"设置输入 {input_tensor.name} 的类型为 int64")
    
    # 创建引擎
    print("正在构建TensorRT引擎...")
    serialized_engine = builder.build_serialized_network(network, config)
    
    # 保存引擎
    print(f"正在保存TensorRT引擎到: {output_path}")
    with open(output_path, 'wb') as f:
        f.write(serialized_engine)
    
    print("TensorRT引擎构建完成！")

def main():
    args = parse_args()
    
    # 确保输出目录存在
    output_dir = os.path.dirname(os.path.abspath(args.output_path))
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # 构建引擎
    build_engine(
        args.onnx_path,
        args.output_path,
        args.fp16,
        args.max_batch_size,
        args.max_workspace_size
    )

if __name__ == "__main__":
    main() 