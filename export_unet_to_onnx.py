import argparse
import torch
import os
from omegaconf import OmegaConf
from typing import Union, Tuple, Optional

from latentsync.models.unet import UNet3DConditionModel
from latentsync.inference.context import LipsyncContext, GLOBAL_CONFIG
from latentsync.utils.util import zero_rank_log

def parse_args():
    parser = argparse.ArgumentParser(description="将UNet模型导出为ONNX格式")
    parser.add_argument(
        "--model_path", 
        type=str, 
        required=True,
        help="UNet模型权重文件路径"
    )
    parser.add_argument(
        "--output_path", 
        type=str, 
        default="unet.onnx",
        help="ONNX输出文件路径"
    )
    parser.add_argument(
        "--sample_batch_size", 
        type=int, 
        default=1,
        help="样本批次大小"
    )
    parser.add_argument(
        "--sample_frames", 
        type=int, 
        default=8,
        help="样本帧数"
    )
    parser.add_argument(
        "--sample_height", 
        type=int, 
        default=512//8,
        help="样本高度"
    )
    parser.add_argument(
        "--sample_width", 
        type=int, 
        default=512//8,
        help="样本宽度"
    )
    parser.add_argument(
        "--seq_length", 
        type=int, 
        default=77,
        help="序列长度"
    )
    parser.add_argument(
        "--use_external_data", 
        action="store_true",
        help="是否将权重存储为外部数据文件"
    )
    parser.add_argument(
        "--fp16", 
        action="store_true",
        help="是否将模型量化为FP16精度"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"正在创建LipsyncContext...")
    context = LipsyncContext()
    
    print(f"正在加载UNet模型: {args.model_path}")
    model_config = OmegaConf.to_container(GLOBAL_CONFIG.unet_config.model)
    unet, _ = UNet3DConditionModel.from_pretrained(
        model_config,
        args.model_path,
        device="cpu"
    )
    
    # 设置为评估模式
    unet.eval()
    
    # 如果需要FP16量化
    if args.fp16:
        print("将模型转换为FP16精度...")
        unet = unet.half()
    
    # 获取cross_attention_dim
    cross_attention_dim = model_config.get("cross_attention_dim", 1280)
    print(f"Cross Attention Dimension: {cross_attention_dim}")
    
    # 创建模型输入的示例数据
    # 样本维度: [batch_size, channels, frames, height, width]
    sample = torch.randn(
        2, 
        13,  # 通道数 
        8, 
        32, 
        32
    )
    
    # 如果使用FP16，也将输入转为half精度
    if args.fp16:
        sample = sample.half()
    
    # 时间步
    timestep = torch.tensor([0])
    
    # 编码器隐藏状态
    encoder_hidden_states = torch.randn(
        16, 
        50, 
        384
    )
    
    # 如果使用FP16，也将编码器隐藏状态转为half精度
    if args.fp16:
        encoder_hidden_states = encoder_hidden_states.half()
    
    # 导出为ONNX格式
    print(f"正在导出UNet模型为ONNX格式: {args.output_path}")
    
    # 定义动态轴
    dynamic_axes = {
        "sample": {0: "batch_size", 2: "frames", 3: "height", 4: "width"},
        "timestep": {0: "batch_size"},
        "encoder_hidden_states": {0: "batch_size", 1: "sequence_length"}
    }
    
    # 确保输出目录存在
    output_dir = os.path.dirname(os.path.abspath(args.output_path))
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # 准备导出参数
    export_kwargs = {
        "f": args.output_path,
        "export_params": True,
        "opset_version": 17,
        "do_constant_folding": True,
        "input_names": ["sample", "timestep", "encoder_hidden_states"],
        "output_names": ["output"],
        "dynamic_axes": dynamic_axes,
        "verbose": False
    }
    
    # 是否使用外部数据
    if args.use_external_data:
        print("使用外部数据格式，权重将存储在单独的文件中")
        # 为大模型设置外部数据选项
        external_data_filename = os.path.basename(args.output_path) + ".weight"
        export_kwargs["external_data_format"] = True
        export_kwargs["external_data_filename"] = external_data_filename
        export_kwargs["size_threshold"] = 1024 * 1024 * 1024  # 1GB以上的张量存储到外部
    else:
        print("使用单一文件格式，所有权重将存储在ONNX文件中")
    
    # 执行导出
    torch.onnx.export(
        unet,  # 要导出的模型
        (sample, timestep, encoder_hidden_states),  # 模型输入
        **export_kwargs
    )
    
    precision_str = "FP16" if args.fp16 else "FP32"
    print(f"UNet模型已成功导出为ONNX格式: {args.output_path} (精度: {precision_str})")
    if args.use_external_data:
        print(f"模型权重已存储在外部文件: {os.path.join(output_dir, external_data_filename)}")

if __name__ == "__main__":
    main() 