import argparse
import warnings
from latentsync.inference.context import LipsyncContext
from latentsync.inference.utils import create_pipeline
from latentsync.inference.utils import set_seed
from latentsync.utils.timer import Timer
import time
import torch
import os
import psutil


def set_cpu_affinity():
    """设置CPU亲和性，优化线程在核心上的分配"""
    try:
        # 获取当前进程对象
        process = psutil.Process(os.getpid())
        
        # 获取CPU核心数
        num_cores = psutil.cpu_count(logical=True)
        
        if num_cores > 2:
            # 获取所有可用的CPU核心
            available_cores = list(range(num_cores))
            
            # 设置CPU亲和性，避免使用超线程带来的性能波动
            # 在多核系统上通常优先使用物理核心
            process.cpu_affinity(available_cores)
            
            print(f"已设置CPU亲和性，使用 {len(available_cores)} 个核心")
            
            # 设置ONNX运行时线程数
            # 通常应该设置为物理核心数的一半到全部
            thread_count = max(4, num_cores // 2)
            os.environ["OMP_NUM_THREADS"] = str(thread_count)
            os.environ["MKL_NUM_THREADS"] = str(thread_count)
            os.environ["OPENBLAS_NUM_THREADS"] = str(thread_count)
            os.environ["VECLIB_MAXIMUM_THREADS"] = str(thread_count)
            os.environ["NUMEXPR_NUM_THREADS"] = str(thread_count)
            
            print(f"ONNX运行时线程数设置为: {thread_count}")
            
    except Exception as e:
        print(f"设置CPU亲和性时出错: {e}")


def warmup_gpu():
    """预热GPU和CUDA环境"""
    if torch.cuda.is_available():
        print("正在预热GPU环境...")
        # 强制初始化CUDA
        torch.cuda.init()
        torch.cuda.empty_cache()
        
        # 创建一些随机张量并进行基础运算，确保CUDA上下文已初始化
        dummy = torch.randn(100, 100, device="cuda")
        for _ in range(5):
            _ = dummy @ dummy.t()
            torch.cuda.synchronize()
        
        print("GPU环境预热完成")


def main(args):
    print(f"Input video path: {args.video_path}")
    print(f"Input audio path: {args.audio_path}")
    
    # 设置CPU亲和性优化
    set_cpu_affinity()
    
    # 预热GPU环境
    warmup_gpu()
    
    # 创建上下文和处理管道
    context = LipsyncContext()
    pipeline = create_pipeline(context)
    
    set_seed(args.seed)
    
    # 运行前等待1秒确保所有资源初始化
    time.sleep(1)
    
    # 主处理流程
    pipeline(
        video_path=args.video_path,
        audio_path=args.audio_path,
        video_out_path=args.video_out_path,
    )

    print(f"Output video path: {args.video_out_path}")
    
    # 打印Timer统计结果
    Timer.summary()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--audio_path", type=str, required=True)
    parser.add_argument("--video_out_path", type=str, required=True)
    parser.add_argument("--seed", type=int, default=1247)
    args = parser.parse_args()

    Timer.enable()

    main(args)
