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
import traceback


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
        print("继续执行，但性能可能不是最优")


def warmup_gpu():
    """预热GPU和CUDA环境"""
    if torch.cuda.is_available():
        try:
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
        except Exception as e:
            print(f"GPU预热过程中出错: {e}")
            print("继续执行，但初始性能可能受影响")


def safe_create_pipeline(context):
    """安全创建处理管道，捕获并处理所有异常"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            print(f"正在创建处理管道 (尝试 {attempt+1}/{max_retries})...")
            pipeline = create_pipeline(context)
            print("处理管道创建成功")
            return pipeline
        except Exception as e:
            print(f"创建处理管道时出错 (尝试 {attempt+1}/{max_retries}): {e}")
            print(traceback.format_exc())
            if attempt < max_retries - 1:
                print(f"等待5秒后重试...")
                time.sleep(5)
                # 清理CUDA缓存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                print("达到最大重试次数，无法创建处理管道")
                raise


def main(args):
    print(f"输入视频路径: {args.video_path}")
    print(f"输入音频路径: {args.audio_path}")
    print(f"输出视频路径: {args.video_out_path}")
    
    try:
        # 设置CPU亲和性优化
        set_cpu_affinity()
        
        # 预热GPU环境
        warmup_gpu()
        
        # 创建上下文和处理管道
        context = LipsyncContext()
        pipeline = safe_create_pipeline(context)
        
        set_seed(args.seed)
        print(f"随机种子设置为: {args.seed}")
        
        # 运行前等待1秒确保所有资源初始化
        time.sleep(1)
        
        # 主处理流程
        print("开始处理视频...")
        pipeline(
            video_path=args.video_path,
            audio_path=args.audio_path,
            video_out_path=args.video_out_path,
        )

        print(f"处理完成，输出视频保存在: {args.video_out_path}")
        
    except Exception as e:
        print(f"处理过程中出现错误: {e}")
        print(traceback.format_exc())
        print("程序异常终止")
        return 1
        
    # 打印Timer统计结果
    print("\n性能统计:")
    Timer.summary()
    
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LatentSync AI唇形合成")
    parser.add_argument("--video_path", type=str, required=True, help="输入视频路径")
    parser.add_argument("--audio_path", type=str, required=True, help="输入音频路径")
    parser.add_argument("--video_out_path", type=str, required=True, help="输出视频路径")
    parser.add_argument("--seed", type=int, default=1247, help="随机种子")
    args = parser.parse_args()

    # 启用性能计时器
    Timer.enable()

    exit_code = main(args)
    exit(exit_code)
