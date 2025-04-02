from latentsync import *
from pathlib import Path
import glob
from typing import List, Tuple
import os
from latentsync.configs.config import GLOBAL_CONFIG
import argparse

def get_video_audio_pairs() -> List[Tuple[str, str]]:
    """获取testset文件夹中子目录中的视频和音频文件对
    
    每个子目录中包含一对对应的视频和音频文件，文件名相同但扩展名不同
    """
    test_dir = Path(GLOBAL_CONFIG.test_dir)
    pairs = []
    
    # 处理子目录中的文件对
    for subdir in test_dir.iterdir():
        if subdir.is_dir():
            # 获取子目录中的视频和音频文件
            subdir_videos = []
            subdir_audios = []
            
            for ext in ['*.mp4', '*.avi', '*.mov']:
                subdir_videos.extend(glob.glob(str(subdir / ext)))
            for ext in ['*.mp3', '*.wav', '*.m4a']:
                subdir_audios.extend(glob.glob(str(subdir / ext)))
            
            # 在子目录中查找匹配的文件对
            for video in subdir_videos:
                video_name = Path(video).stem
                for audio in subdir_audios:
                    audio_name = Path(audio).stem
                    if video_name == audio_name:  # 子目录中的文件通常同名
                        pairs.append((video, audio))
                        break
    
    return pairs

def is_file_processed(video_path: str, output_dir: str) -> bool:
    """检查文件是否已经被处理过
    
    Args:
        video_path: 视频文件路径
        output_dir: 输出目录
    
    Returns:
        bool: 如果文件已处理返回True，否则返回False
    """
    video_name = Path(video_path).stem
    output_name = f"{video_name}_enhanced.mp4"
    output_path = os.path.join(output_dir, output_name)
    return os.path.exists(output_path)

def process_pair(model, video_path: str, audio_path: str, output_dir: str):
    """处理一对视频和音频文件
    
    Args:
        model: 已加载的模型
        video_path: 视频文件路径
        audio_path: 音频文件路径
        output_dir: 输出目录
    """
    # 生成输出文件名
    video_name = Path(video_path).stem
    output_name = f"{video_name}_enhanced.mp4"
    output_path = os.path.join(output_dir, output_name)
    
    print(f"正在处理: {video_path} + {audio_path} -> {output_path}")
    
    try:
        # 直接使用模型进行推理
        model.inference(
            video_path,
            audio_path,
            output_path,
        )
        print(f"成功处理: {output_path}")
    except Exception as e:
        print(f"处理失败 {output_path}: {e}")

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="LatentSync 批量视频唇形同步")
    parser.add_argument("--onnx", action="store_true", help="使用ONNX模型加速")
    parser.add_argument("--trt", action="store_true", help="使用TensorRT模型加速")
    parser.add_argument("--time", action="store_true", help="使用时间统计")
    parser.add_argument("--force", action="store_true", help="强制重新处理所有文件，包括已处理的文件")
    args = parser.parse_args()
    
    model_type = "ONNX" if args.onnx else "TensorRT" if args.trt else "PyTorch"
    print(f"使用{model_type}模型进行批量推理...")

    if args.time:
        Timer.enable()

    # 初始化模型（只加载一次）
    context = LipsyncContext.from_version(
        "v15",
        use_onnx=args.onnx,
        use_trt=args.trt,
    )
    model = LipsyncModel(context)
    
    # 获取所有视频-音频对
    pairs = get_video_audio_pairs()
    print(f"找到 {len(pairs)} 对视频-音频文件待处理")
    
    # 创建输出目录
    output_dir = os.path.join(GLOBAL_CONFIG.output_dir, "enhanced")
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理每一对视频-音频文件
    for video_path, audio_path in pairs:
        if not args.force and is_file_processed(video_path, output_dir):
            print(f"跳过已处理的文件: {video_path}")
            continue
        process_pair(model, video_path, audio_path, output_dir)
    
    if args.time:
        Timer.summary()
    print("批量处理完成！")

if __name__ == "__main__":
    main() 