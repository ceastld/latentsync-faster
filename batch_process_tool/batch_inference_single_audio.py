from latentsync import *
from pathlib import Path
import glob
from typing import List, Tuple, Optional
import os
from latentsync.configs.config import GLOBAL_CONFIG # Keep if needed, else remove
import argparse

def get_video_files(input_dir: str) -> List[str]:
    """获取指定输入目录中所有子目录的视频文件路径。
    
    每个子目录应包含一个视频文件。
    """
    base_dir = Path(input_dir)
    video_files = []
    
    if not base_dir.is_dir():
        print(f"错误：输入目录 {input_dir} 不存在或不是一个目录。")
        return video_files

    for subdir in base_dir.iterdir():
        if subdir.is_dir():
            subdir_video: Optional[str] = None
            
            # 查找视频文件 (只找第一个)
            for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.flv']:
                found_videos = list(subdir.glob(ext))
                if found_videos:
                    if subdir_video is None:
                        subdir_video = str(found_videos[0])
                        if len(found_videos) > 1:
                            print(f"警告：在目录 {subdir} 中找到多个视频文件，将只使用第一个: {subdir_video}")
                    else:
                         print(f"警告：在目录 {subdir} 中找到多个视频文件（不同扩展名），将只使用第一个找到的: {subdir_video}")
                    break # 找到一个视频就停止

            if subdir_video:
                video_files.append(subdir_video)
            else:
                print(f"警告：在目录 {subdir} 中未找到视频文件，跳过此目录。")
                
    return video_files

def is_file_processed(video_path: str, driving_audio_path: str, output_dir: str) -> bool:
    """检查特定视频文件是否已经被处理过
    
    Args:
        video_path: 视频文件路径
        driving_audio_path: 驱动音频文件路径 (用于可能的命名区分，或确认处理过程)
        output_dir: 输出目录
    
    Returns:
        bool: 如果文件已处理返回True，否则返回False
    """
    video_name = Path(video_path).stem
    # 考虑是否需要在输出文件名中包含驱动音频的信息
    # driving_audio_name = Path(driving_audio_path).stem
    # output_name = f"{video_name}_driven_by_{driving_audio_name}_enhanced.mp4"
    # 简化版：只基于视频名
    output_name = f"{video_name}_enhanced.mp4" 
    output_path = os.path.join(output_dir, output_name)
    return os.path.exists(output_path)

def process_video(model, video_path: str, driving_audio_path: str, output_dir: str):
    """处理单个视频文件和指定的驱动音频文件
    
    Args:
        model: 已加载的模型
        video_path: 视频文件路径
        driving_audio_path: 驱动音频文件路径
        output_dir: 输出目录
    """
    video_name = Path(video_path).stem
    # 简化版输出文件名
    output_name = f"{video_name}_enhanced.mp4"
    output_path = os.path.join(output_dir, output_name)
    
    print(f"正在处理: {Path(video_path).name} + {Path(driving_audio_path).name} -> {Path(output_path).name}")
    
    try:
        # 直接使用模型进行推理
        model.inference(
            video_path,
            driving_audio_path,
            output_path,
        )
        print(f"成功处理: {output_path}")
    except Exception as e:
        print(f"处理失败 {output_path}: {e}")
        raise # 重新抛出异常，以便 main 函数可以捕获并计数失败

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="LatentSync 使用单一音频批量处理视频唇形同步")
    parser.add_argument("--input_dir", type=str, required=True, 
                        help="包含子目录的输入文件夹路径，每个子目录应含一个视频文件")
    parser.add_argument("--driving_audio_path", type=str, required=True,
                        help="用于驱动所有视频的单个音频文件路径")
    parser.add_argument("--output_dir", type=str, default=GLOBAL_CONFIG.output_dir, # Consider removing default or making it required
                        help="存放增强后视频的根目录")
    parser.add_argument("--onnx", action="store_true", help="使用ONNX模型加速")
    parser.add_argument("--trt", action="store_true", help="使用TensorRT模型加速")
    parser.add_argument("--time", action="store_true", help="使用时间统计")
    parser.add_argument("--force", action="store_true", help="强制重新处理所有文件，包括已处理的文件")
    args = parser.parse_args()
    
    # 检查驱动音频文件是否存在
    if not Path(args.driving_audio_path).is_file():
        print(f"错误：驱动音频文件 {args.driving_audio_path} 不存在或不是一个文件。")
        return

    model_type = "ONNX" if args.onnx else "TensorRT" if args.trt else "PyTorch"
    print(f"使用{model_type}模型进行批量推理...")
    print(f"输入视频目录: {args.input_dir}")
    print(f"驱动音频: {args.driving_audio_path}")
    print(f"输出目录: {args.output_dir}")

    if args.time:
        Timer.enable()

    # 初始化模型（只加载一次）
    try:
        context = LipsyncContext.from_version(
            "v15", # TODO: Maybe make version configurable?
            use_onnx=args.onnx,
            use_trt=args.trt,
        )
        model = LipsyncModel(context)
    except Exception as e:
        print(f"加载模型失败: {e}")
        return
    
    # 获取所有视频文件
    video_files = get_video_files(args.input_dir)
    if not video_files:
        print("在输入目录中未找到任何视频文件。")
        return
        
    print(f"找到 {len(video_files)} 个视频处理任务")
    
    # 创建输出目录 (直接使用 output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 处理每一个视频文件
    processed_count = 0
    skipped_count = 0
    failed_count = 0
    for video_path in video_files:
        if not args.force and is_file_processed(video_path, args.driving_audio_path, args.output_dir):
            print(f"跳过已处理的文件: {Path(video_path).name}")
            skipped_count += 1
            continue
        try:
            process_video(model, video_path, args.driving_audio_path, args.output_dir)
            processed_count += 1
        except Exception as e:
            # process_video 内已经打印了错误，这里只计数
            failed_count += 1

    print("批量处理完成！")
    print(f"总任务数: {len(video_files)}")
    print(f"成功处理: {processed_count}")
    print(f"跳过处理: {skipped_count}")
    print(f"处理失败: {failed_count}")
    
    if args.time:
        Timer.summary()

if __name__ == "__main__":
    main() 