from latentsync import *
from pathlib import Path
import glob
from typing import List, Tuple, Optional
import os
from latentsync.configs.config import GLOBAL_CONFIG
import argparse

def get_video_multi_audio_pairs(input_dir: str) -> List[Tuple[str, str]]:
    """获取指定输入目录中子目录的视频和多语言音频文件对
    
    每个子目录应包含一个视频文件和多个音频文件。
    """
    base_dir = Path(input_dir)
    pairs = []
    
    if not base_dir.is_dir():
        print(f"错误：输入目录 {input_dir} 不存在或不是一个目录。")
        return pairs

    for subdir in base_dir.iterdir():
        if subdir.is_dir():
            subdir_video: Optional[str] = None
            subdir_audios: List[str] = []
            
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

            if not subdir_video:
                print(f"警告：在目录 {subdir} 中未找到视频文件，跳过此目录。")
                continue

            # 查找所有音频文件
            for ext in ['*.wav']:
                subdir_audios.extend([str(p) for p in subdir.glob(ext)])

            if not subdir_audios:
                print(f"警告：在目录 {subdir} 中未找到音频文件，跳过此目录。")
                continue

            # 为每个音频文件创建一个配对
            for audio in subdir_audios:
                pairs.append((subdir_video, audio))
                
    return pairs

def get_language_from_audio(audio_path: str) -> str:
    """从音频文件名提取语言代码
    
    预期格式为 "任意字符.语言代码.扩展名", 例如 'video_segment.en.wav' -> 'en'
    或者 "语言代码.扩展名", 例如 'es.wav' -> 'es'
    如果 stem 中没有 '.'，则返回整个 stem。
    """
    stem = Path(audio_path).stem
    # 查找 stem 中的最后一个点
    last_dot_index = stem.rfind('.')
    # 如果找到点，并且它不是第一个字符
    if last_dot_index > 0:
        # 提取点之后的部分作为语言代码
        return stem[last_dot_index + 1:]
    else:
        # 如果没有点或点在开头，返回整个 stem 作为备选
        # 或者考虑返回一个默认值或引发错误，取决于具体需求
        print(f"警告：无法从文件名 '{Path(audio_path).name}' 中按预期格式提取语言代码。将使用整个文件名 '{stem}' 作为代码。")
        return stem

def is_file_processed(video_path: str, audio_path: str, output_dir: str) -> bool:
    """检查特定语言的文件是否已经被处理过
    
    Args:
        video_path: 视频文件路径
        audio_path: 音频文件路径 (用于提取语言)
        output_dir: 输出目录
    
    Returns:
        bool: 如果文件已处理返回True，否则返回False
    """
    video_name = Path(video_path).stem
    lang_code = get_language_from_audio(audio_path)
    output_name = f"{video_name}_{lang_code}_enhanced.mp4"
    output_path = os.path.join(output_dir, output_name)
    return os.path.exists(output_path)

def process_pair(model, video_path: str, audio_path: str, output_dir: str):
    """处理一对视频和特定语言的音频文件
    
    Args:
        model: 已加载的模型
        video_path: 视频文件路径
        audio_path: 音频文件路径
        output_dir: 输出目录
    """
    # 生成包含语言信息的输出文件名
    video_name = Path(video_path).stem
    lang_code = get_language_from_audio(audio_path)
    output_name = f"{video_name}_{lang_code}_enhanced.mp4"
    output_path = os.path.join(output_dir, output_name)
    
    print(f"正在处理: {Path(video_path).name} + {Path(audio_path).name} -> {Path(output_path).name}")
    
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
    parser = argparse.ArgumentParser(description="LatentSync 多语言批量视频唇形同步")
    parser.add_argument("--input_dir", type=str, default=GLOBAL_CONFIG.test_dir, 
                        help="包含子目录的输入文件夹路径，每个子目录含一个视频和多个音频")
    parser.add_argument("--output_dir", type=str, default=GLOBAL_CONFIG.output_dir,
                        help="存放增强后视频的根目录")
    parser.add_argument("--onnx", action="store_true", help="使用ONNX模型加速")
    parser.add_argument("--trt", action="store_true", help="使用TensorRT模型加速")
    parser.add_argument("--time", action="store_true", help="使用时间统计")
    parser.add_argument("--force", action="store_true", help="强制重新处理所有文件，包括已处理的文件")
    args = parser.parse_args()
    
    model_type = "ONNX" if args.onnx else "TensorRT" if args.trt else "PyTorch"
    print(f"使用{model_type}模型进行批量推理...")
    print(f"输入目录: {args.input_dir}")
    print(f"输出目录: {args.output_dir}")

    if args.time:
        Timer.enable()

    # 初始化模型（只加载一次）
    context = LipsyncContext.from_version(
        "v15", # TODO: Maybe make version configurable?
        use_onnx=args.onnx,
        use_trt=args.trt,
    )
    model = LipsyncModel(context)
    
    # 获取所有视频-多语言音频对
    pairs = get_video_multi_audio_pairs(args.input_dir)
    print(f"找到 {len(pairs)} 个视频-音频处理任务")
    
    # 创建输出目录
    enhanced_output_dir = os.path.join(args.output_dir, "enhanced_multilang")
    os.makedirs(enhanced_output_dir, exist_ok=True)
    
    # 处理每一对视频-音频文件
    processed_count = 0
    skipped_count = 0
    failed_count = 0
    for video_path, audio_path in pairs:
        if not args.force and is_file_processed(video_path, audio_path, enhanced_output_dir):
            print(f"跳过已处理的文件组合: {Path(video_path).name} + {Path(audio_path).name}")
            skipped_count += 1
            continue
        try:
            process_pair(model, video_path, audio_path, enhanced_output_dir)
            processed_count += 1
        except Exception as e:
            # process_pair 内已经打印了错误，这里只计数
            failed_count += 1

    print("批量处理完成！")
    print(f"总任务数: {len(pairs)}")
    print(f"成功处理: {processed_count}")
    print(f"跳过处理: {skipped_count}")
    print(f"处理失败: {failed_count}")
    
    if args.time:
        Timer.summary()


if __name__ == "__main__":
    main() 