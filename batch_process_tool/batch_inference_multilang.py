from latentsync import *
from pathlib import Path
import glob
from typing import List, Tuple, Optional
import os
from latentsync.configs.config import GLOBAL_CONFIG
import argparse
import tempfile
import cv2
import numpy as np
import librosa
import subprocess
import shutil

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

def preprocess_video(video_path: str, audio_path: str, output_dir: str = None) -> str:
    """将视频长度调整为与音频长度匹配，保持帧率不变
    
    Args:
        video_path: 原始视频文件路径
        audio_path: 音频文件路径
        output_dir: 预处理视频的输出目录，如果提供则保存处理后的文件
        
    Returns:
        str: 处理后的临时视频文件路径
    
    Raises:
        ValueError: 如果视频无法被正确读取或参数无效
    """
    # 生成预处理文件名
    if output_dir:
        video_name = Path(video_path).stem
        lang_code = get_language_from_audio(audio_path)
        preprocessed_dir = os.path.join(output_dir, "preprocessed")
        os.makedirs(preprocessed_dir, exist_ok=True)
        preprocessed_filename = f"{video_name}_{lang_code}_preprocessed.mp4"
        preprocessed_path = os.path.join(preprocessed_dir, preprocessed_filename)
        
        # 如果预处理文件已存在，直接返回
        if os.path.exists(preprocessed_path) and os.path.getsize(preprocessed_path) > 0:
            print(f"发现预处理文件: {preprocessed_path}，跳过处理")
            return preprocessed_path
    else:
        preprocessed_path = None
    
    # 获取音频时长
    audio_duration = librosa.get_duration(path=audio_path)
    if audio_duration <= 0:
        raise ValueError(f"音频文件 {audio_path} 时长无效: {audio_duration}秒")
    
    # 读取视频信息
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        raise ValueError(f"视频帧率无效: {fps}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise ValueError(f"视频帧数无效: {total_frames}")
    
    original_duration = total_frames / fps
    print(f"视频信息: {total_frames}帧, {fps}fps, 时长{original_duration:.2f}秒")
    print(f"音频时长: {audio_duration:.2f}秒")
    
    # 如果时长已经相同，直接返回原视频路径
    if abs(original_duration - audio_duration) < 0.1:  # 允许0.1秒误差
        return video_path
    
    print(f"调整视频时长: 从 {original_duration:.2f}秒 到 {audio_duration:.2f}秒")
    
    # 创建输出文件
    if preprocessed_path:
        output_path = preprocessed_path
    else:
        temp_video = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        output_path = temp_video.name
        temp_video.close()
    
    # 计算需要的帧数
    target_frames = int(audio_duration * fps)
    if target_frames <= 0:
        cap.release()
        raise ValueError(f"计算的目标帧数无效: {target_frames}")
    
    # 创建视频写入器
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        cap.release()
        raise ValueError(f"视频尺寸无效: {width}x{height}")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not out.isOpened():
        cap.release()
        raise ValueError(f"无法创建输出视频文件: {output_path}")
    
    # 对原始视频进行采样，无论是缩短还是延长
    # 使用线性插值对原始视频的时间轴进行采样
    sample_indices = np.linspace(0, total_frames - 1, target_frames, dtype=float)
    
    # 对每个采样点读取相应的帧
    frames_written = 0
    for i in range(len(sample_indices)):
        # 获取采样索引
        frame_idx = int(sample_indices[i])
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            out.write(frame)
            frames_written += 1
        else:
            print(f"警告: 无法读取第 {frame_idx} 帧")
    
    # 释放资源
    cap.release()
    out.release()
    
    if frames_written == 0:
        if not preprocessed_path:  # 只有临时文件才删除
            os.remove(output_path)
        raise ValueError("未能写入任何视频帧")
    
    print(f"处理完成: 写入 {frames_written} 帧到输出视频: {output_path}")
    return output_path

def preprocess_video_ffmpeg(video_path: str, audio_path: str, output_dir: str = None) -> str:
    """使用ffmpeg将视频长度调整为与音频长度匹配，保持帧率不变
    
    Args:
        video_path: 原始视频文件路径
        audio_path: 音频文件路径
        output_dir: 预处理视频的输出目录，如果提供则保存处理后的文件
        
    Returns:
        str: 处理后的视频文件路径
    
    Raises:
        ValueError: 如果视频无法被正确处理
    """
    # 生成预处理文件名
    video_name = Path(video_path).stem
    lang_code = get_language_from_audio(audio_path)
    
    # 检查是否有预处理后的视频文件
    if output_dir:
        preprocessed_dir = os.path.join(output_dir, "preprocessed")
        os.makedirs(preprocessed_dir, exist_ok=True)
        preprocessed_filename = f"{video_name}_{lang_code}_preprocessed.mp4"
        preprocessed_path = os.path.join(preprocessed_dir, preprocessed_filename)
        
        # 如果预处理文件已存在，直接返回
        if os.path.exists(preprocessed_path) and os.path.getsize(preprocessed_path) > 0:
            print(f"发现预处理文件: {preprocessed_path}，跳过处理")
            return preprocessed_path
    else:
        preprocessed_path = None
    
    # 检查ffmpeg是否可用
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except (subprocess.SubprocessError, FileNotFoundError):
        print("警告: ffmpeg未安装或不可用，将回退到OpenCV方法")
        return preprocess_video(video_path, audio_path, output_dir)
    
    # 获取音频时长
    audio_duration = librosa.get_duration(path=audio_path)
    if audio_duration <= 0:
        raise ValueError(f"音频文件 {audio_path} 时长无效: {audio_duration}秒")
    
    # 使用ffprobe获取视频信息
    try:
        # 获取视频时长
        cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", 
               "default=noprint_wrappers=1:nokey=1", video_path]
        video_duration = float(subprocess.check_output(cmd).decode('utf-8').strip())
        
        # 获取视频帧率
        cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", 
               "stream=r_frame_rate", "-of", "default=noprint_wrappers=1:nokey=1", video_path]
        fps_str = subprocess.check_output(cmd).decode('utf-8').strip()
        if '/' in fps_str:
            num, den = map(int, fps_str.split('/'))
            fps = num / den if den != 0 else 0
        else:
            fps = float(fps_str)
        
        if fps <= 0 or video_duration <= 0:
            raise ValueError(f"视频参数无效: 时长={video_duration}秒, 帧率={fps}fps")
            
        print(f"视频信息: 时长={video_duration:.2f}秒, 帧率={fps:.2f}fps")
        print(f"音频时长: {audio_duration:.2f}秒")
        
    except subprocess.SubprocessError as e:
        print(f"获取视频信息失败: {e}")
        return preprocess_video(video_path, audio_path, output_dir)
    
    # 如果时长已经相同，直接返回原视频路径
    if abs(video_duration - audio_duration) < 0.1:  # 允许0.1秒误差
        return video_path
    
    # 设置输出路径 - 如果提供了输出目录则使用持久化路径，否则使用临时文件
    if preprocessed_path:
        output_path = preprocessed_path
    else:
        temp_video = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        output_path = temp_video.name
        temp_video.close()
    
    # 计算原视频和目标视频的总帧数
    original_frames = int(video_duration * fps)
    target_frames = int(audio_duration * fps)
    
    print(f"调整视频时长: 从 {video_duration:.2f}秒 ({original_frames}帧) 到 {audio_duration:.2f}秒 ({target_frames}帧)")
    
    # 计算播放速度比例
    speed_ratio = video_duration / audio_duration
    
    try:
        # 改用atempo滤镜（对视频时间轴进行操作）或使用fps+截断方式
        if target_frames > original_frames:
            # 视频需要延长
            # 方法1：直接调整帧率并限制输出持续时间
            target_fps = target_frames / audio_duration  # 计算需要的帧率以达到目标帧数
            
            cmd = [
                "ffmpeg", "-y", "-i", video_path,
                "-filter:v", f"fps={target_fps}",  # 设置输出帧率
                "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                "-t", str(audio_duration),  # 限制输出持续时间
                output_path
            ]
        else:
            # 视频需要缩短
            # 使用select滤镜选择帧，避免使用不支持的frames参数
            select_ratio = target_frames / original_frames
            # select_expr = f"select='if(lt(random(0),{select_ratio}),1,0)',setpts=N/({fps}*TB)"
            select_expr = f"select='not(mod(n,{int(1/select_ratio)}))',setpts=N/({fps}*TB)"
            
            cmd = [
                "ffmpeg", "-y", "-i", video_path,
                "-filter:v", select_expr,
                "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                output_path
            ]
        
        print(f"执行命令: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)  # 设置10分钟超时
        
        # 检查处理结果
        if result.returncode != 0:
            print(f"ffmpeg错误: {result.stderr}")
            raise subprocess.SubprocessError(f"ffmpeg返回错误代码: {result.returncode}")
        
        # 检查处理后的视频是否存在
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            raise ValueError("处理后的视频文件无效或为空")
            
        print(f"视频处理完成并已保存: {output_path}")
        return output_path
        
    except (subprocess.SubprocessError, subprocess.TimeoutExpired) as e:
        # 清理临时文件
        if not preprocessed_path and os.path.exists(output_path):
            os.remove(output_path)
        print(f"使用ffmpeg处理视频失败: {e}")
        
        # 直接回退到OpenCV方法
        print("回退到OpenCV方法...")
        return preprocess_video(video_path, audio_path, output_dir)

def process_pair(model, video_path: str, audio_path: str, output_dir: str) -> bool:
    """处理一对视频和特定语言的音频文件
    
    Args:
        model: 已加载的模型
        video_path: 视频文件路径
        audio_path: 音频文件路径
        output_dir: 输出目录
        
    Returns:
        bool: 处理成功返回True，失败返回False
    """
    # 生成包含语言信息的输出文件名
    video_name = Path(video_path).stem
    lang_code = get_language_from_audio(audio_path)
    output_name = f"{video_name}_{lang_code}_enhanced.mp4"
    output_path = os.path.join(output_dir, output_name)
    
    print(f"正在处理: {Path(video_path).name} + {Path(audio_path).name} -> {Path(output_path).name}")
    
    processed_video_path = None
    success = False
    
    try:
        # 预处理视频，调整长度与音频匹配（传入输出目录以便持久化存储预处理文件）
        # processed_video_path = preprocess_video_ffmpeg(video_path, audio_path, output_dir)
        
        # 使用预处理后的视频进行推理
        model.inference(
            # processed_video_path,
            video_path,
            audio_path,
            output_path,
        )
        
        # 如果创建了临时文件（不是原视频也不是持久化的预处理文件），则删除
        is_temp_file = processed_video_path != video_path and not processed_video_path.endswith("_preprocessed.mp4")
        if is_temp_file and os.path.exists(processed_video_path):
            os.remove(processed_video_path)
            
        print(f"成功处理: {output_path}")
        success = True
        return success
    except Exception as e:
        # 如果创建了临时文件，确保删除
        is_temp_file = processed_video_path and processed_video_path != video_path and not processed_video_path.endswith("_preprocessed.mp4")
        if is_temp_file and os.path.exists(processed_video_path):
            try:
                os.remove(processed_video_path)
            except Exception as cleanup_error:
                print(f"清理临时文件失败: {cleanup_error}")
        print(f"处理失败 {output_path}: {e}")
        return success

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
            success = process_pair(model, video_path, audio_path, enhanced_output_dir)
            if success:
                processed_count += 1
            else:
                failed_count += 1
        except Exception as e:
            print(f"处理时发生未捕获的异常: {e}")
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