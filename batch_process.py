import os
import glob
from pathlib import Path
from typing import List, Tuple
import subprocess
from latentsync.configs.config import GLOBAL_CONFIG

def get_video_audio_pairs() -> List[Tuple[str, str]]:
    """获取testset文件夹中的视频和音频文件对"""
    test_dir = Path(GLOBAL_CONFIG.test_dir)
    output_dir = Path(GLOBAL_CONFIG.output_dir)
    
    # 获取所有视频文件
    video_files = []
    for ext in ['*.mp4', '*.avi', '*.mov']:
        video_files.extend(glob.glob(str(test_dir / ext)))
    
    # 获取所有音频文件
    audio_files = []
    for ext in ['*.mp3', '*.wav', '*.m4a']:
        audio_files.extend(glob.glob(str(test_dir / ext)))
    
    # 生成配对
    pairs = []
    for video in video_files:
        video_name = Path(video).stem
        # 尝试找到匹配的音频文件
        for audio in audio_files:
            audio_name = Path(audio).stem
            # 如果音频文件名是视频文件名的一部分，或者视频文件名是音频文件名的一部分
            if video_name in audio_name or audio_name in video_name:
                pairs.append((video, audio))
                break
    
    return pairs

def process_pair(video_path: str, audio_path: str, output_path: str):
    """处理一对视频和音频文件"""
    print(f"Processing: {video_path} + {audio_path} -> {output_path}")
    
    # 这里调用您的推理代码
    # 示例：使用subprocess调用inference1.py
    cmd = [
        "python", "latentsync/inference1.py",
        "--video_path", video_path,
        "--audio_path", audio_path,
        "--output_path", output_path
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"Successfully processed: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error processing {output_path}: {e}")

def main():
    # 获取所有视频-音频对
    pairs = get_video_audio_pairs()
    
    # 处理每一对
    for video_path, audio_path in pairs:
        # 生成输出文件名
        video_name = Path(video_path).stem
        audio_name = Path(audio_path).stem
        output_name = f"{video_name}_v15.mp4"
        output_path = os.path.join(GLOBAL_CONFIG.output_dir, output_name)
        
        # 处理文件对
        process_pair(video_path, audio_path, output_path)

if __name__ == "__main__":
    main() 