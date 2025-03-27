import os
import glob
from pathlib import Path
from typing import List, Tuple
import subprocess
from latentsync.configs.config import GLOBAL_CONFIG

def get_video_audio_pairs() -> List[Tuple[str, str, str]]:
    """获取assets文件夹中的视频和音频文件对
    Returns:
        List[Tuple[str, str, str]]: 返回(视频路径, 音频路径, 文件夹名)的列表
    """
    assets_dir = Path(GLOBAL_CONFIG.assets_dir)
    
    # 获取所有子文件夹
    subdirs = [d for d in assets_dir.iterdir() if d.is_dir()]
    
    pairs = []
    for subdir in subdirs:
        # 在每个子文件夹中查找视频和音频文件
        video_files = []
        for ext in ['*.mp4', '*.avi', '*.mov']:
            video_files.extend(list(subdir.glob(ext)))
            
        audio_files = []
        for ext in ['*.mp3', '*.wav', '*.m4a']:
            audio_files.extend(list(subdir.glob(ext)))
        
        # 如果找到视频和音频文件，添加到配对列表
        if video_files and audio_files:
            # 使用第一个找到的视频和音频文件
            pairs.append((
                str(video_files[0]),
                str(audio_files[0]),
                subdir.name
            ))
    
    return pairs

def process_pair(video_path: str, audio_path: str, output_path: str):
    """处理一对视频和音频文件"""
    print(f"Processing: {video_path} + {audio_path} -> {output_path}")
    
    # 这里调用您的推理代码
    # 示例：使用subprocess调用inference1.py
    cmd = [
        "python", "inference1.py",
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
    for video_path, audio_path, folder_name in pairs:
        # 生成输出文件名
        output_name = f"{folder_name}_v15.mp4"
        output_path = os.path.join(GLOBAL_CONFIG.output_dir, output_name)
        
        # 处理文件对
        process_pair(video_path, audio_path, output_path)

if __name__ == "__main__":
    main() 