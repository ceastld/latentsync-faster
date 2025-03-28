import os
import glob
from pathlib import Path
from typing import List, Tuple
import subprocess
from latentsync.configs.config import GLOBAL_CONFIG

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