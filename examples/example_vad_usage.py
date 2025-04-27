import numpy as np
from latentsync.utils.vad import SileroVAD, is_speech
from latentsync.utils.util import read_audio


def example_detect_speech():
    """简单检测音频是否包含语音"""
    # 读取音频文件
    audio_path = "assets/cxk.mp3"
    sampling_rate = 16000
    audio = read_audio(audio_path, audio_sample_rate=sampling_rate)
    
    # 方法1: 使用简便函数
    result = is_speech(audio.numpy())
    print(f"音频包含语音: {result}")
    
    # 方法2: 使用SileroVAD类
    vad = SileroVAD(threshold=0.5, sampling_rate=sampling_rate)
    result = vad.detect(audio.numpy())
    print(f"音频包含语音: {result}")


def example_get_timestamps():
    """获取音频中的语音时间戳"""
    # 读取音频文件
    audio_path = "assets/cxk.mp3"
    sampling_rate = 16000
    audio = read_audio(audio_path, audio_sample_rate=sampling_rate)
    
    # 创建VAD检测器
    vad = SileroVAD(threshold=0.5, sampling_rate=sampling_rate)
    
    # 获取语音段时间戳（以秒为单位）
    timestamps = vad.get_speech_timestamps(audio.numpy(), return_seconds=True)
    
    # 输出语音段信息
    print(f"检测到 {len(timestamps)} 个语音段:")
    for i, ts in enumerate(timestamps):
        start = ts["start"]
        end = ts["end"]
        duration = end - start
        print(f"段 {i+1}: {start:.2f}s - {end:.2f}s (持续时间: {duration:.2f}s)")


if __name__ == "__main__":
    print("示例1: 检测音频是否包含语音")
    example_detect_speech()
    
    print("\n示例2: 获取语音时间戳")
    example_get_timestamps() 