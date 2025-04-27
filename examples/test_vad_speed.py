import torch
import numpy as np
from typing import List, Tuple
from tqdm import tqdm
from latentsync import *
from latentsync.inference._datas import AudioFrame
from latentsync.utils.util import read_audio
from latentsync.utils.timer import Timer
from latentsync.utils.vad import SileroVAD

def test_vad_speed():
    # 使用Timer启用性能测量
    Timer.enable()
    
    # 使用SileroVAD类初始化语音检测器
    threshold = 0.5
    sampling_rate = 16000
    samples_per_frame = 512
    vad = SileroVAD(threshold=threshold, sampling_rate=sampling_rate, window_size_samples=samples_per_frame)
    
    # 读取示例音频文件
    audio_path = "assets/cxk.mp3"
    audio_samples = read_audio(audio_path, audio_sample_rate=sampling_rate)
    audio_array = audio_samples.numpy()
    
    # 将音频分割成帧
    audio_frames = AudioFrame.split_audio(audio_array, samples_per_frame)
    
    print(f"音频总长度: {len(audio_samples) / sampling_rate:.2f}秒")
    print(f"总帧数: {len(audio_frames)}")
    
    print("\n方法1: 逐帧处理")
    # 存储每帧的语音检测结果
    speech_results = []
    
    # 使用Timer测量逐帧处理的时间
    with Timer("method1_frame_by_frame"):
        # 计算有多少帧包含语音
        speech_frames = 0
        for i, frame in enumerate(tqdm(audio_frames)):
            # 使用SileroVAD的_process_frame方法进行检测
            is_speech = vad._process_frame(frame.audio_samples)
            frame.is_speech = is_speech
            
            if is_speech:
                speech_frames += 1
            
            # 保存检测结果
            speech_results.append(is_speech)
    
    print(f"包含语音的帧数: {speech_frames}/{len(audio_frames)} ({speech_frames/len(audio_frames)*100:.2f}%)")
    
    # 可视化语音检测结果，黑色方块表示有语音，白色方块表示没有语音
    def visualize_speech_detection(results):
        """将语音检测结果可视化为一行黑白方块
        
        Args:
            results: 语音检测结果列表 (True表示有语音，False表示无语音)
        """
        # ANSI转义序列颜色代码
        BLACK_BLOCK = "\033[40m  \033[0m"  # 黑色背景
        WHITE_BLOCK = "\033[47m  \033[0m"  # 白色背景
        
        print("\n语音检测结果可视化 (黑色=有语音, 白色=无语音):")
        
        # 所有结果显示在一行
        line = ""
        for is_speech in results:
            if is_speech:
                line += BLACK_BLOCK
            else:
                line += WHITE_BLOCK
        print(line)
    
    # 显示逐帧处理的语音检测结果
    visualize_speech_detection(speech_results)
    
    print("\n方法2: 使用SileroVAD.get_speech_timestamps")
    # 使用Timer测量整体处理的时间
    with Timer("method2_get_speech_timestamps"):
        # 获取语音时间戳
        speech_timestamps = vad.get_speech_timestamps(audio_array, return_seconds=True)
    
    # 输出语音段分析结果
    print(f"检测到的语音段数量: {len(speech_timestamps)}")
    print(f"{'开始(秒)':<10}{'结束(秒)':<10}{'持续时间(秒)':<15}")
    print("-" * 35)
    
    speech_total_time = 0
    for segment in speech_timestamps:
        start_sec = segment["start"]
        end_sec = segment["end"]
        duration = end_sec - start_sec
        speech_total_time += duration
        print(f"{start_sec:<10.4f}{end_sec:<10.4f}{duration:<15.4f}")
    
    total_time = len(audio_array) / sampling_rate
    silence_total_time = total_time - speech_total_time
    
    print("-" * 35)
    print(f"语音总时长: {speech_total_time:.4f}秒 ({speech_total_time/total_time*100:.2f}%)")
    print(f"停顿总时长: {silence_total_time:.4f}秒 ({silence_total_time/total_time*100:.2f}%)")
    print(f"总时长: {total_time:.4f}秒")
    
    # 输出性能测量结果
    Timer.summary()


if __name__ == "__main__":
    test_vad_speed() 