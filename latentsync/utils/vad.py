import torch
import numpy as np
from typing import List, Union, Tuple, Optional


class SileroVAD:
    """基于Silero VAD模型的语音活动检测器"""
    
    def __init__(
        self, 
        threshold: float = 0.5, 
        sampling_rate: int = 16000, 
        window_size_samples: int = 512,
        device: Optional[torch.device] = None
    ):
        """初始化VAD检测器
        
        Args:
            threshold: 语音检测阈值，高于此值被认为是语音
            sampling_rate: 音频采样率，默认为16000Hz
            window_size_samples: 用于语音检测的窗口大小，默认为512
            device: 计算设备，默认为自动选择
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        self.threshold = threshold
        self.sampling_rate = sampling_rate
        self.window_size_samples = window_size_samples
        
        # 加载Silero VAD模型
        self.model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
        )
        self.model = self.model.to(self.device)
    
    def _process_frame(self, frame: np.ndarray) -> bool:
        """处理单个音频帧并检测是否有语音
        
        Args:
            frame: 音频帧数据，形状为(window_size_samples,)
            
        Returns:
            bool: 是否检测到语音
        """
        # 确保帧的长度正确
        if len(frame) != self.window_size_samples:
            if len(frame) > self.window_size_samples:
                # 如果帧太长，取中间部分
                start = (len(frame) - self.window_size_samples) // 2
                frame = frame[start:start+self.window_size_samples]
            else:
                # 如果帧太短，用零填充
                frame = np.pad(frame, (0, self.window_size_samples - len(frame)), mode='constant')
        
        # 将numpy数组转换为torch张量
        audio_tensor = torch.tensor(frame, dtype=torch.float32, device=self.device)
        
        # 使用Silero VAD模型检测语音
        speech_prob = self.model(audio_tensor, self.sampling_rate).item()
        
        # 根据阈值判断是否为语音
        return speech_prob >= self.threshold
    
    def detect(self, audio: np.ndarray) -> bool:
        """判断整个音频是否包含语音
        
        Args:
            audio: 音频数据，形状为(N,)
            
        Returns:
            bool: 音频是否包含语音
        """
        # 将音频分成帧
        num_frames = len(audio) // self.window_size_samples
        if num_frames == 0:
            # 如果音频比一帧还短，则直接处理整个音频
            return self._process_frame(audio)
        
        # 逐帧处理
        for i in range(num_frames):
            start = i * self.window_size_samples
            end = start + self.window_size_samples
            frame = audio[start:end]
            
            # 如果任何一帧包含语音，则整个音频被认为包含语音
            if self._process_frame(frame):
                return True
        
        # 处理最后一个不完整的帧（如果有）
        if len(audio) % self.window_size_samples > 0:
            last_frame = audio[num_frames * self.window_size_samples:]
            if self._process_frame(last_frame):
                return True
        
        return False
    
    def get_speech_timestamps(
        self, 
        audio: np.ndarray, 
        return_seconds: bool = False
    ) -> List[dict]:
        """获取音频中语音段的时间戳
        
        Args:
            audio: 音频数据，形状为(N,)
            return_seconds: 是否以秒为单位返回时间戳，默认为False（以样本数返回）
            
        Returns:
            List[dict]: 语音段的时间戳列表，每项包含'start'和'end'
        """
        # 将音频分成帧
        num_frames = len(audio) // self.window_size_samples
        
        # 存储每帧的检测结果
        speech_frames = []
        
        # 处理每一帧
        for i in range(num_frames):
            start = i * self.window_size_samples
            end = start + self.window_size_samples
            frame = audio[start:end]
            
            speech_frames.append(self._process_frame(frame))
        
        # 处理最后一个不完整的帧（如果有）
        if len(audio) % self.window_size_samples > 0:
            last_frame = audio[num_frames * self.window_size_samples:]
            speech_frames.append(self._process_frame(last_frame))
        
        # 查找连续语音段
        timestamps = []
        in_speech = False
        speech_start = 0
        
        for i, is_speech in enumerate(speech_frames):
            # 检测语音段的开始
            if is_speech and not in_speech:
                speech_start = i
                in_speech = True
            
            # 检测语音段的结束
            elif not is_speech and in_speech:
                # 计算实际样本位置
                start_sample = speech_start * self.window_size_samples
                end_sample = i * self.window_size_samples
                
                if return_seconds:
                    start_sec = start_sample / self.sampling_rate
                    end_sec = end_sample / self.sampling_rate
                    timestamps.append({"start": start_sec, "end": end_sec})
                else:
                    timestamps.append({"start": start_sample, "end": end_sample})
                
                in_speech = False
        
        # 如果音频结束时仍在语音段内
        if in_speech:
            # 计算实际样本位置
            start_sample = speech_start * self.window_size_samples
            end_sample = len(speech_frames) * self.window_size_samples
            if end_sample > len(audio):
                end_sample = len(audio)
                
            if return_seconds:
                start_sec = start_sample / self.sampling_rate
                end_sec = end_sample / self.sampling_rate
                timestamps.append({"start": start_sec, "end": end_sec})
            else:
                timestamps.append({"start": start_sample, "end": end_sample})
        
        return timestamps


def is_speech(audio: np.ndarray, **kwargs) -> bool:
    """检测音频是否包含语音的便捷函数
    
    Args:
        audio: 音频数据
        **kwargs: 传递给SileroVAD构造函数的参数
        
    Returns:
        bool: 是否检测到语音
    """
    vad = SileroVAD(**kwargs)
    return vad.detect(audio) 