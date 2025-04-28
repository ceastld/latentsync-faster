#!/usr/bin/env python
import os
import sys
import argparse
import re
import logging
import json
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Any
import google.generativeai as genai
import tempfile
import requests
from pydub import AudioSegment
import math
import subprocess

# --- 配置常量 ---
DEFAULT_TARGET_LANGS = "zh,ja,ko"
TARGET_LANGUAGE_NAMES = {
    "zh": "Mandarin Chinese",
    "ja": "Japanese",
    "ko": "Korean",
}

# ElevenLabs Voices (与process_video_translate.py相同)
# 这些将被克隆的声音替换
elevenlabs_voices = {          
    "zh": {"male": "4VZIsMPtgggwNg7OXbPY", "female": "Ca5bKgudqKJzq8YRFoAz"},
    "ja": {"male": "Mv8AjrYZCBkdsmDHNwcB", "female": "8EkOjt4xTPGMclNlh1pk"},
    "ko": {"male": "U1cJYS4EdbaHmfR7YzHd", "female": "uyVNoMrnUku1dZyVEXwD"},
    "en": {"male": "1SM7GgM6IMuvQlz2BwM3", "female": "XfNU2rGpBa01ckF309OY"}, 
}
DEFAULT_ELEVENLABS_VOICE_ID = elevenlabs_voices.get("en", {}).get("male", "1SM7GgM6IMuvQlz2BwM3") 
FINAL_AUDIO_LENGTH_SECONDS = 505 # 8分钟25秒 = 505秒
GAP_BETWEEN_SEGMENTS = 300  # 0.3秒的间隔，单位为毫秒

# --- 日志设置 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("TextTranslateProcessor")

# --- ElevenLabs API 相关常量 ---
ELEVENLABS_API_URL = "https://api.elevenlabs.io/v1"
ADD_VOICE_ENDPOINT = f"{ELEVENLABS_API_URL}/voices/add"

# --- 声音克隆相关函数 ---
def extract_audio_from_video(video_path: Path, output_audio_path: Path) -> bool:
    """从视频中提取音频，保存为MP3格式"""
    logger.info(f"从视频 {video_path} 提取音频...")
    
    command = [
        "ffmpeg",
        "-i", str(video_path),
        "-vn",  # 不要视频
        "-codec:a", "libmp3lame",  # 使用MP3编码
        "-q:a", "2",  # 高质量MP3
        "-y",  # 覆盖输出文件（如果存在）
        str(output_audio_path)
    ]
    
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8', errors='ignore')
        logger.info(f"成功提取音频到 {output_audio_path}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"提取音频失败: {e}")
        logger.error(f"错误输出: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"音频提取时发生未知错误: {e}")
        return False

def clone_voice_from_audio(api_key: str, voice_name: str, audio_file_path: Path) -> Optional[str]:
    """使用音频文件克隆声音，返回克隆的声音ID"""
    if not os.path.exists(audio_file_path):
        logger.error(f"音频文件不存在: {audio_file_path}")
        return None
    
    logger.info(f"使用音频 {audio_file_path} 开始克隆声音 '{voice_name}'...")
    
    headers = {
        "xi-api-key": api_key,
    }
    
    # 准备文件上传
    with open(audio_file_path, 'rb') as f:
        files = [('files', (os.path.basename(audio_file_path), f))]
        
        data = {
            "name": voice_name,
        }
        
        try:
            response = requests.post(ADD_VOICE_ENDPOINT, headers=headers, data=data, files=files)
            
            if response.status_code == 200:
                result = response.json()
                voice_id = result.get("voice_id")
                if voice_id:
                    logger.info(f"声音 '{voice_name}' 克隆成功! Voice ID: {voice_id}")
                    return voice_id
                else:
                    logger.error("声音克隆成功，但未在响应中找到voice_id")
                    logger.error(f"响应内容: {response.text}")
                    return None
            else:
                logger.error(f"声音克隆失败，状态码: {response.status_code}")
                logger.error(f"响应内容: {response.text}")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"声音克隆请求错误: {e}")
            return None

def update_voice_ids_with_cloned(cloned_voice_id: str) -> None:
    """使用克隆的声音ID更新所有语言的声音ID"""
    global elevenlabs_voices, DEFAULT_ELEVENLABS_VOICE_ID
    
    # 更新所有语言使用相同的克隆声音ID
    for lang in elevenlabs_voices:
        elevenlabs_voices[lang]["male"] = cloned_voice_id
        elevenlabs_voices[lang]["female"] = cloned_voice_id
    
    # 更新默认声音ID
    DEFAULT_ELEVENLABS_VOICE_ID = cloned_voice_id
    
    logger.info(f"所有声音ID已更新为克隆的声音ID: {cloned_voice_id}")

# --- 时间戳解析 ---
def parse_timestamped_text(file_path: Path) -> List[Tuple[str, str, float, int]]:
    """解析带时间戳的文本文件，返回格式为[(时间戳文本, 内容文本, 时间秒数, 时间毫秒数)]的列表"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 解析时间戳格式 "M:SS: 文本内容"
    pattern = r'(\d+:\d+):\s+(.*?)(?=\n\s*\d+:\d+:|$)'
    matches = re.findall(pattern, content, re.DOTALL)
    
    result = []
    for i, (timestamp, text) in enumerate(matches):
        # 转换时间戳为秒数 - 正确解析为分:秒
        m, s = map(int, timestamp.split(':'))
        seconds = m * 60 + s  # 分钟*60 + 秒数
        milliseconds = seconds * 1000  # 转换为毫秒用于音频处理
        
        # 计算每段的持续时间
        next_seconds = None
        if i < len(matches) - 1:
            next_timestamp = matches[i+1][0]
            next_m, next_s = map(int, next_timestamp.split(':'))
            next_seconds = next_m * 60 + next_s  # 同样正确解析下一个时间戳
            duration = next_seconds - seconds
        else:
            # 最后一段使用估计值
            duration = 30  # 默认30秒
            
        result.append((timestamp, text.strip(), duration, milliseconds))
    
    return result

# --- Gemini翻译处理 ---
class GeminiTextTranslator:
    """处理使用Gemini API的文本翻译"""
    def __init__(self, api_key: str, model: str = "gemini-1.5-flash-latest"):
        if not api_key:
            raise ValueError("需要Gemini API密钥")
        self._model_name = model
        try:
            self._model = genai.GenerativeModel(model)
            logger.info(f"初始化Gemini模型 '{model}'.")
        except Exception as e:
            logger.critical(f"Gemini初始化失败: {e}")
            raise ValueError(f"无法初始化Gemini: {e}") from e

    def translate(self, text_to_translate: str, target_language_name: str, source_language_name: Optional[str] = None) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """翻译文本到目标语言"""
        if not text_to_translate:
            logger.warning("调用翻译时文本为空")
            return None, None, "输入文本为空"

        logger.info(f"开始使用Gemini模型 '{self._model_name}' 翻译到 {target_language_name}")
        
        # 构建提示
        if source_language_name:
            prompt = f"将以下文本从 {source_language_name} 翻译成 {target_language_name}，翻译的简短一些。仅返回翻译文本:\n\n{text_to_translate}"
        else:
            prompt = f"将以下文本翻译成 {target_language_name}。识别源语言。在第一行返回翻译文本，在第二行返回检测到的源语言名称(例如，英语，西班牙语)。\n\n{text_to_translate}"

        contents = [prompt]

        try:
            response = self._model.generate_content(
                contents=contents,
                request_options={'timeout': 300}
            )
            
            if response and response.text:
                lines = response.text.strip().split('\n')
                translated_text = lines[0].strip()
                
                detected_source = None
                if not source_language_name and len(lines) > 1:
                    detected_source = lines[1].strip()
                elif source_language_name:
                    detected_source = source_language_name
                
                if not translated_text:
                    logger.warning("翻译后文本为空")
                    return None, detected_source, "翻译文本为空"
                    
                logger.info(f"翻译成功。源: {detected_source or '自动检测'}。目标: {target_language_name}。翻译文本: {translated_text[:100]}...")
                return translated_text, detected_source, None
            else:
                error_msg = "翻译失败。未从Gemini收到有效回复"
                logger.error(error_msg)
                return None, None, error_msg
                
        except Exception as e:
            error_msg = f"翻译期间发生错误: {e}"
            logger.error(error_msg, exc_info=True)
            return None, None, error_msg

# --- ElevenLabs TTS ---
class ElevenLabsTTS:
    """使用ElevenLabs API处理文本到语音转换"""
    BASE_URL = "https://api.elevenlabs.io/v1"

    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("需要ElevenLabs API密钥")
        self._api_key = api_key

    def run_tts(self, text: str, language_code: str, voice_id: str, output_wav_path: Path, voice_settings: Optional[Dict[str, Any]] = None, speed: float = 1.0) -> Optional[Tuple[str, float]]:
        """使用ElevenLabs生成语音并保存为WAV"""
        if not text: 
            logger.warning("TTS调用时文本为空")
            return None

        tts_url = f"{self.BASE_URL}/text-to-speech/{voice_id}/stream?output_format=mp3_44100_128"
        headers = {"Content-Type": "application/json", "xi-api-key": self._api_key}
        model_id = "eleven_multilingual_v2" if language_code != 'en' else "eleven_english_v2"
        
        model_settings = {"speed": speed}
        
        payload = {
            "text": text, 
            "model_id": model_id,
            "model_settings": model_settings
        }
        
        if voice_settings: payload["voice_settings"] = voice_settings

        logger.info(f"开始ElevenLabs TTS请求。声音: {voice_id}, 模型: {model_id}, 语言: {language_code}, 语速: {speed}. 输出: {output_wav_path}")

        temp_mp3_path = None
        try:
            response = requests.post(tts_url, headers=headers, json=payload, stream=True, timeout=300)

            if response.status_code == 200:
                logger.info(f"TTS请求成功 (状态 {response.status_code})。接收音频流...")
                output_wav_path.parent.mkdir(parents=True, exist_ok=True)

                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_mp3_file_obj:
                    temp_mp3_path = Path(temp_mp3_file_obj.name)
                
                logger.debug(f"将ElevenLabs MP3响应流式传输到临时文件: {temp_mp3_path}")
                bytes_written = 0
                try:
                    with open(temp_mp3_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                bytes_written += len(chunk)
                    
                    logger.info(f"成功下载MP3流到 {temp_mp3_path} ({bytes_written} 字节)")
                    if bytes_written == 0: 
                        logger.error(f"临时MP3文件 {temp_mp3_path} 为空")
                        return None

                    logger.info(f"转换临时MP3 {temp_mp3_path} 为WAV {output_wav_path}")
                    sound = AudioSegment.from_mp3(temp_mp3_path)
                    sound.export(output_wav_path, format="wav")
                    
                    audio_duration = sound.duration_seconds
                    logger.info(f"成功转换为WAV: {output_wav_path} (时长: {audio_duration:.2f}秒)")
                    
                    return str(output_wav_path), audio_duration

                except Exception as e:
                    logger.error(f"MP3下载或转换期间错误: {e}", exc_info=True)
                    return None
                finally:
                    if temp_mp3_path and temp_mp3_path.exists():
                        try: 
                            temp_mp3_path.unlink()
                            logger.debug(f"删除临时MP3文件: {temp_mp3_path}")
                        except OSError as del_e: 
                            logger.warning(f"无法删除临时MP3文件 {temp_mp3_path}: {del_e}")

            else:
                error_text = response.text
                logger.error(f"ElevenLabs TTS请求失败。状态: {response.status_code}, 响应: {error_text}")
                return None

        except requests.exceptions.RequestException as e:
            logger.error(f"ElevenLabs TTS请求期间网络或HTTP错误: {e}", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"ElevenLabs TTS期间发生意外错误: {e}", exc_info=True)
            return None

# --- 主处理逻辑 ---
def process_text_segments(
    segments: List[Tuple[str, str, float, int]], 
    output_dir: Path,
    base_filename: str,
    target_langs: List[str],
    gemini_translator: GeminiTextTranslator,
    elevenlabs_tts: ElevenLabsTTS,
    elevenlabs_voice_type: str,
    elevenlabs_custom_voice_id: Optional[str],
    overwrite_audio: bool
) -> Dict[str, List[Tuple[Path, int, float]]]:
    """
    处理文本段落，进行翻译和语音合成
    返回：Dict[语言代码, List[音频文件路径, 开始时间毫秒, 音频时长秒]]
    """
    logger.info(f"--- 处理 {len(segments)} 个文本段落 ---")
    
    source_language_name = "English"  # 这里我们已知example.txt是英文的
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 存储每种语言的音频文件信息
    audio_segments_by_lang = {}
    
    for target_lang_code in target_langs:
        target_lang_name = TARGET_LANGUAGE_NAMES.get(target_lang_code)
        if not target_lang_name:
            logger.warning(f"跳过未支持的目标语言: {target_lang_code}")
            continue
            
        logger.info(f"== 处理目标语言: {target_lang_name} ({target_lang_code}) ==")
        
        # 确定要使用的声音ID
        voice_id_to_use = elevenlabs_custom_voice_id
        effective_lang_code = target_lang_code
        
        if not voice_id_to_use:
            logger.info(f"尝试为语音类型 '{elevenlabs_voice_type}' 和语言 '{target_lang_code}' 找到声音")
            lang_voices = elevenlabs_voices.get(target_lang_code)
            
            if lang_voices:
                voice_id_to_use = lang_voices.get(elevenlabs_voice_type)
                if not voice_id_to_use:
                    other_gender = "female" if elevenlabs_voice_type == "male" else "male"
                    voice_id_to_use = lang_voices.get(other_gender)
                    if voice_id_to_use:
                        logger.info(f"使用另一种性别 '{other_gender}' 的声音 ID: {voice_id_to_use}")
            
            # 如果仍未找到声音，使用默认英语声音
            if not voice_id_to_use:
                voice_id_to_use = elevenlabs_voices.get("en", {}).get(elevenlabs_voice_type, DEFAULT_ELEVENLABS_VOICE_ID)
                effective_lang_code = "en"
                logger.info(f"无法找到 {target_lang_code} 的声音，使用默认英语声音: {voice_id_to_use}")
        
        # 为该语言创建存储结构
        audio_segments_by_lang[target_lang_code] = []
        
        # 为每个段落处理翻译和TTS
        successful_segments = 0
        for i, (timestamp, text, duration, start_time_ms) in enumerate(segments):
            # 创建临时文件目录
            temp_dir = output_dir / "temp_segments" / target_lang_code
            temp_dir.mkdir(parents=True, exist_ok=True)
            
            # 构建输出文件路径，包含时间戳
            output_wav_path = temp_dir / f"{base_filename}_{target_lang_code}_{timestamp.replace(':', '-')}.wav"
            
            if output_wav_path.exists() and not overwrite_audio:
                # 如果文件已存在，获取其时长
                try:
                    audio = AudioSegment.from_wav(output_wav_path)
                    audio_duration = audio.duration_seconds
                    audio_segments_by_lang[target_lang_code].append((output_wav_path, start_time_ms, audio_duration))
                    logger.info(f"使用已存在的音频文件: {output_wav_path.name} (时长: {audio_duration:.2f}秒)")
                    successful_segments += 1
                    continue
                except Exception as e:
                    logger.warning(f"读取已存在的音频文件时出错: {e}")
                    # 如果读取出错，继续处理生成新文件
                
            logger.info(f"处理段落 {i+1}/{len(segments)}: 时间戳 {timestamp}, 时长约 {duration:.1f}秒")
            
            # 翻译文本
            translated_text, _, error_msg = gemini_translator.translate(
                text, target_lang_name, source_language_name
            )
            
            if error_msg:
                logger.error(f"翻译失败: {error_msg}")
                continue
                
            if not translated_text:
                logger.error("翻译结果为空文本")
                continue
                
            logger.info(f"翻译成功: {translated_text[:100]}...")
            
            # 生成语音 - 直接使用最快语速(1.2)
            voice_settings = {"stability": 0.6, "similarity_boost": 0.75}
            speed = 1.2  # 设置为最快语速
            
            logger.info(f"使用最快语速 {speed} 生成TTS")
            
            # 直接使用最快语速生成TTS
            tts_result = elevenlabs_tts.run_tts(
                text=translated_text,
                language_code=effective_lang_code,
                voice_id=voice_id_to_use,
                output_wav_path=output_wav_path,
                voice_settings=voice_settings,
                speed=speed
            )
            
            if not tts_result:
                logger.error(f"TTS生成失败 (语速: {speed})")
                continue
                
            output_path_str, audio_duration = tts_result
            logger.info(f"TTS成功: {output_path_str} (时长: {audio_duration:.2f}秒)")
            
            # 将音频文件信息添加到列表中
            audio_segments_by_lang[target_lang_code].append((output_wav_path, start_time_ms, audio_duration))
            successful_segments += 1
            
        logger.info(f"=== 完成语言 {target_lang_code} 处理: 成功 {successful_segments}/{len(segments)} 段 ===")
    
    return audio_segments_by_lang

def assemble_complete_audio(
    audio_segments: List[Tuple[Path, int, float]],
    output_path: Path,
    total_duration_ms: int = FINAL_AUDIO_LENGTH_SECONDS * 1000
) -> bool:
    """
    将各个音频片段组装成一个完整的音频文件
    
    参数:
        audio_segments: 列表，每项包含 (音频文件路径, 开始时间毫秒, 音频时长秒)
        output_path: 输出文件路径
        total_duration_ms: 最终音频的总时长(毫秒)
    
    返回:
        成功与否的布尔值
    """
    if not audio_segments:
        logger.error("没有可用的音频片段进行组装")
        return False
    
    try:
        # 创建一个总时长为指定长度的空白音频
        logger.info(f"创建总时长为 {total_duration_ms/1000:.2f}秒 的空白音频")
        complete_audio = AudioSegment.silent(duration=total_duration_ms)
        
        # 对音频片段按开始时间排序
        audio_segments.sort(key=lambda x: x[1])
        
        last_end_time_ms = 0
        
        # 将每个音频片段放置到对应位置
        for i, (audio_path, start_time_ms, duration_sec) in enumerate(audio_segments):
            try:
                # 读取音频文件
                segment_audio = AudioSegment.from_wav(audio_path)
                
                # 调整开始时间，确保不与前一个音频重叠
                adjusted_start_time_ms = max(start_time_ms, last_end_time_ms + GAP_BETWEEN_SEGMENTS)
                
                # 确保不超出总时长
                if adjusted_start_time_ms >= total_duration_ms:
                    logger.warning(f"音频片段 {audio_path.name} 的开始时间 {adjusted_start_time_ms/1000:.2f}秒 超出了总时长 {total_duration_ms/1000:.2f}秒，跳过此片段")
                    continue
                
                # 更新最后结束时间
                last_end_time_ms = adjusted_start_time_ms + len(segment_audio)
                
                # 确保最后结束时间不超出总时长
                if last_end_time_ms > total_duration_ms:
                    # 裁剪音频以适应总时长
                    segment_audio = segment_audio[:total_duration_ms - adjusted_start_time_ms]
                    last_end_time_ms = total_duration_ms
                
                logger.info(f"放置音频片段 {audio_path.name} 到位置 {adjusted_start_time_ms/1000:.2f}秒，结束于 {last_end_time_ms/1000:.2f}秒")
                
                # 叠加音频
                complete_audio = complete_audio.overlay(segment_audio, position=adjusted_start_time_ms)
                
            except Exception as e:
                logger.error(f"处理音频片段 {audio_path} 时出错: {e}")
                continue
        
        # 导出完整音频
        logger.info(f"导出完整音频到 {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        complete_audio.export(output_path, format="wav")
        logger.info(f"成功创建完整音频: {output_path} (总时长: {total_duration_ms/1000:.2f}秒)")
        return True
        
    except Exception as e:
        logger.error(f"合成完整音频时出错: {e}", exc_info=True)
        return False

def main():
    parser = argparse.ArgumentParser(description="从带时间戳的文本文件生成多语言TTS音频，并合成为完整音频文件")
    parser.add_argument("input_file", type=str, help="包含时间戳文本的输入文件路径")
    parser.add_argument("output_dir", type=str, help="保存生成的音频文件的输出目录")
    parser.add_argument("--video_file", type=str, default="batch_process_tool/example.mp4", help="用于声音克隆的视频文件路径")
    parser.add_argument("--target_langs", type=str, default=DEFAULT_TARGET_LANGS,
                      help=f"用于翻译的目标语言代码，逗号分隔 (例如, 'zh,ja,ko'，默认: {DEFAULT_TARGET_LANGS})。")
    parser.add_argument("--gemini_api_key", type=str, default="AIzaSyBl6bVncUpYL9XWGZ-jirSYwNaRy-cW5Rc",
                      help="Google Gemini API密钥。")
    parser.add_argument("--elevenlabs_api_key", type=str, default="sk_dd6498412439f53cd9724717d79fbcbb00d5a320f8122d12",
                      help="ElevenLabs API密钥。")
    parser.add_argument("--elevenlabs_voice_type", type=str, default="male", choices=["male", "female"],
                      help="ElevenLabs TTS的声音类型（男性/女性）。如果未使用克隆声音，则使用此选项。默认：male。")
    parser.add_argument("--overwrite_audio", action="store_true", help="覆盖输出目录中已存在的翻译音频文件。")
    parser.add_argument("--skip_voice_cloning", action="store_true", help="跳过声音克隆步骤，使用默认声音。")
    
    args = parser.parse_args()
    
    # 检查输入文件
    input_path = Path(args.input_file)
    if not input_path.is_file():
        logger.error(f"找不到输入文件: {args.input_file}")
        sys.exit(1)
        
    # 创建输出目录
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 初始化API客户端
    try:
        logger.info("初始化API客户端...")
        genai.configure(api_key=args.gemini_api_key)
        gemini_translator = GeminiTextTranslator(api_key=args.gemini_api_key)
        elevenlabs_tts = ElevenLabsTTS(api_key=args.elevenlabs_api_key)
        logger.info("API客户端初始化成功")
    except Exception as e:
        logger.critical(f"API初始化错误: {e}")
        sys.exit(1)
    
    # 克隆声音（如果未设置跳过）
    cloned_voice_id = None
    if not args.skip_voice_cloning:
        video_path = Path(args.video_file)
        if not video_path.exists():
            logger.error(f"视频文件不存在: {video_path}")
            sys.exit(1)
            
        logger.info("从视频中提取音频用于声音克隆...")
        temp_audio_path = output_path / "temp_audio_for_cloning.mp3"
        
        if not extract_audio_from_video(video_path, temp_audio_path):
            logger.error("无法从视频提取音频。退出。")
            sys.exit(1)
            
        logger.info("开始克隆声音...")
        voice_name = f"Cloned Voice from {video_path.name}"
        cloned_voice_id = clone_voice_from_audio(args.elevenlabs_api_key, voice_name, temp_audio_path)
        
        if cloned_voice_id:
            logger.info(f"声音克隆成功！使用克隆的声音ID: {cloned_voice_id}")
            update_voice_ids_with_cloned(cloned_voice_id)
        else:
            logger.error("声音克隆失败，将使用默认声音。")
    else:
        logger.info("跳过声音克隆，使用默认声音。")
    
    # 解析目标语言
    target_langs_list = [lang.strip().lower() for lang in args.target_langs.split(',') if lang.strip()]
    valid_target_langs = [lang for lang in target_langs_list if lang in TARGET_LANGUAGE_NAMES]
    
    if not valid_target_langs:
        logger.error("未指定有效的目标语言。退出。")
        sys.exit(1)
    
    # 解析时间戳文本
    logger.info(f"解析带时间戳的文本文件: {input_path}")
    segments = parse_timestamped_text(input_path)
    logger.info(f"从文件中解析出 {len(segments)} 个文本段落")
    
    # 处理文本段落，生成各个音频片段
    base_filename = input_path.stem
    audio_segments_by_lang = process_text_segments(
        segments=segments,
        output_dir=output_path,
        base_filename=base_filename,
        target_langs=valid_target_langs,
        gemini_translator=gemini_translator,
        elevenlabs_tts=elevenlabs_tts,
        elevenlabs_voice_type=args.elevenlabs_voice_type,
        elevenlabs_custom_voice_id=cloned_voice_id,  # 使用克隆的声音ID
        overwrite_audio=args.overwrite_audio
    )
    
    # 为每种语言组装完整的音频文件
    for lang_code, audio_segments in audio_segments_by_lang.items():
        if not audio_segments:
            logger.warning(f"语言 {lang_code} 没有可用的音频片段")
            continue
            
        output_complete_path = output_path / f"{base_filename}_{lang_code}_complete.wav"
        
        logger.info(f"组装语言 {lang_code} 的完整音频文件: {output_complete_path}")
        success = assemble_complete_audio(
            audio_segments=audio_segments,
            output_path=output_complete_path,
            total_duration_ms=FINAL_AUDIO_LENGTH_SECONDS * 1000
        )
        
        if success:
            logger.info(f"成功创建语言 {lang_code} 的完整音频文件: {output_complete_path}")
        else:
            logger.error(f"创建语言 {lang_code} 的完整音频文件失败")
    
    logger.info("=== 所有处理完成 ===")
    
if __name__ == "__main__":
    import importlib
    required_modules = ['google.generativeai', 'requests', 'pydub']
    missing_modules = []
    for module_name in required_modules:
        try: importlib.import_module(module_name)
        except ImportError: missing_modules.append(module_name)
    if missing_modules:
        print("错误: 缺少包", file=sys.stderr)
        print(f"  pip install {' '.join(missing_modules)}", file=sys.stderr)
        sys.exit(1)
        
    main() 