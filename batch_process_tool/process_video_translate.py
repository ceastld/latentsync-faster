#!/usr/bin/env python
import os
import sys
import argparse
import subprocess
import asyncio
import logging
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Any
import google.generativeai as genai
import aiohttp
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError
import tempfile # For temporary audio file
import io
import wave
import time
import requests

# --- Configuration Constants ---
VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.mpeg', '.mpg')
DEFAULT_TARGET_FPS = 25
DEFAULT_TARGET_LANGS = "zh,en,es,fr,ru,ja,ko,de,it,uk,pt,pt-br,tr,hi" # Example default languages

# Gemini/ElevenLabs API Keys (It's better to get these from args or env vars)
# GEMINI_API_KEY = "YOUR_GEMINI_API_KEY" 
# ELEVENLABS_API_KEY = "YOUR_ELEVENLABS_API_KEY"

# ElevenLabs Voices (Copied from speech_translator_tts.py, can be customized)
elevenlabs_voices = {          
            "en": {"male": "1SM7GgM6IMuvQlz2BwM3", "female": "XfNU2rGpBa01ckF309OY"}, 
            "en-us": {"male": "uju3wxzG5OhpWcoi3SMy", "female": "yM93hbw8Qtvdma2wCnJG"},
            "en-gb": {"male": "Fahco4VZzobUeiPqni1S", "female": "wJqPPQ618aTW29mptyoc"},
            "es": {"male": "y6WtESLj18d0diFRruBs", "female": "IOyj8WtBHdke2FjQgGAr"}, 
            "es-mx": {"male": "W6Z2FAa578IKOGSVo2sA", "female": "hHjbwzYZW17oh0p05AKv"},
            "es-ar": {"male": "HAsl3FenyWHYwECSP6Hl", "female": "1WXz8v08ntDcSTeVXMN2"},
            "es-cl": {"male": "cMKZRsVE5V7xf6qCp9fF", "female": "JM2A9JbRp8XUJ7bdCXJc"},
            "es-co": {"male": "J2Jb9yZNvpXUNAL3a2bw", "female": "YPh7OporwNAJ28F5IQrm"},
            "zh": {"male": "4VZIsMPtgggwNg7OXbPY", "female": "Ca5bKgudqKJzq8YRFoAz"},
            "fr": {"male": "ohItIVrXTBI80RrUECOD", "female": "R6eR6IR1JzKTAyu3Itp6"},
            "de": {"male": "ukONT0PiO5smfFLmTj12", "female": "otF9rqKzRHFgfwf6serQ"},
            "it": {"male": "W71zT1VwIFFx3mMGH2uZ", "female": "201hPjDVu4Q5DUV7tMQJ"},
            "uk": {"male": "GVRiwBELe0czFUAJj0nX", "female": "3rWBcFHu7rpPUEJQYEqD"},
            "pt": {"male": "36rVQA1AOIPwpA3Hg1tC", "female": "33B4UnXyTNbgLmdEDh5P"},
            "pt-br": {"male": "vr6MVhO51WHYH7ev2Qn9", "female": "lRbfoJL2IRJBT7ma6o7n"},
            "ru": {"male": "blxHPCXhpXOsc7mCKk0P", "female": "AB9XsbSA4eLG12t2myjN"},
            "ja": {"male": "Mv8AjrYZCBkdsmDHNwcB", "female": "8EkOjt4xTPGMclNlh1pk"},
            "ko": {"male": "U1cJYS4EdbaHmfR7YzHd", "female": "uyVNoMrnUku1dZyVEXwD"},
            "tr": {"male": "lxZLq5dcyw12UangGJgN", "female": "mBUB5zYuPwfVE6DTcEjf"},
            "hi": {"male": "Sxk6njaoa7XLsAFT7WcN", "female": "1qEiC6qsybMkmnNdVMbK"},
            "id": {"male": "3mAVBNEqop5UbHtD8oxQ", "female": "LcvlyuBGMjj1h4uAtQjo"},
            "nl": {"male": "AVIlLDn2TVmdaDycgbo3", "female": "YUdpWWny7k5yb4QCeweX"},
            "sv": {"male": "x0u3EW21dbrORJzOq1m9", "female": "aSLKtNoVBZlxQEMsnGL2"},
            "bg": {"male": "kzrsjZhHCumKqmkJl486", "female": "pREMn4INXSs2KOPsNcsD"},
            "cs": {"male": "uYFJyGaibp4N2VwYQshk", "female": "OAAjJsQDvpg3sVjiLgyl"},
            "da": {"male": "ygiXC2Oa1BiHksD3WkJZ", "female": "ZKutKtutnlbOxDxkNlhk"},
            "fi": {"male": "ZKutKtutnlbOxDxkNlhk", "female": "YSabzCJMvEHDduIDMdwV"},
            "el": {"male": "czEPjbZ9jNJoQ7WzdyTa", "female": "czEPjbZ9jNJoQ7WzdyTa"},
            "hu": {"male": "AnNshXL08po8KEaf53gz", "female": "pREMn4INXSs2KOPsNcsD"},
            "ms": {"male": "NpVSXJvYSdIbjOaMbShj", "female": "djUbJhnXETnX31p3rgun"},
            "no": {"male": "2dhHLsmg0MVma2t041qT", "female": "k5IgYJw2jfo6mO5HhagG"},
            "pl": {"male": "gFl0NeqphJUaoBLtWrqM", "female": "Pid5DJleNF2sxsuF6YKD"},
            "ro": {"male": "5asM3ZxsegvXfXI5vqKQ", "female": "gbLy9ep70G3JW53cTzFC"},
            "sk": {"male": "bYqmvVkXUBwLwYpGHGz3", "female": "3K1lqsxxXFiTAXCO09Zv"},
            "vi": {"male": "ueSxRO0nLF1bj93J2hVt", "female": "foH7s9fX31wFFH2yqrFa"},
        }
DEFAULT_ELEVENLABS_VOICE_ID = elevenlabs_voices.get("en", {}).get("male", "1SM7GgM6IMuvQlz2BwM3") 
DEFAULT_ELEVENLABS_LANGUAGE = "en"

# Target language names for Gemini prompts (Copied from speech_translator_tts.py)
TARGET_LANGUAGE_NAMES = {
    "zh": "Mandarin Chinese",
    "en": "English",
    "ru": "Russian",
    "fr": "French",
    "es": "Spanish",
    "ja": "Japanese",
    "ko": "Korean",
    "de": "German",
    "it": "Italian",
    "uk": "Ukrainian",
    "pt": "Portuguese",
    "pt-br": "Portuguese (Brazil)",
    "tr": "Turkish",
    "hi": "Hindi",
    # Add more if needed, ensure they match keys in elevenlabs_voices if using ElevenLabs
}

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)] # Ensure logs go to stdout
)
logger = logging.getLogger("VideoTranslateProcessor")

# --- Helper Functions (To be added: ffmpeg check, video conversion, audio extraction, API classes, etc.) ---

# Function to check if ffmpeg is available
def check_ffmpeg():
    """检查 ffmpeg 命令是否可用"""
    try:
        # Use a list of strings for the command
        subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        logger.info("ffmpeg command found.")
        return True
    except FileNotFoundError:
        # Correctly formatted f-string
        logger.error("错误：找不到 'ffmpeg' 命令。请确保 FFmpeg 已安装并已添加到系统 PATH。")
        return False
    except subprocess.CalledProcessError as e:
        # Correctly formatted f-string
        logger.error(f"错误：执行 'ffmpeg -version' 时出错: {e}")
        return False
    except Exception as e:
        # Correctly formatted f-string
        logger.error(f"检查 ffmpeg 时发生未知错误: {e}")
        return False


# Function to convert video FPS using ffmpeg
def convert_video_fps(input_path: Path, output_path: Path, target_fps: int) -> bool:
    """使用 ffmpeg 将视频转换为目标帧率"""
    command = [
        "ffmpeg",
        "-y",  # 允许覆盖输出文件
        "-i", str(input_path),
        "-filter:v", f"fps=fps={target_fps}",
        "-c:a", "copy", # Copy audio stream without re-encoding
        str(output_path)
    ]
    logger.debug(f"执行视频转换命令: {' '.join(command)}")
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True) # Ensure output dir exists
        # Ensure text=True for text mode, specify encoding and error handling
        result = subprocess.run(command, check=True, text=True, capture_output=True, encoding='utf-8', errors='ignore')
        logger.info(f"成功转换视频帧率: {input_path.name} -> {output_path.name}")
        # logger.debug(f"FFmpeg 输出:\n{result.stderr}") # Uncomment for detailed ffmpeg output
        return True
    except FileNotFoundError:
        logger.error("错误：找不到 ffmpeg 命令。无法转换视频。")
        return False
    except subprocess.CalledProcessError as e:
        logger.error(f"错误：转换 {input_path.name} 时 ffmpeg 返回错误码 {e.returncode}")
        # Ensure stderr is decoded if captured as bytes, or handled correctly if text=True
        logger.error(f"FFmpeg 错误输出:\n{e.stderr}") 
        # Try to delete potentially incomplete output file
        if output_path.exists():
            try:
                output_path.unlink()
                logger.info(f"已删除不完整的输出视频文件: {output_path.name}")
            except OSError as del_e:
                logger.warning(f"警告：无法删除不完整的输出视频文件 {output_path.name}: {del_e}")
        return False
    except Exception as e:
        logger.error(f"转换视频 {input_path.name} 时发生未知错误: {e}", exc_info=True)
        return False

# Function to extract audio using ffmpeg
def extract_audio(video_filepath: Path, output_audio_path: Path) -> bool:
    """
    使用 ffmpeg 从视频文件中提取音频并保存为 MP3。
    """
    command = [
        'ffmpeg',
        '-i', str(video_filepath),
        '-vn', # Disable video recording
        '-codec:a', 'libmp3lame', # Use MP3 codec
        '-q:a', '2', # MP3 quality (0-9, lower is better, 2 is often considered high quality VBR)
        '-y', # Overwrite output file if it exists
        str(output_audio_path)
    ]
    logger.debug(f"执行音频提取命令: {' '.join(command)}")
    try:
        output_audio_path.parent.mkdir(parents=True, exist_ok=True) # Ensure output dir exists
        # Ensure text=True for text mode, specify encoding and error handling
        result = subprocess.run(command, check=True, text=True, capture_output=True, encoding='utf-8', errors='ignore')
        logger.info(f"成功提取音频: {video_filepath.name} -> {output_audio_path.name}")
        # logger.debug(f"FFmpeg 输出:\n{result.stderr}") # Uncomment for detailed ffmpeg output
        # Check if the output file was actually created and is not empty
        if not output_audio_path.exists() or output_audio_path.stat().st_size == 0:
             logger.error(f"音频提取似乎成功，但输出文件 {output_audio_path.name} 不存在或为空。")
             return False
        return True
    except FileNotFoundError:
        logger.error("错误：找不到 ffmpeg 命令。无法提取音频。")
        return False
    except subprocess.CalledProcessError as e:
        logger.error(f"错误：提取音频失败 {video_filepath.name}. FFmpeg 返回错误码 {e.returncode}")
        # Ensure stderr is decoded if captured as bytes, or handled correctly if text=True
        logger.error(f"FFmpeg 错误输出:\n{e.stderr}") 
        # Try to delete potentially incomplete output file
        if output_audio_path.exists():
            try:
                output_audio_path.unlink()
                logger.info(f"已删除不完整的输出音频文件: {output_audio_path.name}")
            except OSError as del_e:
                logger.warning(f"警告：无法删除不完整的输出音频文件 {output_audio_path.name}: {del_e}")
        return False
    except Exception as e:
        logger.error(f"提取音频 {video_filepath.name} 时发生未知错误: {e}", exc_info=True)
        return False

# Function to load audio data (from speech_translator_tts.py)
def load_audio_data(filepath: Path) -> Optional[Tuple[bytes, int]]:
    """Loads audio data and sample rate from WAV or MP3 file. Converts to mono, 16-bit, 16kHz."""
    file_suffix = filepath.suffix.lower()
    logger.info(f"Attempting to load audio file: {filepath}")

    try:
        if file_suffix == '.wav':
            sound = AudioSegment.from_wav(str(filepath))
            logger.info(f"Loaded WAV: {sound.duration_seconds:.2f}s, {sound.frame_rate}Hz, {sound.channels}ch, {sound.sample_width*8}-bit")
        elif file_suffix == '.mp3':
            # Explicitly specify ffmpeg path if needed, otherwise assumes it's in PATH
            # AudioSegment.converter = "/path/to/ffmpeg" 
            sound = AudioSegment.from_mp3(str(filepath))
            logger.info(f"Loaded MP3: {sound.duration_seconds:.2f}s, {sound.frame_rate}Hz, {sound.channels}ch, {sound.sample_width*8}-bit")
        else:
            logger.warning(f"Unsupported audio file format: {file_suffix}. Skipping file.")
            return None

        # Convert to mono
        if sound.channels > 1:
            logger.info("Audio is stereo, converting to mono.")
            sound = sound.set_channels(1)

        # Ensure sample width is 2 (16-bit) - Gemini prefers 16-bit
        if sound.sample_width != 2:
            logger.info(f"Audio sample width is {sound.sample_width}, converting to 16-bit (sample_width=2).")
            sound = sound.set_sample_width(2)
            
        # Gemini prefers 16kHz sample rate for STT
        target_sr = 16000 
        if sound.frame_rate != target_sr:
            logger.info(f"Resampling audio from {sound.frame_rate}Hz to {target_sr}Hz for STT.")
            sound = sound.set_frame_rate(target_sr)

        logger.info(f"Audio loaded and prepared successfully (mono, 16-bit, {sound.frame_rate}Hz). Duration: {sound.duration_seconds:.2f}s")
        # Return raw data (bytes) and sample rate (int)
        return sound.raw_data, sound.frame_rate

    except FileNotFoundError:
        logger.error(f"Audio file not found: {filepath}")
        return None
    except CouldntDecodeError:
        logger.error(f"Failed to decode audio file (check format/corruption, ensure ffmpeg/libav is installed and in PATH): {filepath}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred loading {filepath}: {e}", exc_info=True)
        return None

# Function to write text output (from speech_translator_tts.py)
def write_output(output_file_path: Path, content: str):
    """Writes content to a file, creating parent directories if needed."""
    try:
        output_file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        logger.info(f"Successfully wrote output to: {output_file_path}")
    except IOError as e:
        logger.error(f"Failed to write output file {output_file_path}: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred writing to {output_file_path}: {e}", exc_info=True)


# --- API Processing Classes (Copied/Adapted from speech_translator_tts.py) ---

class GeminiProcessorBase:
    """Base class for Gemini processing operations."""
    def __init__(self, api_key: str, model: str = "gemini-1.5-flash-latest"):
        if not api_key:
            raise ValueError("Gemini API key is required.")
        self._model_name = model
        try:
             self._model = genai.GenerativeModel(model)
             logger.info(f"Initialized Gemini model '{model}'.")
        except Exception as e:
             logger.critical(f"Failed Gemini Init: {e}"); raise ValueError(f"Could not init Gemini: {e}") from e

    def _send_to_gemini(self, contents: List, generation_config: Optional[genai.types.GenerationConfig] = None) -> Optional[genai.types.GenerateContentResponse]:
        """Sends request to Gemini API synchronously and handles potential errors."""
        logger.debug(f"Sending sync request to Gemini model: {self._model_name}...")
        response = None
        try:
            # Direct synchronous call to the library method
            response = self._model.generate_content(
                contents=contents,
                generation_config=generation_config,
                request_options={'timeout': 300} 
            )
            logger.debug("Gemini sync request completed.")
            if not response: logger.warning("Empty response from Gemini."); return None
            if response.prompt_feedback and response.prompt_feedback.block_reason: 
                 logger.error(f"Gemini blocked: {response.prompt_feedback.block_reason}..."); return None
            if response.candidates:
                 # ... check finish reason ...
                 pass 
            elif not hasattr(response, 'text') or not response.text:
                 logger.error("Gemini response lacks candidates and text."); return None
            
            return response

        except Exception as e:
            logger.error(f"Error during sync Gemini API call ({self._model_name}): {e}", exc_info=True)
            return None


class GeminiSTT(GeminiProcessorBase):
    """Handles Speech-to-Text using Gemini API."""
    def __init__(self, api_key: str, model: str = "gemini-1.5-flash-latest"):
        super().__init__(api_key, model)

    def _create_wav_in_memory(self, audio_data: bytes, sample_rate: int) -> Optional[bytes]:
        """Creates a WAV file format in memory from raw audio data."""
        try:
            buffer = io.BytesIO()
            with wave.open(buffer, 'wb') as wf:
                wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sample_rate)
                wf.writeframes(audio_data)
            return buffer.getvalue()
        except Exception as e:
            logger.error(f"Error creating WAV in memory: {e}", exc_info=True); return None

    def recognize(self, audio_data: bytes, sample_rate: int) -> Tuple[Optional[str], Optional[str]]:
        """Performs STT synchronously, detecting language.
        
        Returns:
            Tuple (recognized_text, detected_language_code) or (None, None)
        """
        logger.info(f"Starting STT recognition with Gemini model '{self._model_name}'. Audio duration: {len(audio_data) / (sample_rate * 2):.2f}s")
        wav_data = self._create_wav_in_memory(audio_data, sample_rate)
        if not wav_data: return None, None

        audio_file = None
        try:
            logger.debug("Uploading audio data to Gemini File API...")
            audio_file = genai.upload_file(path=io.BytesIO(wav_data), mime_type="audio/wav")
            logger.info(f"Uploaded audio for STT, Gemini file name: {audio_file.name}")

            logger.debug(f"Waiting for Gemini file {audio_file.name} to become active...")
            processing_laps = 0
            while audio_file.state.name == "PROCESSING":
                 print('.', end='', flush=True)
                 processing_laps += 1
                 time.sleep(min(5, 1 + processing_laps // 2))
                 audio_file = genai.get_file(name=audio_file.name)
                 if processing_laps > 60: raise TimeoutError("Gemini file processing timeout")
            print()

            if audio_file.state.name == "FAILED": logger.error(f"Gemini audio file processing failed: {audio_file.name}"); return None, None
            if audio_file.state.name != "ACTIVE": logger.error(f"Gemini audio file {audio_file.name} finished in unexpected state: {audio_file.state.name}"); return None, None
            logger.info(f"Gemini audio file ready: {audio_file.name}")

            # Original prompt for transcription and language detection
            prompt = (
                "Transcribe this audio file precisely. "
                "Identify the main language spoken and return its ISO 639-1 code (e.g., en, zh, es). "
                "Respond ONLY with the following two lines in this exact order:\n"
                "1. Transcription\n"
                "2. Language Code (or 'unknown')"
            )
            contents = [audio_file, prompt]
            response = self._send_to_gemini(contents)

            # Original response parsing for text and language
            response_text = None
            if hasattr(response, 'text') and response.text: response_text = response.text
            elif response and response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                 response_text = "".join(part.text for part in response.candidates[0].content.parts if hasattr(part, 'text'))

            if response_text:
                 try:
                     lines = response_text.strip().split('\n')
                     recognized_text = lines[0].strip() if len(lines) >= 1 else None
                     detected_language_code = None
                     if len(lines) >= 2:
                          lang_code_raw = lines[1].strip().lower()
                          if lang_code_raw and lang_code_raw != 'unknown' and len(lang_code_raw) <= 3 and lang_code_raw.islower():
                               detected_language_code = lang_code_raw
                          elif lang_code_raw == 'unknown': logger.info("Gemini reported language as 'unknown'.")
                          else: logger.warning(f"Received potentially invalid language code '{lang_code_raw}', treating as unknown.")
                     
                     if not recognized_text: logger.warning("STT response text was empty after stripping."); return None, None
                     logger.info(f"STT successful. Lang: {detected_language_code or 'Unknown'}. Text: {recognized_text[:100]}...")
                     return recognized_text, detected_language_code
                 except Exception as e:
                      logger.error(f"Error parsing STT response: {e}. Response text: {response_text}", exc_info=True)
                      if response_text: return response_text.strip(), None # Fallback
                      return None, None
            else:
                 logger.error("STT failed. No valid response text received from Gemini.")
                 return None, None

        except Exception as e:
             logger.error(f"An error occurred during STT processing: {e}", exc_info=True)
             return None, None
        finally:
            if audio_file and hasattr(audio_file, 'name') and audio_file.name:
                try:
                    logger.debug(f"Attempting to delete temporary Gemini file: {audio_file.name}")
                    genai.delete_file(name=audio_file.name)
                    logger.info(f"Deleted temporary Gemini file: {audio_file.name}")
                except Exception as e: logger.warning(f"Could not delete Gemini file {audio_file.name}: {e}")


class GeminiTextTranslator(GeminiProcessorBase):
    """Handles text translation using Gemini API."""
    def __init__(self, api_key: str, model: str = "gemini-1.5-flash-latest"):
        super().__init__(api_key, model)

    def translate(self, text_to_translate: str, target_language_name: str, source_language_name: Optional[str] = None) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Translates text to the target language.

        Args:
            text_to_translate: The text to translate.
            target_language_name: The full name of the target language (e.g., "Mandarin Chinese").
            source_language_name: Optional. The full name of the source language.

        Returns:
            A tuple: (translated_text, detected_source_language_name, error_message).
            Returns (None, None, error_message) on failure.
        """
        if not text_to_translate:
            logger.warning("Translate called with empty text.")
            return None, None, "Input text was empty."

        logger.info(f"Starting translation to {target_language_name} using Gemini model '{self._model_name}'.")
        
        # Construct the prompt
        if source_language_name:
             prompt = f"Translate the following text from {source_language_name} to {target_language_name}. Respond ONLY with the translated text:\n\n{text_to_translate}"
        else:
             prompt = f"Translate the following text to {target_language_name}. Identify the source language. Respond with the translated text on the first line, and the detected source language name (e.g., English, Spanish) on the second line.\n\n{text_to_translate}"

        contents = [prompt]
        # Consider adjusting generation config (e.g., temperature for creativity vs accuracy)
        # generation_config = genai.types.GenerationConfig(temperature=0.3) 
        generation_config=None

        response = self._send_to_gemini(contents, generation_config=generation_config)

        if response and response.text:
            try:
                lines = response.text.strip().split('\n')
                translated_text = lines[0].strip()
                
                detected_source = None
                if not source_language_name and len(lines) > 1:
                    detected_source = lines[1].strip()
                elif source_language_name:
                     detected_source = source_language_name # We provided it

                if not translated_text:
                     logger.warning("Translation response text was empty after stripping.")
                     return None, detected_source, "Translated text was empty."
                     
                logger.info(f"Translation successful. Source: {detected_source or 'Auto-detected'}. Target: {target_language_name}. Translated text: {translated_text[:100]}...")
                return translated_text, detected_source, None
            except Exception as e:
                error_msg = f"Error parsing translation response: {e}. Response text: {response.text}"
                logger.error(error_msg, exc_info=True)
                # Fallback: return the raw text if parsing failed but text exists? Or just fail? Let's fail clearly.
                return None, None, error_msg # Return raw text as translated? Maybe not useful.
        else:
            error_msg = "Translation failed. No valid response received from Gemini."
            logger.error(error_msg)
            return None, None, error_msg


class ElevenLabsTTS:
    """Handles Text-to-Speech using ElevenLabs API (Synchronous)."""
    BASE_URL = "https://api.elevenlabs.io/v1"

    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("ElevenLabs API key is required.")
        self._api_key = api_key
        # self._session = requests.Session() # Optionally use a session

    def run_tts(self, text: str, language_code: str, voice_id: str, output_wav_path: Path, voice_settings: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Generates speech synchronously using ElevenLabs and saves as WAV."""
        if not text: logger.warning("TTS called with empty text."); return None

        tts_url = f"{self.BASE_URL}/text-to-speech/{voice_id}/stream?output_format=mp3_44100_128"
        headers = { "Content-Type": "application/json", "xi-api-key": self._api_key }
        model_id = "eleven_multilingual_v2" if language_code != 'en' else "eleven_english_v2"
        payload = {"text": text, "model_id": model_id}
        if voice_settings: payload["voice_settings"] = voice_settings

        logger.info(f"Starting TTS request to ElevenLabs. Voice: {voice_id}, Model: {model_id}, Lang: {language_code}. Output: {output_wav_path}")

        temp_mp3_path = None
        try:
            # Use requests.post with stream=True for synchronous streaming
            response = requests.post(tts_url, headers=headers, json=payload, stream=True, timeout=300)

            if response.status_code == 200:
                logger.info(f"TTS request successful (Status {response.status_code}). Receiving audio stream...")
                output_wav_path.parent.mkdir(parents=True, exist_ok=True)

                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_mp3_file_obj:
                    temp_mp3_path = Path(temp_mp3_file_obj.name)
                
                logger.debug(f"Streaming ElevenLabs MP3 response to temporary file: {temp_mp3_path}")
                bytes_written = 0
                try:
                    # Write stream content synchronously using iter_content
                    with open(temp_mp3_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk: # filter out keep-alive new chunks
                                f.write(chunk)
                                bytes_written += len(chunk)
                    
                    logger.info(f"Successfully downloaded MP3 stream to {temp_mp3_path} ({bytes_written} bytes)")
                    if bytes_written == 0: logger.error(f"Temporary MP3 file {temp_mp3_path} is empty."); return None

                    # Convert MP3 to WAV (remains synchronous)
                    logger.info(f"Converting temporary MP3 {temp_mp3_path} to WAV {output_wav_path}")
                    sound = AudioSegment.from_mp3(temp_mp3_path)
                    sound.export(output_wav_path, format="wav")
                    logger.info(f"Successfully converted to WAV: {output_wav_path}")
                    return str(output_wav_path)

                except Exception as e: # Handle download/conversion errors
                    logger.error(f"Error during MP3 download or conversion: {e}", exc_info=True); return None
                finally: # Cleanup temp MP3
                    # Ensure temporary MP3 is deleted after conversion attempt
                    if temp_mp3_path and temp_mp3_path.exists():
                         try: temp_mp3_path.unlink(); logger.debug(f"Deleted temporary MP3 file: {temp_mp3_path}")
                         except OSError as del_e: logger.warning(f"Could not delete temporary MP3 file {temp_mp3_path}: {del_e}")

            else: # Handle HTTP errors
                error_text = response.text # Get error text synchronously
                logger.error(f"ElevenLabs TTS request failed. Status: {response.status_code}, Response: {error_text}")
                return None

        except requests.exceptions.RequestException as e: # Catch requests exceptions
            logger.error(f"Network or HTTP error during ElevenLabs TTS request: {e}", exc_info=True); return None
        except Exception as e: # Catch other errors
            logger.error(f"An unexpected error occurred during ElevenLabs TTS: {e}", exc_info=True); return None


# --- Main Processing Logic (Synchronous) ---

def process_translation_and_tts_for_lang( # Synchronous
    original_text: str,
    detected_source_lang_name: Optional[str],
    target_lang_code: str,
    target_lang_name: str,
    output_dir: Path,
    base_filename: str,
    gemini_translator_client: GeminiTextTranslator,
    elevenlabs_tts_client: ElevenLabsTTS,
    elevenlabs_voice_type: str,
    elevenlabs_custom_voice_id: Optional[str],
    overwrite_audio: bool
) -> bool:
    """Handles translation and TTS synchronously. No text saving.
       Uses specified voice type or custom ID.
    """
    output_wav_path = output_dir / f"{base_filename}_{target_lang_code}.wav"

    if output_wav_path.exists() and not overwrite_audio:
         logger.info(f"Skipping: Translated audio file '{output_wav_path.name}' already exists.")
         return True 

    # --- 1. Translate Text --- 
    logger.info(f"Translating text to {target_lang_name}...")
    # Synchronous call
    translated_text, _, error_msg = gemini_translator_client.translate( 
        original_text, target_lang_name, detected_source_lang_name
    )
    if error_msg: logger.error(f"Translation failed: {error_msg}"); return False
    if not translated_text: logger.error("Translation resulted in empty text."); return False
    logger.info(f"Translation successful.")

    # --- 2. Synthesize Speech (TTS) --- 
    logger.info(f"Synthesizing speech for {target_lang_name}... Output: {output_wav_path.name}")
    voice_id_to_use = elevenlabs_custom_voice_id
    chosen_logic = "custom ID" 
    effective_lang_code = target_lang_code

    if voice_id_to_use:
        logger.info(f"Using custom voice ID specified: {voice_id_to_use}")
    else:
        logger.info(f"Attempting to find voice for type: '{elevenlabs_voice_type}' for language '{target_lang_code}'")
        chosen_logic = f"voice type arg ({elevenlabs_voice_type})"
        lang_voices = elevenlabs_voices.get(target_lang_code)

        if lang_voices:
            voice_id_to_use = lang_voices.get(elevenlabs_voice_type)
            if voice_id_to_use:
                 logger.info(f"Found voice ID for specified type '{elevenlabs_voice_type}' in '{target_lang_code}': {voice_id_to_use}")
            else:
                 # Fallback 1: Try the *other* gender if specified type failed
                 logger.warning(f"Voice for type '{elevenlabs_voice_type}' not found for '{target_lang_code}'. Trying other gender.")
                 other_gender = "female" if elevenlabs_voice_type == "male" else "male"
                 voice_id_to_use = lang_voices.get(other_gender)
                 if voice_id_to_use:
                      logger.info(f"Found voice ID for other gender '{other_gender}' in '{target_lang_code}': {voice_id_to_use}")
                      chosen_logic = f"other gender ({other_gender})"
                 else:
                      logger.warning(f"Could not find any voice for language '{target_lang_code}' after trying both genders.")
        else:
             logger.warning(f"No voices configured at all for language '{target_lang_code}'.")

        # Fallback 2: Use default English voice if still no voice found
        if not voice_id_to_use:
             logger.warning(f"No voice found for '{target_lang_code}'. Falling back to default English.")
             default_voice_id_candidate = elevenlabs_voices.get(DEFAULT_ELEVENLABS_LANGUAGE, {}).get(elevenlabs_voice_type)
             if default_voice_id_candidate:
                  voice_id_to_use = default_voice_id_candidate
                  chosen_logic = f"default English ({elevenlabs_voice_type})"
             else:
                  # Absolute fallback to the hardcoded default ID
                  voice_id_to_use = DEFAULT_ELEVENLABS_VOICE_ID 
                  chosen_logic = f"hardcoded default English (id: {DEFAULT_ELEVENLABS_VOICE_ID})"
             
             effective_lang_code = DEFAULT_ELEVENLABS_LANGUAGE
             logger.info(f"Using default English voice: {voice_id_to_use} (Lang: {effective_lang_code}) chosen via [{chosen_logic}] logic.")


    if not voice_id_to_use:
         logger.error("FATAL: Could not determine a valid ElevenLabs Voice ID after all fallbacks.")
         return False

    voice_settings = {"stability": 0.6, "similarity_boost": 0.75}

    # Synchronous call
    tts_output_path_str = elevenlabs_tts_client.run_tts(
        text=translated_text, language_code=effective_lang_code, voice_id=voice_id_to_use,
        output_wav_path=output_wav_path, voice_settings=voice_settings
    )

    if tts_output_path_str:
        logger.info(f"TTS successful (Logic: {chosen_logic}): {tts_output_path_str}")
        return True
    else:
        logger.error(f"TTS failed (Logic: {chosen_logic}, Voice: {voice_id_to_use})")
        return False


def process_video_file( # Synchronous
    video_file_path: Path, 
    output_root_dir: Path,
    target_fps: int,
    target_langs: List[str],
    gemini_stt_client: GeminiSTT, 
    gemini_translator_client: GeminiTextTranslator, 
    elevenlabs_tts_client: ElevenLabsTTS, 
    elevenlabs_voice_type: str, 
    elevenlabs_custom_voice_id: Optional[str],
    overwrite_video: bool,
    overwrite_audio: bool
):
    """Processes a single video file synchronously. No text saving."""
    video_filename = video_file_path.name
    base_filename = video_file_path.stem
    logger.info(f"--- Processing Video File: {video_filename} ---")
    output_subdir = output_root_dir / base_filename 
    output_subdir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output will be saved to: {output_subdir}")

    # 1. Convert Video FPS (Synchronous call via subprocess.run)
    output_video_path = output_subdir / video_filename 
    logger.info(f"Target video output path: {output_video_path}")
    if not output_video_path.exists() or overwrite_video:
        logger.info(f"Converting video FPS for {video_filename}...")
        if not convert_video_fps(video_file_path, output_video_path, target_fps):
            logger.error(f"Video conversion failed. Skipping."); return
        logger.info(f"Video conversion successful.")
    else: logger.info(f"Output video exists. Skipping conversion.")

    # 2. Extract Audio (Synchronous call via subprocess.run)
    temp_audio_path = None 
    audio_processing_successful = False
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_audio_file_obj: temp_audio_path = Path(temp_audio_file_obj.name)
        logger.info(f"Extracting audio to temp: {temp_audio_path}")
        if not extract_audio(video_file_path, temp_audio_path):
            logger.error(f"Audio extraction failed. Skipping."); return
        logger.info("Audio extraction successful.")
        if not temp_audio_path.exists() or temp_audio_path.stat().st_size < 100: 
            logger.error(f"Temp audio file is missing/small. Skipping."); return
            
        # 3. Load audio (Synchronous)
        audio_data_tuple = load_audio_data(temp_audio_path)
        if not audio_data_tuple: logger.error(f"Failed to load audio. Skipping."); return
        audio_data, sample_rate = audio_data_tuple
        logger.info(f"Audio loaded for STT.")

        # 4. STT (Synchronous call)
        logger.info("Performing STT...")
        original_text, detected_language_code = gemini_stt_client.recognize(audio_data, sample_rate)
        if not original_text: logger.error(f"STT failed. Skipping."); return
        logger.info(f"STT successful. Lang: {detected_language_code or 'Unknown'}.")
        detected_language_name = TARGET_LANGUAGE_NAMES.get(detected_language_code) if detected_language_code else None
        
        # 5. Translate & TTS for each target language (Synchronous loop)
        successful_translations = 0
        total_languages_to_process = 0
        languages_processed = []
        for lang_code in target_langs:
             if detected_language_code and lang_code == detected_language_code: continue 
             target_lang_name = TARGET_LANGUAGE_NAMES.get(lang_code)
             if not target_lang_name: continue 
             output_wav_path = output_subdir / f"{base_filename}_{lang_code}.wav"
             if output_wav_path.exists() and not overwrite_audio: continue 
             
             total_languages_to_process += 1
             logger.info(f"-- Processing language: {target_lang_name} ({lang_code}) --")
             try:
                 # Synchronous call
                 success = process_translation_and_tts_for_lang( 
                     original_text=original_text,
                     detected_source_lang_name=detected_language_name,
                     target_lang_code=lang_code, target_lang_name=target_lang_name,
                     output_dir=output_subdir, base_filename=base_filename,
                     gemini_translator_client=gemini_translator_client,
                     elevenlabs_tts_client=elevenlabs_tts_client,
                     elevenlabs_voice_type=elevenlabs_voice_type,
                     elevenlabs_custom_voice_id=elevenlabs_custom_voice_id,
                     overwrite_audio=overwrite_audio
                 )
                 if success: successful_translations += 1
                 languages_processed.append(lang_code if success else f"{lang_code}(failed)")
             except Exception as lang_e:
                  logger.error(f"Error processing lang {lang_code}: {lang_e}", exc_info=True)
                  languages_processed.append(f"{lang_code}(failed)")

        # Log summary
        if total_languages_to_process > 0:
             logger.info(f"Finished TTS for {base_filename}. Success: {successful_translations}/{total_languages_to_process}. Processed: {languages_processed}")
             audio_processing_successful = successful_translations > 0
        else:
             logger.info(f"No new TTS tasks needed for {base_filename}."); audio_processing_successful = True 

    except Exception as e: logger.error(f"Error during audio processing: {e}", exc_info=True); audio_processing_successful = False
    finally:
        if temp_audio_path and temp_audio_path.exists():
            try: temp_audio_path.unlink(); logger.debug(f"Deleted temp audio: {temp_audio_path}")
            except OSError as e: logger.warning(f"Could not delete temp audio {temp_audio_path}: {e}")

    if audio_processing_successful: logger.info(f"--- Finished processing Video: {video_filename} ---")
    else: logger.error(f"--- Finished processing Video: {video_filename} with errors. ---")


def main(): # Synchronous
    # Update description
    parser = argparse.ArgumentParser(description="Process video files directly in a folder: convert FPS, extract audio, translate audio, and synthesize speech synchronously.")
    # Update input_dir help text
    parser.add_argument("--input_dir", type=str, help="Directory containing video files to process.") 
    parser.add_argument("--output_dir", type=str, help="Directory to save the processed results (in subfolders named after videos).")
    parser.add_argument("--target_fps", type=int, default=DEFAULT_TARGET_FPS, help=f"Target video frame rate (default: {DEFAULT_TARGET_FPS}).")
    parser.add_argument("--target_langs", type=str, default=DEFAULT_TARGET_LANGS,
                        help=f"Comma-separated list of target language codes for translation (e.g., 'zh,fr,es', default: {DEFAULT_TARGET_LANGS}). See TARGET_LANGUAGE_NAMES.")
    parser.add_argument("--gemini_api_key", type=str, default="AIzaSyBl6bVncUpYL9XWGZ-jirSYwNaRy-cW5Rc", 
                        help="Google Gemini API Key (can also be set via GEMINI_API_KEY env var). Required.")
    parser.add_argument("--elevenlabs_api_key", type=str, default="sk_dd6498412439f53cd9724717d79fbcbb00d5a320f8122d12", 
                        help="ElevenLabs API Key (can also be set via ELEVENLABS_API_KEY env var). Required.")
    parser.add_argument("--gemini_stt_model", type=str, default="gemini-1.5-flash-latest", help="Gemini model for STT.")
    parser.add_argument("--gemini_translate_model", type=str, default="gemini-1.5-flash-latest", help="Gemini model for Translation.")
    # Update help text for voice type/id
    parser.add_argument("--elevenlabs_voice_type", type=str, default="male", choices=["male", "female"], 
                        help="Voice type (male/female) for ElevenLabs TTS. Used if custom voice ID is not provided. Default: male.") 
    parser.add_argument("--elevenlabs_custom_voice_id", type=str, default=None, 
                        help="Specific ElevenLabs Voice ID to use (overrides voice_type selection). Default: None.") 
    parser.add_argument("--overwrite_video", action="store_true", help="Overwrite existing video files in the output directory.")
    parser.add_argument("--overwrite_audio", action="store_true", help="Overwrite existing translated audio files (wav) in the output directory.")

    args = parser.parse_args()

    # (Keep validation, ffmpeg check, Gemini config) ...
    if not args.gemini_api_key: logger.error("Gemini API Key is required."); sys.exit(1)
    if not args.elevenlabs_api_key: logger.error("ElevenLabs API Key is required."); sys.exit(1)
    try: genai.configure(api_key=args.gemini_api_key); logger.debug("Gemini configured.")
    except Exception as e: logger.critical(f"Failed Gemini config: {e}"); sys.exit(1)
    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    if not input_path.is_dir(): logger.error(f"Input directory not found: {args.input_dir}"); sys.exit(1)
    if not check_ffmpeg(): sys.exit(1)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Validate target languages
    target_langs_list = [lang.strip().lower() for lang in args.target_langs.split(',') if lang.strip()]
    valid_target_langs = [lang for lang in target_langs_list if lang in TARGET_LANGUAGE_NAMES]
    skipped_langs = [lang for lang in target_langs_list if lang not in TARGET_LANGUAGE_NAMES]
    if skipped_langs: logger.warning(f"Skipping unsupported target languages: {skipped_langs}")
    if not valid_target_langs: logger.error("No valid target languages specified or configured. Exiting."); sys.exit(1)
    
    logger.info(f"Input Dir: {input_path.resolve()}")
    logger.info(f"Output Dir: {output_path.resolve()}")
    logger.info(f"Target FPS: {args.target_fps}")
    logger.info(f"Target Languages: {valid_target_langs}")
    logger.info(f"Voice Type: {args.elevenlabs_voice_type}")
    logger.info(f"Custom Voice ID: {args.elevenlabs_custom_voice_id or 'Not specified'}")
    logger.info(f"Overwrite Video: {args.overwrite_video}")
    logger.info(f"Overwrite Audio: {args.overwrite_audio}")

    # --- Initialize API Clients (Synchronous) --- 
    try:
        logger.info("Initializing API clients...")
        gemini_stt = GeminiSTT(api_key=args.gemini_api_key, model=args.gemini_stt_model)
        gemini_translator = GeminiTextTranslator(api_key=args.gemini_api_key, model=args.gemini_translate_model)
        elevenlabs_tts = ElevenLabsTTS(api_key=args.elevenlabs_api_key) # No session needed
        logger.info("API clients initialized successfully.")

        # --- Find and Process Video Files Synchronously --- 
        video_files_to_process = []
        logger.info(f"Scanning for video files in: {input_path}")
        for item in input_path.iterdir():
            # Check if it's a file and has a supported video extension
            if item.is_file() and item.suffix.lower() in VIDEO_EXTENSIONS:
                 video_files_to_process.append(item)
            elif item.is_file():
                 logger.debug(f"Skipping non-video file: {item.name}")
            elif item.is_dir():
                 logger.debug(f"Skipping directory: {item.name}")
        
        video_files_to_process.sort() # Process in predictable order
        total_files = len(video_files_to_process)
        if total_files == 0:
             logger.warning(f"No video files found in {input_path}. Nothing to process.")
             sys.exit(0)
        logger.info(f"Found {total_files} video files to process.")
        
        logger.info(f"Starting sequential processing...")
        successful_files, failed_files = 0, 0

        # Loop through found video files and process them
        for i, video_file in enumerate(video_files_to_process):
            logger.info(f"=== Processing file {i+1}/{total_files}: {video_file.name} ===")
            try:
                # Synchronous call to process the video file
                process_video_file( 
                    video_file_path=video_file,
                    output_root_dir=output_path,
                    target_fps=args.target_fps,
                    target_langs=valid_target_langs,
                    gemini_stt_client=gemini_stt,
                    gemini_translator_client=gemini_translator,
                    elevenlabs_tts_client=elevenlabs_tts,
                    elevenlabs_voice_type=args.elevenlabs_voice_type,
                    elevenlabs_custom_voice_id=args.elevenlabs_custom_voice_id,
                    overwrite_video=args.overwrite_video,
                    overwrite_audio=args.overwrite_audio
                )
                # Assuming success if no exception is raised, adjust if needed
                successful_files += 1 
            except Exception as e:
                 # Catch errors during the processing of a single file
                 logger.error(f"--- Error processing file {video_file.name}: {e} ---", exc_info=True)
                 failed_files += 1
        
        logger.info("=== All processing finished. ===")
        logger.info(f"Summary: Successful files: {successful_files}, Failed files: {failed_files}")

    except ValueError as e: logger.critical(f"API Init Error: {e}"); sys.exit(1)
    except Exception as e: logger.critical(f"Unexpected main error: {e}", exc_info=True); sys.exit(1)


if __name__ == "__main__":
    # (Keep imports check)
    import importlib
    required_modules = ['google.generativeai', 'requests', 'pydub'] # Check requests now
    missing_modules = []
    for module_name in required_modules:
        try: importlib.import_module(module_name)
        except ImportError: missing_modules.append(module_name)
    if missing_modules:
        print("ERROR: Missing packages", file=sys.stderr)
        print(f"  pip install {' '.join(missing_modules)}", file=sys.stderr)
        sys.exit(1)

    # Removed asyncio event loop policy

    # Synchronous call
    main() 