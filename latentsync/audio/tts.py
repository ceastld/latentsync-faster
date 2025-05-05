#!/usr/bin/env python
import os
import requests
import tempfile
from pathlib import Path
from typing import Dict, Optional, Any, Union, List
import logging
import json
import time
from pydub import AudioSegment
import io
from dotenv import load_dotenv
import numpy as np

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger("ElevenLabsTTS")

# Default voice settings
DEFAULT_VOICE_SETTINGS = {
    "stability": 0.5,
    "similarity_boost": 0.75
}

# Default voice ID (Adam - male voice)
DEFAULT_VOICE_ID = "pNInz6obpgDQGcFmaJgB"

class ElevenLabsRealTimeTTS:
    """
    Real-time Text-to-Speech implementation using ElevenLabs API
    """
    BASE_URL = "https://api.elevenlabs.io/v1"
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        voice_id: str = DEFAULT_VOICE_ID,
        model_id: str = "eleven_multilingual_v2",
        stability: float = 0.5,
        similarity_boost: float = 0.75,
        style: float = 0.0,
        use_speaker_boost: bool = True,
        output_format: str = "mp3_44100_128"
    ):
        """
        Initialize the ElevenLabs TTS client
        
        Args:
            api_key: ElevenLabs API key (if None, will be loaded from .env)
            voice_id: Voice ID to use for synthesis
            model_id: Model ID to use for synthesis
            stability: Voice stability (0.0-1.0)
            similarity_boost: Voice similarity boost (0.0-1.0)
            style: Speaking style (0.0-1.0)
            use_speaker_boost: Whether to use speaker boost
            output_format: Output audio format
        """
        # Load API key from .env if not provided
        if api_key is None:
            load_dotenv()
            api_key = os.getenv("ELEVEN_API_KEY")
            
        if not api_key:
            raise ValueError("ElevenLabs API key is required. Set it in .env file or pass directly.")
            
        self._api_key = api_key
        self.voice_id = voice_id
        self.model_id = model_id
        self.output_format = output_format
        
        # Configure voice settings
        self.voice_settings = {
            "stability": stability,
            "similarity_boost": similarity_boost,
            "style": style,
            "use_speaker_boost": use_speaker_boost
        }
        
        # Initialize session for better performance in streaming scenarios
        self._session = requests.Session()
        
        # Validate the API key and voice ID
        self._validate_setup()
        
    def _validate_setup(self) -> None:
        """Validate API key and get available voices"""
        try:
            # Check available voices
            voices_url = f"{self.BASE_URL}/voices"
            headers = {"xi-api-key": self._api_key}
            
            response = self._session.get(voices_url, headers=headers)
            if response.status_code != 200:
                raise ValueError(f"Failed to validate ElevenLabs API key. Status: {response.status_code}")
                
            voices_data = response.json()
            
            # Check if selected voice_id exists
            voice_ids = [voice["voice_id"] for voice in voices_data.get("voices", [])]
            if self.voice_id not in voice_ids:
                logger.warning(f"Voice ID {self.voice_id} not found. Available voices: {voice_ids}")
                if voice_ids:
                    self.voice_id = voice_ids[0]
                    logger.info(f"Using first available voice: {self.voice_id}")
                    
            logger.info(f"ElevenLabs API validated successfully. Using voice: {self.voice_id}")
        except Exception as e:
            logger.error(f"Failed to validate ElevenLabs setup: {e}")
            raise
            
    def get_available_voices(self) -> List[Dict[str, Any]]:
        """Get list of available voices from ElevenLabs"""
        try:
            voices_url = f"{self.BASE_URL}/voices"
            headers = {"xi-api-key": self._api_key}
            
            response = self._session.get(voices_url, headers=headers)
            if response.status_code != 200:
                logger.error(f"Failed to get voices. Status: {response.status_code}, Response: {response.text}")
                return []
                
            voices_data = response.json()
            return voices_data.get("voices", [])
        except Exception as e:
            logger.error(f"Error getting available voices: {e}")
            return []
            
    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available models from ElevenLabs"""
        try:
            models_url = f"{self.BASE_URL}/models"
            headers = {"xi-api-key": self._api_key}
            
            response = self._session.get(models_url, headers=headers)
            if response.status_code != 200:
                logger.error(f"Failed to get models. Status: {response.status_code}, Response: {response.text}")
                return []
                
            models_data = response.json()
            return models_data
        except Exception as e:
            logger.error(f"Error getting available models: {e}")
            return []
            
    def text_to_speech(
        self, 
        text: str,
        voice_id: Optional[str] = None,
        model_id: Optional[str] = None,
        voice_settings: Optional[Dict[str, Any]] = None,
        output_path: Optional[Union[str, Path]] = None,
        output_format: Optional[str] = None,
        stream: bool = True
    ) -> Optional[Union[bytes, str]]:
        """
        Convert text to speech using ElevenLabs API
        
        Args:
            text: Text to convert to speech
            voice_id: Voice ID to use (overrides the instance voice_id)
            model_id: Model ID to use (overrides the instance model_id)
            voice_settings: Voice settings to use (overrides the instance voice_settings)
            output_path: Path to save the audio file (if None, audio is returned as bytes)
            output_format: Output format (overrides the instance output_format)
            stream: Whether to stream the response
            
        Returns:
            Audio bytes if output_path is None, otherwise the path to the saved file
        """
        if not text:
            logger.warning("TTS called with empty text.")
            return None
            
        # Use instance defaults if parameters are not provided
        voice_id = voice_id or self.voice_id
        model_id = model_id or self.model_id
        voice_settings = voice_settings or self.voice_settings
        output_format = output_format or self.output_format
        
        tts_url = f"{self.BASE_URL}/text-to-speech/{voice_id}"
        if stream:
            tts_url += f"/stream?output_format={output_format}"
            
        headers = {
            "Content-Type": "application/json",
            "xi-api-key": self._api_key
        }
        
        payload = {
            "text": text,
            "model_id": model_id,
            "voice_settings": voice_settings
        }
        
        logger.info(f"Starting TTS request. Voice: {voice_id}, Model: {model_id}")
        
        try:
            response = self._session.post(tts_url, headers=headers, json=payload, stream=stream)
            
            if response.status_code != 200:
                logger.error(f"TTS request failed. Status: {response.status_code}, Response: {response.text}")
                return None
                
            logger.info(f"TTS request successful (Status {response.status_code})")
            
            # If no output path is provided, return the audio bytes
            if not output_path:
                if stream:
                    # Collect streaming response into bytes
                    audio_bytes = b""
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            audio_bytes += chunk
                    return audio_bytes
                else:
                    # Return the full response content
                    return response.content
            
            # Save the audio to the specified output path
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            if stream:
                # Stream the response to a file
                with open(output_file, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            else:
                # Write the full response to a file
                with open(output_file, 'wb') as f:
                    f.write(response.content)
                    
            logger.info(f"Saved audio to {output_file}")
            return str(output_file)
            
        except Exception as e:
            logger.error(f"Error during TTS request: {e}")
            return None
            
    def text_to_speech_stream(
        self, 
        text: str,
        chunk_callback=None,
        voice_id: Optional[str] = None,
        model_id: Optional[str] = None,
        voice_settings: Optional[Dict[str, Any]] = None,
        output_format: Optional[str] = None
    ) -> bool:
        """
        Stream audio chunks as they're received from the API
        
        Args:
            text: Text to convert to speech
            chunk_callback: Callback function to process each audio chunk
            voice_id: Voice ID to use (overrides the instance voice_id)
            model_id: Model ID to use (overrides the instance model_id)
            voice_settings: Voice settings to use (overrides the instance voice_settings)
            output_format: Output format (overrides the instance output_format)
            
        Returns:
            True if successful, False otherwise
        """
        if not text:
            logger.warning("TTS stream called with empty text.")
            return False
            
        if not chunk_callback:
            logger.warning("No chunk callback provided for streaming TTS.")
            return False
            
        # Use instance defaults if parameters are not provided
        voice_id = voice_id or self.voice_id
        model_id = model_id or self.model_id
        voice_settings = voice_settings or self.voice_settings
        output_format = output_format or self.output_format
        
        tts_url = f"{self.BASE_URL}/text-to-speech/{voice_id}/stream?output_format={output_format}"
        headers = {
            "Content-Type": "application/json",
            "xi-api-key": self._api_key
        }
        
        payload = {
            "text": text,
            "model_id": model_id,
            "voice_settings": voice_settings
        }
        
        logger.info(f"Starting streaming TTS request. Voice: {voice_id}, Model: {model_id}")
        
        try:
            response = self._session.post(tts_url, headers=headers, json=payload, stream=True)
            
            if response.status_code != 200:
                logger.error(f"Streaming TTS request failed. Status: {response.status_code}, Response: {response.text}")
                return False
                
            logger.info(f"Streaming TTS request successful (Status {response.status_code})")
            
            # Process the stream in chunks
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    # Call the callback with each chunk
                    chunk_callback(chunk)
                    
            return True
            
        except Exception as e:
            logger.error(f"Error during streaming TTS request: {e}")
            return False
            
    def text_to_speech_to_numpy(
        self, 
        text: str,
        voice_id: Optional[str] = None,
        model_id: Optional[str] = None,
        voice_settings: Optional[Dict[str, Any]] = None,
        sample_rate: int = 44100
    ) -> Optional[np.ndarray]:
        """
        Convert text to speech and return as numpy array
        
        Args:
            text: Text to convert to speech
            voice_id: Voice ID to use (overrides the instance voice_id)
            model_id: Model ID to use (overrides the instance model_id)
            voice_settings: Voice settings to use (overrides the instance voice_settings)
            sample_rate: Sample rate for the returned audio
            
        Returns:
            Numpy array with audio data or None if failed
        """
        # Get audio as bytes
        audio_bytes = self.text_to_speech(
            text=text,
            voice_id=voice_id,
            model_id=model_id,
            voice_settings=voice_settings,
            output_path=None
        )
        
        if not audio_bytes:
            return None
            
        try:
            # Convert bytes to AudioSegment
            audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes), format="mp3")
            
            # Convert to numpy array
            samples = np.array(audio_segment.get_array_of_samples())
            
            # Handle stereo to mono conversion if needed
            if audio_segment.channels == 2:
                samples = samples.reshape((-1, 2)).mean(axis=1)
                
            # Convert to float32 and normalize
            samples = samples.astype(np.float32) / np.iinfo(samples.dtype).max
            
            return samples
        except Exception as e:
            logger.error(f"Error converting audio bytes to numpy array: {e}")
            return None


# Example usage
if __name__ == "__main__":
    # Initialize TTS with default parameters
    tts = ElevenLabsRealTimeTTS()
    
    # List available voices
    voices = tts.get_available_voices()
    print(f"Available voices: {len(voices)}")
    for voice in voices[:5]:  # Show first 5 voices
        print(f"- {voice['name']} (ID: {voice['voice_id']})")
    
    # Convert text to speech
    text = "Hello, this is a test of the ElevenLabs real-time TTS system."
    
    # Save to file
    output_path = "test_output.mp3"
    result = tts.text_to_speech(text, output_path=output_path)
    
    if result:
        print(f"Audio saved to {result}")
    else:
        print("Failed to generate speech")
        
    # Stream example
    def process_chunk(chunk):
        print(f"Received chunk of size: {len(chunk)} bytes")
    
    print("Streaming example:")
    tts.text_to_speech_stream(
        "This is a streaming test of the ElevenLabs TTS system.",
        chunk_callback=process_chunk
    )
    
    # Convert to numpy array
    print("Converting to numpy array:")
    numpy_audio = tts.text_to_speech_to_numpy("This is a test for numpy conversion.")
    if numpy_audio is not None:
        print(f"Audio array shape: {numpy_audio.shape}, dtype: {numpy_audio.dtype}")
