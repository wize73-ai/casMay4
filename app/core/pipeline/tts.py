# app/core/pipeline/tts.py
"""
Text-to-Speech Module for CasaLingua

This module provides speech synthesis capabilities, converting processed text
into natural-sounding audio in multiple languages and with various voices.

Author: Exygy Development Team
Version: 1.0.0
License: MIT
"""

import os
import io
import uuid
import logging
import asyncio
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple, BinaryIO

# Fallback imports for gTTS and pydub
try:
    from gtts import gTTS
except ImportError:
    gTTS = None

try:
    from pydub import AudioSegment
except ImportError:
    AudioSegment = None

from app.services.models.manager import ModelManager
from app.utils.logging import get_logger
from app.core.pipeline.tokenizer import TokenizerPipeline
from app.services.models.loader import ModelRegistry

logger = get_logger("core.tts")

class TTSPipeline:
    """
    Text-to-Speech Pipeline for converting text to audio.
    
    Features:
    - Support for multiple languages and voices
    - Adjustable speech parameters (speed, pitch)
    - Multiple output formats
    - Caching for efficiency
    - Fallback mechanisms for reliability
    """
    
    def __init__(self, model_manager: ModelManager, config: Dict[str, Any] = None):
        """
        Initialize the TTS pipeline.
        
        Args:
            model_manager: Model manager for accessing TTS models
            config: Configuration dictionary
        """
        self.model_manager = model_manager
        self.config = config or {}
        self.initialized = False
        # Dynamically load tokenizer using ModelRegistry
        registry = ModelRegistry()
        model_name, tokenizer_name = registry.get_model_and_tokenizer("tts_preprocessor")
        self.tokenizer = TokenizerPipeline(model_name=tokenizer_name, task_type="tts_preprocessing")
        
        # Output settings
        self.default_format = "mp3"
        self.supported_formats = ["mp3", "wav", "ogg"]
        
        # Voice settings
        self.default_voices = {
            "en": "en-us-1",  # English
            "es": "es-es-1",  # Spanish
            "fr": "fr-fr-1",  # French
            "de": "de-de-1",  # German
            "it": "it-it-1",  # Italian
            "pt": "pt-br-1",  # Portuguese
            "nl": "nl-nl-1",  # Dutch
            "ru": "ru-ru-1",  # Russian
            "zh": "zh-cn-1",  # Chinese
            "ja": "ja-jp-1",  # Japanese
            "ko": "ko-kr-1",  # Korean
            "ar": "ar-sa-1",  # Arabic
            "hi": "hi-in-1",  # Hindi
        }
        
        # Cached models
        self.tts_models = {}
        
        # Audio cache
        self.cache_enabled = self.config.get("tts_cache_enabled", True)
        self.cache_dir = Path(self.config.get("tts_cache_dir", "cache/tts"))
        self.cache_size_limit = self.config.get("tts_cache_size_mb", 500) * 1024 * 1024  # Convert MB to bytes
        
        # Ensure cache directory exists
        if self.cache_enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Temp directory for output
        self.temp_dir = Path(self.config.get("temp_dir", "temp"))
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("TTS pipeline created (not yet initialized)")
    
    async def initialize(self) -> None:
        """
        Initialize the TTS pipeline.
        
        This loads necessary models and prepares the pipeline.
        """
        if self.initialized:
            logger.warning("TTS pipeline already initialized")
            return
        
        logger.info("Initializing TTS pipeline")
        
        # Verify TTS models are available
        primary_model = await self._get_tts_model()
        if not primary_model:
            logger.warning("No TTS models available, functionality will be limited")
        else:
            logger.info(f"Primary TTS model loaded: {primary_model.get('name', 'unknown')}")
            
        # Clean up cache if enabled
        if self.cache_enabled:
            await self._cleanup_cache()
            
            # Start background task for periodic cleanup
            asyncio.create_task(self._periodic_cache_cleanup())
        
        self.initialized = True
        logger.info("TTS pipeline initialization complete")
    
    async def synthesize(self, 
                        text: str, 
                        language: str = "en",
                        voice: Optional[str] = None,
                        speed: float = 1.0,
                        pitch: float = 1.0,
                        output_format: str = None,
                        output_path: Optional[str] = None) -> Tuple[Optional[str], Dict[str, Any]]:
        """
        Synthesize speech from text.
        
        Args:
            text: Text to convert to speech
            language: Language code (e.g., 'en', 'es')
            voice: Voice identifier (if None, uses default for language)
            speed: Speech rate multiplier (0.5-2.0)
            pitch: Voice pitch adjustment (0.5-2.0)
            output_format: Output audio format (mp3, wav, ogg)
            output_path: Path to save audio file (if None, returns audio content)
            
        Returns:
            Tuple of (file_path, format_info) where:
            - file_path: Path to the generated audio file or None if output_path not provided
            - format_info: Dictionary with format information
        """
        if not self.initialized:
            raise RuntimeError("TTS pipeline not initialized")
        
        # Validate and normalize parameters
        language = language.lower()
        if voice is None:
            voice = self.default_voices.get(language, self.default_voices.get("en"))
        
        speed = max(0.5, min(2.0, speed))  # Limit to reasonable range
        pitch = max(0.5, min(2.0, pitch))  # Limit to reasonable range
        
        # Set output format
        output_format = output_format or self.default_format
        if output_format not in self.supported_formats:
            logger.warning(f"Unsupported format: {output_format}, using {self.default_format}")
            output_format = self.default_format
        
        # Generate cache key if caching is enabled
        cache_key = None
        if self.cache_enabled:
            cache_key = self._generate_cache_key(text, language, voice, speed, pitch, output_format)
            
            # Check cache
            cached_path = self._get_cached_audio(cache_key, output_format)
            if cached_path:
                logger.debug(f"Using cached audio for: '{text[:30]}...' → {cached_path}")
                
                # If output path is specified, copy the cached file
                if output_path:
                    import shutil
                    output_file = Path(output_path)
                    output_file.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(cached_path, output_file)
                    return str(output_file), self._get_audio_info(output_file)
                
                # Otherwise, return the cached path
                return str(cached_path), self._get_audio_info(cached_path)
        
        # No cache hit, generate new audio
        try:
            logger.debug(f"Synthesizing speech for text: '{text[:50]}...'")

            # Optional: tokenize input if tokenizer is available
            token_ids = None
            if self.tokenizer:
                token_ids = self.tokenizer.encode(text)
                logger.debug(f"Tokenized input: {token_ids}")

            # Get TTS model
            tts_model = await self._get_tts_model(language)

            if not tts_model:
                raise ValueError(f"No TTS model available for language: {language}")

            # Prepare input data
            input_data = {
                "text": text,
                "language": language,
                "voice": voice,
                "speed": speed,
                "pitch": pitch,
                "format": output_format
            }
            if token_ids is not None:
                input_data["tokens"] = token_ids

            # Run TTS model
            start_time = time.time()
            result = await self.model_manager.run_model(
                tts_model,
                "synthesize",
                input_data
            )
            processing_time = time.time() - start_time

            # Get audio content
            audio_content = result.get("audio_content")
            if not audio_content:
                raise ValueError("TTS model did not return audio content")

            # Create output file path if not specified
            if not output_path:
                audio_id = str(uuid.uuid4())
                output_path = str(self.temp_dir / f"tts_{audio_id}.{output_format}")

            # Ensure output directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            # Write audio to file
            with open(output_path, "wb") as f:
                f.write(audio_content)

            # Get audio information
            audio_info = {
                "format": output_format,
                "language": language,
                "voice": voice,
                "duration": result.get("duration", processing_time),
                "sample_rate": result.get("sample_rate", 22050),
                "channels": result.get("channels", 1),
                "processing_time": processing_time
            }

            logger.debug(f"Speech synthesis completed in {processing_time:.2f}s")

            # Cache the result if enabled
            if self.cache_enabled and cache_key:
                self._cache_audio(cache_key, output_path, audio_content)

            return output_path, audio_info

        except Exception as e:
            logger.error("Error synthesizing speech", exc_info=True)

            # Try fallback approach
            try:
                result = await self._fallback_synthesis(
                    text, language, voice, speed, pitch, output_format, output_path
                )
                if result:
                    return result
            except Exception as fallback_e:
                logger.error("Fallback synthesis failed", exc_info=True)

            raise ValueError(f"Speech synthesis failed: {str(e)}")
    
    async def get_available_voices(self, language: Optional[str] = None) -> Dict[str, Any]:
        """
        Get available voices for speech synthesis.
        
        Args:
            language: Optional language filter
            
        Returns:
            Dict with available voices
        """
        # Get primary TTS model
        tts_model = await self._get_tts_model()
        
        if not tts_model:
            return {"status": "error", "message": "No TTS models available"}
        
        try:
            # Get voice information from model
            result = await self.model_manager.run_model(
                tts_model,
                "get_voices",
                {"language": language}
            )
            
            voices = result.get("voices", [])
            
            # If no voices returned, use default mappings
            if not voices:
                voices = [
                    {
                        "id": voice_id,
                        "language": lang,
                        "name": f"Voice {i+1}",
                        "gender": "female" if i % 2 == 0 else "male"
                    }
                    for i, (lang, voice_id) in enumerate(self.default_voices.items())
                    if language is None or lang == language
                ]
            
            return {
                "status": "success",
                "voices": voices,
                "default_voice": self.default_voices.get(language or "en")
            }
            
        except Exception as e:
            logger.error(f"Error getting available voices: {str(e)}", exc_info=True)
            
            # Return default voices
            return {
                "status": "warning",
                "message": f"Error getting voices: {str(e)}",
                "voices": [
                    {
                        "id": voice_id,
                        "language": lang,
                        "name": f"Voice {i+1}",
                        "gender": "female" if i % 2 == 0 else "male"
                    }
                    for i, (lang, voice_id) in enumerate(self.default_voices.items())
                    if language is None or lang == language
                ]
            }
    
    async def _get_tts_model(self, language: str = None) -> Optional[Dict[str, Any]]:
        """
        Get TTS model for a specific language.
        
        Args:
            language: Language code
            
        Returns:
            TTS model or None if not available
        """
        # Check if we have a language-specific model cached
        if language and f"tts_{language}" in self.tts_models:
            return self.tts_models[f"tts_{language}"]
        
        # Check if we have a general model cached
        if "tts" in self.tts_models:
            return self.tts_models["tts"]
        
        # Try to get language-specific model
        if language:
            model = await self.model_manager.get_model(f"tts_{language}")
            if model:
                self.tts_models[f"tts_{language}"] = model
                return model
        
        # Fall back to general TTS model
        model = await self.model_manager.get_model("tts")
        if model:
            self.tts_models["tts"] = model
            return model
        
        # No model available
        logger.warning(f"No TTS model available for language: {language or 'any'}")
        return None
    
    def _generate_cache_key(self,
                          text: str,
                          language: str,
                          voice: str,
                          speed: float,
                          pitch: float,
                          output_format: str) -> str:
        """
        Generate a cache key for TTS output.
        
        Args:
            text: Text to synthesize
            language: Language code
            voice: Voice identifier
            speed: Speech rate
            pitch: Voice pitch
            output_format: Output format
            
        Returns:
            Cache key string
        """
        import hashlib
        
        # Create a string with all parameters
        params = f"{text}|{language}|{voice}|{speed:.1f}|{pitch:.1f}|{output_format}"
        
        # Hash the parameters
        return hashlib.md5(params.encode()).hexdigest()
    
    def _get_cached_audio(self, cache_key: str, output_format: str) -> Optional[str]:
        """
        Get cached audio file if available.
        
        Args:
            cache_key: Cache key
            output_format: Output format
            
        Returns:
            Path to cached file or None
        """
        if not self.cache_enabled:
            return None
        
        # Check if cache file exists
        cache_file = self.cache_dir / f"{cache_key}.{output_format}"
        
        if cache_file.exists():
            # Update access time
            os.utime(cache_file, None)
            return str(cache_file)
        
        return None
    
    def _cache_audio(self, cache_key: str, output_path: str, audio_content: bytes) -> None:
        """
        Cache audio file for future use.
        
        Args:
            cache_key: Cache key
            output_path: Path to audio file
            audio_content: Audio content as bytes
        """
        if not self.cache_enabled:
            return
        
        try:
            # Get output format from path
            output_format = os.path.splitext(output_path)[1].lstrip('.')
            
            # Create cache file
            cache_file = self.cache_dir / f"{cache_key}.{output_format}"
            
            # Write to cache
            with open(cache_file, "wb") as f:
                f.write(audio_content)
                
            logger.debug(f"Cached audio file: {cache_file}")
            
        except Exception as e:
            logger.error(f"Error caching audio: {str(e)}", exc_info=True)
    
    def _get_audio_info(self, file_path: str) -> Dict[str, Any]:
        """
        Get audio file information.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Dict with audio information
        """
        # Get format from file extension
        file_format = os.path.splitext(file_path)[1].lstrip('.')
        
        # Basic info
        info = {
            "format": file_format,
            "filesize": os.path.getsize(file_path)
        }
        
        # Try to get more detailed information
        try:
            # Use audioread if available
            import audioread
            with audioread.audio_open(file_path) as f:
                info["duration"] = f.duration
                info["sample_rate"] = f.samplerate
                info["channels"] = f.channels
        except (ImportError, Exception):
            # Fallback to basic estimation
            info["duration"] = info["filesize"] / (32000 * 2 / 8)  # Rough estimate
        
        return info
    
    async def _cleanup_cache(self) -> None:
        """Clean up TTS cache to stay within size limits."""
        if not self.cache_enabled:
            return
        
        try:
            # Get all cache files
            cache_files = list(self.cache_dir.glob("*.*"))
            
            # Calculate total cache size
            total_size = sum(f.stat().st_size for f in cache_files)
            
            # Check if we need to clean up
            if total_size <= self.cache_size_limit:
                return
            
            logger.info(f"TTS cache size ({total_size / 1024 / 1024:.1f} MB) exceeds limit "
                       f"({self.cache_size_limit / 1024 / 1024:.1f} MB), cleaning up")
            
            # Sort by access time (oldest first)
            cache_files.sort(key=lambda f: f.stat().st_atime)
            
            # Remove files until we're under the limit
            for file in cache_files:
                if total_size <= self.cache_size_limit * 0.8:  # Clean up to 80% of limit
                    break
                
                file_size = file.stat().st_size
                file.unlink()
                total_size -= file_size
                logger.debug(f"Removed cache file: {file.name}")
            
            logger.info(f"TTS cache cleanup complete, new size: {total_size / 1024 / 1024:.1f} MB")
            
        except Exception as e:
            logger.error(f"Error cleaning up TTS cache: {str(e)}", exc_info=True)
    
    async def _periodic_cache_cleanup(self) -> None:
        """Periodically clean up TTS cache."""
        if not self.cache_enabled:
            return
        
        # Run cleanup every hour
        while True:
            try:
                await asyncio.sleep(3600)  # 1 hour
                await self._cleanup_cache()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic cache cleanup: {str(e)}", exc_info=True)
                await asyncio.sleep(300)  # Wait 5 minutes before retrying
    
    async def _fallback_synthesis(self,
                                  text: str,
                                  language: str,
                                  voice: str,
                                  speed: float,
                                  pitch: float,
                                  output_format: str,
                                  output_path: Optional[str]) -> Optional[Tuple[str, Dict[str, Any]]]:
        """
        Fallback TTS implementation when primary model fails.

        Args:
            text: Text to synthesize
            language: Language code
            voice: Voice ID
            speed: Speech rate
            pitch: Voice pitch
            output_format: Output format
            output_path: Output file path

        Returns:
            Tuple of (file_path, format_info) or None on failure
        """
        logger.info("Using fallback TTS implementation")

        try:
            # Try to use alternative TTS model
            fallback_model = await self.model_manager.get_model("tts_fallback")

            if fallback_model:
                # Use alternative model
                input_data = {
                    "text": text,
                    "language": language,
                    "voice": voice,
                    "speed": speed,
                    "pitch": pitch,
                    "format": output_format
                }

                result = await self.model_manager.run_model(
                    fallback_model,
                    "synthesize",
                    input_data
                )

                audio_content = result.get("audio_content")

                if audio_content:
                    # Create output file
                    if not output_path:
                        audio_id = str(uuid.uuid4())
                        output_path = str(self.temp_dir / f"tts_fallback_{audio_id}.{output_format}")

                    # Ensure directory exists
                    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

                    # Write audio to file
                    with open(output_path, "wb") as f:
                        f.write(audio_content)

                    audio_info = {
                        "format": output_format,
                        "language": language,
                        "voice": voice,
                        "fallback": True
                    }

                    return output_path, audio_info

            # No fallback model available, try gTTS if installed
            if gTTS is not None:
                # Create output file
                if not output_path:
                    audio_id = str(uuid.uuid4())
                    output_path = str(self.temp_dir / f"tts_gtts_{audio_id}.mp3")

                # Ensure directory exists
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)

                # Map language code to gTTS format if needed
                gtts_lang = language
                if len(language) > 2 and '-' in language:
                    gtts_lang = language.split('-')[0]

                # Create gTTS object
                tts = gTTS(text=text, lang=gtts_lang, slow=(speed < 0.8))

                # Save to file
                tts.save(output_path)

                # Convert to requested format if needed
                if output_format != "mp3":
                    if AudioSegment is not None:
                        # Load MP3
                        audio = AudioSegment.from_file(output_path, format="mp3")
                        # Apply pitch adjustment
                        if pitch != 1.0:
                            octaves = pitch - 1.0
                            new_sample_rate = int(audio.frame_rate * (2.0 ** octaves))
                            audio = audio._spawn(audio.raw_data, overrides={
                                "frame_rate": new_sample_rate
                            })
                        # Save in requested format
                        converted_path = output_path.replace(".mp3", f".{output_format}")
                        audio.export(converted_path, format=output_format)
                        # Use converted file
                        output_path = converted_path
                    else:
                        logger.warning("pydub not available for format conversion in fallback TTS")

                audio_info = {
                    "format": "mp3" if output_format not in ["wav", "ogg"] else output_format,
                    "language": language,
                    "fallback": True,
                    "engine": "gtts"
                }

                return output_path, audio_info
            else:
                logger.warning("gTTS not available for fallback synthesis")

        except Exception as e:
            logger.error("Error in fallback synthesis", exc_info=True)

        return None
    
    async def cleanup(self) -> None:
        """
        Clean up TTS resources.
        
        This should be called before application shutdown.
        """
        logger.info("Cleaning up TTS resources")
        
        # Clear model cache
        self.tts_models.clear()
        
        # Final cache cleanup
        if self.cache_enabled:
            await self._cleanup_cache()