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

from app.utils.logging import get_logger

logger = get_logger("casalingua.core.tts")

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
    
    def __init__(self, model_manager, config: Dict[str, Any] = None, registry_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the TTS pipeline.
        
        Args:
            model_manager: Model manager for accessing TTS models
            config: Configuration dictionary
            registry_config: Registry configuration for models
        """
        self.model_manager = model_manager
        self.config = config or {}
        self.registry_config = registry_config or {}
        self.initialized = False
        
        # Model type for TTS
        self.model_type = "tts"
        
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
        try:
            # Load the TTS model through the model manager
            model_info = await self.model_manager.load_model(self.model_type)
            if model_info:
                logger.info(f"TTS model loaded successfully")
            else:
                logger.warning("TTS model loading failed, will use fallbacks")
        except Exception as e:
            logger.warning(f"Error loading TTS model: {str(e)}")
            logger.warning("TTS functionality will be limited to fallbacks")
            
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
                        output_path: Optional[str] = None) -> Dict[str, Any]:
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
            Dict with synthesis results:
            - audio_file: Path to the generated audio file
            - audio_content: Binary audio data (if requested)
            - format: Audio format
            - duration: Audio duration in seconds
            - model_used: Name of model used
        """
        if not self.initialized:
            await self.initialize()
        
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
                    
                    # Get audio info
                    audio_info = self._get_audio_info(output_file)
                    
                    return {
                        "audio_file": str(output_file),
                        "format": output_format,
                        "duration": audio_info.get("duration", 0),
                        "model_used": "cache",
                        "cache_hit": True
                    }
                
                # Get audio info
                audio_info = self._get_audio_info(cached_path)
                
                # Read audio content
                with open(cached_path, "rb") as f:
                    audio_content = f.read()
                
                return {
                    "audio_file": str(cached_path),
                    "audio_content": audio_content,
                    "format": output_format,
                    "duration": audio_info.get("duration", 0),
                    "model_used": "cache",
                    "cache_hit": True
                }
        
        # No cache hit, generate new audio
        try:
            logger.debug(f"Synthesizing speech for text: '{text[:50]}...'")
            
            # Prepare input data for model
            input_data = {
                "text": text,
                "source_language": language,
                "parameters": {
                    "voice": voice,
                    "speed": speed,
                    "pitch": pitch,
                    "format": output_format
                }
            }
            
            # Run TTS model through model manager
            start_time = time.time()
            result = await self.model_manager.run_model(
                self.model_type,
                "process",
                input_data
            )
            processing_time = time.time() - start_time
            
            # Extract result
            if isinstance(result, dict) and "result" in result:
                # The result could be audio content directly or a path
                if isinstance(result["result"], bytes):
                    audio_content = result["result"]
                elif isinstance(result["result"], str) and os.path.exists(result["result"]):
                    # Read file from provided path
                    with open(result["result"], "rb") as f:
                        audio_content = f.read()
                else:
                    raise ValueError("TTS model returned invalid result format")
            else:
                raise ValueError("TTS model did not return expected result format")
            
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
            audio_info = self._get_audio_info(output_path)
            
            # Extract metadata from result
            metadata = result.get("metadata", {})
            
            # Prepare result
            synthesis_result = {
                "audio_file": output_path,
                "audio_content": audio_content,
                "format": output_format,
                "language": language,
                "voice": voice,
                "duration": audio_info.get("duration", metadata.get("duration", processing_time)),
                "model_used": metadata.get("model_used", self.model_type),
                "processing_time": processing_time
            }
            
            logger.debug(f"Speech synthesis completed in {processing_time:.2f}s")
            
            # Cache the result if enabled
            if self.cache_enabled and cache_key:
                self._cache_audio(cache_key, output_path, audio_content)
            
            return synthesis_result
            
        except Exception as e:
            logger.error(f"Error synthesizing speech: {str(e)}", exc_info=True)
            
            # Try fallback approach
            try:
                fallback_result = await self._fallback_synthesis(
                    text, language, voice, speed, pitch, output_format, output_path
                )
                
                if fallback_result:
                    # Mark as fallback
                    fallback_result["fallback"] = True
                    return fallback_result
                    
            except Exception as fallback_e:
                logger.error(f"Fallback synthesis failed: {str(fallback_e)}", exc_info=True)
            
            # Return error if all approaches failed
            return {
                "status": "error",
                "error": f"Speech synthesis failed: {str(e)}",
                "text": text,
                "language": language
            }
    
    async def get_available_voices(self, language: Optional[str] = None) -> Dict[str, Any]:
        """
        Get available voices for speech synthesis.
        
        Args:
            language: Optional language filter
            
        Returns:
            Dict with available voices
        """
        if not self.initialized:
            await self.initialize()
        
        try:
            # Prepare input data
            input_data = {
                "parameters": {
                    "language": language
                }
            }
            
            # Try to get voices from model
            result = await self.model_manager.run_model(
                self.model_type,
                "get_voices",
                input_data
            )
            
            # Extract voices from result
            if isinstance(result, dict) and "result" in result:
                voices = result["result"]
                
                # Handle different result formats
                if isinstance(voices, list):
                    return {
                        "status": "success",
                        "voices": voices,
                        "default_voice": self.default_voices.get(language or "en")
                    }
                elif isinstance(voices, dict) and "voices" in voices:
                    return {
                        "status": "success",
                        "voices": voices["voices"],
                        "default_voice": voices.get("default_voice", self.default_voices.get(language or "en"))
                    }
            
            # If no voices or unexpected format, fallback to defaults
            logger.warning("Model did not return expected voice format, using defaults")
            
        except Exception as e:
            logger.error(f"Error getting available voices: {str(e)}", exc_info=True)
        
        # Fallback - return default voices
        voices = []
        for lang, voice_id in self.default_voices.items():
            if language is None or lang == language:
                voices.append({
                    "id": voice_id,
                    "language": lang,
                    "name": f"{lang.upper()} Voice {voice_id.split('-')[-1]}",
                    "gender": "female" if int(voice_id.split('-')[-1]) % 2 == 0 else "male"
                })
        
        return {
            "status": "success",
            "voices": voices,
            "default_voice": self.default_voices.get(language or "en")
        }
    
    async def _fallback_synthesis(self,
                                  text: str,
                                  language: str,
                                  voice: str,
                                  speed: float,
                                  pitch: float,
                                  output_format: str,
                                  output_path: Optional[str]) -> Optional[Dict[str, Any]]:
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
            Dict with synthesis results or None on failure
        """
        logger.info("Using fallback TTS implementation")

        try:
            # Try to use alternative TTS model
            fallback_model_type = "tts_fallback"
            
            # Prepare input for fallback model
            input_data = {
                "text": text,
                "source_language": language,
                "parameters": {
                    "voice": voice,
                    "speed": speed,
                    "pitch": pitch,
                    "format": output_format
                }
            }
            
            # Try to run fallback model
            result = await self.model_manager.run_model(
                fallback_model_type,
                "process",
                input_data
            )
            
            # Extract result
            if isinstance(result, dict) and "result" in result:
                audio_content = result["result"]
                
                # Create output file
                if not output_path:
                    audio_id = str(uuid.uuid4())
                    output_path = str(self.temp_dir / f"tts_fallback_{audio_id}.{output_format}")
                
                # Ensure directory exists
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                
                # Write audio to file
                with open(output_path, "wb") as f:
                    f.write(audio_content)
                
                # Get audio info
                audio_info = self._get_audio_info(output_path)
                
                return {
                    "audio_file": output_path,
                    "audio_content": audio_content,
                    "format": output_format,
                    "language": language,
                    "voice": voice,
                    "duration": audio_info.get("duration", 0),
                    "model_used": "fallback"
                }
            
        except Exception as e:
            logger.error(f"Error in fallback synthesis: {str(e)}", exc_info=True)
            
            # Try gTTS as a last resort
            try:
                # Import gTTS only if needed
                from gtts import gTTS
                
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
                    try:
                        from pydub import AudioSegment
                        
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
                    except ImportError:
                        logger.warning("pydub not available for format conversion in fallback TTS")
                
                # Read audio content
                with open(output_path, "rb") as f:
                    audio_content = f.read()
                
                # Get audio info
                audio_info = self._get_audio_info(output_path)
                
                return {
                    "audio_file": output_path,
                    "audio_content": audio_content,
                    "format": output_format,
                    "language": language,
                    "voice": voice,
                    "duration": audio_info.get("duration", 0),
                    "model_used": "gtts"
                }
                
            except Exception as gtts_e:
                logger.error(f"Error in gTTS fallback: {str(gtts_e)}", exc_info=True)

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
    
    async def cleanup(self) -> None:
        """
        Clean up TTS resources.
        
        This should be called before application shutdown.
        """
        logger.info("Cleaning up TTS resources")
        
        # Final cache cleanup
        if self.cache_enabled:
            await self._cleanup_cache()