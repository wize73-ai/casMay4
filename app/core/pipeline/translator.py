"""
Translation Pipeline for CasaLingua

This module handles translation between different languages, with
support for context-aware and domain-specific translation.

Author: Exygy Development Team
Version: 1.0.0
License: MIT
"""

import logging
import time
from typing import Dict, Any, List, Optional, Tuple, Union

from app.api.schemas.language import LanguageDetectionRequest
from app.api.schemas.translation import TranslationRequest, TranslationResult
from app.utils.logging import get_logger

logger = get_logger("casalingua.core.translator")


class TranslationPipeline:
    """
    Translation pipeline for text content.
    
    Features:
    - Language detection
    - Context-aware translation
    - Domain-specific translation
    - Automatic model selection
    """
    
    def __init__(
        self,
        model_manager,
        config: Dict[str, Any] = None,
        registry_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the translation pipeline.
        
        Args:
            model_manager: Model manager for accessing translation models
            config: Configuration dictionary
            registry_config: Registry configuration for models
        """
        self.model_manager = model_manager
        self.config = config or {}
        self.registry_config = registry_config or {}
        self.initialized = False
        
        # Language code mapping (ISO 639-1 to full names)
        self.language_names = {
            "en": "English",
            "es": "Spanish",
            "fr": "French",
            "de": "German",
            "it": "Italian",
            "pt": "Portuguese",
            "nl": "Dutch",
            "ru": "Russian",
            "zh": "Chinese",
            "ja": "Japanese",
            "ko": "Korean",
            "ar": "Arabic",
            "hi": "Hindi",
            "bn": "Bengali",
            "vi": "Vietnamese",
            "th": "Thai"
        }
        
        # Default model types
        self.translation_model_type = "translation"
        self.language_detection_model_type = "language_detection"
        
        # Domain-specific vocabulary
        self.domain_vocabulary = {}
        
        logger.info("Translation pipeline created (not yet initialized)")
    
    async def initialize(self) -> None:
        """
        Initialize the translation pipeline.
        
        This loads language detection models and prepares
        translation capabilities.
        """
        if self.initialized:
            logger.warning("Translation pipeline already initialized")
            return
        
        logger.info("Initializing translation pipeline")
        
        # Load language detection model
        try:
            logger.info("Loading language detection model")
            await self.model_manager.load_model(self.language_detection_model_type)
            logger.info("Language detection model ready")
        except Exception as e:
            logger.warning(f"Error loading language detection model: {str(e)}")
            logger.warning("Will use fallback language detection")
        
        # Load domain vocabulary if available
        domain_vocab_path = self.config.get("domain_vocabulary_path")
        if domain_vocab_path:
            logger.info(f"Loading domain vocabulary from {domain_vocab_path}")
            self._load_domain_vocabulary(domain_vocab_path)
        
        self.initialized = True
        logger.info("Translation pipeline initialization complete")
    
    async def detect_language(self, request: LanguageDetectionRequest) -> Dict[str, Any]:
        """
        Detect the language of text.
        
        Args:
            request: Language detection request
            
        Returns:
            Dict with detected language and confidence
        """
        if not self.initialized:
            await self.initialize()
        
        text = request.text
        
        if not text:
            return {"language": "en", "confidence": 0.0}
        
        logger.debug(f"Detecting language for text of length {len(text)}")
        
        try:
            # Prepare input for language detection model
            input_data = {
                "text": text,
                "parameters": {
                    "detailed": request.detailed
                }
            }
            
            # Run language detection model
            result = await self.model_manager.run_model(
                self.language_detection_model_type,
                "process",
                input_data
            )
            
            # Extract language detection results
            if isinstance(result, dict) and "result" in result:
                detection_result = result["result"]
                
                # Handle possible result formats
                if isinstance(detection_result, dict):
                    language = detection_result.get("language", "en")
                    confidence = detection_result.get("confidence", 0.0)
                elif isinstance(detection_result, list) and detection_result:
                    # Take first result if it's a list
                    language = detection_result[0].get("language", "en")
                    confidence = detection_result[0].get("confidence", 0.0)
                else:
                    language = "en"
                    confidence = 0.0
                
                logger.debug(f"Language detected: {language} (confidence: {confidence:.2f})")
                
                if request.detailed:
                    return {
                        "language": language,
                        "confidence": confidence,
                        "name": self.language_names.get(language, language),
                        "alternatives": detection_result.get("alternatives", [])
                    }
                else:
                    return {
                        "language": language,
                        "confidence": confidence
                    }
            else:
                # Fallback to default
                logger.warning("Unexpected language detection result format")
                return {"language": "en", "confidence": 0.0}
            
        except Exception as e:
            logger.error(f"Error detecting language: {str(e)}", exc_info=True)
            return {"language": "en", "confidence": 0.0}
    
    async def translate(self, request: TranslationRequest) -> TranslationResult:
        """
        Translate text from source to target language.
        
        Args:
            request: Translation request with text, languages, and options
            
        Returns:
            TranslationResult with translated text and metadata
        """
        if not self.initialized:
            await self.initialize()
        
        text = request.text
        source_language = request.source_language
        target_language = request.target_language
        
        # Same language, no translation needed
        if source_language == target_language:
            return TranslationResult(
                translated_text=text,
                source_language=source_language,
                target_language=target_language,
                confidence=1.0
            )
        
        if not text:
            return TranslationResult(
                translated_text="",
                source_language=source_language,
                target_language=target_language,
                confidence=0.0
            )
        
        logger.debug(f"Translating from {source_language} to {target_language}")
        
        try:
            # Get model ID if specified
            model_id = request.model_id or self.translation_model_type
            
            # Prepare translation input
            input_data = {
                "text": text,
                "source_language": source_language,
                "target_language": target_language,
                "parameters": {
                    "preserve_formatting": request.preserve_formatting,
                    "formality": request.formality,
                    "domain": request.domain,
                    "glossary_id": request.glossary_id
                }
            }
            
            # Add context if provided
            if request.context:
                input_data["context"] = request.context
            
            # Run translation model
            start_time = time.time()
            result = await self.model_manager.run_model(
                model_id,
                "process",
                input_data
            )
            processing_time = time.time() - start_time
            
            # Extract translation results
            if isinstance(result, dict) and "result" in result:
                translated_text = result["result"]
                
                # Handle different result formats
                if isinstance(translated_text, list) and translated_text:
                    translated_text = translated_text[0]
                elif not isinstance(translated_text, str):
                    translated_text = str(translated_text)
                
                # Extract confidence if available
                confidence = 0.0
                if "metadata" in result and isinstance(result["metadata"], dict):
                    confidence = result["metadata"].get("confidence", 0.0)
                
                # Get model used
                model_used = model_id
                if "metadata" in result and isinstance(result["metadata"], dict):
                    model_used = result["metadata"].get("model_used", model_id)
                
                # Apply domain-specific vocabulary if available
                domain = request.domain
                if domain and domain in self.domain_vocabulary:
                    translated_text = self._apply_domain_vocabulary(
                        translated_text,
                        target_language,
                        domain
                    )
                
                logger.debug(f"Translation completed in {processing_time:.3f}s")
                
                return TranslationResult(
                    translated_text=translated_text,
                    source_language=source_language,
                    target_language=target_language,
                    confidence=confidence,
                    model_used=model_used
                )
            else:
                # Fallback for unexpected result format
                logger.warning("Unexpected translation result format")
                if isinstance(result, str):
                    translated_text = result
                else:
                    translated_text = text  # Fall back to original text
                
                return TranslationResult(
                    translated_text=translated_text,
                    source_language=source_language,
                    target_language=target_language,
                    confidence=0.0
                )
            
        except Exception as e:
            logger.error(f"Translation error: {str(e)}", exc_info=True)
            raise
    
    async def translate_text(
        self,
        text: str,
        source_language: str,
        target_language: str,
        model_id: Optional[str] = None,
        glossary_id: Optional[str] = None,
        preserve_formatting: bool = True,
        formality: Optional[str] = None,
        verify: bool = False,
        user_id: Optional[str] = None,
        request_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Unified entry point for translation requests from the processor.
        
        Args:
            text: Text to translate
            source_language: Source language code
            target_language: Target language code
            model_id: Optional specific model to use
            glossary_id: Optional glossary ID
            preserve_formatting: Whether to preserve formatting
            formality: Optional formality level
            verify: Whether to verify translation
            user_id: Optional user ID for tracking
            request_id: Optional request ID for tracking
            
        Returns:
            Dict with translation results
        """
        # Create translation request
        request = TranslationRequest(
            text=text,
            source_language=source_language,
            target_language=target_language,
            model_id=model_id,
            glossary_id=glossary_id,
            preserve_formatting=preserve_formatting,
            formality=formality,
            context=[],  # No context in simplified interface
            domain=None  # No domain in simplified interface
        )
        
        # Process the translation
        result = await self.translate(request)
        
        # Return as dictionary for compatibility
        return {
            "translated_text": result.translated_text,
            "source_language": result.source_language,
            "target_language": result.target_language,
            "confidence": result.confidence,
            "model_used": result.model_used,
            "request_id": request_id
        }
    
    def _load_domain_vocabulary(self, vocabulary_path: str) -> None:
        """
        Load domain-specific vocabulary for better translations.
        
        Args:
            vocabulary_path: Path to vocabulary file
        """
        try:
            import json
            
            with open(vocabulary_path, 'r', encoding='utf-8') as f:
                self.domain_vocabulary = json.load(f)
                
            logger.info(f"Loaded domain vocabulary with {len(self.domain_vocabulary)} domains")
            
        except Exception as e:
            logger.error(f"Error loading domain vocabulary: {str(e)}", exc_info=True)
            self.domain_vocabulary = {}
    
    def _apply_domain_vocabulary(self, 
                               text: str,
                               language: str,
                               domain: str) -> str:
        """
        Apply domain-specific vocabulary to translation.
        
        Args:
            text: Translated text
            language: Target language
            domain: Domain (legal, medical, etc.)
            
        Returns:
            Text with domain-specific terminology
        """
        if not self.domain_vocabulary or domain not in self.domain_vocabulary:
            return text
        
        domain_terms = self.domain_vocabulary.get(domain, {}).get(language, {})
        
        # Replace general terms with domain-specific terms
        processed_text = text
        for general_term, specific_term in domain_terms.items():
            processed_text = processed_text.replace(general_term, specific_term)
        
        return processed_text
    
    def get_supported_languages(self) -> List[Dict[str, str]]:
        """
        Get list of supported languages.
        
        Returns:
            List of dictionaries with language codes and names
        """
        return [
            {"code": code, "name": name}
            for code, name in self.language_names.items()
        ]