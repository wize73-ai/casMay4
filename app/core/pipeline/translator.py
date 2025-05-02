# app/core/pipeline/translator.py
"""
Translation Pipeline for CasaLingua

This module handles translation between different languages, with
support for context-aware and domain-specific translation.
"""

import logging
import time
from typing import Dict, Any, List, Optional, Tuple, Union

from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

from app.core.pipeline.tokenizer import TokenizerPipeline

from app.services.models.manager import ModelManager
# Import ModelRegistry for dynamic tokenizer loading

__all__ = ["TranslationPipeline"]

logger = logging.getLogger("casalingua.core.translator")


# NLLB Translator wrapper for HuggingFace M2M100 models
class NLLBTranslator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    async def translate(self, input_data):
        text = input_data["text"]
        src_lang = input_data["source_language"]
        tgt_lang = input_data["target_language"]

        # Normalize source and target language codes if necessary
        if "_" in src_lang:
            src_lang = src_lang.split("_")[0]
        if "_" in tgt_lang:
            tgt_lang = tgt_lang.split("_")[0]

        # Map ISO-639-1 codes to M2M100 language codes
        lang_code_map = {
            "en": "en_XX",
            "es": "es_XX",
            "fr": "fr_XX",
            "de": "de_DE",
            "it": "it_IT",
            "pt": "pt_XX",
            "nl": "nl_XX",
            "ru": "ru_RU",
            "zh": "zh_CN",
            "ja": "ja_XX",
            "ko": "ko_KR",
            "ar": "ar_AR",
            "hi": "hi_IN",
            "bn": "bn_IN",
            "vi": "vi_VN",
            "th": "th_TH"
        }
        src_lang = lang_code_map.get(src_lang, src_lang)
        tgt_lang = lang_code_map.get(tgt_lang, tgt_lang)

        self.tokenizer.src_lang = src_lang
        encoded = self.tokenizer(text, return_tensors="pt", padding=True)
        if tgt_lang not in self.tokenizer.lang_code_to_id:
            raise ValueError(f"Target language code '{tgt_lang}' not supported by tokenizer.")
        generated_tokens = self.model.generate(**encoded, forced_bos_token_id=self.tokenizer.lang_code_to_id[tgt_lang])
        translated = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        return {"translated_text": translated[0], "confidence": 1.0}

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
        model_manager: ModelManager,
        config: Dict[str, Any] = None,
        registry_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the translation pipeline.
        
        Args:
            model_manager: Model manager for accessing translation models
            config: Configuration dictionary
        """
        self.model_manager = model_manager
        self.config = config or {}
        self.initialized = False
        # Use registry_config to get model and tokenizer names for translation
        registry_config = registry_config or {}
        model_info = registry_config.get("translation", {})
        model_name = model_info.get("model_name")
        tokenizer_name = model_info.get("tokenizer_name")
        self.tokenizer = TokenizerPipeline(model_name=tokenizer_name, task_type="translation")
        
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
        
        # Initialize language detection model
        self.language_detection_model = None
        
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
        
        logger.debug("TranslationPipeline initializing language detection model using loader-managed config.")
        # Load language detection model with better error handling
        logger.info("Loading language detection model")
        try:
            # First try direct model loading
            self.language_detection_model = await self.model_manager.get_model("language_detection")
            logger.info("Language detection model ready")
        except Exception as e:
            logger.warning(f"Error loading language detection model: {str(e)}")
            self.language_detection_model = None
        # Load domain vocabulary if available
        domain_vocab_path = self.config.get("domain_vocabulary_path")
        if domain_vocab_path:
            logger.info(f"Loading domain vocabulary from {domain_vocab_path}")
            self._load_domain_vocabulary(domain_vocab_path)
        self.initialized = True
        logger.info("Translation pipeline initialization complete")    
        
    async def detect_language(self, text: str) -> Tuple[str, float]:
        """
        Detect the language of text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Tuple of (language_code, confidence)
        """
        if not self.initialized:
            raise RuntimeError("Translation pipeline not initialized")
        
        if not text:
            return "en", 0.0
        
        logger.debug(f"Detecting language for text of length {len(text)}")
        
        try:
            # Use loaded language detection model
            if self.language_detection_model:
                # Truncate text if it's too long
                sample_text = text[:min(len(text), 1000)]
                
                # Get prediction from model
                result = await self.model_manager.run_model(
                    self.language_detection_model,
                    "detect_language",
                    {"text": sample_text}
                )
                
                detected_lang = result.get("language", "en")
                confidence = result.get("confidence", 0.0)
                
                logger.debug(f"Language detected: {detected_lang} (confidence: {confidence:.2f})")
                return detected_lang, confidence
            
            # Fallback to simple heuristic detection
            logger.warning("Language detection model not available, using fallback")
            
            best_lang = "en"
            highest_score = 0
            
            # Simple language detection based on character frequencies
            for lang, words in self._get_language_markers().items():
                score = sum(1 for word in words if f" {word} " in f" {text.lower()} ")
                if score > highest_score:
                    highest_score = score
                    best_lang = lang
            
            confidence = min(highest_score / 10, 0.9)
            logger.info(f"Fallback language detection selected: {best_lang} with confidence {confidence:.2f}")
            return best_lang, confidence
            
        except Exception as e:
            logger.error(f"Error detecting language: {str(e)}", exc_info=True)
            return "en", 0.0
    
    async def translate(self, 
                       text: str, 
                       source_language: str, 
                       target_language: str,
                       options: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Translate text from source to target language.
        
        Args:
            text: Text to translate
            source_language: Source language code
            target_language: Target language code
            options: Additional options
                - context: Additional context for translation
                - domain: Specific domain (legal, medical, etc.)
                - preserve_formatting: Whether to preserve formatting
                - model_name: Specific model to use
                
        Returns:
            Dict containing:
            - translated_text: Translated text
            - model_used: Name of model used
            - source_language: Detected or provided source language
            - target_language: Target language
            - confidence: Translation confidence
        """
        if not self.initialized:
            raise RuntimeError("Translation pipeline not initialized")
        
        if not text:
            return {"translated_text": "", "model_used": "none", "confidence": 0.0}
        
        # Same language, no translation needed
        if source_language == target_language:
            return {
                "translated_text": text,
                "model_used": "none",
                "source_language": source_language,
                "target_language": target_language,
                "confidence": 1.0
            }
        
        options = options or {}
        logger.debug(f"Translating from {source_language} to {target_language}")
        
        try:
            # 1. Select appropriate translation model
            model_name = options.get("model_name")
            translation_model = await self._get_translation_model(
                source_language, 
                target_language,
                model_name
            )
            
            if not translation_model:
                raise ValueError(f"No translation model available for {source_language} to {target_language}")
            
            # 2. Prepare translation context
            context = options.get("context", [])
            domain = options.get("domain")
            
            input_data = {
                "text": text,
                "source_language": source_language,
                "target_language": target_language,
                "context": context,
                "domain": domain
            }
            # Shared tokenizer injection
            if self.tokenizer:
                try:
                    # Fallback basic token mapping if custom prep method not available
                    input_data["tokens"] = self.tokenizer.tokenize(text)
                    input_data["token_type"] = "raw"
                except Exception as e:
                    logger.warning(f"Tokenizer fallback tokenization failed: {str(e)}")
            
            # 3. Run translation
            start_time = time.time()
            
            if hasattr(translation_model, "translate") and callable(getattr(translation_model, "translate", None)):
                result = await translation_model.translate(input_data)
            else:
                logger.error(f"Translation model '{translation_model}' does not implement 'translate'")
                raise AttributeError("Translation model does not implement 'translate'")
            
            processing_time = time.time() - start_time
            logger.debug(f"Translation completed in {processing_time:.3f}s")
            
            # 4. Post-process translation
            translated_text = result.get("translated_text", result.get("translation", ""))
            confidence = result.get("confidence", 0.0)
            
            # 5. Apply domain-specific vocabulary if available
            if domain and domain in self.domain_vocabulary:
                translated_text = self._apply_domain_vocabulary(
                    translated_text,
                    target_language,
                    domain
                )
            
            return {
                "translated_text": translated_text,
                "model_used": model_name or getattr(translation_model, "name", "unknown"),
                "source_language": source_language,
                "target_language": target_language,
                "confidence": confidence,
                "processing_time": processing_time
            }
            
        except Exception as e:
            logger.error(f"Translation error: {str(e)}", exc_info=True)
            logger.exception("Exception occurred during translation.")
            raise
    
    async def _get_translation_model(
        self,
        source_language: str,
        target_language: str,
        model_name: Optional[str] = None
    ) -> Optional[Any]:
        """
        Get the general-purpose translation model under the 'translation' key.

        Args:
            source_language: Source language code
            target_language: Target language code
            model_name: Optional specific model name

        Returns:
            Translation model instance or None
        """
        try:
            model_key = "translation"
            logger.debug(f"Attempting to load translation model: {model_key}")
            model_result = await self.model_manager.get_model(model_key)
            model = model_result[0] if isinstance(model_result, tuple) else model_result

            if not model:
                logger.error(f"No translation model found for key '{model_key}'")
                return None

            # Acquire tokenizer from model_result tuple if present
            tokenizer = model_result[1] if isinstance(model_result, tuple) and len(model_result) > 1 else None

            if tokenizer is None:
                logger.error(f"Tokenizer for model key '{model_key}' is not available or failed to load.")
                return None

            # Always wrap model with NLLBTranslator
            return NLLBTranslator(model=model, tokenizer=tokenizer)

        except Exception as e:
            logger.exception(f"Failed to get translation model '{model_key}': {e}")
            return None
    
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
    
    def _get_language_markers(self) -> Dict[str, List[str]]:
        """
        Get common words for each supported language.
        
        Returns:
            Dict mapping language codes to lists of common words
        """
        return {
            "en": ["the", "and", "to", "of", "a", "in", "is", "that", "for", "it"],
            "es": ["el", "la", "de", "que", "y", "en", "un", "una", "es", "por"],
            "fr": ["le", "la", "de", "et", "est", "en", "un", "une", "qui", "dans"],
            "de": ["der", "die", "das", "und", "ist", "in", "ein", "eine", "zu", "mit"],
            "it": ["il", "la", "di", "e", "è", "un", "una", "che", "per", "con"],
            "pt": ["o", "a", "de", "e", "é", "um", "uma", "que", "para", "com"],
            "nl": ["de", "het", "een", "in", "is", "en", "van", "op", "te", "dat"]
        }
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
        """
        options = {
            "model_name": model_id,
            "glossary_id": glossary_id,
            "preserve_formatting": preserve_formatting,
            "formality": formality,
            "verify": verify
        }
        return await self.translate(
            text=text,
            source_language=source_language,
            target_language=target_language,
            options=options
        )