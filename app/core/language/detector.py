"""
Language Detection Module for CasaLingua

Provides robust language detection capabilities,
integrating with the model system architecture.

Author: Exygy Development Team
Version: 1.0.0
License: MIT
"""

import logging
from typing import Tuple, List, Dict, Any, Optional
from colorama import Fore, Style, init as colorama_init

# Initialize colorama
colorama_init(autoreset=True)

# Setup logging
from app.utils.logging import get_logger
logger = get_logger(__name__)

# Try to import langdetect (preferred option)
try:
    from langdetect import detect, DetectorFactory, detect_langs, LangDetectException
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    logger.warning("langdetect not available, using fallback detection method")

class FastLanguageDetector:
    """
    Language detector using langdetect library with probabilistic confidence.
    """
    
    def __init__(self):
        """
        Initialize the language detector.
        """
        # Make detection deterministic if possible
        if LANGDETECT_AVAILABLE:
            try:
                DetectorFactory.seed = 42
            except Exception as e:
                logger.warning(f"Could not set langdetect seed: {e}")
                
        # Attempt to use fasttext model if langdetect is not available
        try:
            # pylint: disable=import-outside-toplevel,import-error
            import fasttext
            from app.services.models.registry import load_registry_config

            registry = load_registry_config()
            model_path = registry.get("language_detection", {}).get("model_name")

            if model_path:
                self.fasttext_model = fasttext.load_model(model_path)
                logger.info(f"Loaded fastText model from: {model_path}")
            else:
                self.fasttext_model = None
                logger.warning("No fastText model path found in registry config")
        except Exception as e:
            self.fasttext_model = None
            logger.warning(f"fastText model not loaded: {e}")
                
        # Language name mapping
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
        
        # Common words in various languages for fallback detection
        self.language_markers = {
            "en": ["the", "and", "of", "to", "in", "is", "you", "that", "it", "he"],
            "es": ["el", "la", "de", "que", "y", "a", "en", "un", "ser", "se"],
            "fr": ["le", "la", "de", "et", "est", "en", "que", "un", "une", "du"],
            "de": ["der", "die", "das", "und", "ist", "in", "zu", "den", "mit", "nicht"],
            "it": ["il", "la", "di", "e", "è", "un", "una", "che", "per", "con"],
            "pt": ["o", "a", "de", "e", "é", "um", "uma", "que", "para", "com"],
            "nl": ["de", "het", "een", "in", "is", "en", "van", "op", "te", "dat"]
        }
        
        logger.info("Language detector initialized")

    async def detect_language(self, text: str) -> Tuple[str, float]:
        """
        Detect the language of input text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Tuple of (language_code, confidence)
        """
        if not text or len(text.strip()) < 10:
            logger.warning("Text too short for reliable language detection")
            return "en", 0.5  # Default to English with low confidence
            
        # Try to use langdetect if available
        if LANGDETECT_AVAILABLE:
            try:
                result = self.detect_with_confidence(text)
                if result:
                    # Get top result
                    top_lang = result[0]
                    return top_lang["lang"], top_lang["prob"]
            except Exception as e:
                logger.warning(f"Language detection error: {e}, using fallback method")
                
        # Fallback to fasttext or marker-based detection
        if hasattr(self, 'fasttext_model') and self.fasttext_model:
            prediction = self.fasttext_model.predict(text)
            if prediction and isinstance(prediction[0], list):
                lang_code = prediction[0][0].replace("__label__", "")
                confidence = float(prediction[1][0])
                return lang_code, confidence
        return self._detect_with_markers(text)
        
    def detect(self, text: str) -> str:
        """
        Detect language with colorful console output.
        
        Args:
            text: Text to analyze
            
        Returns:
            Language code
        """
        try:
            if LANGDETECT_AVAILABLE:
                lang = detect(text)
                lang_name = self.language_names.get(lang, lang)
                print(f"{Fore.CYAN}[LanguageDetector]{Style.RESET_ALL} Detected language: {Fore.YELLOW}{lang}{Style.RESET_ALL} ({lang_name})")
                return lang
            else:
                lang, _ = self._detect_with_markers(text)
                lang_name = self.language_names.get(lang, lang)
                print(f"{Fore.CYAN}[LanguageDetector]{Style.RESET_ALL} Detected language: {Fore.YELLOW}{lang}{Style.RESET_ALL} ({lang_name}) (fallback method)")
                return lang
        except Exception:
            print(f"{Fore.RED}[LanguageDetector]{Style.RESET_ALL} Failed to detect language.")
            return "unknown"

    def detect_with_confidence(self, text: str) -> List[Dict[str, float]]:
        """
        Detect language with confidence scores and colorful output.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of language probabilities
        """
        try:
            if not LANGDETECT_AVAILABLE:
                lang, conf = self._detect_with_markers(text)
                result = [{"lang": lang, "prob": conf}]
                print(f"{Fore.CYAN}[LanguageDetector]{Style.RESET_ALL} Ranked languages (fallback method):")
                print(f"  {Fore.YELLOW}{lang}{Style.RESET_ALL}: {Fore.GREEN}{conf:.2f}{Style.RESET_ALL}")
                return result
                
            results = detect_langs(text)
            print(f"{Fore.CYAN}[LanguageDetector]{Style.RESET_ALL} Ranked languages:")
            
            language_results = []
            for r in results:
                lang_name = self.language_names.get(r.lang, r.lang)
                print(f"  {Fore.YELLOW}{r.lang}{Style.RESET_ALL} ({lang_name}): {Fore.GREEN}{r.prob:.2f}{Style.RESET_ALL}")
                language_results.append({"lang": r.lang, "prob": r.prob})
                
            return language_results
            
        except Exception as e:
            print(f"{Fore.RED}[LanguageDetector]{Style.RESET_ALL} Failed to detect languages: {e}")
            return [{"lang": "unknown", "prob": 0.0}]
            
    def _detect_with_markers(self, text: str) -> Tuple[str, float]:
        """
        Detect language using marker words.
        
        Args:
            text: Text to analyze
            
        Returns:
            Tuple of (language_code, confidence)
        """
        text = text.lower()
        scores = {lang: 0 for lang in self.language_markers}
        
        # Count occurrences of marker words
        for lang, markers in self.language_markers.items():
            for word in markers:
                if f" {word} " in f" {text} ":
                    scores[lang] += 1
        
        # Calculate confidence
        total_markers = sum(scores.values())
        
        if total_markers == 0:
            logger.warning("No language markers found in text")
            return "en", 0.5  # Default to English with low confidence
            
        # Find language with highest score
        best_lang = max(scores.items(), key=lambda x: x[1])
        lang_code = best_lang[0]
        confidence = best_lang[1] / total_markers if total_markers > 0 else 0.5
        
        return lang_code, min(confidence * 2, 0.95)  # Scale up but cap at 0.95

# Entry point for direct usage
if __name__ == "__main__":
    # Example usage
    detector = FastLanguageDetector()
    
    test_texts = [
        "This is an example sentence in English.",
        "Esto es una frase de ejemplo en español.",
        "Ceci est une phrase d'exemple en français.",
        "Dies ist ein Beispielsatz auf Deutsch.",
        "これは日本語の例文です。",
        "这是一个中文例句。"
    ]
    
    for text in test_texts:
        lang = detector.detect(text)
        print(f"Detected: {lang} for text: '{text[:30]}...'")
        print("-" * 60)