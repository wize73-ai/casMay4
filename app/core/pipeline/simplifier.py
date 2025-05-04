"""
Text Simplification Pipeline for CasaLingua

This module handles text simplification to different levels of complexity,
with support for target grade levels and domain-specific simplification.

Author: Exygy Development Team
Version: 1.0.0
License: MIT
"""

import re
import logging
import time
from typing import Dict, Any, List, Optional, Tuple, Union

from app.utils.logging import get_logger

logger = get_logger("casalingua.core.simplifier")

class SimplificationPipeline:
    """
    Text simplification pipeline.
    
    Features:
    - Multiple simplification levels
    - Grade-level targeting
    - Context-aware simplification
    - Domain-specific simplification
    """
    
    def __init__(
        self,
        model_manager,
        config: Dict[str, Any] = None,
        registry_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the simplification pipeline.
        
        Args:
            model_manager: Model manager for accessing simplification models
            config: Configuration dictionary
            registry_config: Registry configuration for models
        """
        self.model_manager = model_manager
        self.config = config or {}
        self.registry_config = registry_config or {}
        self.initialized = False
        
        # Model type for simplification
        self.model_type = "simplifier"
        
        # Simplification levels map to grade levels
        self.level_grade_map: Dict[int, int] = {
            1: 12,  # College level
            2: 10,  # High school 
            3: 8,   # Middle school
            4: 6,   # Elementary school
            5: 4    # Early elementary
        }
        
        # Load grade level vocabulary
        self.grade_level_vocabulary: Dict[int, Dict[str, Dict[str, str]]] = {}
        
        logger.info("Simplification pipeline created (not yet initialized)")
    
    async def initialize(self) -> None:
        """
        Initialize the simplification pipeline.
        
        This loads necessary models and prepares the pipeline.
        """
        if self.initialized:
            logger.warning("Simplification pipeline already initialized")
            return
        
        logger.info("Initializing simplification pipeline")
        
        # Load simplification model
        try:
            logger.info(f"Loading simplification model ({self.model_type})")
            await self.model_manager.load_model(self.model_type)
            logger.info("Simplification model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading simplification model: {str(e)}")
            logger.warning("Pipeline will function with reduced capabilities")
        
        # Initialize grade level vocabulary
        await self._load_grade_level_vocabulary()
        
        self.initialized = True
        logger.info("Simplification pipeline initialization complete")
    
    async def simplify(self, 
                      text: str, 
                      language: str,
                      level: int = 3,
                      grade_level: Optional[int] = None,
                      options: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Simplify text to the specified level.
        
        Args:
            text: Text to simplify
            language: Language code
            level: Simplification level (1-5, where 5 is simplest)
            grade_level: Target grade level (1-12)
            options: Additional options
                - context: Additional context for simplification
                - domain: Specific domain (legal, medical, etc.)
                - preserve_formatting: Whether to preserve formatting
                - model_name: Specific model to use
                
        Returns:
            Dict containing:
            - simplified_text: Simplified text
            - model_used: Name of model used
            - level: Simplification level used
            - grade_level: Target grade level
            - metrics: Readability metrics
        """
        if not self.initialized:
            await self.initialize()
        
        if not text:
            return {"simplified_text": "", "model_used": "none"}
        
        options = options or {}
        
        # Determine target grade level
        if grade_level:
            # If grade level is specified directly, use it
            target_grade = grade_level
            # Find the closest simplification level
            level = min(self.level_grade_map.items(), key=lambda x: abs(x[1] - target_grade))[0]
        else:
            # Convert level to grade level
            level = max(1, min(5, level))  # Ensure level is between 1 and 5
            target_grade = self.level_grade_map.get(level, 8)
        
        logger.debug(f"Simplifying text to level {level} (grade {target_grade})")
        
        try:
            # Get model ID if specified, otherwise use default
            model_id = options.get("model_name", self.model_type)
            
            # Prepare simplification input
            input_data = {
                "text": text,
                "source_language": language,
                "parameters": {
                    "level": level,
                    "grade_level": target_grade,
                    "domain": options.get("domain"),
                    "preserve_formatting": options.get("preserve_formatting", True)
                }
            }
            
            # Add context if provided
            if "context" in options:
                input_data["context"] = options["context"]
            
            # Run simplification model
            start_time = time.time()
            result = await self.model_manager.run_model(
                model_id,
                "process",
                input_data
            )
            processing_time = time.time() - start_time
            
            # Extract simplification results
            if isinstance(result, dict) and "result" in result:
                simplified_text = result["result"]
                
                # Handle different result formats
                if isinstance(simplified_text, list) and simplified_text:
                    simplified_text = simplified_text[0]
                elif not isinstance(simplified_text, str):
                    simplified_text = str(simplified_text)
            else:
                # Handle unexpected result format
                simplified_text = str(result) if result else text
            
            # Apply grade level vocabulary if available
            if target_grade in self.grade_level_vocabulary:
                simplified_text = self._apply_grade_level_vocabulary(
                    simplified_text,
                    language,
                    target_grade
                )
            
            # Calculate readability metrics
            metrics = self._calculate_readability_metrics(simplified_text, language)
            
            logger.debug(f"Simplification completed in {processing_time:.3f}s")
            
            return {
                "simplified_text": simplified_text,
                "model_used": model_id,
                "level": level,
                "grade_level": target_grade,
                "metrics": metrics,
                "processing_time": processing_time
            }
            
        except Exception as e:
            logger.error(f"Simplification error: {str(e)}", exc_info=True)
            raise
    
    async def _load_grade_level_vocabulary(self) -> None:
        """
        Load grade-level vocabulary for different languages.
        
        This loads vocabulary appropriate for different grade levels
        to ensure simplified text uses appropriate vocabulary.
        """
        try:
            # In a real implementation, this would load from files
            # For now, we'll use a simple dictionary
            
            self.grade_level_vocabulary = {
                # Grade 4 vocabulary (elementary)
                4: {
                    "en": {
                        "utilize": "use",
                        "purchase": "buy",
                        "indicate": "show",
                        "sufficient": "enough",
                        "assist": "help",
                        "obtain": "get",
                        "require": "need",
                        "additional": "more",
                        "prior to": "before",
                        "subsequently": "later"
                    }
                },
                # Grade 8 vocabulary (middle school)
                8: {
                    "en": {
                        "utilize": "use",
                        "purchase": "buy",
                        "indicate": "show",
                        "sufficient": "enough",
                        "subsequently": "later"
                    }
                }
            }
            
            logger.info(f"Loaded grade level vocabulary for {len(self.grade_level_vocabulary)} grade levels")
            
        except Exception as e:
            logger.error(f"Error loading grade level vocabulary: {str(e)}", exc_info=True)
            self.grade_level_vocabulary = {}
    
    def _apply_grade_level_vocabulary(self, 
                                    text: str,
                                    language: str,
                                    grade_level: int) -> str:
        """
        Apply grade-level appropriate vocabulary.
        
        Args:
            text: Text to process
            language: Language code
            grade_level: Target grade level
            
        Returns:
            Text with grade-appropriate vocabulary
        """
        # Find closest available grade level
        available_grades = list(self.grade_level_vocabulary.keys())
        if not available_grades:
            return text
        
        closest_grade = min(available_grades, key=lambda g: abs(g - grade_level))
        grade_vocab = self.grade_level_vocabulary.get(closest_grade, {}).get(language, {})
        
        if not grade_vocab:
            return text
        
        # Apply vocabulary replacements
        processed_text = text
        for complex_word, simple_word in grade_vocab.items():
            pattern = r'\b' + re.escape(complex_word) + r'\b'
            processed_text = re.sub(pattern, simple_word, processed_text, flags=re.IGNORECASE)
        
        return processed_text
    
    def _calculate_readability_metrics(self, text: str, language: str) -> Dict[str, float]:
        """
        Calculate readability metrics for the simplified text.
        
        Args:
            text: Text to analyze
            language: Language code
            
        Returns:
            Dict with readability metrics
        """
        # Only English is fully supported for now
        if language != "en":
            return {"estimated_grade_level": 0}
        
        try:
            # Count words and sentences
            words = len(re.findall(r'\b\w+\b', text))
            sentences = len(re.findall(r'[.!?]+', text)) or 1
            
            # Count syllables (rough approximation)
            syllables = 0
            for word in re.findall(r'\b\w+\b', text):
                word = word.lower()
                if len(word) <= 3:
                    syllables += 1
                else:
                    # Count vowel groups as syllables
                    vowels = "aeiouy"
                    count = 0
                    prev_is_vowel = False
                    for char in word:
                        is_vowel = char in vowels
                        if is_vowel and not prev_is_vowel:
                            count += 1
                        prev_is_vowel = is_vowel
                    
                    # Adjust for common patterns
                    if word.endswith('e'):
                        count -= 1
                    if word.endswith('le') and len(word) > 2 and word[-3] not in vowels:
                        count += 1
                    if count == 0:
                        count = 1
                    
                    syllables += count
            
            # Calculate Flesch-Kincaid Grade Level
            words_per_sentence = words / sentences
            syllables_per_word = syllables / words
            fk_grade = 0.39 * words_per_sentence + 11.8 * syllables_per_word - 15.59
            
            # Ensure grade level is reasonable
            fk_grade = max(1, min(12, fk_grade))
            
            return {
                "estimated_grade_level": round(fk_grade, 1),
                "words_per_sentence": round(words_per_sentence, 1),
                "syllables_per_word": round(syllables_per_word, 1),
                "words": words,
                "sentences": sentences
            }
            
        except Exception as e:
            logger.error(f"Error calculating readability metrics: {str(e)}", exc_info=True)
            return {"estimated_grade_level": 0}