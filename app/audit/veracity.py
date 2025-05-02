"""
Veracity Auditing Module for CasaLingua

This module provides quality assessment, verification, and auditing
capabilities for translations and other language processing outputs
to ensure accuracy and consistency.

Author: Exygy Development Team
Version: 1.0.0
License: MIT
"""

import re
import json
import time
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime

import numpy as np

from app.utils.config import load_config, get_config_value
from app.utils.logging import get_logger
from app.services.models.manager import ModelManager


JSONDict = Dict[str, Any]

logger = get_logger(__name__)

class VeracityAuditor:
    """
    Handles verification and quality assessment of translations
    and other language processing outputs for CasaLingua.
    """
    
    def __init__(
        self,
        model_manager: Optional[ModelManager] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the veracity auditor.
        
        Args:
            model_manager: Model manager instance
            config: Application configuration
        """
        self.config = config or load_config()
        self.model_manager = model_manager
        
        # Load veracity configuration
        self.veracity_config = get_config_value(self.config, "veracity", {})
        self.enabled = get_config_value(self.veracity_config, "enabled", True)
        self.threshold = get_config_value(self.veracity_config, "threshold", 0.75)
        self.max_sample_size = get_config_value(self.veracity_config, "max_sample_size", 1000)
        self.min_confidence = get_config_value(self.veracity_config, "min_confidence", 0.7)
        
        # Initialize quality metrics
        self.metrics: Dict[str, Dict[str, Any]] = {}
        self.quality_statistics: Dict[str, Dict[str, float]] = {}
        
        # Load reference data
        self.reference_embeddings: Dict[str, Dict[str, np.ndarray]] = {}
        self.language_pairs: List[Tuple[str, str]] = []
        
        logger.info("Veracity auditor initialized")
        
    async def initialize(self) -> None:
        """Initialize the veracity auditor and load reference data."""
        if not self.enabled:
            logger.info("Veracity auditing is disabled")
            return
            
        logger.info("Initializing veracity auditor...")
        
        # Load reference embeddings if available
        embeddings_path = get_config_value(
            self.veracity_config, "reference_embeddings_path", None
        )
        
        if embeddings_path:
            await self._load_reference_embeddings(embeddings_path)
            
        # Initialize language pairs
        self.language_pairs = get_config_value(
            self.veracity_config, "language_pairs", []
        )
        
        # If no language pairs specified, use all available
        if not self.language_pairs:
            if self.model_manager:
                registry = self.model_manager._model_registry
                if registry:
                    languages = registry.get_supported_languages("translation")
                    self.language_pairs = [
                        (src, tgt) for src in languages for tgt in languages if src != tgt
                    ]
            
        logger.info(f"Veracity auditor initialized with {len(self.language_pairs)} language pairs")
        
    async def _load_reference_embeddings(self, path: str) -> None:
        """
        Load reference embeddings from file.
        
        Args:
            path: Path to the embeddings file
        """
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            # Convert lists to numpy arrays
            for lang_pair, embeddings in data.items():
                source_lang, target_lang = lang_pair.split("-")
                
                if lang_pair not in self.reference_embeddings:
                    self.reference_embeddings[lang_pair] = {}
                    
                for key, embedding_list in embeddings.items():
                    self.reference_embeddings[lang_pair][key] = np.array(embedding_list)
                    
            logger.info(f"Loaded reference embeddings for {len(self.reference_embeddings)} language pairs")
        except Exception as e:
            logger.error(f"Error loading reference embeddings: {str(e)}", exc_info=True)
            
    async def verify_translation(
        self,
        source_text: str,
        translation: str,
        source_lang: str,
        target_lang: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Verify the quality and accuracy of a translation.
        
        Args:
            source_text: Original text
            translation: Translated text
            source_lang: Source language code
            target_lang: Target language code
            metadata: Additional metadata about the translation
            
        Returns:
            Dictionary with verification results
        """
        if not self.enabled:
            return {
                "verified": True,
                "score": 1.0,
                "confidence": 1.0,
                "issues": []
            }
            
        start_time = time.time()
        lang_pair = f"{source_lang}-{target_lang}"
        
        # Initialize result structure
        result = {
            "verified": False,
            "score": 0.0,
            "confidence": 0.0,
            "issues": [],
            "metrics": {}
        }
        
        try:
            # Basic validation
            result.update(await self._perform_basic_validation(
                source_text, translation, source_lang, target_lang
            ))
            
            # Semantic verification (if model manager available)
            if self.model_manager:
                semantic_result = await self._perform_semantic_verification(
                    source_text, translation, source_lang, target_lang
                )
                
                # Update results, preserving any issues from basic validation
                for key, value in semantic_result.items():
                    if key == "issues":
                        result["issues"].extend(value)
                    else:
                        result[key] = value
                        
            # Content integrity checks
            integrity_result = await self._check_content_integrity(
                source_text, translation, source_lang, target_lang
            )
            result["metrics"].update(integrity_result["metrics"])
            result["issues"].extend(integrity_result["issues"])
            
            # Determine final verification status
            result["verified"] = (
                result["score"] >= self.threshold and 
                result["confidence"] >= self.min_confidence and
                len([i for i in result["issues"] if i["severity"] == "critical"]) == 0
            )
            
            # Track metrics
            self._update_quality_statistics(lang_pair, result)
            
        except Exception as e:
            logger.error(f"Error verifying translation: {str(e)}", exc_info=True)
            result["issues"].append({
                "type": "system_error",
                "severity": "critical",
                "message": f"Verification error: {str(e)}"
            })
            result["verified"] = False
            result["score"] = 0.0
            result["confidence"] = 0.0
            
        # Record execution time
        result["verification_time"] = time.time() - start_time
        
        # Record metadata if provided
        if metadata:
            result["metadata"] = metadata
            
        return result
        
    async def _perform_basic_validation(
        self,
        source_text: str,
        translation: str,
        source_lang: str,
        target_lang: str
    ) -> JSONDict:
        """
        Perform basic validation checks on the translation.

        Args:
            source_text: Original text
            translation: Translated text
            source_lang: Source language code
            target_lang: Target language code

        Returns:
            Dictionary with validation results
        """
        issues = []

        # Check for empty translation
        if not translation.strip():
            issues.append({
                "type": "empty_translation",
                "severity": "critical",
                "message": "Translation is empty"
            })
            logger.warning(f"[Basic Validation] Empty translation detected for {source_lang}->{target_lang}")
            return {
                "verified": False,
                "score": 0.0,
                "confidence": 1.0,
                "issues": issues
            }

        # Check for untranslated content
        if source_text.strip().lower() == translation.strip().lower():
            issues.append({
                "type": "untranslated",
                "severity": "critical",
                "message": "Translation is identical to source text"
            })
            logger.warning(f"[Basic Validation] Untranslated content detected for {source_lang}->{target_lang}")
            return {
                "verified": False,
                "score": 0.0,
                "confidence": 1.0,
                "issues": issues
            }

        # Check if language appears to be correct
        # (Basic check based on language-specific characters)
        if not self._check_language_characters(translation, target_lang):
            issues.append({
                "type": "wrong_language",
                "severity": "warning",
                "message": f"Translation may not be in {target_lang}"
            })
            logger.info(f"[Basic Validation] Language character check failed for {target_lang}")

        # Check length ratio
        source_length = len(source_text.split())
        translation_length = len(translation.split())

        # Get expected ratio range for this language pair
        expected_ratio = self._estimate_length_ratio(source_lang, target_lang)
        actual_ratio = translation_length / max(1, source_length)

        if actual_ratio < expected_ratio * 0.5 or actual_ratio > expected_ratio * 2.0:
            issues.append({
                "type": "length_mismatch",
                "severity": "warning",
                "message": f"Translation length ratio ({actual_ratio:.2f}) is outside expected range",
                "expected_ratio": expected_ratio,
                "actual_ratio": actual_ratio
            })
            logger.info(f"[Basic Validation] Length ratio outside expected range for {source_lang}->{target_lang}")

        # Basic checks passed
        return {
            "verified": len(issues) == 0,
            "score": 1.0 if len(issues) == 0 else 0.5,
            "confidence": 0.7,
            "issues": issues,
            "metrics": {
                "length_ratio": actual_ratio,
                "source_length": source_length,
                "translation_length": translation_length
            }
        }
        
    async def _perform_semantic_verification(
        self,
        source_text: str,
        translation: str,
        source_lang: str,
        target_lang: str
    ) -> JSONDict:
        """
        Verify the semantic equivalence of source and translation.

        Args:
            source_text: Original text
            translation: Translated text
            source_lang: Source language code
            target_lang: Target language code

        Returns:
            Dictionary with verification results
        """
        issues = []
        lang_pair = f"{source_lang}-{target_lang}"

        try:
            # Limit text length for processing
            source_sample = self._get_text_sample(source_text)
            translation_sample = self._get_text_sample(translation)

            # Get embeddings for source and translation
            source_embedding = await self.model_manager.create_embeddings(
                source_sample, model_key="embedding_model"
            )
            translation_embedding = await self.model_manager.create_embeddings(
                translation_sample, model_key="embedding_model"
            )

            # Calculate semantic similarity
            similarity = self._calculate_similarity(
                source_embedding[0], translation_embedding[0]
            )

            # Compare with reference data if available
            confidence = 0.8  # Default confidence
            if lang_pair in self.reference_embeddings:
                confidence = self._compare_with_reference(
                    source_embedding[0],
                    translation_embedding[0],
                    lang_pair
                )

            # Check for semantic issues
            if similarity < 0.5:
                issues.append({
                    "type": "low_semantic_similarity",
                    "severity": "critical",
                    "message": "Translation meaning differs significantly from source",
                    "similarity": float(similarity)
                })
                logger.warning(f"[Semantic Verification] Low semantic similarity ({similarity:.2f}) for {lang_pair}")
            elif similarity < 0.7:
                issues.append({
                    "type": "moderate_semantic_divergence",
                    "severity": "warning",
                    "message": "Translation may have semantic differences from source",
                    "similarity": float(similarity)
                })
                logger.info(f"[Semantic Verification] Moderate semantic divergence ({similarity:.2f}) for {lang_pair}")

            # Determine verification score based on similarity
            score = float(similarity)

            return {
                "verified": similarity >= self.threshold,
                "score": score,
                "confidence": float(confidence),
                "issues": issues,
                "metrics": {
                    "semantic_similarity": float(similarity)
                }
            }

        except Exception as e:
            logger.error(f"[Semantic Verification] Error: {str(e)}", exc_info=True)
            issues.append({
                "type": "semantic_verification_error",
                "severity": "warning",
                "message": f"Could not perform semantic verification: {str(e)}"
            })

            return {
                "verified": False,
                "score": 0.5,
                "confidence": 0.3,
                "issues": issues,
                "metrics": {}
            }
            
    async def _check_content_integrity(
        self,
        source_text: str,
        translation: str,
        source_lang: str,
        target_lang: str
    ) -> JSONDict:
        """
        Check for content integrity issues in translation.

        Args:
            source_text: Original text
            translation: Translated text
            source_lang: Source language code
            target_lang: Target language code

        Returns:
            Dictionary with check results
        """
        issues = []
        metrics = {}

        # Check for missing numbers
        source_numbers = self._extract_numbers(source_text)
        translation_numbers = self._extract_numbers(translation)

        missing_numbers = [n for n in source_numbers if n not in translation_numbers]
        if missing_numbers:
            issues.append({
                "type": "missing_numbers",
                "severity": "critical" if len(missing_numbers) > 0 else "warning",
                "message": f"Translation is missing {len(missing_numbers)} numbers from source",
                "missing": missing_numbers[:5]  # Show first few missing numbers
            })
            logger.warning(f"[Content Integrity] Missing numbers detected: {missing_numbers[:5]}")

        metrics["missing_numbers_count"] = len(missing_numbers)

        # Check for missing named entities (simplified approach)
        source_entities = self._extract_potential_entities(source_text)
        matched_count = 0

        for entity in source_entities:
            # Skip single-letter entities
            if len(entity) <= 1:
                continue
            if entity in translation:
                matched_count += 1

        if source_entities:
            entity_preservation = matched_count / len(source_entities)
            metrics["entity_preservation"] = entity_preservation
            if entity_preservation < 0.7:
                issues.append({
                    "type": "missing_entities",
                    "severity": "warning",
                    "message": "Translation may be missing important named entities"
                })
                logger.info(f"[Content Integrity] Entity preservation low: {entity_preservation:.2f}")

        # Check for hallucinated content (simplified approach)
        source_token_count = len(source_text.split())
        translation_token_count = len(translation.split())

        if translation_token_count > source_token_count * 1.5 and source_token_count > 10:
            issues.append({
                "type": "possible_hallucination",
                "severity": "warning",
                "message": "Translation contains significantly more content than source"
            })
            logger.info(f"[Content Integrity] Possible hallucination detected ({translation_token_count} vs {source_token_count} tokens)")

        return {
            "issues": issues,
            "metrics": metrics
        }
        
    def _get_text_sample(self, text: str) -> str:
        """
        Get a sample of text, limiting to max sample size.

        Args:
            text: Input text

        Returns:
            Sampled text string
        """
        words = text.split()
        if len(words) <= self.max_sample_size:
            return text

        # Take first and last parts to capture important context
        first_part = words[:self.max_sample_size // 2]
        last_part = words[-(self.max_sample_size // 2):]

        return " ".join(first_part + last_part)
        
    def _calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.

        Args:
            embedding1: First embedding array
            embedding2: Second embedding array

        Returns:
            Cosine similarity as float
        """
        # Normalize embeddings
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return np.dot(embedding1, embedding2) / (norm1 * norm2)
        
    def _compare_with_reference(
        self,
        source_embedding: np.ndarray,
        translation_embedding: np.ndarray,
        lang_pair: str
    ) -> float:
        """
        Compare embeddings with reference data to determine confidence.

        Args:
            source_embedding: Source text embedding
            translation_embedding: Translation embedding
            lang_pair: Language pair identifier

        Returns:
            Confidence score (0-1)
        """
        reference = self.reference_embeddings.get(lang_pair, {})

        if "positive_pairs" not in reference or "negative_pairs" not in reference:
            return 0.8  # Default confidence

        # Calculate similarity
        similarity = self._calculate_similarity(source_embedding, translation_embedding)

        # Compare with reference positive and negative pairs
        positive_similarities = []
        for i in range(len(reference["positive_pairs"]) // 2):
            pos_source = reference["positive_pairs"][i*2]
            pos_translation = reference["positive_pairs"][i*2 + 1]
            pos_similarity = self._calculate_similarity(pos_source, pos_translation)
            positive_similarities.append(pos_similarity)

        negative_similarities = []
        for i in range(len(reference["negative_pairs"]) // 2):
            neg_source = reference["negative_pairs"][i*2]
            neg_translation = reference["negative_pairs"][i*2 + 1]
            neg_similarity = self._calculate_similarity(neg_source, neg_translation)
            negative_similarities.append(neg_similarity)

        # Calculate confidence based on position relative to positive/negative distributions
        if not positive_similarities or not negative_similarities:
            return 0.8

        avg_positive = np.mean(positive_similarities)
        avg_negative = np.mean(negative_similarities)
        std_positive = np.std(positive_similarities) or 0.1

        # Higher confidence when similarity is close to positive examples
        z_score = (similarity - avg_positive) / std_positive
        confidence = 1.0 / (1.0 + np.exp(z_score))  # Sigmoid function

        # Adjust confidence based on separation between positive and negative
        separation = (avg_positive - avg_negative) / std_positive
        if separation > 2:
            # Good separation, higher confidence
            confidence = min(0.9, confidence + 0.1)
        elif separation < 1:
            # Poor separation, lower confidence
            confidence = max(0.5, confidence - 0.1)

        return float(confidence)
        
    def _check_language_characters(self, text: str, language: str) -> bool:
        """
        Check if text contains characters typical of the target language.

        Args:
            text: Text to check
            language: Target language code

        Returns:
            True if text appears to be in target language
        """
        # Language-specific character patterns
        language_patterns = {
            "zh": r'[\u4e00-\u9fff]',  # Chinese
            "ja": r'[\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff]',  # Japanese
            "ko": r'[\uac00-\ud7af\u1100-\u11ff]',  # Korean
            "ru": r'[а-яА-ЯёЁ]',  # Russian
            "ar": r'[\u0600-\u06ff]',  # Arabic
            "he": r'[\u0590-\u05ff]',  # Hebrew
            "th": r'[\u0e00-\u0e7f]',  # Thai
            # Add more languages as needed
        }

        # If language has a specific pattern, check for it
        if language in language_patterns:
            pattern = language_patterns[language]
            matched = re.search(pattern, text)
            return matched is not None

        # For languages without specific patterns, return True
        return True
        
    def _estimate_length_ratio(self, source_lang: str, target_lang: str) -> float:
        """
        Estimate expected length ratio between source and target languages.

        Args:
            source_lang: Source language code
            target_lang: Target language code

        Returns:
            Expected ratio of target to source length
        """
        # Approximate length ratios based on language pairs
        # These are averages and can vary by content type
        length_ratios = {
            "en-es": 1.3,  # English to Spanish
            "en-fr": 1.3,  # English to French
            "en-de": 1.1,  # English to German
            "en-zh": 0.7,  # English to Chinese
            "en-ja": 0.6,  # English to Japanese
            "en-ru": 0.8,  # English to Russian
            "zh-en": 1.5,  # Chinese to English
            "ja-en": 1.7,  # Japanese to English
            "es-en": 0.8,  # Spanish to English
            "fr-en": 0.8,  # French to English
            "de-en": 0.9,  # German to English
            "ru-en": 1.3,  # Russian to English
            # Add more language pairs as needed
        }

        lang_pair = f"{source_lang}-{target_lang}"

        # Return configured ratio if available
        if lang_pair in length_ratios:
            return length_ratios[lang_pair]

        # Return default ratio (1.0) for unknown pairs
        return 1.0
        
    def _extract_numbers(self, text: str) -> List[str]:
        """
        Extract numbers from text.

        Args:
            text: Text to extract numbers from

        Returns:
            List of numbers as strings
        """
        # Match integers, decimals, and numbers with commas
        number_pattern = r'\b\d+(?:,\d+)*(?:\.\d+)?\b'
        return re.findall(number_pattern, text)
        
    def _extract_potential_entities(self, text: str) -> List[str]:
        """
        Extract potential named entities from text.

        This is a simplified approach that looks for capitalized words.
        For more accurate entity detection, use a dedicated NER model.

        Args:
            text: Text to extract entities from

        Returns:
            List of potential entity strings
        """
        # Simple heuristic: words starting with capital letters
        # This is a simplification and will miss many entities
        words = re.findall(r'\b[A-Z][a-zA-Z]*\b', text)

        # Remove duplicates
        return list(set(words))
        
    def _update_quality_statistics(self, lang_pair: str, result: JSONDict) -> None:
        """
        Update quality statistics for a language pair.

        Args:
            lang_pair: Language pair identifier
            result: Verification result
        Returns:
            None
        """
        if lang_pair not in self.quality_statistics:
            self.quality_statistics[lang_pair] = {
                "verified_count": 0,
                "total_count": 0,
                "average_score": 0.0,
                "average_confidence": 0.0,
                "issue_counts": {}
            }

        stats = self.quality_statistics[lang_pair]

        # Update counts
        stats["total_count"] += 1
        if result["verified"]:
            stats["verified_count"] += 1

        # Update averages
        stats["average_score"] = (
            (stats["average_score"] * (stats["total_count"] - 1) + result["score"]) /
            stats["total_count"]
        )

        stats["average_confidence"] = (
            (stats["average_confidence"] * (stats["total_count"] - 1) + result["confidence"]) /
            stats["total_count"]
        )

        # Update issue counts
        for issue in result["issues"]:
            issue_type = issue["type"]
            if issue_type not in stats["issue_counts"]:
                stats["issue_counts"][issue_type] = 0
            stats["issue_counts"][issue_type] += 1
            
    def get_quality_statistics(self) -> JSONDict:
        """
        Get quality statistics for all language pairs.

        Returns:
            Dictionary with quality statistics
        """
        stats = {
            "overall": {
                "verified_count": 0,
                "total_count": 0,
                "verification_rate": 0.0,
                "average_score": 0.0,
                "average_confidence": 0.0,
                "top_issues": []
            },
            "by_language_pair": self.quality_statistics
        }

        # Calculate overall statistics
        total_verified = 0
        total_count = 0
        total_score = 0.0
        total_confidence = 0.0
        all_issues: Dict[str, int] = {}

        for lang_pair, pair_stats in self.quality_statistics.items():
            total_verified += pair_stats["verified_count"]
            total_count += pair_stats["total_count"]
            total_score += pair_stats["average_score"] * pair_stats["total_count"]
            total_confidence += pair_stats["average_confidence"] * pair_stats["total_count"]

            # Aggregate issues
            for issue_type, count in pair_stats["issue_counts"].items():
                if issue_type not in all_issues:
                    all_issues[issue_type] = 0
                all_issues[issue_type] += count

        if total_count > 0:
            stats["overall"]["verified_count"] = total_verified
            stats["overall"]["total_count"] = total_count
            stats["overall"]["verification_rate"] = total_verified / total_count
            stats["overall"]["average_score"] = total_score / total_count
            stats["overall"]["average_confidence"] = total_confidence / total_count

            # Get top issues
            top_issues = sorted(
                [{"type": k, "count": v} for k, v in all_issues.items()],
                key=lambda x: x["count"],
                reverse=True
            )[:5]

            stats["overall"]["top_issues"] = top_issues

        return stats