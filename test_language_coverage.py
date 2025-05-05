#!/usr/bin/env python3
"""
Language Coverage Test for CasaLingua

This script tests the language coverage of the CasaLingua translation system,
verifying which language pairs are supported and properly functioning.
It creates a comprehensive language support matrix and identifies any gaps
or issues in language coverage.

Usage:
    python test_language_coverage.py --output-file language_coverage_matrix.json

Author: Exygy Development Team
Version: 1.0.0
License: MIT
"""

import os
import sys
import json
import logging
import asyncio
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime
import csv
import matplotlib.pyplot as plt
import numpy as np

# Add project root to path for imports
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("language_coverage.log")
    ]
)
logger = logging.getLogger("language_coverage")

# Define commonly supported languages with their names
COMMON_LANGUAGES = {
    "ar": "Arabic",
    "bn": "Bengali",
    "zh": "Chinese",
    "nl": "Dutch",
    "en": "English",
    "fr": "French",
    "de": "German",
    "el": "Greek",
    "he": "Hebrew",
    "hi": "Hindi",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "ms": "Malay",
    "pt": "Portuguese",
    "ru": "Russian",
    "es": "Spanish",
    "sw": "Swahili",
    "tl": "Tagalog",
    "ta": "Tamil",
    "th": "Thai",
    "tr": "Turkish",
    "ur": "Urdu",
    "vi": "Vietnamese"
}

# Test phrases for each language (greeting + simple statement)
TEST_PHRASES = {
    "ar": "مرحبا. كيف حالك؟",  # Hello. How are you?
    "bn": "হ্যালো। আপনি কেমন আছেন?",  # Hello. How are you?
    "zh": "你好。你好吗？",  # Hello. How are you?
    "nl": "Hallo. Hoe gaat het met je?",  # Hello. How are you?
    "en": "Hello. How are you?",
    "fr": "Bonjour. Comment allez-vous?",  # Hello. How are you?
    "de": "Hallo. Wie geht es dir?",  # Hello. How are you?
    "el": "Γεια σας. Πώς είστε;",  # Hello. How are you?
    "he": "שלום. מה שלומך?",  # Hello. How are you?
    "hi": "नमस्ते। आप कैसे हैं?",  # Hello. How are you?
    "id": "Halo. Apa kabar?",  # Hello. How are you?
    "it": "Ciao. Come stai?",  # Hello. How are you?
    "ja": "こんにちは。お元気ですか？",  # Hello. How are you?
    "ko": "안녕하세요. 어떻게 지내세요?",  # Hello. How are you?
    "ms": "Helo. Apa khabar?",  # Hello. How are you?
    "pt": "Olá. Como vai você?",  # Hello. How are you?
    "ru": "Здравствуйте. Как дела?",  # Hello. How are you?
    "es": "Hola. ¿Cómo estás?",  # Hello. How are you?
    "sw": "Habari. Je, unafanyaje?",  # Hello. How are you?
    "tl": "Kumusta. Kamusta ka?",  # Hello. How are you?
    "ta": "வணக்கம். எப்படி இருக்கிறீர்கள்?",  # Hello. How are you?
    "th": "สวัสดี. คุณสบายดีไหม?",  # Hello. How are you?
    "tr": "Merhaba. Nasılsın?",  # Hello. How are you?
    "ur": "ہیلو۔ آپ کیسے ہیں؟",  # Hello. How are you?
    "vi": "Xin chào. Bạn có khỏe không?"  # Hello. How are you?
}

class LanguageCoverageTest:
    """Tests language coverage for the CasaLingua translation system."""
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        output_dir: Optional[str] = None,
        languages: Optional[List[str]] = None
    ):
        """
        Initialize language coverage test.
        
        Args:
            config_path: Path to configuration file
            output_dir: Directory for test results
            languages: List of language codes to test (default: all common languages)
        """
        # Import app modules
        from app.utils.config import load_config
        
        # Load configuration
        self.config = load_config(config_path) if config_path else load_config()
        
        # Set up output directory
        self.output_dir = Path(output_dir) if output_dir else Path("language_coverage_results")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set languages to test
        self.languages = languages if languages else list(COMMON_LANGUAGES.keys())
        
        # Initialize components
        self.processor = None
        self.model_manager = None
        
        # Store test results
        self.coverage_matrix = {}
        self.language_detection_results = {}
        self.supported_languages = set()
        self.unsupported_languages = set()
    
    async def initialize(self) -> None:
        """Initialize test components."""
        from app.services.models.loader import ModelLoader, load_registry_config
        from app.services.models.manager import EnhancedModelManager
        from app.core.pipeline.processor import UnifiedProcessor
        from app.audit.logger import AuditLogger
        from app.audit.metrics import MetricsCollector
        
        logger.info("Initializing test components...")
        
        try:
            # Create model loader
            model_loader = ModelLoader(config=self.config)
            
            # Create hardware info dict
            hardware_info = {
                "memory": {"total_gb": 16, "available_gb": 12},
                "system": {"processor_type": "apple_silicon"}
            }
            
            # Create audit logger and metrics collector
            audit_logger = AuditLogger(config=self.config)
            metrics = MetricsCollector(config=self.config)
            
            # Load model registry configuration
            registry_config = load_registry_config(self.config)
            
            # Create model manager
            self.model_manager = EnhancedModelManager(
                model_loader, hardware_info, self.config
            )
            
            # Create processor
            self.processor = UnifiedProcessor(
                self.model_manager, audit_logger, metrics, 
                self.config, registry_config
            )
            
            # Initialize processor
            await self.processor.initialize()
            logger.info("Test components initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing test components: {e}", exc_info=True)
            raise
    
    async def test_language_detection(self) -> Dict[str, Dict[str, Any]]:
        """
        Test language detection for each language.
        
        Returns:
            Dictionary of language detection results by language code
        """
        logger.info("Testing language detection...")
        
        # Check that processor is initialized
        if not self.processor:
            raise ValueError("Processor not initialized. Call initialize() first.")
        
        results = {}
        
        # Test each language
        for lang_code, lang_name in COMMON_LANGUAGES.items():
            if lang_code not in TEST_PHRASES:
                logger.warning(f"No test phrase for {lang_code} ({lang_name}), skipping")
                continue
            
            # Get test phrase
            test_phrase = TEST_PHRASES[lang_code]
            
            try:
                # Detect language
                detection_result = await self.processor.detect_language(
                    text=test_phrase, 
                    detailed=True
                )
                
                # Extract detected language and confidence
                detected_lang = detection_result.get("detected_language", "unknown")
                confidence = detection_result.get("confidence", 0.0)
                alternatives = detection_result.get("alternatives", [])
                
                # Check if detection was correct
                is_correct = detected_lang == lang_code
                
                # Add to results
                results[lang_code] = {
                    "language": lang_name,
                    "test_phrase": test_phrase,
                    "detected_language": detected_lang,
                    "detected_language_name": COMMON_LANGUAGES.get(detected_lang, "Unknown"),
                    "confidence": confidence,
                    "alternatives": alternatives,
                    "correct": is_correct
                }
                
                logger.info(f"  {lang_code} ({lang_name}): Detected as {detected_lang} with confidence {confidence:.2f} - {'✓' if is_correct else '✗'}")
                
            except Exception as e:
                logger.error(f"  Error detecting language {lang_code}: {e}")
                
                # Add error to results
                results[lang_code] = {
                    "language": lang_name,
                    "test_phrase": test_phrase,
                    "error": str(e),
                    "correct": False
                }
        
        # Save results
        self.language_detection_results = results
        
        # Save to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = self.output_dir / f"language_detection_results_{timestamp}.json"
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
        
        # Calculate summary
        correct_count = sum(1 for result in results.values() if result.get("correct", False))
        total_count = len(results)
        accuracy = correct_count / total_count if total_count > 0 else 0
        
        logger.info(f"Language detection accuracy: {correct_count}/{total_count} ({accuracy:.2%})")
        
        return results
    
    async def test_translation_coverage(self) -> Dict[str, Dict[str, Any]]:
        """
        Test translation coverage for all language pairs.
        
        Returns:
            Dictionary containing the coverage matrix
        """
        logger.info("Testing translation coverage...")
        
        # Check that processor is initialized
        if not self.processor:
            raise ValueError("Processor not initialized. Call initialize() first.")
        
        # Initialize coverage matrix
        coverage_matrix = {}
        
        # Get list of actually supported source and target languages
        supported_sources = set()
        supported_targets = set()
        
        # Test each language pair
        total_pairs = len(self.languages) * len(self.languages)
        completed = 0
        
        for source_lang in self.languages:
            coverage_matrix[source_lang] = {}
            source_name = COMMON_LANGUAGES.get(source_lang, "Unknown")
            
            for target_lang in self.languages:
                # Skip self-translation
                if source_lang == target_lang:
                    coverage_matrix[source_lang][target_lang] = {
                        "supported": True,  # Self-translation is always "supported"
                        "quality": 1.0,     # Perfect quality for self-translation
                        "source_text": TEST_PHRASES.get(source_lang, ""),
                        "translated_text": TEST_PHRASES.get(source_lang, "")
                    }
                    completed += 1
                    continue
                
                target_name = COMMON_LANGUAGES.get(target_lang, "Unknown")
                logger.info(f"Testing translation {source_lang} ({source_name}) -> {target_lang} ({target_name}) [{completed}/{total_pairs}]")
                
                # Get test phrase for source language
                source_text = TEST_PHRASES.get(source_lang, "Hello. How are you?")
                
                try:
                    # Attempt translation
                    translation_result = await self.processor.process_translation(
                        text=source_text,
                        source_language=source_lang,
                        target_language=target_lang
                    )
                    
                    # Extract translated text and confidence
                    translated_text = translation_result.get("translated_text", "")
                    confidence = translation_result.get("confidence", 0.0)
                    
                    # Check if translation succeeded
                    is_supported = bool(translated_text)
                    
                    # Add to coverage matrix
                    coverage_matrix[source_lang][target_lang] = {
                        "supported": is_supported,
                        "quality": confidence,
                        "source_text": source_text,
                        "translated_text": translated_text,
                        "model_id": translation_result.get("model_id", "unknown")
                    }
                    
                    # Update supported languages sets
                    if is_supported:
                        supported_sources.add(source_lang)
                        supported_targets.add(target_lang)
                        self.supported_languages.add((source_lang, target_lang))
                    else:
                        self.unsupported_languages.add((source_lang, target_lang))
                    
                    logger.info(f"  {'✓' if is_supported else '✗'} {translated_text[:50]}...")
                    
                except Exception as e:
                    logger.error(f"  Error translating {source_lang} -> {target_lang}: {e}")
                    
                    # Add error to coverage matrix
                    coverage_matrix[source_lang][target_lang] = {
                        "supported": False,
                        "error": str(e),
                        "source_text": source_text,
                        "translated_text": ""
                    }
                    
                    self.unsupported_languages.add((source_lang, target_lang))
                
                completed += 1
        
        # Save coverage matrix
        self.coverage_matrix = {
            "timestamp": datetime.now().isoformat(),
            "languages_tested": len(self.languages),
            "pairs_tested": total_pairs,
            "supported_sources": list(supported_sources),
            "supported_targets": list(supported_targets),
            "supported_pairs": len(self.supported_languages),
            "coverage_percent": len(self.supported_languages) / (len(self.languages) * (len(self.languages) - 1)) * 100,
            "matrix": coverage_matrix
        }
        
        # Save to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = self.output_dir / f"language_coverage_matrix_{timestamp}.json"
        with open(json_path, "w") as f:
            json.dump(self.coverage_matrix, f, indent=2)
        
        # Generate coverage reports
        self._generate_coverage_reports()
        
        return self.coverage_matrix
    
    def _generate_coverage_reports(self) -> None:
        """Generate coverage reports and visualizations."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create CSV summary
        csv_path = self.output_dir / f"language_coverage_summary_{timestamp}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            
            # Write header with target languages
            header = ["Source Language"]
            for target_lang in self.languages:
                header.append(f"{target_lang} ({COMMON_LANGUAGES.get(target_lang, 'Unknown')})")
            writer.writerow(header)
            
            # Write data for each source language
            for source_lang in self.languages:
                row = [f"{source_lang} ({COMMON_LANGUAGES.get(source_lang, 'Unknown')})"]
                
                for target_lang in self.languages:
                    if source_lang == target_lang:
                        row.append("N/A")  # Self-translation
                    elif (source_lang, target_lang) in self.supported_languages:
                        row.append("Yes")  # Supported
                    else:
                        row.append("No")   # Not supported
                
                writer.writerow(row)
        
        # Generate heat map visualization
        if 'plt' in globals():
            self._generate_coverage_heatmap()
    
    def _generate_coverage_heatmap(self) -> None:
        """Generate a heatmap visualization of language coverage."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create data matrix
        matrix_data = np.zeros((len(self.languages), len(self.languages)))
        
        # Fill matrix
        for i, source_lang in enumerate(self.languages):
            for j, target_lang in enumerate(self.languages):
                if source_lang == target_lang:
                    matrix_data[i, j] = 0.5  # Diagonal (self-translation)
                elif (source_lang, target_lang) in self.supported_languages:
                    matrix_data[i, j] = 1.0  # Supported
                else:
                    matrix_data[i, j] = 0.0  # Not supported
        
        # Create figure
        plt.figure(figsize=(12, 10))
        
        # Create heatmap
        plt.imshow(matrix_data, cmap='RdYlGn', interpolation='nearest')
        
        # Add labels
        language_labels = [f"{lang} ({COMMON_LANGUAGES.get(lang, 'Unknown')})" for lang in self.languages]
        plt.xticks(range(len(self.languages)), language_labels, rotation=90)
        plt.yticks(range(len(self.languages)), language_labels)
        
        # Add colorbar
        cbar = plt.colorbar()
        cbar.set_ticks([0, 0.5, 1])
        cbar.set_ticklabels(['Not Supported', 'Self-Translation', 'Supported'])
        
        # Add labels and title
        plt.xlabel('Target Language')
        plt.ylabel('Source Language')
        plt.title('Language Pair Coverage Matrix')
        
        # Add coverage percentage to title
        coverage_percent = len(self.supported_languages) / (len(self.languages) * (len(self.languages) - 1)) * 100
        plt.suptitle(f'Coverage: {coverage_percent:.1f}% of possible language pairs')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        chart_path = self.output_dir / f"language_coverage_heatmap_{timestamp}.png"
        plt.savefig(chart_path, dpi=300)
        plt.close()
        
        logger.info(f"Coverage heatmap saved to {chart_path}")
    
    async def test_language_support(self) -> Dict[str, Any]:
        """
        Run comprehensive language coverage tests.
        
        Returns:
            Dictionary of test results
        """
        # Check that processor is initialized
        if not self.processor:
            raise ValueError("Processor not initialized. Call initialize() first.")
        
        # Test language detection
        detection_results = await self.test_language_detection()
        
        # Test translation coverage
        coverage_matrix = await self.test_translation_coverage()
        
        # Combine results
        combined_results = {
            "timestamp": datetime.now().isoformat(),
            "language_detection": {
                "accuracy": sum(1 for r in detection_results.values() if r.get("correct", False)) / len(detection_results) if detection_results else 0,
                "results": detection_results
            },
            "translation_coverage": coverage_matrix
        }
        
        # Save combined results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = self.output_dir / f"language_support_results_{timestamp}.json"
        with open(json_path, "w") as f:
            json.dump(combined_results, f, indent=2)
        
        # Generate summary report
        self._generate_summary_report(combined_results)
        
        return combined_results
    
    def _generate_summary_report(self, results: Dict[str, Any]) -> None:
        """
        Generate a summary report of language support.
        
        Args:
            results: Language support test results
        """
        # Extract data
        detection_accuracy = results["language_detection"]["accuracy"]
        coverage_percent = results["translation_coverage"]["coverage_percent"]
        supported_pairs = results["translation_coverage"]["supported_pairs"]
        total_pairs = len(self.languages) * (len(self.languages) - 1)  # Excluding self-translation
        
        # Create summary text
        summary = [
            "# Language Support Summary",
            "",
            f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Language Detection",
            f"- Accuracy: {detection_accuracy:.2%}",
            f"- Languages Tested: {len(results['language_detection']['results'])}",
            "",
            "## Translation Coverage",
            f"- Coverage: {coverage_percent:.2f}% ({supported_pairs}/{total_pairs} language pairs)",
            f"- Languages Tested: {len(self.languages)}",
            "",
            "## Most Supported Source Languages",
            *self._get_most_supported_sources(5),
            "",
            "## Most Supported Target Languages",
            *self._get_most_supported_targets(5),
            "",
            "## Least Supported Source Languages",
            *self._get_least_supported_sources(5),
            "",
            "## Least Supported Target Languages",
            *self._get_least_supported_targets(5),
            ""
        ]
        
        # Save summary
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_path = self.output_dir / f"language_support_summary_{timestamp}.md"
        with open(summary_path, "w") as f:
            f.write("\n".join(summary))
        
        logger.info(f"Summary report saved to {summary_path}")
    
    def _get_most_supported_sources(self, count: int = 5) -> List[str]:
        """
        Get the most supported source languages.
        
        Args:
            count: Number of languages to return
            
        Returns:
            List of formatted strings for report
        """
        source_counts = {}
        
        for source_lang, target_lang in self.supported_languages:
            if source_lang not in source_counts:
                source_counts[source_lang] = 0
            source_counts[source_lang] += 1
        
        # Sort by count
        sorted_sources = sorted(source_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Format for report
        result = []
        for i, (lang, count) in enumerate(sorted_sources[:count], 1):
            lang_name = COMMON_LANGUAGES.get(lang, "Unknown")
            result.append(f"{i}. {lang} ({lang_name}): {count} target languages")
        
        return result
    
    def _get_most_supported_targets(self, count: int = 5) -> List[str]:
        """
        Get the most supported target languages.
        
        Args:
            count: Number of languages to return
            
        Returns:
            List of formatted strings for report
        """
        target_counts = {}
        
        for source_lang, target_lang in self.supported_languages:
            if target_lang not in target_counts:
                target_counts[target_lang] = 0
            target_counts[target_lang] += 1
        
        # Sort by count
        sorted_targets = sorted(target_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Format for report
        result = []
        for i, (lang, count) in enumerate(sorted_targets[:count], 1):
            lang_name = COMMON_LANGUAGES.get(lang, "Unknown")
            result.append(f"{i}. {lang} ({lang_name}): {count} source languages")
        
        return result
    
    def _get_least_supported_sources(self, count: int = 5) -> List[str]:
        """
        Get the least supported source languages.
        
        Args:
            count: Number of languages to return
            
        Returns:
            List of formatted strings for report
        """
        source_counts = {lang: 0 for lang in self.languages}
        
        for source_lang, target_lang in self.supported_languages:
            source_counts[source_lang] += 1
        
        # Sort by count
        sorted_sources = sorted(source_counts.items(), key=lambda x: x[1])
        
        # Format for report
        result = []
        for i, (lang, count) in enumerate(sorted_sources[:count], 1):
            lang_name = COMMON_LANGUAGES.get(lang, "Unknown")
            result.append(f"{i}. {lang} ({lang_name}): {count} target languages")
        
        return result
    
    def _get_least_supported_targets(self, count: int = 5) -> List[str]:
        """
        Get the least supported target languages.
        
        Args:
            count: Number of languages to return
            
        Returns:
            List of formatted strings for report
        """
        target_counts = {lang: 0 for lang in self.languages}
        
        for source_lang, target_lang in self.supported_languages:
            target_counts[target_lang] += 1
        
        # Sort by count
        sorted_targets = sorted(target_counts.items(), key=lambda x: x[1])
        
        # Format for report
        result = []
        for i, (lang, count) in enumerate(sorted_targets[:count], 1):
            lang_name = COMMON_LANGUAGES.get(lang, "Unknown")
            result.append(f"{i}. {lang} ({lang_name}): {count} source languages")
        
        return result
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        logger.info("Cleaning up resources...")
        
        if self.processor:
            try:
                await self.processor.shutdown()
            except Exception as e:
                logger.warning(f"Error during processor shutdown: {e}")

async def main():
    """Main entry point for language coverage test script."""
    parser = argparse.ArgumentParser(description="CasaLingua Language Coverage Test")
    
    parser.add_argument(
        "--config", 
        type=str, 
        help="Path to configuration file"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="language_coverage_results", 
        help="Directory for test results"
    )
    parser.add_argument(
        "--languages", 
        type=str, 
        help="Comma-separated list of language codes to test (default: common languages)"
    )
    parser.add_argument(
        "--detection-only", 
        action="store_true", 
        help="Only test language detection, not translation"
    )
    
    args = parser.parse_args()
    
    # Parse languages if provided
    languages = None
    if args.languages:
        languages = args.languages.split(",")
    
    # Initialize test
    coverage_test = LanguageCoverageTest(
        config_path=args.config,
        output_dir=args.output_dir,
        languages=languages
    )
    
    try:
        # Initialize components
        await coverage_test.initialize()
        
        if args.detection_only:
            # Only test language detection
            await coverage_test.test_language_detection()
        else:
            # Run comprehensive test
            await coverage_test.test_language_support()
        
    finally:
        # Clean up
        await coverage_test.cleanup()

if __name__ == "__main__":
    asyncio.run(main())