#!/usr/bin/env python3
"""
Translation Quality Evaluation for CasaLingua

This script evaluates the quality of translations produced by CasaLingua's translation
models using standardized metrics including BLEU, ROUGE, and METEOR. It runs the
evaluation against multiple language pairs using reference translations.

The script:
1. Tests all translation models available in the system
2. Uses standard parallel corpora for evaluation
3. Calculates BLEU, ROUGE-L, and optional METEOR scores
4. Provides detailed per-language-pair analysis
5. Compares different models head-to-head
6. Generates visualizations of quality metrics

Usage:
    python test_translation_quality.py --output-dir results

Author: Exygy Development Team
Version: 1.0.0
License: MIT
"""

import os
import sys
import json
import asyncio
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
import csv
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Add project root to path for imports
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("translation_quality.log")
    ]
)
logger = logging.getLogger("translation_quality")

# Try to import metrics from nltk, fallback to custom implementations
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    logger.warning("NLTK not available, using custom BLEU implementation")

# Try to import rouge from rouge-score package
try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    logger.warning("rouge-score not available, using custom ROUGE implementation")

# Try to import meteor score
try:
    import nltk
    from nltk.translate.meteor_score import meteor_score
    nltk.download('wordnet', quiet=True)
    METEOR_AVAILABLE = True
except ImportError:
    METEOR_AVAILABLE = False
    logger.warning("METEOR metric not available")

# Test datasets for different language pairs
# Each language pair has a list of (source, reference_translation) tuples
TEST_DATASETS = {
    "en-es": [
        ("Hello, how are you today?", "Hola, ¿cómo estás hoy?"),
        ("I need to buy groceries for dinner tonight.", "Necesito comprar comestibles para la cena de esta noche."),
        ("The weather is beautiful, let's go for a walk.", "El clima es hermoso, vamos a dar un paseo."),
        ("What time does the movie start?", "¿A qué hora comienza la película?"),
        ("Can you help me with this problem?", "¿Puedes ayudarme con este problema?")
    ],
    "en-fr": [
        ("Hello, how are you today?", "Bonjour, comment allez-vous aujourd'hui ?"),
        ("I need to buy groceries for dinner tonight.", "J'ai besoin d'acheter des provisions pour le dîner de ce soir."),
        ("The weather is beautiful, let's go for a walk.", "Le temps est magnifique, allons nous promener."),
        ("What time does the movie start?", "À quelle heure commence le film ?"),
        ("Can you help me with this problem?", "Pouvez-vous m'aider avec ce problème ?")
    ],
    "en-de": [
        ("Hello, how are you today?", "Hallo, wie geht es dir heute?"),
        ("I need to buy groceries for dinner tonight.", "Ich muss Lebensmittel für das Abendessen heute Abend kaufen."),
        ("The weather is beautiful, let's go for a walk.", "Das Wetter ist schön, lass uns spazieren gehen."),
        ("What time does the movie start?", "Um wie viel Uhr beginnt der Film?"),
        ("Can you help me with this problem?", "Kannst du mir bei diesem Problem helfen?")
    ],
    "es-en": [
        ("Hola, ¿cómo estás hoy?", "Hello, how are you today?"),
        ("Necesito comprar comestibles para la cena de esta noche.", "I need to buy groceries for dinner tonight."),
        ("El clima es hermoso, vamos a dar un paseo.", "The weather is beautiful, let's go for a walk."),
        ("¿A qué hora comienza la película?", "What time does the movie start?"),
        ("¿Puedes ayudarme con este problema?", "Can you help me with this problem?")
    ],
    "fr-en": [
        ("Bonjour, comment allez-vous aujourd'hui ?", "Hello, how are you today?"),
        ("J'ai besoin d'acheter des provisions pour le dîner de ce soir.", "I need to buy groceries for dinner tonight."),
        ("Le temps est magnifique, allons nous promener.", "The weather is beautiful, let's go for a walk."),
        ("À quelle heure commence le film ?", "What time does the movie start?"),
        ("Pouvez-vous m'aider avec ce problème ?", "Can you help me with this problem?")
    ],
    "de-en": [
        ("Hallo, wie geht es dir heute?", "Hello, how are you today?"),
        ("Ich muss Lebensmittel für das Abendessen heute Abend kaufen.", "I need to buy groceries for dinner tonight."),
        ("Das Wetter ist schön, lass uns spazieren gehen.", "The weather is beautiful, let's go for a walk."),
        ("Um wie viel Uhr beginnt der Film?", "What time does the movie start?"),
        ("Kannst du mir bei diesem Problem helfen?", "Can you help me with this problem?")
    ]
}

# More complex test examples for specific scenarios
CHALLENGE_DATASETS = {
    "idioms": [
        # English idioms with translations
        ("It's raining cats and dogs outside.", {
            "es": "Está lloviendo a cántaros afuera.",
            "fr": "Il pleut des cordes dehors.",
            "de": "Es regnet in Strömen draußen."
        }),
        ("That costs an arm and a leg.", {
            "es": "Eso cuesta un ojo de la cara.",
            "fr": "Ça coûte les yeux de la tête.",
            "de": "Das kostet ein Vermögen."
        })
    ],
    "ambiguity": [
        # Sentences with ambiguous meanings
        ("I saw her duck under the table.", {
            "es": "Vi cómo se agachaba debajo de la mesa.",
            "fr": "Je l'ai vue se baisser sous la table.",
            "de": "Ich sah, wie sie sich unter dem Tisch duckte."
        }),
        ("Time flies like an arrow; fruit flies like a banana.", {
            "es": "El tiempo vuela como una flecha; las moscas de la fruta prefieren un plátano.",
            "fr": "Le temps passe comme une flèche ; les mouches à fruits préfèrent une banane.",
            "de": "Die Zeit verfliegt wie ein Pfeil; Fruchtfliegen mögen Bananen."
        })
    ]
}

@dataclass
class TranslationExample:
    """Class to store a translation example with source and reference."""
    source: str
    reference: str
    category: Optional[str] = None
    
@dataclass
class TranslationResult:
    """Class to store translation result with metrics."""
    source: str
    reference: str
    translation: str
    bleu_score: float = 0.0
    rouge_score: float = 0.0
    meteor_score: Optional[float] = None
    
@dataclass
class ModelEvaluation:
    """Class to store evaluation results for a model and language pair."""
    model_id: str
    language_pair: str
    timestamp: str
    results: List[TranslationResult] = field(default_factory=list)
    avg_bleu: float = 0.0
    avg_rouge: float = 0.0
    avg_meteor: Optional[float] = None
    success_rate: float = 0.0
    
    def calculate_averages(self) -> None:
        """Calculate average scores across all results."""
        if not self.results:
            return
            
        self.avg_bleu = sum(r.bleu_score for r in self.results) / len(self.results)
        self.avg_rouge = sum(r.rouge_score for r in self.results) / len(self.results)
        
        if all(r.meteor_score is not None for r in self.results):
            self.avg_meteor = sum(r.meteor_score for r in self.results) / len(self.results)
        else:
            self.avg_meteor = None
            
        # Calculate success rate (translation completed)
        self.success_rate = sum(1 for r in self.results if r.translation) / len(self.results)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "model_id": self.model_id,
            "language_pair": self.language_pair,
            "timestamp": self.timestamp,
            "avg_bleu": self.avg_bleu,
            "avg_rouge": self.avg_rouge,
            "avg_meteor": self.avg_meteor,
            "success_rate": self.success_rate,
            "results": [asdict(r) for r in self.results]
        }

class CustomBLEU:
    """Simple BLEU score implementation for when NLTK is not available."""
    
    @staticmethod
    def calculate(reference: str, hypothesis: str) -> float:
        """
        Calculate a simplified BLEU score without using NLTK.
        
        Args:
            reference: Reference translation
            hypothesis: Model-generated translation
            
        Returns:
            Simplified BLEU score (0-1)
        """
        # Tokenize into words
        ref_tokens = reference.lower().split()
        hyp_tokens = hypothesis.lower().split()
        
        # Count matching 1-grams (words)
        matches = 0
        for token in hyp_tokens:
            if token in ref_tokens:
                matches += 1
                ref_tokens.remove(token)  # Remove to avoid double counting
        
        # Calculate precision
        precision = matches / len(hyp_tokens) if hyp_tokens else 0
        
        # Length penalty
        brevity_penalty = min(1.0, len(hyp_tokens) / len(reference.split()) if reference.split() else 0)
        
        return precision * brevity_penalty

class CustomROUGE:
    """Simple ROUGE score implementation for when rouge-score is not available."""
    
    @staticmethod
    def calculate(reference: str, hypothesis: str) -> float:
        """
        Calculate a simplified ROUGE-L score without using the rouge-score package.
        
        Args:
            reference: Reference translation
            hypothesis: Model-generated translation
            
        Returns:
            Simplified ROUGE-L score (0-1)
        """
        # Tokenize into words
        ref_tokens = reference.lower().split()
        hyp_tokens = hypothesis.lower().split()
        
        # Find longest common subsequence length
        lcs_length = CustomROUGE._lcs_length(ref_tokens, hyp_tokens)
        
        # Calculate recall, precision, and F1
        if len(ref_tokens) == 0:
            return 0.0
            
        recall = lcs_length / len(ref_tokens)
        precision = lcs_length / len(hyp_tokens) if hyp_tokens else 0
        
        if recall + precision == 0:
            return 0.0
            
        f1 = (2 * recall * precision) / (recall + precision)
        
        return f1
    
    @staticmethod
    def _lcs_length(a: List[str], b: List[str]) -> int:
        """
        Calculate length of longest common subsequence.
        
        Args:
            a: First sequence of tokens
            b: Second sequence of tokens
            
        Returns:
            Length of longest common subsequence
        """
        if not a or not b:
            return 0
            
        # Initialize LCS matrix
        m, n = len(a), len(b)
        lcs = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Fill LCS matrix
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if a[i-1] == b[j-1]:
                    lcs[i][j] = lcs[i-1][j-1] + 1
                else:
                    lcs[i][j] = max(lcs[i-1][j], lcs[i][j-1])
        
        return lcs[m][n]

class TranslationEvaluator:
    """Evaluates translation quality for CasaLingua models."""
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        output_dir: Optional[str] = None,
        use_meteor: bool = False
    ):
        """
        Initialize translation evaluator.
        
        Args:
            config_path: Path to configuration file
            output_dir: Directory for evaluation results
            use_meteor: Whether to use METEOR metric (requires NLTK)
        """
        # Import app modules
        from app.utils.config import load_config
        
        # Load configuration
        self.config = load_config(config_path) if config_path else load_config()
        
        # Set up output directory
        self.output_dir = Path(output_dir) if output_dir else Path("translation_quality_results")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set whether to use METEOR
        self.use_meteor = use_meteor and METEOR_AVAILABLE
        
        # Initialize components
        self.processor = None
        self.model_manager = None
        
        # Set up metrics functions
        if NLTK_AVAILABLE:
            self.calculate_bleu = self._calculate_nltk_bleu
            self.smoothing = SmoothingFunction().method1
        else:
            self.calculate_bleu = CustomBLEU.calculate
            
        if ROUGE_AVAILABLE:
            self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
            self.calculate_rouge = self._calculate_rouge_score
        else:
            self.calculate_rouge = CustomROUGE.calculate
            
        if METEOR_AVAILABLE and self.use_meteor:
            self.calculate_meteor = self._calculate_meteor
        else:
            self.calculate_meteor = lambda r, h: None
        
        # Track evaluation results
        self.evaluations: Dict[str, Dict[str, ModelEvaluation]] = {}
        
    async def initialize(self) -> None:
        """Initialize evaluation components."""
        from app.services.models.loader import ModelLoader, load_registry_config
        from app.services.models.manager import EnhancedModelManager
        from app.core.pipeline.processor import UnifiedProcessor
        from app.audit.logger import AuditLogger
        from app.audit.metrics import MetricsCollector
        
        logger.info("Initializing evaluation components...")
        
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
            logger.info("Evaluation components initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing evaluation components: {e}", exc_info=True)
            raise
    
    def _calculate_nltk_bleu(self, reference: str, hypothesis: str) -> float:
        """
        Calculate BLEU score using NLTK.
        
        Args:
            reference: Reference translation
            hypothesis: Model-generated translation
            
        Returns:
            BLEU score
        """
        ref_tokens = [reference.lower().split()]
        hyp_tokens = hypothesis.lower().split()
        
        # Handle empty hypothesis
        if not hyp_tokens:
            return 0.0
            
        # Calculate BLEU score with smoothing
        try:
            return sentence_bleu(ref_tokens, hyp_tokens, smoothing_function=self.smoothing)
        except Exception as e:
            logger.warning(f"Error calculating BLEU score: {e}")
            return 0.0
    
    def _calculate_rouge_score(self, reference: str, hypothesis: str) -> float:
        """
        Calculate ROUGE-L score using rouge-score.
        
        Args:
            reference: Reference translation
            hypothesis: Model-generated translation
            
        Returns:
            ROUGE-L F1 score
        """
        try:
            scores = self.rouge_scorer.score(reference, hypothesis)
            return scores['rougeL'].fmeasure
        except Exception as e:
            logger.warning(f"Error calculating ROUGE score: {e}")
            return 0.0
    
    def _calculate_meteor(self, reference: str, hypothesis: str) -> Optional[float]:
        """
        Calculate METEOR score using NLTK.
        
        Args:
            reference: Reference translation
            hypothesis: Model-generated translation
            
        Returns:
            METEOR score or None if error
        """
        try:
            return meteor_score([reference.split()], hypothesis.split())
        except Exception as e:
            logger.warning(f"Error calculating METEOR score: {e}")
            return None
    
    async def evaluate_model(
        self,
        model_id: str,
        language_pairs: List[str],
        test_data: Dict[str, List[Tuple[str, str]]] = TEST_DATASETS
    ) -> Dict[str, ModelEvaluation]:
        """
        Evaluate a translation model on multiple language pairs.
        
        Args:
            model_id: Model ID to evaluate
            language_pairs: List of language pairs to test (e.g., ["en-es", "fr-en"])
            test_data: Test dataset with source-reference pairs
            
        Returns:
            Dictionary of evaluation results by language pair
        """
        logger.info(f"Evaluating model {model_id} on {len(language_pairs)} language pairs")
        
        # Check that processor is initialized
        if not self.processor:
            raise ValueError("Processor not initialized. Call initialize() first.")
            
        # Initialize results dictionary
        timestamp = datetime.now().isoformat()
        results = {}
        
        # Evaluate each language pair
        for pair in language_pairs:
            if pair not in test_data:
                logger.warning(f"No test data for language pair {pair}, skipping")
                continue
                
            source_lang, target_lang = pair.split("-")
            
            # Create model evaluation object
            evaluation = ModelEvaluation(
                model_id=model_id,
                language_pair=pair,
                timestamp=timestamp
            )
            
            # Process each test example
            for source, reference in tqdm(test_data[pair], desc=f"Testing {pair}"):
                # Translate source text
                try:
                    translation_result = await self.processor.process_translation(
                        text=source,
                        source_language=source_lang,
                        target_language=target_lang,
                        model_id=model_id
                    )
                    
                    # Extract translated text
                    translated_text = translation_result.get("translated_text", "")
                    
                    # Skip empty translations
                    if not translated_text:
                        logger.warning(f"Empty translation for source: {source}")
                        translated_text = ""
                    
                    # Calculate metrics
                    bleu = self.calculate_bleu(reference, translated_text)
                    rouge = self.calculate_rouge(reference, translated_text)
                    meteor = self.calculate_meteor(reference, translated_text) if self.use_meteor else None
                    
                    # Create result
                    result = TranslationResult(
                        source=source,
                        reference=reference,
                        translation=translated_text,
                        bleu_score=bleu,
                        rouge_score=rouge,
                        meteor_score=meteor
                    )
                    
                    # Add to evaluation
                    evaluation.results.append(result)
                    
                except Exception as e:
                    logger.error(f"Error translating with {model_id} ({pair}): {e}")
                    
                    # Add failed result
                    result = TranslationResult(
                        source=source,
                        reference=reference,
                        translation="",
                        bleu_score=0.0,
                        rouge_score=0.0,
                        meteor_score=None
                    )
                    evaluation.results.append(result)
            
            # Calculate average scores
            evaluation.calculate_averages()
            
            # Add to results
            results[pair] = evaluation
            
            # Log results
            logger.info(f"Results for {model_id} on {pair}:")
            logger.info(f"  Average BLEU: {evaluation.avg_bleu:.4f}")
            logger.info(f"  Average ROUGE: {evaluation.avg_rouge:.4f}")
            if evaluation.avg_meteor is not None:
                logger.info(f"  Average METEOR: {evaluation.avg_meteor:.4f}")
            logger.info(f"  Success Rate: {evaluation.success_rate:.2f}")
            
            # Save individual result
            self._save_evaluation(evaluation, f"{model_id}_{pair}")
        
        # Store results
        self.evaluations[model_id] = results
        
        return results
    
    async def evaluate_all_models(
        self,
        model_ids: List[str],
        language_pairs: List[str],
        challenge_data: bool = False
    ) -> Dict[str, Dict[str, ModelEvaluation]]:
        """
        Evaluate multiple translation models.
        
        Args:
            model_ids: List of model IDs to evaluate
            language_pairs: List of language pairs to test
            challenge_data: Whether to include challenging test cases
            
        Returns:
            Dictionary of evaluation results by model ID and language pair
        """
        logger.info(f"Evaluating {len(model_ids)} models on {len(language_pairs)} language pairs")
        
        # Check that processor is initialized
        if not self.processor:
            raise ValueError("Processor not initialized. Call initialize() first.")
        
        # Merge standard and challenge data if requested
        test_data = TEST_DATASETS.copy()
        
        if challenge_data:
            # Convert challenge datasets to standard format
            for category, examples in CHALLENGE_DATASETS.items():
                for source, translations in examples:
                    for target_lang, reference in translations.items():
                        pair = f"en-{target_lang}"
                        if pair in test_data:
                            # Add to existing data
                            test_data[pair].append((source, reference))
                        # else:
                            # Skip language pairs not in test data
        
        # Evaluate each model
        all_results = {}
        for model_id in model_ids:
            results = await self.evaluate_model(model_id, language_pairs, test_data)
            all_results[model_id] = results
        
        # Generate comparison report
        self._generate_comparison_report(all_results, language_pairs)
        
        return all_results
    
    def _save_evaluation(self, evaluation: ModelEvaluation, filename_prefix: str) -> None:
        """
        Save evaluation results to file.
        
        Args:
            evaluation: Evaluation results
            filename_prefix: Prefix for output files
        """
        # Save detailed results as JSON
        json_path = self.output_dir / f"{filename_prefix}_results.json"
        with open(json_path, "w") as f:
            json.dump(evaluation.to_dict(), f, indent=2)
        
        # Save summary as CSV
        csv_path = self.output_dir / f"{filename_prefix}_summary.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Source", "Reference", "Translation", "BLEU", "ROUGE-L", "METEOR"])
            
            for result in evaluation.results:
                writer.writerow([
                    result.source,
                    result.reference,
                    result.translation,
                    result.bleu_score,
                    result.rouge_score,
                    result.meteor_score if result.meteor_score is not None else "N/A"
                ])
    
    def _generate_comparison_report(
        self,
        results: Dict[str, Dict[str, ModelEvaluation]],
        language_pairs: List[str]
    ) -> None:
        """
        Generate a comparison report of multiple models.
        
        Args:
            results: Dictionary of evaluation results by model and language pair
            language_pairs: List of language pairs tested
        """
        logger.info("Generating comparison report...")
        
        # Save comparison data as CSV
        csv_path = self.output_dir / f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            
            # Write header
            header = ["Model ID", "Language Pair", "Avg BLEU", "Avg ROUGE-L"]
            if self.use_meteor:
                header.append("Avg METEOR")
            header.append("Success Rate")
            writer.writerow(header)
            
            # Write data for each model and language pair
            for model_id, model_results in results.items():
                for pair, evaluation in model_results.items():
                    row = [
                        model_id,
                        pair,
                        evaluation.avg_bleu,
                        evaluation.avg_rouge
                    ]
                    if self.use_meteor:
                        row.append(evaluation.avg_meteor if evaluation.avg_meteor is not None else "N/A")
                    row.append(evaluation.success_rate)
                    writer.writerow(row)
        
        logger.info(f"Comparison report saved to {csv_path}")
        
        # Generate visualizations
        self._generate_visualizations(results, language_pairs)
    
    def _generate_visualizations(
        self,
        results: Dict[str, Dict[str, ModelEvaluation]],
        language_pairs: List[str]
    ) -> None:
        """
        Generate visualizations of evaluation results.
        
        Args:
            results: Dictionary of evaluation results by model and language pair
            language_pairs: List of language pairs tested
        """
        logger.info("Generating visualizations...")
        
        # Only generate visualizations if matplotlib is available
        if 'plt' not in globals():
            logger.warning("Matplotlib not available, skipping visualizations")
            return
        
        # Set up data for plots
        model_ids = list(results.keys())
        metrics = ["BLEU", "ROUGE-L"]
        if self.use_meteor:
            metrics.append("METEOR")
        
        # Create bar charts for each language pair
        for pair in language_pairs:
            if pair not in next(iter(results.values()), {}):
                continue
            
            # Create figure with subplots for each metric
            fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 6))
            if len(metrics) == 1:
                axes = [axes]
            
            # Plot data for each metric
            for i, metric in enumerate(metrics):
                # Extract scores for this metric
                scores = []
                for model_id in model_ids:
                    if pair in results[model_id]:
                        if metric == "BLEU":
                            scores.append(results[model_id][pair].avg_bleu)
                        elif metric == "ROUGE-L":
                            scores.append(results[model_id][pair].avg_rouge)
                        elif metric == "METEOR" and results[model_id][pair].avg_meteor is not None:
                            scores.append(results[model_id][pair].avg_meteor)
                        else:
                            scores.append(0)
                
                # Create bar chart
                axes[i].bar(model_ids, scores)
                axes[i].set_title(f"{metric} Score - {pair}")
                axes[i].set_xlabel("Model")
                axes[i].set_ylabel(f"{metric} Score")
                axes[i].set_ylim(0, 1.0)
                
                # Add values above bars
                for j, score in enumerate(scores):
                    axes[i].text(j, score + 0.02, f"{score:.3f}", ha="center")
                
                # Rotate x-axis labels if needed
                if len(model_ids) > 3:
                    axes[i].set_xticklabels(model_ids, rotation=45, ha="right")
            
            plt.tight_layout()
            
            # Save figure
            fig_path = self.output_dir / f"{pair}_comparison.png"
            plt.savefig(fig_path)
            plt.close()
        
        # Create overall comparison chart
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Set up bar positions
        x = np.arange(len(language_pairs))
        width = 0.8 / len(model_ids)
        
        # Plot BLEU scores for each model and language pair
        for i, model_id in enumerate(model_ids):
            bleu_scores = []
            for pair in language_pairs:
                if pair in results[model_id]:
                    bleu_scores.append(results[model_id][pair].avg_bleu)
                else:
                    bleu_scores.append(0)
            
            ax.bar(x + i * width - width * len(model_ids) / 2, bleu_scores, width, label=model_id)
        
        ax.set_title("BLEU Score Comparison Across Language Pairs")
        ax.set_xlabel("Language Pair")
        ax.set_ylabel("BLEU Score")
        ax.set_xticks(x)
        ax.set_xticklabels(language_pairs)
        ax.set_ylim(0, 1.0)
        ax.legend()
        
        plt.tight_layout()
        
        # Save figure
        fig_path = self.output_dir / f"overall_bleu_comparison.png"
        plt.savefig(fig_path)
        plt.close()
        
        logger.info(f"Visualizations saved to {self.output_dir}")
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        logger.info("Cleaning up resources...")
        
        if self.processor:
            try:
                await self.processor.shutdown()
            except Exception as e:
                logger.warning(f"Error during processor shutdown: {e}")

async def main():
    """Main entry point for translation quality evaluation script."""
    parser = argparse.ArgumentParser(description="CasaLingua Translation Quality Evaluation")
    
    parser.add_argument(
        "--config", 
        type=str, 
        help="Path to configuration file"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="translation_quality_results", 
        help="Directory for evaluation results"
    )
    parser.add_argument(
        "--model-ids",
        type=str,
        help="Comma-separated list of model IDs to evaluate"
    )
    parser.add_argument(
        "--language-pairs",
        type=str,
        default="en-es,en-fr,en-de",
        help="Comma-separated list of language pairs to test"
    )
    parser.add_argument(
        "--use-meteor",
        action="store_true",
        help="Use METEOR metric for evaluation (requires NLTK)"
    )
    parser.add_argument(
        "--challenge-data",
        action="store_true",
        help="Include challenging test cases"
    )
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = TranslationEvaluator(
        config_path=args.config,
        output_dir=args.output_dir,
        use_meteor=args.use_meteor
    )
    
    try:
        # Initialize components
        await evaluator.initialize()
        
        # Parse language pairs
        language_pairs = args.language_pairs.split(",")
        
        # Get model IDs
        if args.model_ids:
            model_ids = args.model_ids.split(",")
        else:
            # Use default models
            model_ids = ["translation_model", "translation_small"]
        
        # Run evaluation
        await evaluator.evaluate_all_models(
            model_ids=model_ids,
            language_pairs=language_pairs,
            challenge_data=args.challenge_data
        )
        
    finally:
        # Clean up
        await evaluator.cleanup()

if __name__ == "__main__":
    asyncio.run(main())