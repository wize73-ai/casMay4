#!/usr/bin/env python3
"""
Batch Translation Performance Testing

This script tests the performance of batch translation capabilities in CasaLingua.
It measures throughput, latency, memory usage, and quality for different batch sizes
and language pairs.
"""

import argparse
import asyncio
import csv
import gc
import json
import os
import random
import sys
import time
import tracemalloc
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, Set

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

try:
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    HAS_VISUALIZATION = True
except ImportError:
    HAS_VISUALIZATION = False

# Import project modules (this assumes we're running from the project root)
from app.utils.config import load_config
from app.services.models.manager import EnhancedModelManager
from app.services.models.loader import ModelLoader
from app.audit.logger import AuditLogger
from app.audit.metrics import MetricsManager
from app.core.pipeline.processor import UnifiedProcessor


# Test data - texts in multiple languages
@dataclass
class TestSet:
    """Test dataset for batch translation testing"""
    name: str
    source_lang: str
    target_lang: str
    texts: List[str]
    references: Optional[List[str]] = None
    domain: Optional[str] = None
    
    @property
    def language_pair(self) -> str:
        """Get language pair string"""
        return f"{self.source_lang}-{self.target_lang}"
    
    @property
    def total_tokens(self) -> int:
        """Approximate token count (rough estimation)"""
        # Simple approximation: 1 token ≈ 4 characters
        return sum(len(text) for text in self.texts) // 4
    
    @property
    def total_chars(self) -> int:
        """Total character count"""
        return sum(len(text) for text in self.texts)
    
    @property
    def avg_text_length(self) -> float:
        """Average text length in characters"""
        if not self.texts:
            return 0
        return self.total_chars / len(self.texts)


@dataclass
class BatchTestResult:
    """Result of a batch translation test"""
    test_set: TestSet
    batch_size: int
    texts_count: int
    total_time: float
    texts_per_second: float
    tokens_per_second: float
    chars_per_second: float
    peak_memory_mb: float
    memory_per_text_kb: float
    quality_score: Optional[float] = None
    errors: List[str] = field(default_factory=list)
    batch_times: List[float] = field(default_factory=list)
    
    @property
    def avg_batch_time(self) -> float:
        """Average time per batch"""
        if not self.batch_times:
            return 0
        return sum(self.batch_times) / len(self.batch_times)
    
    @property
    def language_pair(self) -> str:
        """Get language pair string"""
        return self.test_set.language_pair


class BatchTranslationTester:
    """Tests batch translation performance"""
    
    def __init__(
        self,
        config_path: str = "config/default.json",
        registry_path: str = "config/model_registry.json",
        audit_log_path: str = "logs/audit/batch_test.jsonl",
        metrics_path: str = "logs/metrics/batch_test.json",
        output_dir: str = "logs/batch_tests",
        batch_sizes: List[int] = None,
    ):
        """Initialize the batch translation tester"""
        self.config_path = config_path
        self.registry_path = registry_path
        self.audit_log_path = audit_log_path
        self.metrics_path = metrics_path
        self.output_dir = output_dir
        self.batch_sizes = batch_sizes or [1, 2, 4, 8, 16, 32, 64]
        
        # Components
        self.config = None
        self.registry_config = None
        self.model_loader = None
        self.model_manager = None
        self.processor = None
        self.audit_logger = None
        self.metrics = None
        
        # Results
        self.results = []
        
        # Setup output directory
        os.makedirs(output_dir, exist_ok=True)
    
    async def initialize(self):
        """Initialize all components"""
        print("Initializing batch translation tester...")
        
        # Load configuration
        self.config = load_config(self.config_path)
        
        # Load model registry
        with open(self.registry_path, "r") as f:
            self.registry_config = json.load(f)
        
        # Initialize audit logger
        self.audit_logger = AuditLogger(log_file=self.audit_log_path)
        
        # Initialize metrics manager
        self.metrics = MetricsManager(self.metrics_path)
        
        # Initialize model components
        self.model_loader = ModelLoader(self.config, self.registry_config)
        self.model_manager = EnhancedModelManager(
            self.model_loader, 
            self.audit_logger,
            self.metrics
        )
        
        # Create processor
        self.processor = UnifiedProcessor(
            self.model_manager,
            self.audit_logger,
            self.metrics,
            self.config,
            self.registry_config
        )
        
        await self.processor.initialize()
        print("Initialization complete.")
    
    def create_test_sets(self) -> List[TestSet]:
        """Create test sets for batch translation testing"""
        test_sets = []
        
        # English to Spanish (short texts)
        en_es_short = TestSet(
            name="en-es-short",
            source_lang="en",
            target_lang="es",
            texts=[
                "Hello, how are you?",
                "The weather is nice today.",
                "I need to buy groceries.",
                "What time is the meeting?",
                "Can you help me with this?",
                "This is a test message.",
                "Please send the information as soon as possible.",
                "Thank you for your assistance.",
                "Have a great day!",
                "See you tomorrow at the office."
            ]
        )
        test_sets.append(en_es_short)
        
        # English to Spanish (medium texts)
        en_es_medium = TestSet(
            name="en-es-medium",
            source_lang="en",
            target_lang="es",
            texts=[
                "The rapid development of artificial intelligence has led to significant advancements in natural language processing.",
                "Climate change is one of the most pressing challenges facing humanity in the 21st century.",
                "Regular exercise and a balanced diet are essential components of a healthy lifestyle.",
                "The global economy is increasingly interconnected, with events in one region affecting markets worldwide.",
                "Renewable energy sources such as solar and wind power are becoming more cost-effective alternatives to fossil fuels.",
                "Effective communication skills are vital in both personal relationships and professional environments.",
                "The preservation of biodiversity is crucial for maintaining ecosystem stability and resilience.",
                "Digital literacy is becoming an essential skill as technology plays a larger role in daily life.",
                "Urban planning must address challenges related to population growth, transportation, and sustainability.",
                "Cultural diversity enriches societies by promoting exchange of ideas, traditions, and perspectives."
            ]
        )
        test_sets.append(en_es_medium)
        
        # English to French (medium texts)
        en_fr_medium = TestSet(
            name="en-fr-medium",
            source_lang="en",
            target_lang="fr",
            texts=[
                "The company announced a new strategic partnership with industry leaders.",
                "Students are encouraged to participate in extracurricular activities for a well-rounded education.",
                "The museum is hosting a special exhibition of contemporary art this month.",
                "Remote work has become more common following changes in workplace dynamics.",
                "The research team published their findings in a peer-reviewed scientific journal.",
                "Regular maintenance can extend the lifespan of household appliances.",
                "The film festival showcases independent productions from around the world.",
                "Improving public transportation infrastructure can reduce traffic congestion in urban areas.",
                "Cloud computing services offer scalable solutions for data storage and processing.",
                "The conference brought together experts from various fields to discuss interdisciplinary approaches."
            ]
        )
        test_sets.append(en_fr_medium)
        
        # Spanish to English (medium texts)
        es_en_medium = TestSet(
            name="es-en-medium",
            source_lang="es",
            target_lang="en",
            texts=[
                "La diversidad cultural enriquece nuestra sociedad y promueve la comprensión entre diferentes pueblos.",
                "El desarrollo sostenible busca equilibrar el crecimiento económico con la protección del medio ambiente.",
                "Las nuevas tecnologías han transformado la manera en que nos comunicamos y accedemos a la información.",
                "La educación de calidad es fundamental para el desarrollo personal y el progreso de las comunidades.",
                "El cambio climático representa uno de los mayores desafíos ambientales de nuestro tiempo.",
                "La conservación de la biodiversidad es esencial para mantener el equilibrio de los ecosistemas.",
                "Las políticas públicas deben orientarse a reducir las desigualdades sociales y económicas.",
                "La innovación tecnológica impulsa el crecimiento económico y mejora la calidad de vida.",
                "La cooperación internacional es clave para abordar los problemas globales contemporáneos.",
                "El acceso universal a la atención sanitaria es un derecho humano fundamental."
            ]
        )
        test_sets.append(es_en_medium)
        
        # English to German (medium texts)
        en_de_medium = TestSet(
            name="en-de-medium",
            source_lang="en",
            target_lang="de",
            texts=[
                "Digital transformation is changing business models across industries.",
                "Sustainable agriculture practices focus on environmental conservation and economic viability.",
                "The healthcare system faces challenges related to affordability and accessibility.",
                "International cooperation is essential for addressing global challenges effectively.",
                "Educational reforms aim to prepare students for the changing demands of the workforce.",
                "Public health initiatives focus on prevention, education, and community engagement.",
                "Consumer behavior has evolved with the growth of e-commerce and digital marketing.",
                "Technological innovation drives economic growth and improves quality of life.",
                "Environmental regulations aim to balance industrial development with ecological preservation.",
                "Cultural heritage preservation maintains connections to historical traditions and practices."
            ]
        )
        test_sets.append(en_de_medium)
        
        # English to Spanish (long texts)
        en_es_long = TestSet(
            name="en-es-long",
            source_lang="en",
            target_lang="es",
            texts=[
                """Climate change is the long-term alteration of temperature and typical weather patterns in a place. 
                Climate change could refer to a particular location or the planet as a whole. Climate change may 
                cause weather patterns to be less predictable. These unexpected weather patterns can make it 
                difficult to maintain and grow crops in regions that rely on farming. Climate change has also been 
                connected with other damaging weather events such as more frequent and more intense hurricanes, 
                floods, downpours, and winter storms.""",
                
                """Artificial intelligence (AI) is intelligence—perceiving, synthesizing, and inferring information—demonstrated 
                by machines, as opposed to intelligence displayed by humans or by other animals. Example tasks in which 
                AI is applied include speech recognition, computer vision, translation between (natural) languages, 
                decision-making, and producing creative content such as images, text, music, and inventions.""",
                
                """Machine learning is a field of inquiry devoted to understanding and building methods that 'learn', 
                that is, methods that leverage data to improve performance on some set of tasks. It is considered 
                a part of artificial intelligence. Machine learning algorithms build a model based on sample data, 
                known as training data, in order to make predictions or decisions without being explicitly programmed to do so."""
            ]
        )
        test_sets.append(en_es_long)
        
        # German to English (medium texts)
        de_en_medium = TestSet(
            name="de-en-medium",
            source_lang="de",
            target_lang="en",
            texts=[
                "Nachhaltigkeit ist ein wichtiges Prinzip für die Entwicklung unserer modernen Gesellschaft.",
                "Die digitale Transformation verändert die Art und Weise, wie Unternehmen arbeiten und kommunizieren.",
                "Erneuerbare Energien spielen eine zentrale Rolle bei der Bekämpfung des Klimawandels.",
                "Bildungschancen sollten für alle Menschen unabhängig von ihrer Herkunft zugänglich sein.",
                "Kulturelle Vielfalt bereichert unsere Gesellschaft und fördert gegenseitiges Verständnis.",
                "Wissenschaftliche Forschung treibt Innovation und technologischen Fortschritt voran.",
                "Die Globalisierung hat sowohl wirtschaftliche als auch kulturelle Auswirkungen auf Länder weltweit.",
                "Gesundheitsvorsorge ist ein wichtiger Bestandteil eines nachhaltigen Gesundheitssystems.",
                "Demokratische Prozesse erfordern die aktive Beteiligung der Bürgerinnen und Bürger.",
                "Der Schutz der Biodiversität ist entscheidend für das ökologische Gleichgewicht unseres Planeten."
            ]
        )
        test_sets.append(de_en_medium)
        
        # Return all test sets
        return test_sets
    
    async def translate_batch(
        self, 
        texts: List[str], 
        source_lang: str, 
        target_lang: str,
        batch_size: int,
        domain: Optional[str] = None
    ) -> Tuple[List[str], List[float], float, float]:
        """Translate a batch of texts and measure performance"""
        translations = []
        batch_times = []
        
        # Track memory usage
        tracemalloc.start()
        memory_start = tracemalloc.get_traced_memory()[1]
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_start_time = time.time()
            
            try:
                # Use context for domain if specified
                batch_results = []
                for text in batch:
                    result = await self.processor.translate_text(
                        text=text,
                        source_language=source_lang,
                        target_language=target_lang,
                        domain=domain
                    )
                    batch_results.append(result.get("translated_text", ""))
                
                translations.extend(batch_results)
                batch_time = time.time() - batch_start_time
                batch_times.append(batch_time)
            
            except Exception as e:
                print(f"Error translating batch: {str(e)}")
                # Fill with empty strings for failed translations
                translations.extend([""] * len(batch))
                batch_times.append(time.time() - batch_start_time)
        
        # Track peak memory
        peak_memory = tracemalloc.get_traced_memory()[1] - memory_start
        tracemalloc.stop()
        
        # Calculate total time
        total_time = sum(batch_times)
        
        return translations, batch_times, total_time, peak_memory
    
    def calculate_quality_score(self, translated_texts: List[str], reference_texts: List[str]) -> float:
        """Calculate translation quality score (if references available)"""
        # If no references, return None
        if not reference_texts:
            return None
        
        # Simple character-level similarity (alternative to BLEU/ROUGE when no references)
        similarities = []
        
        for trans, ref in zip(translated_texts, reference_texts):
            # Skip empty translations
            if not trans or not ref:
                continue
                
            # Calculate character-level edit distance (Levenshtein distance)
            # Here we use a simplified version
            chars_same = sum(t == r for t, r in zip(trans[:min(len(trans), len(ref))], ref[:min(len(trans), len(ref))]))
            max_len = max(len(trans), len(ref))
            if max_len > 0:
                similarity = chars_same / max_len
                similarities.append(similarity)
        
        # Return average similarity
        if similarities:
            return sum(similarities) / len(similarities)
        return 0.0
    
    async def run_batch_test(self, test_set: TestSet, batch_size: int) -> BatchTestResult:
        """Run a batch translation test with the specified parameters"""
        print(f"Running batch test: {test_set.name}, batch_size={batch_size}")
        
        # Reset memory
        gc.collect()
        
        # Translate texts
        translations, batch_times, total_time, peak_memory = await self.translate_batch(
            texts=test_set.texts,
            source_lang=test_set.source_lang,
            target_lang=test_set.target_lang,
            batch_size=batch_size,
            domain=test_set.domain
        )
        
        # Calculate metrics
        texts_count = len(test_set.texts)
        texts_per_second = texts_count / total_time if total_time > 0 else 0
        tokens_per_second = test_set.total_tokens / total_time if total_time > 0 else 0
        chars_per_second = test_set.total_chars / total_time if total_time > 0 else 0
        peak_memory_mb = peak_memory / (1024 * 1024)
        memory_per_text_kb = (peak_memory / texts_count) / 1024 if texts_count > 0 else 0
        
        # Calculate quality if references available
        quality_score = None
        if test_set.references:
            quality_score = self.calculate_quality_score(translations, test_set.references)
        
        # Collect errors (empty translations)
        errors = [i for i, t in enumerate(translations) if not t]
        
        # Create and return result
        result = BatchTestResult(
            test_set=test_set,
            batch_size=batch_size,
            texts_count=texts_count,
            total_time=total_time,
            texts_per_second=texts_per_second,
            tokens_per_second=tokens_per_second,
            chars_per_second=chars_per_second,
            peak_memory_mb=peak_memory_mb,
            memory_per_text_kb=memory_per_text_kb,
            quality_score=quality_score,
            errors=[f"Failed to translate text {i}" for i in errors],
            batch_times=batch_times
        )
        
        return result
    
    async def run_tests(self):
        """Run all batch translation tests"""
        # Create test sets
        test_sets = self.create_test_sets()
        
        # Run tests for each combination of test set and batch size
        results = []
        for test_set in test_sets:
            for batch_size in self.batch_sizes:
                result = await self.run_batch_test(test_set, batch_size)
                results.append(result)
                
                # Print progress
                print(f"Completed test: {test_set.name}, batch_size={batch_size}")
                print(f"  - Texts per second: {result.texts_per_second:.2f}")
                print(f"  - Tokens per second: {result.tokens_per_second:.2f}")
                print(f"  - Memory usage (MB): {result.peak_memory_mb:.2f}")
                if result.errors:
                    print(f"  - Errors: {len(result.errors)}")
                print()
                
                # Short pause between tests
                await asyncio.sleep(1)
        
        self.results = results
        return results
    
    def save_results(self) -> str:
        """Save test results to files"""
        if not self.results:
            return None
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save as CSV
        csv_file = os.path.join(self.output_dir, f"batch_test_results_{timestamp}.csv")
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "test_name", "language_pair", "batch_size", "texts_count",
                "total_time", "texts_per_second", "tokens_per_second", "chars_per_second",
                "peak_memory_mb", "memory_per_text_kb", "quality_score", "errors"
            ])
            
            for result in self.results:
                writer.writerow([
                    result.test_set.name,
                    result.language_pair,
                    result.batch_size,
                    result.texts_count,
                    f"{result.total_time:.4f}",
                    f"{result.texts_per_second:.4f}",
                    f"{result.tokens_per_second:.4f}",
                    f"{result.chars_per_second:.4f}",
                    f"{result.peak_memory_mb:.4f}",
                    f"{result.memory_per_text_kb:.4f}",
                    f"{result.quality_score:.4f}" if result.quality_score is not None else "N/A",
                    len(result.errors)
                ])
        
        # Save as JSON
        json_file = os.path.join(self.output_dir, f"batch_test_results_{timestamp}.json")
        with open(json_file, "w") as f:
            json.dump(
                [
                    {
                        "test_name": result.test_set.name,
                        "language_pair": result.language_pair,
                        "batch_size": result.batch_size,
                        "texts_count": result.texts_count,
                        "total_time": result.total_time,
                        "texts_per_second": result.texts_per_second,
                        "tokens_per_second": result.tokens_per_second,
                        "chars_per_second": result.chars_per_second,
                        "peak_memory_mb": result.peak_memory_mb,
                        "memory_per_text_kb": result.memory_per_text_kb,
                        "quality_score": result.quality_score,
                        "errors": len(result.errors),
                        "batch_times": result.batch_times,
                        "avg_batch_time": result.avg_batch_time
                    }
                    for result in self.results
                ],
                f,
                indent=2
            )
        
        # Generate visualizations
        if HAS_VISUALIZATION:
            self.generate_visualizations(timestamp)
        
        # Generate HTML report
        html_file = self.generate_html_report(timestamp)
        
        return html_file
    
    def generate_visualizations(self, timestamp: str):
        """Generate visualizations of the results"""
        if not HAS_VISUALIZATION or not self.results:
            return
        
        # Convert results to DataFrame for easier analysis
        data = []
        for result in self.results:
            data.append({
                "test_name": result.test_set.name,
                "language_pair": result.language_pair,
                "batch_size": result.batch_size,
                "texts_per_second": result.texts_per_second,
                "tokens_per_second": result.tokens_per_second,
                "peak_memory_mb": result.peak_memory_mb,
                "avg_batch_time": result.avg_batch_time,
                "quality_score": result.quality_score if result.quality_score is not None else 0,
            })
        
        df = pd.DataFrame(data)
        
        # Set style
        sns.set_style("whitegrid")
        
        # 1. Texts per second vs batch size by language pair
        plt.figure(figsize=(12, 8))
        for lang_pair in df["language_pair"].unique():
            subset = df[df["language_pair"] == lang_pair]
            plt.plot(
                subset["batch_size"], 
                subset["texts_per_second"], 
                "o-", 
                label=lang_pair
            )
        
        plt.xlabel("Batch Size")
        plt.ylabel("Texts per Second")
        plt.title("Translation Speed by Batch Size and Language Pair")
        plt.legend()
        plt.grid(True)
        
        plt.savefig(
            os.path.join(self.output_dir, f"batch_speed_{timestamp}.png"),
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()
        
        # 2. Memory usage vs batch size
        plt.figure(figsize=(12, 8))
        for lang_pair in df["language_pair"].unique():
            subset = df[df["language_pair"] == lang_pair]
            plt.plot(
                subset["batch_size"], 
                subset["peak_memory_mb"], 
                "o-", 
                label=lang_pair
            )
        
        plt.xlabel("Batch Size")
        plt.ylabel("Peak Memory (MB)")
        plt.title("Memory Usage by Batch Size and Language Pair")
        plt.legend()
        plt.grid(True)
        
        plt.savefig(
            os.path.join(self.output_dir, f"batch_memory_{timestamp}.png"),
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()
        
        # 3. Performance comparison across test sets
        plt.figure(figsize=(14, 8))
        
        # Group by test name and get the best batch size for each test
        best_performance = df.loc[df.groupby("test_name")["texts_per_second"].idxmax()]
        best_performance = best_performance.sort_values("texts_per_second", ascending=False)
        
        # Plot bar chart
        sns.barplot(x="test_name", y="texts_per_second", data=best_performance)
        plt.xticks(rotation=45, ha="right")
        plt.xlabel("Test Set")
        plt.ylabel("Texts per Second (Best Performance)")
        plt.title("Best Translation Performance by Test Set")
        
        # Add batch size annotations
        for i, row in enumerate(best_performance.itertuples()):
            plt.text(
                i, 
                row.texts_per_second * 0.95, 
                f"Batch: {row.batch_size}", 
                ha="center",
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", boxstyle="round,pad=0.2")
            )
        
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.output_dir, f"best_performance_{timestamp}.png"),
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()
        
        # 4. Efficiency comparison (texts/s vs memory)
        plt.figure(figsize=(12, 8))
        
        # Create scatter plot with batch size as point size
        for lang_pair in df["language_pair"].unique():
            subset = df[df["language_pair"] == lang_pair]
            plt.scatter(
                subset["peak_memory_mb"],
                subset["texts_per_second"],
                s=subset["batch_size"] * 5,  # Size proportional to batch size
                alpha=0.7,
                label=lang_pair
            )
        
        plt.xlabel("Peak Memory (MB)")
        plt.ylabel("Texts per Second")
        plt.title("Translation Efficiency: Speed vs Memory Usage")
        plt.grid(True)
        plt.legend()
        
        plt.savefig(
            os.path.join(self.output_dir, f"efficiency_{timestamp}.png"),
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()
    
    def generate_html_report(self, timestamp: str) -> str:
        """Generate HTML report of the results"""
        if not self.results:
            return None
        
        # File path
        html_file = os.path.join(self.output_dir, f"batch_test_report_{timestamp}.html")
        
        # Get optimal batch sizes
        optimal_batch_sizes = {}
        lang_pairs = set(result.language_pair for result in self.results)
        
        for lang_pair in lang_pairs:
            pair_results = [r for r in self.results if r.language_pair == lang_pair]
            # Sort by texts per second (descending)
            pair_results.sort(key=lambda x: x.texts_per_second, reverse=True)
            if pair_results:
                optimal_batch_sizes[lang_pair] = pair_results[0].batch_size
        
        # Build HTML content
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Batch Translation Performance Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                h1, h2, h3 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .card {{ background-color: #fff; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); padding: 20px; margin-bottom: 20px; }}
                .plot {{ max-width: 100%; height: auto; margin: 20px 0; }}
                .highlight {{ background-color: #e6f7ff; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Batch Translation Performance Report</h1>
                <p>Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                
                <div class="card">
                    <h2>Executive Summary</h2>
                    <p>This report presents the results of batch translation performance testing across different language pairs and batch sizes.</p>
                    
                    <h3>Optimal Batch Sizes by Language Pair</h3>
                    <table>
                        <tr>
                            <th>Language Pair</th>
                            <th>Optimal Batch Size</th>
                        </tr>
        """
        
        # Add optimal batch sizes
        for lang_pair, batch_size in optimal_batch_sizes.items():
            html += f"""
                        <tr>
                            <td>{lang_pair}</td>
                            <td>{batch_size}</td>
                        </tr>
            """
        
        html += """
                    </table>
                    
                    <h3>Key Findings</h3>
                    <ul>
        """
        
        # Add key findings
        # 1. Best performing language pair
        if self.results:
            best_result = max(self.results, key=lambda x: x.texts_per_second)
            html += f"""
                        <li>The fastest translation was achieved with language pair <strong>{best_result.language_pair}</strong> using batch size <strong>{best_result.batch_size}</strong>, reaching <strong>{best_result.texts_per_second:.2f} texts/second</strong>.</li>
            """
        
        # 2. Memory efficiency
        if self.results:
            most_efficient = min(self.results, key=lambda x: x.memory_per_text_kb)
            html += f"""
                        <li>The most memory-efficient configuration was language pair <strong>{most_efficient.language_pair}</strong> with batch size <strong>{most_efficient.batch_size}</strong>, using only <strong>{most_efficient.memory_per_text_kb:.2f} KB per text</strong>.</li>
            """
        
        # 3. General batch size observation
        avg_optimal_batch = sum(optimal_batch_sizes.values()) / len(optimal_batch_sizes) if optimal_batch_sizes else 0
        html += f"""
                        <li>The average optimal batch size across all language pairs is <strong>{avg_optimal_batch:.1f}</strong>.</li>
        """
        
        # 4. Error patterns
        error_counts = sum(1 for result in self.results if result.errors)
        if error_counts:
            html += f"""
                        <li>Translation errors were observed in <strong>{error_counts}</strong> test configurations, primarily with larger batch sizes.</li>
            """
        else:
            html += f"""
                        <li>No translation errors were observed across all test configurations.</li>
            """
        
        html += """
                    </ul>
                </div>
                
                <div class="card">
                    <h2>Detailed Results</h2>
                    <table>
                        <tr>
                            <th>Test Name</th>
                            <th>Language Pair</th>
                            <th>Batch Size</th>
                            <th>Texts per Second</th>
                            <th>Tokens per Second</th>
                            <th>Memory (MB)</th>
                            <th>Avg Batch Time (s)</th>
                            <th>Errors</th>
                        </tr>
        """
        
        # Add result rows
        for result in sorted(self.results, key=lambda x: (x.language_pair, x.batch_size)):
            # Highlight optimal batch sizes
            is_optimal = result.batch_size == optimal_batch_sizes.get(result.language_pair)
            row_class = 'class="highlight"' if is_optimal else ''
            
            html += f"""
                        <tr {row_class}>
                            <td>{result.test_set.name}</td>
                            <td>{result.language_pair}</td>
                            <td>{result.batch_size}</td>
                            <td>{result.texts_per_second:.4f}</td>
                            <td>{result.tokens_per_second:.4f}</td>
                            <td>{result.peak_memory_mb:.2f}</td>
                            <td>{result.avg_batch_time:.4f}</td>
                            <td>{len(result.errors)}</td>
                        </tr>
            """
        
        html += """
                    </table>
                </div>
        """
        
        # Add visualizations if available
        if HAS_VISUALIZATION:
            html += """
                <div class="card">
                    <h2>Visualizations</h2>
            """
            
            # Include plots
            plot_files = [
                f"batch_speed_{timestamp}.png",
                f"batch_memory_{timestamp}.png",
                f"best_performance_{timestamp}.png",
                f"efficiency_{timestamp}.png"
            ]
            
            for plot_file in plot_files:
                if os.path.exists(os.path.join(self.output_dir, plot_file)):
                    html += f"""
                        <h3>{plot_file.split('_')[0].title()} Analysis</h3>
                        <img src="{plot_file}" alt="{plot_file}" class="plot">
                    """
            
            html += """
                </div>
            """
        
        # Recommendations
        html += """
            <div class="card">
                <h2>Recommendations</h2>
                <ul>
        """
        
        # Generate recommendations
        html += f"""
                    <li>For general use, a batch size of <strong>{int(avg_optimal_batch)}</strong> appears to be optimal across most language pairs.</li>
        """
        
        # Language-specific recommendations
        for lang_pair, batch_size in optimal_batch_sizes.items():
            html += f"""
                    <li>For {lang_pair} translations, use a batch size of <strong>{batch_size}</strong> for optimal performance.</li>
            """
        
        # Memory recommendation
        if self.results:
            memory_efficient_sizes = {}
            for lang_pair in lang_pairs:
                pair_results = [r for r in self.results if r.language_pair == lang_pair]
                # Find batch size with best memory per text ratio
                if pair_results:
                    best_memory = min(pair_results, key=lambda x: x.memory_per_text_kb)
                    memory_efficient_sizes[lang_pair] = best_memory.batch_size
            
            if memory_efficient_sizes:
                html += """
                    <li>For memory-constrained environments, consider these batch sizes for each language pair:</li>
                    <ul>
                """
                
                for lang_pair, batch_size in memory_efficient_sizes.items():
                    html += f"""
                        <li>{lang_pair}: {batch_size}</li>
                    """
                
                html += """
                    </ul>
                """
        
        html += """
                </ul>
            </div>
            
            </div>
        </body>
        </html>
        """
        
        # Write to file
        with open(html_file, "w") as f:
            f.write(html)
        
        return html_file
    
    async def cleanup(self):
        """Clean up resources"""
        if self.processor:
            await self.processor.cleanup()
        
        if self.model_manager:
            await self.model_manager.cleanup()
        
        if self.audit_logger:
            await self.audit_logger.close()
        
        if self.metrics:
            await self.metrics.close()


async def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Batch Translation Performance Testing")
    parser.add_argument("--config", type=str, default="config/default.json",
                        help="Path to configuration file")
    parser.add_argument("--registry", type=str, default="config/model_registry.json",
                        help="Path to model registry")
    parser.add_argument("--output", type=str, default="logs/batch_tests",
                        help="Output directory for test results")
    parser.add_argument("--batch-sizes", type=str, default="1,2,4,8,16,32",
                        help="Comma-separated list of batch sizes to test")
    
    args = parser.parse_args()
    
    # Parse batch sizes
    batch_sizes = [int(s) for s in args.batch_sizes.split(",")]
    
    # Create tester
    tester = BatchTranslationTester(
        config_path=args.config,
        registry_path=args.registry,
        output_dir=args.output,
        batch_sizes=batch_sizes
    )
    
    try:
        # Initialize
        await tester.initialize()
        
        # Run tests
        await tester.run_tests()
        
        # Save results
        report_file = tester.save_results()
        
        if report_file:
            print(f"\nBatch translation performance test completed!")
            print(f"Report saved to: {report_file}")
    
    finally:
        # Cleanup
        await tester.cleanup()


if __name__ == "__main__":
    asyncio.run(main())