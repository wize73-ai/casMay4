#!/usr/bin/env python3
"""
Comprehensive Performance Benchmarking Script for CasaLingua

This script performs extensive performance benchmarking across all models in the
CasaLingua language processing pipeline. It tests translation, simplification,
anonymization, language detection, summarization and other model capabilities,
measuring execution time, memory usage, and throughput for each operation.

Usage:
    python test_model_performance.py --output-dir results --test-all

Author: Exygy Development Team
Version: 1.0.0
License: MIT
"""

import os
import sys
import time
import json
import asyncio
import argparse
import psutil
import csv
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
from datetime import datetime
import statistics
import platform
import gc

import torch
import numpy as np
from tqdm import tqdm

# Add project root to path for imports
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import from the benchmark module
from scripts.benchmark import BenchmarkResult

# Import app components
from app.utils.config import load_config
from app.services.models.loader import ModelLoader
from app.services.models.manager import EnhancedModelManager
from app.core.pipeline.processor import UnifiedProcessor
from app.audit.logger import AuditLogger
from app.audit.metrics import MetricsCollector

# Configure logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("model_benchmark.log")
    ]
)
logger = logging.getLogger("model_benchmark")

# Test datasets for different model types
TEST_DATASETS = {
    "translation": [
        {"id": "t1", "text": "Hello, how are you doing today? I hope you're having a wonderful day."},
        {"id": "t2", "text": "The quick brown fox jumps over the lazy dog. This pangram contains all letters of the English alphabet."},
        {"id": "t3", "text": "Machine learning is a method of data analysis that automates analytical model building."},
        {"id": "t4", "text": "Cloud computing is the on-demand availability of computer system resources, especially data storage and computing power."},
        {"id": "t5", "text": "The United Nations is an international organization founded in 1945 after the Second World War."}
    ],
    "simplification": [
        {"id": "s1", "text": "The mitochondrion is a double membrane-bound organelle found in most eukaryotic organisms. Mitochondria use aerobic respiration to generate most of the cell's supply of adenosine triphosphate (ATP), which is used as a source of chemical energy."},
        {"id": "s2", "text": "Quantum mechanics is a fundamental theory in physics that provides a description of the physical properties of nature at the scale of atoms and subatomic particles."},
        {"id": "s3", "text": "The complex interactions between various atmospheric phenomena can precipitate meteorological perturbations that manifest as adverse climatic conditions."},
        {"id": "s4", "text": "The endocrine system is a network of glands and organs that produce, store, and secrete hormones. When functioning optimally, the endocrine system works with other systems to regulate your body's healthy development and function throughout life."},
        {"id": "s5", "text": "The judicial branch interprets laws, applies laws to individual cases, and decides if laws violate the Constitution. The judicial branch determines whether the accused is guilty or not, but it is the responsibility of the executive branch to carry out the punishment."}
    ],
    "language_detection": [
        {"id": "ld1", "text": "Hello, how are you doing today?"},
        {"id": "ld2", "text": "Hola, ¿cómo estás hoy?"},
        {"id": "ld3", "text": "Bonjour, comment ça va aujourd'hui?"},
        {"id": "ld4", "text": "Hallo, wie geht es dir heute?"},
        {"id": "ld5", "text": "Ciao, come stai oggi?"},
        {"id": "ld6", "text": "Olá, como você está hoje?"},
        {"id": "ld7", "text": "Привет, как дела сегодня?"},
        {"id": "ld8", "text": "你好，今天怎么样？"},
        {"id": "ld9", "text": "こんにちは、今日はどうですか？"},
        {"id": "ld10", "text": "안녕하세요, 오늘 어떻게 지내세요?"}
    ],
    "anonymization": [
        {"id": "a1", "text": "John Smith lives at 123 Main St, New York and his email is john.smith@example.com. His phone number is (555) 123-4567."},
        {"id": "a2", "text": "Patient: Jane Doe, DOB: 01/15/1982, SSN: 123-45-6789, Insurance ID: INS-987654321"},
        {"id": "a3", "text": "Please transfer $5,000 to account #12345678, Bank of America, routing number 123456789."},
        {"id": "a4", "text": "Dr. Michael Johnson can be reached at mjohnson@hospital.org or at 555-987-6543"},
        {"id": "a5", "text": "The meeting will be held at 123 Corporate Drive, Chicago, IL 60601 on December 15, 2025."}
    ],
    "summarization": [
        {"id": "sum1", "text": "Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to natural intelligence displayed by animals including humans. AI research has been defined as the field of study of intelligent agents, which refers to any system that perceives its environment and takes actions that maximize its chance of achieving its goals. The term 'artificial intelligence' had previously been used to describe machines that mimic and display 'human' cognitive skills that are associated with the human mind, such as 'learning' and 'problem-solving.' This definition has since been rejected by major AI researchers who now describe AI in terms of rationality and acting rationally, which does not limit how intelligence can be articulated. AI applications include advanced web search engines, recommendation systems, understanding human speech, self-driving cars, automated decision-making, and competing at the highest level in strategic game systems."},
        {"id": "sum2", "text": "Climate change includes both global warming driven by human-induced emissions of greenhouse gases and the resulting large-scale shifts in weather patterns. Though there have been previous periods of climatic change, since the mid-20th century humans have had an unprecedented impact on Earth's climate system and caused change on a global scale. The largest driver of warming is the emission of gases that create a greenhouse effect, of which more than 90% are carbon dioxide and methane. Fossil fuel burning for energy consumption is the main source of these emissions, with additional contributions from agriculture, deforestation, and manufacturing. The human cause of climate change is not disputed by any scientific body of national or international standing. Temperature rise is accelerated or tempered by climate feedbacks, such as loss of sunlight-reflecting snow and ice cover, increased water vapor, and changes to land and ocean carbon sinks."},
        {"id": "sum3", "text": "The United Nations (UN) is an intergovernmental organization aiming to maintain international peace and security, develop friendly relations among nations, achieve international cooperation, and be a centre for harmonizing the actions of nations. It is the largest, most familiar, most internationally represented and most powerful intergovernmental organization in the world. The UN is headquartered on international territory in New York City, with its other main offices in Geneva, Nairobi, Vienna, and The Hague. The UN was established after World War II with the aim of preventing future wars, succeeding the ineffective League of Nations. On 25 April 1945, 50 governments met in San Francisco for a conference and started drafting the UN Charter, which was adopted on 25 June 1945 and took effect on 24 October 1945, when the UN began operations. The organization's objectives include maintaining international peace and security, protecting human rights, delivering humanitarian aid, promoting sustainable development, and upholding international law."}
    ],
    "analysis": [
        {"id": "an1", "text": "I absolutely love this product! It's the best purchase I've made all year. The quality is outstanding and it works perfectly."},
        {"id": "an2", "text": "I'm very disappointed with this service. The customer support was unhelpful and the product arrived damaged. I would not recommend it to anyone."},
        {"id": "an3", "text": "Apple Inc. announced today that it will open a new campus in Austin, Texas. The company's CEO, Tim Cook, said the expansion will create thousands of jobs."},
        {"id": "an4", "text": "The restaurant was okay. The food was decent but nothing special. The service was neither good nor bad. I might go back if I'm in the area."},
        {"id": "an5", "text": "The new environmental policy will limit carbon emissions from major industries. Environmental groups praised the move, while industry representatives expressed concerns about implementation costs."}
    ]
}

# Language pairs for translation benchmarking
LANGUAGE_PAIRS = [
    ("en", "es"),  # English to Spanish
    ("en", "fr"),  # English to French
    ("en", "de"),  # English to German
    ("es", "en"),  # Spanish to English
    ("fr", "en"),  # French to English
    ("de", "en")   # German to English
]

class ComprehensiveBenchmark:
    """Comprehensive benchmark for all CasaLingua models and pipelines."""
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        output_dir: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Initialize comprehensive benchmark.
        
        Args:
            config_path: Path to configuration file
            output_dir: Directory for benchmark results
            device: Device to use for benchmarking (cpu, cuda, mps)
        """
        # Load configuration
        self.config = load_config(config_path) if config_path else load_config()
        
        # Set up output directory
        self.output_dir = Path(output_dir) if output_dir else Path("benchmark_results")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set device for testing
        self.device = device
        if not device:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch, "backends") and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
                
        logger.info(f"Using device: {self.device}")
        
        # Initialize components
        self.processor = None
        self.model_manager = None
        self.model_loader = None
        
        # Track system info
        self.system_info = self._get_system_info()
        logger.info(f"System info: {json.dumps(self.system_info, indent=2)}")
        
        # Track benchmark results
        self.results: Dict[str, BenchmarkResult] = {}
        
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information for benchmarking context."""
        info = {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "cpu_count": psutil.cpu_count(logical=False),
            "total_memory_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "torch_version": torch.__version__,
            "device": self.device
        }
        
        # Add GPU info if available
        if self.device == "cuda" and torch.cuda.is_available():
            try:
                info["gpu_name"] = torch.cuda.get_device_name(0)
                info["gpu_count"] = torch.cuda.device_count()
                info["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
            except Exception as e:
                logger.warning(f"Failed to get CUDA device info: {e}")
                
        # Add Mac M-series GPU info if available
        elif self.device == "mps":
            info["gpu_type"] = "Apple Silicon"
            
        return info
        
    async def initialize(self) -> None:
        """Initialize benchmark components."""
        logger.info("Initializing benchmark components...")
        
        try:
            # Initialize model loader
            self.model_loader = ModelLoader(config=self.config)
            
            # Create hardware info dict
            hardware_info = {
                "memory": {"total_gb": self.system_info["total_memory_gb"], "available_gb": self.system_info["total_memory_gb"] * 0.8},
                "system": {"processor_type": "apple_silicon" if self.device == "mps" else "cpu"}
            }
            
            if self.device == "cuda":
                hardware_info["gpu"] = {"available": True, "count": torch.cuda.device_count()}
                
            # Create metrics collector and audit logger
            audit_logger = AuditLogger(config=self.config)
            metrics = MetricsCollector(config=self.config)
            
            # Load model registry configuration 
            from app.services.models.loader import load_registry_config
            registry_config = load_registry_config(self.config)
            
            # Create model manager
            self.model_manager = EnhancedModelManager(self.model_loader, hardware_info, self.config)
            
            # Create processor
            self.processor = UnifiedProcessor(self.model_manager, audit_logger, metrics, self.config, registry_config)
            
            # Initialize processor
            await self.processor.initialize()
            logger.info("Benchmark components initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing benchmark: {e}", exc_info=True)
            raise
    
    async def benchmark_all(self) -> Dict[str, BenchmarkResult]:
        """
        Run comprehensive benchmarks for all model types.
        
        Returns:
            Dictionary of benchmark results by model type
        """
        # Verify processor is initialized
        if not self.processor:
            raise ValueError("Processor is not initialized. Call initialize() first.")
            
        # Create aggregated results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        all_results = {}
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create master results file
        master_results_path = self.output_dir / f"benchmark_master_{timestamp}.json"
        
        try:
            # Run translation benchmarks
            logger.info("Starting translation benchmarks...")
            translation_results = await self.benchmark_translation()
            all_results["translation"] = translation_results
            
            # Run language detection benchmarks
            logger.info("Starting language detection benchmarks...")
            detection_results = await self.benchmark_language_detection()
            all_results["language_detection"] = detection_results
            
            # Run simplification benchmarks
            logger.info("Starting simplification benchmarks...")
            simplification_results = await self.benchmark_simplification()
            all_results["simplification"] = simplification_results
            
            # Run anonymization benchmarks
            logger.info("Starting anonymization benchmarks...")
            anonymization_results = await self.benchmark_anonymization()
            all_results["anonymization"] = anonymization_results
            
            # Run summarization benchmarks
            logger.info("Starting summarization benchmarks...")
            summarization_results = await self.benchmark_summarization()
            all_results["summarization"] = summarization_results
            
            # Run analysis benchmarks
            logger.info("Starting analysis benchmarks...")
            analysis_results = await self.benchmark_analysis()
            all_results["analysis"] = analysis_results
            
            # Generate summary report
            summary_report = self._generate_summary_report(all_results)
            
            # Save master results
            with open(master_results_path, "w") as f:
                json.dump({
                    "timestamp": timestamp,
                    "system_info": self.system_info,
                    "summary": summary_report,
                    "results": {k: v.to_dict() for k, v in all_results.items()}
                }, f, indent=2)
                
            logger.info(f"Master benchmark results saved to {master_results_path}")
            
            # Print summary
            self._print_summary_report(summary_report)
            
            return all_results
            
        except Exception as e:
            logger.error(f"Error in comprehensive benchmark: {e}", exc_info=True)
            
            # Save partial results
            with open(master_results_path, "w") as f:
                json.dump({
                    "timestamp": timestamp,
                    "system_info": self.system_info,
                    "error": str(e),
                    "partial_results": {k: v.to_dict() for k, v in all_results.items()}
                }, f, indent=2)
                
            logger.info(f"Partial benchmark results saved to {master_results_path}")
            raise
    
    async def benchmark_translation(self) -> BenchmarkResult:
        """
        Benchmark translation models.
        
        Returns:
            Benchmark result for translation
        """
        # Create benchmark result
        result = BenchmarkResult(
            name="translation_benchmark",
            description="Benchmark of translation capabilities"
        )
        
        # Get test data
        test_texts = TEST_DATASETS["translation"]
        
        try:
            # Benchmark each language pair
            for source_lang, target_lang in LANGUAGE_PAIRS:
                logger.info(f"Benchmarking translation {source_lang} -> {target_lang}")
                
                for item in test_texts:
                    text = item["text"]
                    item_id = item["id"]
                    task_name = f"translate_{source_lang}_{target_lang}_{item_id}"
                    
                    # Track memory before
                    memory_before = self._get_memory_usage()
                    
                    # Measure time
                    start_time = time.time()
                    
                    # Translate text
                    translation_result = await self.processor.process_translation(
                        text=text,
                        source_language=source_lang,
                        target_language=target_lang
                    )
                    
                    # Calculate duration
                    duration = time.time() - start_time
                    
                    # Track memory after
                    memory_after = self._get_memory_usage()
                    memory_used = memory_after - memory_before if memory_before and memory_after else None
                    
                    # Extract translated text
                    translated_text = translation_result.get("translated_text", "")
                    
                    # Add result
                    result.add_task_result(
                        task_name=task_name,
                        duration=duration,
                        input_size=len(text),
                        output_size=len(translated_text),
                        memory_usage=memory_used,
                        metrics={
                            "source_language": source_lang,
                            "target_language": target_lang,
                            "model_id": translation_result.get("model_id", "default")
                        },
                        metadata={
                            "source_text": text,
                            "translated_text": translated_text,
                            "confidence": translation_result.get("confidence", 0)
                        }
                    )
                    
                    # Garbage collection to prevent memory buildup
                    gc.collect()
                    if self.device == "cuda":
                        torch.cuda.empty_cache()
                    elif self.device == "mps":
                        torch.mps.empty_cache()
            
            # Complete benchmark
            result.complete()
            
            # Save result
            result_path = self.output_dir / f"translation_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            result.save_to_file(result_path)
            
            # Print summary
            result.print_summary()
            
            return result
            
        except Exception as e:
            logger.error(f"Error in translation benchmark: {e}", exc_info=True)
            
            # Complete with available data
            result.complete()
            return result
    
    async def benchmark_language_detection(self) -> BenchmarkResult:
        """
        Benchmark language detection models.
        
        Returns:
            Benchmark result for language detection
        """
        # Create benchmark result
        result = BenchmarkResult(
            name="language_detection_benchmark",
            description="Benchmark of language detection capabilities"
        )
        
        # Get test data
        test_texts = TEST_DATASETS["language_detection"]
        
        try:
            # Test with and without detailed results
            for detailed in [False, True]:
                for item in test_texts:
                    text = item["text"]
                    item_id = item["id"]
                    task_name = f"detect_language_{item_id}_detailed_{detailed}"
                    
                    # Track memory before
                    memory_before = self._get_memory_usage()
                    
                    # Measure time
                    start_time = time.time()
                    
                    # Detect language
                    detection_result = await self.processor.detect_language(
                        text=text,
                        detailed=detailed
                    )
                    
                    # Calculate duration
                    duration = time.time() - start_time
                    
                    # Track memory after
                    memory_after = self._get_memory_usage()
                    memory_used = memory_after - memory_before if memory_before and memory_after else None
                    
                    # Add result
                    result.add_task_result(
                        task_name=task_name,
                        duration=duration,
                        input_size=len(text),
                        output_size=0,  # No significant output size for language detection
                        memory_usage=memory_used,
                        metrics={
                            "detailed": detailed,
                            "model_id": detection_result.get("model_id", "default"),
                            "confidence": detection_result.get("confidence", 0)
                        },
                        metadata={
                            "text": text,
                            "detected_language": detection_result.get("detected_language"),
                            "alternatives": detection_result.get("alternatives", [])
                        }
                    )
                    
                    # Garbage collection
                    gc.collect()
            
            # Complete benchmark
            result.complete()
            
            # Save result
            result_path = self.output_dir / f"language_detection_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            result.save_to_file(result_path)
            
            # Print summary
            result.print_summary()
            
            return result
            
        except Exception as e:
            logger.error(f"Error in language detection benchmark: {e}", exc_info=True)
            
            # Complete with available data
            result.complete()
            return result
    
    async def benchmark_simplification(self) -> BenchmarkResult:
        """
        Benchmark text simplification models.
        
        Returns:
            Benchmark result for simplification
        """
        # Create benchmark result
        result = BenchmarkResult(
            name="simplification_benchmark",
            description="Benchmark of text simplification capabilities"
        )
        
        # Get test data
        test_texts = TEST_DATASETS["simplification"]
        
        try:
            # Test simplification with different target levels
            for level in ["simple", "elementary", "intermediate"]:
                for item in test_texts:
                    text = item["text"]
                    item_id = item["id"]
                    task_name = f"simplify_{item_id}_level_{level}"
                    
                    # Track memory before
                    memory_before = self._get_memory_usage()
                    
                    # Measure time
                    start_time = time.time()
                    
                    # Simplify text using appropriate method
                    if hasattr(self.processor, "simplify_text"):
                        simplification_result = await self.processor.simplify_text(
                            text=text,
                            target_level=level,
                            language="en"
                        )
                    elif hasattr(self.processor, "process_simplification"):
                        simplification_result = await self.processor.process_simplification(
                            text=text,
                            level=level,
                            language="en"
                        )
                    else:
                        # Fallback to _process_text
                        simplification_result = await self.processor._process_text(
                            text=text,
                            options={
                                "simplify": True,
                                "target_level": level,
                                "source_language": "en"
                            },
                            metadata={}
                        )
                    
                    # Calculate duration
                    duration = time.time() - start_time
                    
                    # Track memory after
                    memory_after = self._get_memory_usage()
                    memory_used = memory_after - memory_before if memory_before and memory_after else None
                    
                    # Extract simplified text
                    if isinstance(simplification_result, dict):
                        if "simplified_text" in simplification_result:
                            simplified_text = simplification_result["simplified_text"]
                        elif "processed_text" in simplification_result:
                            simplified_text = simplification_result["processed_text"]
                        else:
                            simplified_text = str(simplification_result)
                    else:
                        simplified_text = str(simplification_result)
                    
                    # Add result
                    result.add_task_result(
                        task_name=task_name,
                        duration=duration,
                        input_size=len(text),
                        output_size=len(simplified_text),
                        memory_usage=memory_used,
                        metrics={
                            "target_level": level,
                            "language": "en"
                        },
                        metadata={
                            "original_text": text,
                            "simplified_text": simplified_text
                        }
                    )
                    
                    # Garbage collection
                    gc.collect()
                    if self.device == "cuda":
                        torch.cuda.empty_cache()
                    elif self.device == "mps":
                        torch.mps.empty_cache()
            
            # Complete benchmark
            result.complete()
            
            # Save result
            result_path = self.output_dir / f"simplification_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            result.save_to_file(result_path)
            
            # Print summary
            result.print_summary()
            
            return result
            
        except Exception as e:
            logger.error(f"Error in simplification benchmark: {e}", exc_info=True)
            
            # Complete with available data
            result.complete()
            return result
    
    async def benchmark_anonymization(self) -> BenchmarkResult:
        """
        Benchmark text anonymization models.
        
        Returns:
            Benchmark result for anonymization
        """
        # Create benchmark result
        result = BenchmarkResult(
            name="anonymization_benchmark",
            description="Benchmark of text anonymization capabilities"
        )
        
        # Get test data
        test_texts = TEST_DATASETS["anonymization"]
        
        try:
            # Test anonymization with different strategies
            for strategy in ["mask", "redact", "pseudonymize"]:
                for item in test_texts:
                    text = item["text"]
                    item_id = item["id"]
                    task_name = f"anonymize_{item_id}_strategy_{strategy}"
                    
                    # Track memory before
                    memory_before = self._get_memory_usage()
                    
                    # Measure time
                    start_time = time.time()
                    
                    # Anonymize text using appropriate method
                    if hasattr(self.processor, "anonymize_text"):
                        anonymization_result = await self.processor.anonymize_text(
                            text=text,
                            language="en",
                            strategy=strategy
                        )
                    elif hasattr(self.processor, "process_anonymization"):
                        anonymization_result = await self.processor.process_anonymization(
                            text=text,
                            language="en",
                            strategy=strategy
                        )
                    else:
                        # Fallback to _process_text
                        anonymization_result = await self.processor._process_text(
                            text=text,
                            options={
                                "anonymize": True,
                                "anonymization_strategy": strategy,
                                "source_language": "en"
                            },
                            metadata={}
                        )
                    
                    # Calculate duration
                    duration = time.time() - start_time
                    
                    # Track memory after
                    memory_after = self._get_memory_usage()
                    memory_used = memory_after - memory_before if memory_before and memory_after else None
                    
                    # Extract anonymized text
                    if isinstance(anonymization_result, dict):
                        if "anonymized_text" in anonymization_result:
                            anonymized_text = anonymization_result["anonymized_text"]
                        elif "processed_text" in anonymization_result:
                            anonymized_text = anonymization_result["processed_text"]
                        else:
                            anonymized_text = str(anonymization_result)
                    elif isinstance(anonymization_result, tuple) and len(anonymization_result) == 2:
                        # Some anonymization methods return (text, entities)
                        anonymized_text = anonymization_result[0]
                    else:
                        anonymized_text = str(anonymization_result)
                    
                    # Add result
                    result.add_task_result(
                        task_name=task_name,
                        duration=duration,
                        input_size=len(text),
                        output_size=len(anonymized_text),
                        memory_usage=memory_used,
                        metrics={
                            "strategy": strategy,
                            "language": "en"
                        },
                        metadata={
                            "original_text": text,
                            "anonymized_text": anonymized_text
                        }
                    )
                    
                    # Garbage collection
                    gc.collect()
                    if self.device == "cuda":
                        torch.cuda.empty_cache()
                    elif self.device == "mps":
                        torch.mps.empty_cache()
            
            # Complete benchmark
            result.complete()
            
            # Save result
            result_path = self.output_dir / f"anonymization_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            result.save_to_file(result_path)
            
            # Print summary
            result.print_summary()
            
            return result
            
        except Exception as e:
            logger.error(f"Error in anonymization benchmark: {e}", exc_info=True)
            
            # Complete with available data
            result.complete()
            return result
    
    async def benchmark_summarization(self) -> BenchmarkResult:
        """
        Benchmark text summarization models.
        
        Returns:
            Benchmark result for summarization
        """
        # Create benchmark result
        result = BenchmarkResult(
            name="summarization_benchmark",
            description="Benchmark of text summarization capabilities"
        )
        
        # Get test data
        test_texts = TEST_DATASETS["summarization"]
        
        try:
            # Test summarization
            for item in test_texts:
                text = item["text"]
                item_id = item["id"]
                task_name = f"summarize_{item_id}"
                
                # Track memory before
                memory_before = self._get_memory_usage()
                
                # Measure time
                start_time = time.time()
                
                # Summarize text
                summarization_result = await self.processor.process_summarization(
                    text=text,
                    language="en"
                )
                
                # Calculate duration
                duration = time.time() - start_time
                
                # Track memory after
                memory_after = self._get_memory_usage()
                memory_used = memory_after - memory_before if memory_before and memory_after else None
                
                # Extract summary
                summary = summarization_result.get("summary", "")
                
                # Add result
                result.add_task_result(
                    task_name=task_name,
                    duration=duration,
                    input_size=len(text),
                    output_size=len(summary),
                    memory_usage=memory_used,
                    metrics={
                        "language": "en",
                        "model_id": summarization_result.get("model_id", "default")
                    },
                    metadata={
                        "original_text": text,
                        "summary": summary
                    }
                )
                
                # Garbage collection
                gc.collect()
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                elif self.device == "mps":
                    torch.mps.empty_cache()
            
            # Complete benchmark
            result.complete()
            
            # Save result
            result_path = self.output_dir / f"summarization_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            result.save_to_file(result_path)
            
            # Print summary
            result.print_summary()
            
            return result
            
        except Exception as e:
            logger.error(f"Error in summarization benchmark: {e}", exc_info=True)
            
            # Complete with available data
            result.complete()
            return result
    
    async def benchmark_analysis(self) -> BenchmarkResult:
        """
        Benchmark text analysis models.
        
        Returns:
            Benchmark result for analysis
        """
        # Create benchmark result
        result = BenchmarkResult(
            name="analysis_benchmark",
            description="Benchmark of text analysis capabilities"
        )
        
        # Get test data
        test_texts = TEST_DATASETS["analysis"]
        
        try:
            # Test different analysis types
            analysis_types = [
                ["sentiment"],
                ["entities"],
                ["topics"],
                ["sentiment", "entities"],
                ["sentiment", "entities", "topics"]
            ]
            
            for analyses in analysis_types:
                analysis_name = "_".join(analyses)
                
                for item in test_texts:
                    text = item["text"]
                    item_id = item["id"]
                    task_name = f"analyze_{item_id}_{analysis_name}"
                    
                    # Track memory before
                    memory_before = self._get_memory_usage()
                    
                    # Measure time
                    start_time = time.time()
                    
                    # Analyze text
                    analysis_result = await self.processor.analyze_text(
                        text=text,
                        language="en",
                        analyses=analyses
                    )
                    
                    # Calculate duration
                    duration = time.time() - start_time
                    
                    # Track memory after
                    memory_after = self._get_memory_usage()
                    memory_used = memory_after - memory_before if memory_before and memory_after else None
                    
                    # Add result
                    result.add_task_result(
                        task_name=task_name,
                        duration=duration,
                        input_size=len(text),
                        output_size=0,  # No significant output size for analysis
                        memory_usage=memory_used,
                        metrics={
                            "language": "en",
                            "analyses": analyses,
                            "model_id": analysis_result.get("model_id", "default")
                        },
                        metadata={
                            "text": text,
                            "sentiment": analysis_result.get("sentiment"),
                            "entities": analysis_result.get("entities"),
                            "topics": analysis_result.get("topics")
                        }
                    )
                    
                    # Garbage collection
                    gc.collect()
                    if self.device == "cuda":
                        torch.cuda.empty_cache()
                    elif self.device == "mps":
                        torch.mps.empty_cache()
            
            # Complete benchmark
            result.complete()
            
            # Save result
            result_path = self.output_dir / f"analysis_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            result.save_to_file(result_path)
            
            # Print summary
            result.print_summary()
            
            return result
            
        except Exception as e:
            logger.error(f"Error in analysis benchmark: {e}", exc_info=True)
            
            # Complete with available data
            result.complete()
            return result
    
    def _get_memory_usage(self) -> Optional[float]:
        """
        Get current memory usage.
        
        Returns:
            Memory usage in MB or None if not available
        """
        try:
            if self.device == "cuda" and torch.cuda.is_available():
                return torch.cuda.memory_allocated() / (1024 * 1024)
            elif self.device == "mps" and hasattr(torch.mps, "current_allocated_memory"):
                return torch.mps.current_allocated_memory() / (1024 * 1024)
            else:
                # System memory
                process = psutil.Process(os.getpid())
                return process.memory_info().rss / (1024 * 1024)
        except Exception:
            return None
    
    def _generate_summary_report(self, results: Dict[str, BenchmarkResult]) -> Dict[str, Any]:
        """
        Generate summary report from benchmark results.
        
        Args:
            results: Dictionary of benchmark results
            
        Returns:
            Summary report dictionary
        """
        summary = {
            "timestamp": datetime.now().isoformat(),
            "system_info": self.system_info,
            "model_types": [],
            "performance": {}
        }
        
        # Summarize each model type
        for model_type, result in results.items():
            model_summary = {
                "name": model_type,
                "task_count": result.metrics.get("total_tasks", 0),
                "avg_duration": result.metrics.get("avg_duration", 0),
                "p95_duration": result.metrics.get("p95_duration", 0),
                "avg_memory_mb": result.metrics.get("avg_memory_usage", 0),
                "throughput": result.metrics.get("throughput_items_per_second", 0)
            }
            
            summary["model_types"].append(model_summary)
            
            # Store raw performance data
            summary["performance"][model_type] = {
                "durations": [task["duration"] for task in result.tasks],
                "memory_usage": [task["memory_usage"] for task in result.tasks if task["memory_usage"] is not None]
            }
        
        # Calculate overall stats
        all_durations = []
        for model_type in results.values():
            all_durations.extend(task["duration"] for task in model_type.tasks)
            
        if all_durations:
            summary["overall"] = {
                "total_tasks": len(all_durations),
                "avg_duration": statistics.mean(all_durations),
                "median_duration": statistics.median(all_durations),
                "p95_duration": statistics.quantiles(sorted(all_durations), n=20)[18] if len(all_durations) >= 20 else max(all_durations),
                "min_duration": min(all_durations),
                "max_duration": max(all_durations)
            }
        
        return summary
    
    def _print_summary_report(self, summary: Dict[str, Any]) -> None:
        """
        Print summary report to console.
        
        Args:
            summary: Summary report dictionary
        """
        logger.info("\n======== BENCHMARK SUMMARY ========")
        logger.info(f"System: {summary['system_info']['platform']} | Device: {summary['system_info']['device']}")
        
        if "overall" in summary:
            logger.info(f"\nOverall Performance:")
            logger.info(f"  Total Tasks: {summary['overall']['total_tasks']}")
            logger.info(f"  Avg Duration: {summary['overall']['avg_duration']:.4f}s")
            logger.info(f"  P95 Duration: {summary['overall']['p95_duration']:.4f}s")
            logger.info(f"  Min/Max: {summary['overall']['min_duration']:.4f}s / {summary['overall']['max_duration']:.4f}s")
        
        logger.info("\nModel Type Performance:")
        for model in summary["model_types"]:
            logger.info(f"  {model['name']}:")
            logger.info(f"    Tasks: {model['task_count']}")
            logger.info(f"    Avg Duration: {model['avg_duration']:.4f}s")
            logger.info(f"    P95 Duration: {model['p95_duration']:.4f}s")
            logger.info(f"    Avg Memory: {model['avg_memory_mb']:.2f} MB")
            logger.info(f"    Throughput: {model['throughput']:.2f} items/sec")
        
        logger.info("====================================")
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        logger.info("Cleaning up resources...")
        
        if self.processor:
            try:
                await self.processor.shutdown()
            except Exception as e:
                logger.warning(f"Error during processor shutdown: {e}")
                
        # Force garbage collection
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()
        elif self.device == "mps":
            torch.mps.empty_cache()

async def main():
    """Main entry point for benchmark script."""
    parser = argparse.ArgumentParser(description="CasaLingua Comprehensive Model Benchmark")
    
    parser.add_argument(
        "--config", 
        type=str, 
        help="Path to configuration file"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="benchmark_results", 
        help="Directory for benchmark results"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        choices=["cpu", "cuda", "mps"], 
        help="Device to use for benchmarking"
    )
    parser.add_argument(
        "--test-all", 
        action="store_true", 
        help="Run all benchmarks"
    )
    parser.add_argument(
        "--test-translation", 
        action="store_true", 
        help="Run translation benchmarks"
    )
    parser.add_argument(
        "--test-detection", 
        action="store_true", 
        help="Run language detection benchmarks"
    )
    parser.add_argument(
        "--test-simplification", 
        action="store_true", 
        help="Run simplification benchmarks"
    )
    parser.add_argument(
        "--test-anonymization", 
        action="store_true", 
        help="Run anonymization benchmarks"
    )
    parser.add_argument(
        "--test-summarization", 
        action="store_true", 
        help="Run summarization benchmarks"
    )
    parser.add_argument(
        "--test-analysis", 
        action="store_true", 
        help="Run analysis benchmarks"
    )
    
    args = parser.parse_args()
    
    # Initialize benchmark
    benchmark = ComprehensiveBenchmark(
        config_path=args.config,
        output_dir=args.output_dir,
        device=args.device
    )
    
    try:
        # Initialize components
        await benchmark.initialize()
        
        # Run selected benchmarks
        if args.test_all:
            await benchmark.benchmark_all()
        else:
            if args.test_translation:
                await benchmark.benchmark_translation()
            if args.test_detection:
                await benchmark.benchmark_language_detection()
            if args.test_simplification:
                await benchmark.benchmark_simplification()
            if args.test_anonymization:
                await benchmark.benchmark_anonymization()
            if args.test_summarization:
                await benchmark.benchmark_summarization()
            if args.test_analysis:
                await benchmark.benchmark_analysis()
                
            # If no specific test was selected, run all tests
            if not any([
                args.test_translation, args.test_detection, args.test_simplification,
                args.test_anonymization, args.test_summarization, args.test_analysis
            ]):
                await benchmark.benchmark_all()
    finally:
        # Clean up
        await benchmark.cleanup()

if __name__ == "__main__":
    asyncio.run(main())