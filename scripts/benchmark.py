"""
CasaLingua Benchmarking Tool

This module provides performance benchmarking capabilities for the 
CasaLingua language processing platform, allowing comparison of different
models, configurations, and optimization techniques.

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
import csv
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
from datetime import datetime
import statistics

import torch
import numpy as np
from tqdm import tqdm

# ===============================
# CasaLingua Benchmarking Tool
#
# Usage:
#   python benchmark.py --model translation/medium --test-data test.txt --source-lang es --target-lang en
#
# CLI Flags:
#   --config           Optional path to a custom config file
#   --output-dir       Output directory for benchmark results
#   --device           Device to use: cpu or cuda
#   --model            Translation model ID to benchmark
#   --test-data        Path to input file (.txt, .json, .csv)
#   --source-lang      Source language code (e.g., en)
#   --target-lang      Target language code (e.g., fr)
#   --batch-size       Size of batch (default=1)
#   --verify           Enable veracity checking (if configured)
#   --compare-models   Comma-separated list of models to compare
#   --batch-sizes      Comma-separated list of batch sizes to test
#
# Ladder Logic:
#   [Start] --> [Parse CLI] --> [Load Config + Init Benchmark] --> [Load Test Data]
#            --> ┌──────────────┐
#                │compare-models│ --┐
#                └──────────────┘  │
#                    │             │
#   [Single Model Benchmark]   [Benchmark Batch Sizes]
#                    ↓             ↓
#           [Benchmark Loop Over Texts] --┐
#                    ↓                   ↓
#          [Translate + Verify]     [Metrics + Mem]
#                    ↓                   ↓
#            [Save JSON/CSV] <---- [Print Summary]
#                    ↓
#                 [Cleanup]
# ===============================

# Add project root to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.utils.config import load_config
from app.model.loader import ModelLoader, get_model_loader
from app.model.manager import ModelManager
from app.audit.veracity import VeracityAuditor
from app.utils.logging import get_logger
logger = get_logger(__name__)

class BenchmarkResult:
    """Class to store and analyze benchmark results."""
    
    def __init__(self, name: str, description: str = ""):
        """
        Initialize benchmark result.
        
        Args:
            name: Benchmark name
            description: Benchmark description
        """
        self.name = name
        self.description = description
        self.tasks: List[Dict[str, Any]] = []
        self.start_time = time.time()
        self.end_time: Optional[float] = None
        self.metrics: Dict[str, Any] = {}
        
    def add_task_result(
        self,
        task_name: str,
        duration: float,
        input_size: int,
        output_size: int,
        memory_usage: Optional[float] = None,
        metrics: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add a task result to the benchmark.
        
        Args:
            task_name: Name of the task
            duration: Task duration in seconds
            input_size: Size of input (characters, tokens, etc.)
            output_size: Size of output (characters, tokens, etc.)
            memory_usage: Peak memory usage in MB
            metrics: Additional task metrics
            metadata: Additional task metadata
        """
        self.tasks.append({
            "task_name": task_name,
            "duration": duration,
            "input_size": input_size,
            "output_size": output_size,
            "memory_usage": memory_usage,
            "metrics": metrics or {},
            "metadata": metadata or {},
            "timestamp": time.time()
        })
        
    def complete(self) -> None:
        """Mark benchmark as complete and calculate aggregate metrics."""
        self.end_time = time.time()
        self.calculate_metrics()
        
    def calculate_metrics(self) -> None:
        """Calculate aggregate metrics from task results."""
        if not self.tasks:
            self.metrics = {
                "total_tasks": 0,
                "total_duration": 0,
                "avg_duration": 0
            }
            return
            
        # Extract durations
        durations = [task["duration"] for task in self.tasks]
        
        # Calculate basic metrics
        self.metrics = {
            "total_tasks": len(self.tasks),
            "total_duration": sum(durations),
            "avg_duration": statistics.mean(durations),
            "min_duration": min(durations),
            "max_duration": max(durations),
            "median_duration": statistics.median(durations),
            "stddev_duration": statistics.stdev(durations) if len(durations) > 1 else 0,
            "total_input_size": sum(task["input_size"] for task in self.tasks),
            "total_output_size": sum(task["output_size"] for task in self.tasks),
        }
        
        # Calculate percentiles
        self.metrics["p90_duration"] = self._percentile(durations, 90)
        self.metrics["p95_duration"] = self._percentile(durations, 95)
        self.metrics["p99_duration"] = self._percentile(durations, 99)
        
        # Calculate throughput
        total_seconds = sum(durations)
        if total_seconds > 0:
            self.metrics["throughput_items_per_second"] = len(self.tasks) / total_seconds
            total_input = self.metrics["total_input_size"]
            self.metrics["throughput_input_per_second"] = total_input / total_seconds
            
        # Calculate memory usage if available
        memory_values = [task["memory_usage"] for task in self.tasks if task["memory_usage"] is not None]
        if memory_values:
            self.metrics["avg_memory_usage"] = statistics.mean(memory_values)
            self.metrics["max_memory_usage"] = max(memory_values)
        
    def _percentile(self, data: List[float], percentile: float) -> float:
        """
        Calculate percentile value from data.
        
        Args:
            data: List of values
            percentile: Percentile to calculate (0-100)
            
        Returns:
            Percentile value
        """
        if not data:
            return 0.0
            
        sorted_data = sorted(data)
        index = (percentile / 100.0) * (len(sorted_data) - 1)
        
        # Interpolate between two values if needed
        floor_index = int(index)
        ceil_index = min(floor_index + 1, len(sorted_data) - 1)
        
        if floor_index == ceil_index:
            return sorted_data[floor_index]
            
        floor_value = sorted_data[floor_index]
        ceil_value = sorted_data[ceil_index]
        
        # Linear interpolation
        fraction = index - floor_index
        return floor_value + (ceil_value - floor_value) * fraction
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert benchmark result to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.end_time - self.start_time if self.end_time else None,
            "metrics": self.metrics,
            "tasks": self.tasks
        }
        
    def save_to_file(self, file_path: Union[str, Path]) -> None:
        """
        Save benchmark result to file.
        
        Args:
            file_path: Path to output file
        """
        with open(file_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
            
    def print_summary(self) -> None:
        """Print benchmark summary to console."""
        logger.info(f"\n===== Benchmark Result: {self.name} =====")
        if self.description:
            logger.info(f"Description: {self.description}")
        logger.info(f"Total tasks: {self.metrics.get('total_tasks', 0)}")
        logger.info(f"Total duration: {self.metrics.get('total_duration', 0):.2f}s")
        logger.info(f"Average duration: {self.metrics.get('avg_duration', 0):.4f}s")
        logger.info(f"Median duration: {self.metrics.get('median_duration', 0):.4f}s")
        logger.info(f"P95 duration: {self.metrics.get('p95_duration', 0):.4f}s")
        logger.info(f"Min duration: {self.metrics.get('min_duration', 0):.4f}s")
        logger.info(f"Max duration: {self.metrics.get('max_duration', 0):.4f}s")
        if "throughput_items_per_second" in self.metrics:
            logger.info(f"Throughput: {self.metrics['throughput_items_per_second']:.2f} items/second")
        if "avg_memory_usage" in self.metrics:
            logger.info(f"Average memory usage: {self.metrics['avg_memory_usage']:.2f} MB")
            logger.info(f"Peak memory usage: {self.metrics['max_memory_usage']:.2f} MB")
        logger.info("=" * (20 + len(self.name)))

class TranslationBenchmark:
    """Benchmark for translation models and configurations."""
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        output_dir: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Initialize translation benchmark.
        
        Args:
            config_path: Path to configuration file
            output_dir: Directory for benchmark results
            device: Device to use for benchmarking (cpu, cuda)
        """
        # Load configuration
        self.config = load_config(config_path) if config_path else load_config()
        
        # Set up output directory
        self.output_dir = Path(output_dir) if output_dir else Path("benchmark_results")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set device
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # Initialize components
        self.model_loader = None
        self.model_manager = None
        self.veracity_auditor = None
        
        # Track benchmark results
        self.results: Dict[str, BenchmarkResult] = {}
        
    async def initialize(self) -> None:
        """Initialize benchmark components."""
        logger.info(f"Initializing benchmark on device: {self.device}")
        
        # Initialize model loader
        self.model_loader = get_model_loader(
            config=self.config,
            device=self.device
        )
        
        # Initialize model manager (minimal implementation for benchmarking)
        self.model_manager = ModelManager(
            model_registry=None,  # Not needed for benchmarking
            hardware_info={"gpu_count": 1 if self.device.startswith("cuda") else 0},
            config=self.config
        )
        self.model_manager.loader = self.model_loader
        
        # Initialize veracity auditor if needed
        if self.config.get("veracity", {}).get("enabled", False):
            self.veracity_auditor = VeracityAuditor(
                model_manager=self.model_manager,
                config=self.config
            )
            await self.veracity_auditor.initialize()
            
    async def benchmark_model(
        self,
        model_id: str,
        test_texts: List[Dict[str, str]],
        source_lang: str,
        target_lang: str,
        batch_size: int = 1,
        verify: bool = False,
        benchmark_name: Optional[str] = None,
        description: Optional[str] = None
    ) -> BenchmarkResult:
        """
        Benchmark a translation model.
        
        Args:
            model_id: Model identifier
            test_texts: List of test texts with keys 'text' and 'id'
            source_lang: Source language code
            target_lang: Target language code
            batch_size: Batch size for processing
            verify: Whether to verify translations
            benchmark_name: Custom benchmark name
            description: Benchmark description
            
        Returns:
            Benchmark result
        """
        # Create benchmark name if not provided
        if not benchmark_name:
            benchmark_name = f"{model_id}_{source_lang}-{target_lang}_b{batch_size}"
            if verify:
                benchmark_name += "_verified"
                
        # Create description if not provided
        if not description:
            description = f"Translation benchmark for model {model_id} ({source_lang}->{target_lang})"
            
        # Create benchmark result
        result = BenchmarkResult(benchmark_name, description)
        
        try:
            # Load model
            logger.info(f"Loading model: {model_id}")
            model, tokenizer = await self.model_loader.load_model(model_id)
            
            # Process texts
            logger.info(f"Benchmarking {len(test_texts)} texts...")
            for i in tqdm(range(0, len(test_texts), batch_size)):
                # Get batch
                batch = test_texts[i:i+batch_size]
                
                # Process each text in batch
                for item in batch:
                    text = item["text"]
                    item_id = item.get("id", f"text_{i}")
                    
                    # Measure memory before translation
                    start_memory = self._get_gpu_memory_usage() if self.device.startswith("cuda") or self.device == "mps" else None
                    
                    # Measure translation time
                    start_time = time.time()
                    
                    # Tokenize input
                    inputs = tokenizer(
                        text,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=512
                    ).to(self.device)
                    
                    # Generate translation
                    with torch.no_grad():
                        outputs = model.generate(**inputs)
                        
                    # Decode output
                    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    
                    # Measure translation time
                    translation_time = time.time() - start_time
                    
                    # Verify translation if requested
                    verification_time = 0
                    verification_result = None
                    
                    if verify and self.veracity_auditor:
                        verification_start = time.time()
                        verification_result = await self.veracity_auditor.verify_translation(
                            source_text=text,
                            translation=translated_text,
                            source_language=source_lang,
                            target_language=target_lang
                        )
                        verification_time = time.time() - verification_start
                        
                    # Get memory after translation
                    end_memory = self._get_gpu_memory_usage() if self.device.startswith("cuda") or self.device == "mps" else None
                    memory_usage = end_memory - start_memory if start_memory is not None and end_memory is not None else None
                    
                    # Add task result
                    result.add_task_result(
                        task_name=f"translate_{item_id}",
                        duration=translation_time + verification_time,
                        input_size=len(text),
                        output_size=len(translated_text),
                        memory_usage=memory_usage,
                        metrics={
                            "translation_time": translation_time,
                            "verification_time": verification_time
                        },
                        metadata={
                            "source_text_sample": text[:100] + "..." if len(text) > 100 else text,
                            "translated_text_sample": translated_text[:100] + "..." if len(translated_text) > 100 else translated_text,
                            "verification_result": verification_result
                        }
                    )
                    
            # Complete benchmark
            result.complete()
            
            # Save result
            result_path = self.output_dir / f"{benchmark_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            result.save_to_file(result_path)
            
            # Print summary
            result.print_summary()
            
            # Store result
            self.results[benchmark_name] = result
            
            return result
            
        except Exception as e:
            logger.info(f"Error in benchmark: {str(e)}")
            import traceback
            logger.info(traceback.format_exc())
            # Complete benchmark with available data
            result.complete()
            return result
            
    async def compare_models(
        self,
        model_ids: List[str],
        test_texts: List[Dict[str, str]],
        source_lang: str,
        target_lang: str,
        batch_size: int = 1,
        verify: bool = False,
        output_file: Optional[str] = None
    ) -> Dict[str, BenchmarkResult]:
        """
        Compare multiple translation models.
        
        Args:
            model_ids: List of model identifiers
            test_texts: List of test texts
            source_lang: Source language code
            target_lang: Target language code
            batch_size: Batch size for processing
            verify: Whether to verify translations
            output_file: Path to output comparison file
            
        Returns:
            Dictionary of benchmark results by model
        """
        results = {}
        
        # Benchmark each model
        for model_id in model_ids:
            logger.info(f"\nBenchmarking model: {model_id}")
            result = await self.benchmark_model(
                model_id=model_id,
                test_texts=test_texts,
                source_lang=source_lang,
                target_lang=target_lang,
                batch_size=batch_size,
                verify=verify
            )
            results[model_id] = result
            
        # Generate comparison report
        if output_file:
            self._generate_comparison_report(results, output_file)
            
        return results
        
    def _generate_comparison_report(
        self,
        results: Dict[str, BenchmarkResult],
        output_file: str
    ) -> None:
        """
        Generate a comparison report of benchmark results.
        
        Args:
            results: Dictionary of benchmark results
            output_file: Path to output file
        """
        # Prepare data for CSV
        rows = []
        metrics = [
            "avg_duration", "median_duration", "p95_duration",
            "min_duration", "max_duration", "throughput_items_per_second"
        ]
        
        # Add header row
        header = ["Model"] + metrics
        rows.append(header)
        
        # Add data rows
        for model_id, result in results.items():
            row = [model_id]
            for metric in metrics:
                row.append(result.metrics.get(metric, "N/A"))
            rows.append(row)
            
        # Write to CSV
        with open(output_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(rows)
            
        logger.info(f"Comparison report saved to {output_file}")
        
    def _get_gpu_memory_usage(self) -> Optional[float]:
        """
        Get GPU memory usage in MB.
        Returns:
            Memory usage in MB or None if not available
        """
        # Mac M4 GPU (MPS)
        if hasattr(torch, "backends") and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            try:
                return torch.mps.current_allocated_memory() / (1024 * 1024)
            except Exception:
                return None
        # CUDA GPU
        elif torch.cuda.is_available():
            try:
                return torch.cuda.memory_allocated() / (1024 * 1024)
            except Exception:
                return None
        return None
            
    async def benchmark_batch_sizes(
        self,
        model_id: str,
        test_texts: List[Dict[str, str]],
        source_lang: str,
        target_lang: str,
        batch_sizes: List[int] = [1, 2, 4, 8, 16],
        output_file: Optional[str] = None
    ) -> Dict[int, BenchmarkResult]:
        """
        Benchmark a model with different batch sizes.
        
        Args:
            model_id: Model identifier
            test_texts: List of test texts
            source_lang: Source language code
            target_lang: Target language code
            batch_sizes: List of batch sizes to test
            output_file: Path to output comparison file
            
        Returns:
            Dictionary of benchmark results by batch size
        """
        results = {}
        
        # Benchmark each batch size
        for batch_size in batch_sizes:
            logger.info(f"\nBenchmarking with batch size: {batch_size}")
            result = await self.benchmark_model(
                model_id=model_id,
                test_texts=test_texts,
                source_lang=source_lang,
                target_lang=target_lang,
                batch_size=batch_size,
                benchmark_name=f"{model_id}_b{batch_size}",
                description=f"Batch size {batch_size} benchmark for model {model_id}"
            )
            results[batch_size] = result
            
        # Generate comparison report
        if output_file:
            # Prepare data for CSV
            rows = []
            metrics = [
                "avg_duration", "throughput_items_per_second", 
                "throughput_input_per_second", "max_memory_usage"
            ]
            
            # Add header row
            header = ["Batch Size"] + metrics
            rows.append(header)
            
            # Add data rows
            for batch_size, result in results.items():
                row = [batch_size]
                for metric in metrics:
                    row.append(result.metrics.get(metric, "N/A"))
                rows.append(row)
                
            # Write to CSV
            with open(output_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(rows)
                
            logger.info(f"Batch size comparison saved to {output_file}")
            
        return results
        
    async def cleanup(self) -> None:
        """Clean up resources."""
        if self.model_loader:
            await self.model_loader.unload_all_models()

async def load_test_data(file_path: str) -> List[Dict[str, str]]:
    """
    Load test data from a file.
    
    Args:
        file_path: Path to test data file
        
    Returns:
        List of test texts
    """
    # Determine file type from extension
    ext = Path(file_path).suffix.lower()
    
    if ext == ".json":
        # Load JSON file
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        # Check if it's an array of strings or objects
        if isinstance(data, list):
            if all(isinstance(item, str) for item in data):
                # Convert strings to objects
                return [{"id": f"text_{i}", "text": text} for i, text in enumerate(data)]
            elif all(isinstance(item, dict) and "text" in item for item in data):
                # Already in the right format
                return data
                
    elif ext == ".csv":
        # Load CSV file
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            
            # Find text column
            text_col = None
            id_col = None
            
            for i, col in enumerate(header):
                if col.lower() in ["text", "source", "input"]:
                    text_col = i
                elif col.lower() in ["id", "text_id"]:
                    id_col = i
                    
            if text_col is None:
                text_col = 0  # Default to first column
                
            # Read data
            data = []
            for i, row in enumerate(reader):
                if text_col < len(row):
                    item = {
                        "text": row[text_col],
                        "id": row[id_col] if id_col is not None and id_col < len(row) else f"text_{i}"
                    }
                    data.append(item)
                    
            return data
            
    elif ext == ".txt":
        # Load text file (one text per line)
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            
        return [{"id": f"text_{i}", "text": line.strip()} for i, line in enumerate(lines) if line.strip()]
        
    # Default empty list if format not recognized
    return []

async def main():
    """Main entry point for benchmarking tool."""
    parser = argparse.ArgumentParser(description="CasaLingua Benchmarking Tool")
    
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
        choices=["cpu", "cuda"], 
        help="Device to use for benchmarking"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        required=True, 
        help="Model to benchmark"
    )
    parser.add_argument(
        "--test-data", 
        type=str, 
        required=True, 
        help="Path to test data file"
    )
    parser.add_argument(
        "--source-lang", 
        type=str, 
        required=True, 
        help="Source language code"
    )
    parser.add_argument(
        "--target-lang", 
        type=str, 
        required=True, 
        help="Target language code"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=1, 
        help="Batch size for processing"
    )
    parser.add_argument(
        "--verify", 
        action="store_true", 
        help="Verify translations"
    )
    parser.add_argument(
        "--compare-models", 
        type=str, 
        help="Comma-separated list of models to compare"
    )
    parser.add_argument(
        "--batch-sizes", 
        type=str, 
        help="Comma-separated list of batch sizes to test"
    )
    
    args = parser.parse_args()
    
    # Initialize benchmark
    benchmark = TranslationBenchmark(
        config_path=args.config,
        output_dir=args.output_dir,
        device=args.device
    )
    await benchmark.initialize()
    try:
        # Load test data
        test_texts = await load_test_data(args.test_data)
        logger.info(f"Loaded {len(test_texts)} test texts from {args.test_data}")
        if args.compare_models:
            # Compare multiple models
            model_ids = [m.strip() for m in args.compare_models.split(",")]
            output_file = Path(args.output_dir) / f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            await benchmark.compare_models(
                model_ids=model_ids,
                test_texts=test_texts,
                source_lang=args.source_lang,
                target_lang=args.target_lang,
                batch_size=args.batch_size,
                verify=args.verify,
                output_file=str(output_file)
            )
        elif args.batch_sizes:
            # Test different batch sizes
            batch_sizes = [int(b.strip()) for b in args.batch_sizes.split(",")]
            output_file = Path(args.output_dir) / f"batch_size_comparison_{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            await benchmark.benchmark_batch_sizes(
                model_id=args.model,
                test_texts=test_texts,
                source_lang=args.source_lang,
                target_lang=args.target_lang,
                batch_sizes=batch_sizes,
                output_file=str(output_file)
            )
        else:
            # Benchmark single model
            await benchmark.benchmark_model(
                model_id=args.model,
                test_texts=test_texts,
                source_lang=args.source_lang,
                target_lang=args.target_lang,
                batch_size=args.batch_size,
                verify=args.verify
            )
    finally:
        # Clean up
        await benchmark.cleanup()

if __name__ == "__main__":
    asyncio.run(main())