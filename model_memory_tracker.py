#!/usr/bin/env python3
"""
Memory Usage Tracker for ML Models in CasaLingua

This script monitors and tracks memory usage of machine learning models in the
CasaLingua application. It helps optimize resource allocation by providing
detailed memory consumption analysis for different models and operations.

Features:
- Real-time memory tracking for models during loading and inference
- GPU and CPU memory profiling
- Memory usage visualization and reports
- Memory leak detection
- Optimization recommendations

Usage:
    python model_memory_tracker.py

Author: Exygy Development Team
Version: 1.0.0
License: MIT
"""

import os
import sys
import json
import time
import asyncio
import logging
import argparse
import psutil
import gc
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import tracemalloc

# Add project root to path for imports
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("memory_tracking.log")
    ]
)
logger = logging.getLogger("memory_tracker")

# Try to import torch for GPU memory tracking
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available, GPU memory tracking disabled")

class MemorySnapshot:
    """Represents a memory snapshot at a specific point in time."""
    
    def __init__(
        self,
        timestamp: float,
        cpu_usage: Dict[str, float],
        gpu_usage: Optional[Dict[str, float]] = None,
        tracemalloc_snapshot: Optional[Any] = None
    ):
        """
        Initialize memory snapshot.
        
        Args:
            timestamp: Unix timestamp
            cpu_usage: CPU memory usage in bytes
            gpu_usage: GPU memory usage in bytes (if available)
            tracemalloc_snapshot: Python object memory snapshot (if tracemalloc enabled)
        """
        self.timestamp = timestamp
        self.cpu_usage = cpu_usage
        self.gpu_usage = gpu_usage
        self.tracemalloc_snapshot = tracemalloc_snapshot
    
    @staticmethod
    def take_snapshot() -> 'MemorySnapshot':
        """
        Take a memory snapshot of current state.
        
        Returns:
            Memory snapshot object
        """
        timestamp = time.time()
        
        # Get CPU memory usage
        process = psutil.Process(os.getpid())
        cpu_usage = {
            "rss": process.memory_info().rss,  # Resident Set Size
            "vms": process.memory_info().vms,  # Virtual Memory Size
            "shared": getattr(process.memory_info(), "shared", 0),  # Shared memory
            "data": getattr(process.memory_info(), "data", 0),  # Data segment
            "system_percent": psutil.virtual_memory().percent,  # System memory usage percent
            "system_available": psutil.virtual_memory().available  # Available system memory
        }
        
        # Get GPU memory usage if available
        gpu_usage = None
        if TORCH_AVAILABLE:
            gpu_usage = {}
            
            # CUDA GPU
            if torch.cuda.is_available():
                gpu_usage["allocated"] = torch.cuda.memory_allocated()
                gpu_usage["reserved"] = torch.cuda.memory_reserved()
                gpu_usage["max_allocated"] = torch.cuda.max_memory_allocated()
                
                # Get per-device stats if multiple GPUs
                if torch.cuda.device_count() > 1:
                    gpu_usage["devices"] = {}
                    for i in range(torch.cuda.device_count()):
                        with torch.cuda.device(i):
                            gpu_usage["devices"][i] = {
                                "allocated": torch.cuda.memory_allocated(),
                                "reserved": torch.cuda.memory_reserved(),
                                "name": torch.cuda.get_device_name(i)
                            }
            
            # Apple Silicon (MPS)
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                if hasattr(torch.mps, "current_allocated_memory"):
                    gpu_usage["allocated"] = torch.mps.current_allocated_memory()
                if hasattr(torch.mps, "driver_allocated_memory"):
                    gpu_usage["reserved"] = torch.mps.driver_allocated_memory()
                if hasattr(torch.mps, "max_memory"):
                    gpu_usage["max_memory"] = torch.mps.max_memory()
        
        # Get tracemalloc snapshot if enabled
        tracemalloc_snapshot = None
        if tracemalloc.is_tracing():
            tracemalloc_snapshot = tracemalloc.take_snapshot()
        
        return MemorySnapshot(timestamp, cpu_usage, gpu_usage, tracemalloc_snapshot)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert snapshot to dictionary for serialization.
        
        Returns:
            Dictionary representation of snapshot
        """
        result = {
            "timestamp": self.timestamp,
            "cpu_usage": self.cpu_usage,
        }
        
        if self.gpu_usage:
            result["gpu_usage"] = self.gpu_usage
        
        # We don't include tracemalloc_snapshot as it's not serializable
        
        return result
    
    def get_rss_mb(self) -> float:
        """
        Get resident set size in MB.
        
        Returns:
            RSS memory usage in MB
        """
        return self.cpu_usage["rss"] / (1024 * 1024)
    
    def get_gpu_allocated_mb(self) -> Optional[float]:
        """
        Get allocated GPU memory in MB.
        
        Returns:
            Allocated GPU memory in MB or None if not available
        """
        if self.gpu_usage and "allocated" in self.gpu_usage:
            return self.gpu_usage["allocated"] / (1024 * 1024)
        return None

class MemoryTracker:
    """Tracks memory usage of models and operations."""
    
    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize memory tracker.
        
        Args:
            output_dir: Directory for saving memory tracking results
        """
        # Set up output directory
        self.output_dir = Path(output_dir) if output_dir else Path("memory_tracking_results")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize tracking state
        self.tracking_active = False
        self.baseline_snapshot = None
        self.snapshots: Dict[str, List[MemorySnapshot]] = {}
        self.current_operation = None
        
        # Start tracing with tracemalloc if available
        try:
            tracemalloc.start()
            logger.info("Tracemalloc started for detailed Python object tracking")
        except Exception as e:
            logger.warning(f"Failed to start tracemalloc: {e}")
    
    def start_tracking(self, operation: str) -> None:
        """
        Start tracking memory for an operation.
        
        Args:
            operation: Name of operation being tracked
        """
        # Force garbage collection to get accurate baseline
        gc.collect()
        if TORCH_AVAILABLE:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
        
        # Take baseline snapshot
        self.baseline_snapshot = MemorySnapshot.take_snapshot()
        
        # Set up tracking for this operation
        self.current_operation = operation
        self.snapshots[operation] = [self.baseline_snapshot]
        self.tracking_active = True
        
        logger.info(f"Started memory tracking for '{operation}'")
        
        # Log baseline memory
        rss_mb = self.baseline_snapshot.get_rss_mb()
        gpu_mb = self.baseline_snapshot.get_gpu_allocated_mb()
        
        logger.info(f"Baseline memory - RSS: {rss_mb:.2f} MB")
        if gpu_mb is not None:
            logger.info(f"Baseline memory - GPU: {gpu_mb:.2f} MB")
    
    def take_snapshot(self, label: Optional[str] = None) -> None:
        """
        Take a memory snapshot during tracking.
        
        Args:
            label: Optional label for this snapshot
        """
        if not self.tracking_active:
            logger.warning("Tried to take snapshot but tracking is not active")
            return
        
        # Take snapshot
        snapshot = MemorySnapshot.take_snapshot()
        
        # Add to snapshots
        self.snapshots[self.current_operation].append(snapshot)
        
        # Log memory change from baseline
        rss_mb = snapshot.get_rss_mb()
        baseline_rss_mb = self.baseline_snapshot.get_rss_mb()
        rss_diff = rss_mb - baseline_rss_mb
        
        gpu_mb = snapshot.get_gpu_allocated_mb()
        baseline_gpu_mb = self.baseline_snapshot.get_gpu_allocated_mb()
        gpu_diff = None
        if gpu_mb is not None and baseline_gpu_mb is not None:
            gpu_diff = gpu_mb - baseline_gpu_mb
        
        snapshot_label = f" ({label})" if label else ""
        logger.info(f"Memory snapshot{snapshot_label} - RSS: {rss_mb:.2f} MB ({rss_diff:+.2f} MB)")
        if gpu_diff is not None:
            logger.info(f"Memory snapshot{snapshot_label} - GPU: {gpu_mb:.2f} MB ({gpu_diff:+.2f} MB)")
    
    def stop_tracking(self) -> Dict[str, Any]:
        """
        Stop tracking and analyze results.
        
        Returns:
            Analysis results dictionary
        """
        if not self.tracking_active:
            logger.warning("Tried to stop tracking but tracking is not active")
            return {}
        
        # Take final snapshot
        final_snapshot = MemorySnapshot.take_snapshot()
        self.snapshots[self.current_operation].append(final_snapshot)
        
        # Calculate memory changes
        baseline_rss = self.baseline_snapshot.get_rss_mb()
        final_rss = final_snapshot.get_rss_mb()
        rss_diff = final_rss - baseline_rss
        
        baseline_gpu = self.baseline_snapshot.get_gpu_allocated_mb()
        final_gpu = final_snapshot.get_gpu_allocated_mb()
        gpu_diff = None
        if baseline_gpu is not None and final_gpu is not None:
            gpu_diff = final_gpu - baseline_gpu
        
        # Log results
        logger.info(f"Stopped memory tracking for '{self.current_operation}'")
        logger.info(f"Final memory - RSS: {final_rss:.2f} MB ({rss_diff:+.2f} MB from baseline)")
        if gpu_diff is not None:
            logger.info(f"Final memory - GPU: {final_gpu:.2f} MB ({gpu_diff:+.2f} MB from baseline)")
        
        # Analyze memory usage
        analysis = self._analyze_memory_usage(self.current_operation)
        
        # Reset tracking state
        self.tracking_active = False
        self.current_operation = None
        
        return analysis
    
    def _analyze_memory_usage(self, operation: str) -> Dict[str, Any]:
        """
        Analyze memory usage for an operation.
        
        Args:
            operation: Name of operation to analyze
            
        Returns:
            Analysis results dictionary
        """
        if operation not in self.snapshots:
            return {}
        
        snapshots = self.snapshots[operation]
        if len(snapshots) < 2:
            return {}
        
        baseline = snapshots[0]
        final = snapshots[-1]
        
        # Calculate CPU memory metrics
        rss_values = [s.get_rss_mb() for s in snapshots]
        
        cpu_analysis = {
            "baseline_mb": baseline.get_rss_mb(),
            "final_mb": final.get_rss_mb(),
            "change_mb": final.get_rss_mb() - baseline.get_rss_mb(),
            "change_percent": ((final.get_rss_mb() - baseline.get_rss_mb()) / baseline.get_rss_mb()) * 100 if baseline.get_rss_mb() > 0 else 0,
            "peak_mb": max(rss_values),
            "peak_diff_mb": max(rss_values) - baseline.get_rss_mb()
        }
        
        # Calculate GPU memory metrics if available
        gpu_analysis = None
        if all(s.get_gpu_allocated_mb() is not None for s in snapshots):
            gpu_values = [s.get_gpu_allocated_mb() for s in snapshots]
            
            gpu_analysis = {
                "baseline_mb": baseline.get_gpu_allocated_mb(),
                "final_mb": final.get_gpu_allocated_mb(),
                "change_mb": final.get_gpu_allocated_mb() - baseline.get_gpu_allocated_mb(),
                "change_percent": ((final.get_gpu_allocated_mb() - baseline.get_gpu_allocated_mb()) / baseline.get_gpu_allocated_mb()) * 100 if baseline.get_gpu_allocated_mb() > 0 else 0,
                "peak_mb": max(gpu_values),
                "peak_diff_mb": max(gpu_values) - baseline.get_gpu_allocated_mb()
            }
        
        # Check for memory leak indicators
        memory_leak_detected = cpu_analysis["change_mb"] > 100  # More than 100MB left allocated
        
        # Build analysis result
        analysis = {
            "operation": operation,
            "timestamp": datetime.now().isoformat(),
            "snapshot_count": len(snapshots),
            "duration_seconds": snapshots[-1].timestamp - snapshots[0].timestamp,
            "cpu_memory": cpu_analysis,
            "gpu_memory": gpu_analysis,
            "memory_leak_detected": memory_leak_detected
        }
        
        # Save analysis results
        self._save_analysis(operation, analysis)
        
        # Generate memory usage chart
        self._generate_memory_chart(operation)
        
        return analysis
    
    def _save_analysis(self, operation: str, analysis: Dict[str, Any]) -> None:
        """
        Save analysis results to file.
        
        Args:
            operation: Operation name
            analysis: Analysis results dictionary
        """
        # Create timestamp suffix
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save analysis as JSON
        json_path = self.output_dir / f"{operation}_memory_analysis_{timestamp}.json"
        with open(json_path, "w") as f:
            json.dump(analysis, f, indent=2)
            
        logger.info(f"Memory analysis saved to {json_path}")
        
        # Save raw snapshot data for reference
        snapshots_data = [s.to_dict() for s in self.snapshots[operation]]
        snapshots_path = self.output_dir / f"{operation}_memory_snapshots_{timestamp}.json"
        with open(snapshots_path, "w") as f:
            json.dump(snapshots_data, f, indent=2)
    
    def _generate_memory_chart(self, operation: str) -> None:
        """
        Generate memory usage chart for an operation.
        
        Args:
            operation: Operation name
        """
        # Only generate chart if matplotlib is available
        if 'plt' not in globals():
            logger.warning("Matplotlib not available, skipping chart generation")
            return
        
        if operation not in self.snapshots:
            return
        
        snapshots = self.snapshots[operation]
        if len(snapshots) < 2:
            return
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Prepare data
        timestamps = [(s.timestamp - snapshots[0].timestamp) for s in snapshots]
        rss_values = [s.get_rss_mb() for s in snapshots]
        
        # Plot CPU memory usage
        plt.plot(timestamps, rss_values, 'b-', label='RSS Memory (MB)')
        
        # Plot GPU memory usage if available
        if all(s.get_gpu_allocated_mb() is not None for s in snapshots):
            gpu_values = [s.get_gpu_allocated_mb() for s in snapshots]
            plt.plot(timestamps, gpu_values, 'r-', label='GPU Memory (MB)')
        
        # Add labels and title
        plt.xlabel('Time (seconds)')
        plt.ylabel('Memory Usage (MB)')
        plt.title(f'Memory Usage During {operation}')
        plt.grid(True)
        plt.legend()
        
        # Save figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        chart_path = self.output_dir / f"{operation}_memory_chart_{timestamp}.png"
        plt.savefig(chart_path)
        plt.close()
        
        logger.info(f"Memory usage chart saved to {chart_path}")

class ModelMemoryProfiler:
    """Profiles memory usage of CasaLingua models during loading and inference."""
    
    def __init__(self, config_path: Optional[str] = None, output_dir: Optional[str] = None):
        """
        Initialize model memory profiler.
        
        Args:
            config_path: Path to configuration file
            output_dir: Directory for profiling results
        """
        # Import app modules
        from app.utils.config import load_config
        
        # Load configuration
        self.config = load_config(config_path) if config_path else load_config()
        
        # Set up memory tracker
        self.memory_tracker = MemoryTracker(output_dir)
        
        # Initialize components
        self.processor = None
        self.model_manager = None
        
        # Track profiling results
        self.profile_results = {}
    
    async def initialize(self) -> None:
        """Initialize profiling components."""
        from app.services.models.loader import ModelLoader, load_registry_config
        from app.services.models.manager import EnhancedModelManager
        from app.core.pipeline.processor import UnifiedProcessor
        from app.audit.logger import AuditLogger
        from app.audit.metrics import MetricsCollector
        
        logger.info("Initializing profiling components...")
        
        try:
            # Create model loader
            self.model_loader = ModelLoader(config=self.config)
            
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
            
            # Create enhanced model manager that tracks memory usage
            self.model_manager = EnhancedModelManager(
                self.model_loader, hardware_info, self.config
            )
            
            # Create processor
            self.processor = UnifiedProcessor(
                self.model_manager, audit_logger, metrics, 
                self.config, registry_config
            )
            
            # Wrap model loading methods to track memory
            await self._wrap_model_methods()
            
            # Initialize processor
            self.memory_tracker.start_tracking("processor_initialization")
            await self.processor.initialize()
            initialization_analysis = self.memory_tracker.stop_tracking()
            self.profile_results["initialization"] = initialization_analysis
            
            logger.info("Profiling components initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing profiling components: {e}", exc_info=True)
            raise
    
    async def _wrap_model_methods(self) -> None:
        """Wrap model loading and inference methods to track memory usage."""
        # Save original methods
        original_load_model = self.model_loader.load_model
        
        # Wrap load_model method to track memory
        async def memory_tracked_load_model(model_id: str, *args, **kwargs):
            self.memory_tracker.start_tracking(f"load_model_{model_id}")
            try:
                result = await original_load_model(model_id, *args, **kwargs)
                return result
            finally:
                analysis = self.memory_tracker.stop_tracking()
                self.profile_results[f"load_model_{model_id}"] = analysis
        
        # Replace methods with wrapped versions
        self.model_loader.load_model = memory_tracked_load_model
    
    async def profile_model_loading(self, model_ids: List[str]) -> Dict[str, Any]:
        """
        Profile memory usage during model loading.
        
        Args:
            model_ids: List of model IDs to profile
            
        Returns:
            Dictionary of profiling results by model ID
        """
        logger.info(f"Profiling memory usage for loading {len(model_ids)} models")
        
        results = {}
        
        for model_id in model_ids:
            logger.info(f"Loading model: {model_id}")
            
            # Unload all models first to start with clean state
            await self.model_manager.unload_all_models()
            
            # Force garbage collection
            gc.collect()
            if TORCH_AVAILABLE:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                elif hasattr(torch.mps, "empty_cache"):
                    torch.mps.empty_cache()
            
            # Start memory tracking
            self.memory_tracker.start_tracking(f"model_loading_{model_id}")
            
            try:
                # Load model
                model, tokenizer = await self.model_loader.load_model(model_id)
                
                # Take a snapshot after loading
                self.memory_tracker.take_snapshot("after_loading")
                
                # Get model size info if available
                model_size_mb = None
                if hasattr(model, "get_memory_footprint"):
                    model_size_mb = model.get_memory_footprint() / (1024 * 1024)
                    logger.info(f"Model memory footprint: {model_size_mb:.2f} MB")
                
                # Stop tracking and get analysis
                analysis = self.memory_tracker.stop_tracking()
                
                # Add model size info to analysis
                if model_size_mb is not None:
                    analysis["model_footprint_mb"] = model_size_mb
                
                results[model_id] = analysis
                
            except Exception as e:
                logger.error(f"Error loading model {model_id}: {e}")
                self.memory_tracker.stop_tracking()
        
        return results
    
    async def profile_inference(
        self,
        model_id: str,
        sample_texts: List[str],
        operation: str,
        batch_size: int = 1
    ) -> Dict[str, Any]:
        """
        Profile memory usage during model inference.
        
        Args:
            model_id: Model ID to profile
            sample_texts: List of sample texts for inference
            operation: Operation type (translate, simplify, etc.)
            batch_size: Batch size for processing
            
        Returns:
            Dictionary of profiling results
        """
        logger.info(f"Profiling memory usage for {operation} with model {model_id}")
        
        # Check that processor is initialized
        if not self.processor:
            raise ValueError("Processor not initialized. Call initialize() first.")
        
        # Make sure model is loaded
        if not await self.model_manager.is_model_loaded(model_id):
            logger.info(f"Loading model {model_id} before profiling")
            await self.model_loader.load_model(model_id)
        
        # Force garbage collection
        gc.collect()
        if TORCH_AVAILABLE:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
        
        # Start memory tracking
        self.memory_tracker.start_tracking(f"{operation}_{model_id}")
        
        try:
            # Process each sample text
            for i in range(0, len(sample_texts), batch_size):
                batch = sample_texts[i:i+batch_size]
                
                # Take snapshot before processing
                self.memory_tracker.take_snapshot(f"before_batch_{i}")
                
                # Process text based on operation type
                if operation == "translate":
                    for text in batch:
                        await self.processor.process_translation(
                            text=text,
                            source_language="en",
                            target_language="es",
                            model_id=model_id
                        )
                elif operation == "simplify":
                    for text in batch:
                        if hasattr(self.processor, "simplify_text"):
                            await self.processor.simplify_text(
                                text=text,
                                target_level="simple",
                                language="en"
                            )
                elif operation == "anonymize":
                    for text in batch:
                        if hasattr(self.processor, "anonymize_text"):
                            await self.processor.anonymize_text(
                                text=text,
                                language="en"
                            )
                elif operation == "detect":
                    for text in batch:
                        await self.processor.detect_language(text=text)
                elif operation == "analyze":
                    for text in batch:
                        await self.processor.analyze_text(
                            text=text,
                            language="en",
                            analyses=["sentiment", "entities"]
                        )
                elif operation == "summarize":
                    for text in batch:
                        await self.processor.process_summarization(
                            text=text,
                            language="en"
                        )
                
                # Take snapshot after processing
                self.memory_tracker.take_snapshot(f"after_batch_{i}")
        
        except Exception as e:
            logger.error(f"Error during {operation} profiling: {e}")
        
        # Stop tracking and get analysis
        analysis = self.memory_tracker.stop_tracking()
        
        return analysis
    
    async def profile_memory_with_batch_sizes(
        self,
        model_id: str,
        operation: str,
        batch_sizes: List[int] = [1, 2, 4, 8]
    ) -> Dict[str, Any]:
        """
        Profile memory usage with different batch sizes.
        
        Args:
            model_id: Model ID to profile
            operation: Operation type
            batch_sizes: List of batch sizes to test
            
        Returns:
            Dictionary of profiling results by batch size
        """
        results = {}
        
        # Sample texts for testing
        sample_texts = [
            "This is a test of the model memory profiling system.",
            "The quick brown fox jumps over the lazy dog.",
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
            "Machine learning models can consume significant memory.",
            "Memory profiling helps optimize resource allocation.",
            "Efficient models run faster and use less memory.",
            "The mitochondrion is the powerhouse of the cell.",
            "Python is a widely used programming language.",
            "Cloud computing provides scalable resources.",
            "Neural networks can learn complex patterns."
        ]
        
        # Test each batch size
        for batch_size in batch_sizes:
            logger.info(f"Profiling {operation} with batch size {batch_size}")
            
            # Unload all models and load just the one we need
            await self.model_manager.unload_all_models()
            await self.model_loader.load_model(model_id)
            
            # Force garbage collection
            gc.collect()
            if TORCH_AVAILABLE:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                elif hasattr(torch.mps, "empty_cache"):
                    torch.mps.empty_cache()
            
            # Run profiling with this batch size
            analysis = await self.profile_inference(
                model_id=model_id,
                sample_texts=sample_texts,
                operation=operation,
                batch_size=batch_size
            )
            
            results[batch_size] = analysis
        
        # Generate comparison chart
        self._generate_batch_comparison_chart(model_id, operation, results)
        
        return results
    
    def _generate_batch_comparison_chart(
        self,
        model_id: str,
        operation: str,
        results: Dict[int, Dict[str, Any]]
    ) -> None:
        """
        Generate chart comparing memory usage with different batch sizes.
        
        Args:
            model_id: Model ID profiled
            operation: Operation type
            results: Dictionary of profiling results by batch size
        """
        # Only generate chart if matplotlib is available
        if 'plt' not in globals():
            return
        
        # Extract batch sizes and memory usage
        batch_sizes = sorted(results.keys())
        
        # Extract CPU peak memory
        cpu_peaks = [results[bs]["cpu_memory"]["peak_mb"] for bs in batch_sizes]
        
        # Extract GPU peak memory if available
        gpu_peaks = None
        if all(results[bs].get("gpu_memory") is not None for bs in batch_sizes):
            gpu_peaks = [results[bs]["gpu_memory"]["peak_mb"] for bs in batch_sizes]
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Plot CPU memory
        plt.plot(batch_sizes, cpu_peaks, 'b-o', label='CPU Peak Memory (MB)')
        
        # Plot GPU memory if available
        if gpu_peaks:
            plt.plot(batch_sizes, gpu_peaks, 'r-o', label='GPU Peak Memory (MB)')
        
        # Add labels and title
        plt.xlabel('Batch Size')
        plt.ylabel('Peak Memory Usage (MB)')
        plt.title(f'Memory Usage vs Batch Size for {operation.capitalize()} ({model_id})')
        plt.grid(True)
        plt.legend()
        
        # Set x-axis ticks to match batch sizes
        plt.xticks(batch_sizes)
        
        # Save figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        chart_path = self.memory_tracker.output_dir / f"{operation}_{model_id}_batch_comparison_{timestamp}.png"
        plt.savefig(chart_path)
        plt.close()
        
        logger.info(f"Batch comparison chart saved to {chart_path}")
    
    async def profile_all_models(self) -> Dict[str, Any]:
        """
        Profile memory usage for all models in the system.
        
        Returns:
            Dictionary of profiling results by model ID
        """
        # Get available model IDs
        model_ids = await self.model_manager.get_available_models()
        
        # Sort into types
        model_types = {
            "translation": [],
            "simplification": [],
            "detection": [],
            "anonymization": [],
            "summarization": [],
            "analysis": []
        }
        
        # Categorize models by type
        for model_id in model_ids:
            if "translation" in model_id:
                model_types["translation"].append(model_id)
            elif "simplifier" in model_id or "t5" in model_id.lower():
                model_types["simplification"].append(model_id)
            elif "detection" in model_id or "detector" in model_id:
                model_types["detection"].append(model_id)
            elif "anonymizer" in model_id or "anonymization" in model_id:
                model_types["anonymization"].append(model_id)
            elif "summarizer" in model_id or "summarization" in model_id:
                model_types["summarization"].append(model_id)
            elif "analysis" in model_id or "analyzer" in model_id:
                model_types["analysis"].append(model_id)
            else:
                # Default to translation
                model_types["translation"].append(model_id)
        
        # Profile loading memory usage for each model
        loading_results = await self.profile_model_loading(model_ids)
        
        # Profile inference memory usage
        inference_results = {}
        
        # Sample text for testing
        sample_text = "This is a sample text for testing model memory usage during inference."
        
        operations = {
            "translation": "translate",
            "simplification": "simplify",
            "detection": "detect",
            "anonymization": "anonymize",
            "summarization": "summarize",
            "analysis": "analyze"
        }
        
        # Profile one model per type
        for model_type, models in model_types.items():
            if models:
                model_id = models[0]  # Use first model of each type
                operation = operations.get(model_type, "translate")
                
                inference_results[model_id] = await self.profile_inference(
                    model_id=model_id,
                    sample_texts=[sample_text] * 5,  # 5 repetitions
                    operation=operation
                )
        
        # Combine results
        all_results = {
            "timestamp": datetime.now().isoformat(),
            "model_loading": loading_results,
            "model_inference": inference_results
        }
        
        # Generate summary report
        self._generate_summary_report(all_results)
        
        return all_results
    
    def _generate_summary_report(self, results: Dict[str, Any]) -> None:
        """
        Generate a summary report of memory profiling results.
        
        Args:
            results: Dictionary of profiling results
        """
        # Create timestamp suffix
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save full results as JSON
        json_path = self.memory_tracker.output_dir / f"memory_profiling_results_{timestamp}.json"
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
        
        # Create CSV summary for loading
        loading_csv_path = self.memory_tracker.output_dir / f"model_loading_summary_{timestamp}.csv"
        with open(loading_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow([
                "Model ID", "CPU Baseline (MB)", "CPU Final (MB)", "CPU Change (MB)", 
                "CPU Peak (MB)", "GPU Baseline (MB)", "GPU Final (MB)", "GPU Change (MB)",
                "GPU Peak (MB)"
            ])
            
            # Write data for each model
            for model_id, analysis in results["model_loading"].items():
                cpu_memory = analysis.get("cpu_memory", {})
                gpu_memory = analysis.get("gpu_memory", {})
                
                row = [
                    model_id.replace("load_model_", ""),
                    cpu_memory.get("baseline_mb", "N/A"),
                    cpu_memory.get("final_mb", "N/A"),
                    cpu_memory.get("change_mb", "N/A"),
                    cpu_memory.get("peak_mb", "N/A")
                ]
                
                if gpu_memory:
                    row.extend([
                        gpu_memory.get("baseline_mb", "N/A"),
                        gpu_memory.get("final_mb", "N/A"),
                        gpu_memory.get("change_mb", "N/A"),
                        gpu_memory.get("peak_mb", "N/A")
                    ])
                else:
                    row.extend(["N/A", "N/A", "N/A", "N/A"])
                
                writer.writerow(row)
        
        # Create CSV summary for inference
        inference_csv_path = self.memory_tracker.output_dir / f"model_inference_summary_{timestamp}.csv"
        with open(inference_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow([
                "Model ID", "Operation", "CPU Baseline (MB)", "CPU Final (MB)", 
                "CPU Change (MB)", "CPU Peak (MB)", "Memory Leak Detected"
            ])
            
            # Write data for each model
            for model_id, analysis in results["model_inference"].items():
                cpu_memory = analysis.get("cpu_memory", {})
                
                writer.writerow([
                    model_id,
                    analysis.get("operation", "").replace(f"_{model_id}", ""),
                    cpu_memory.get("baseline_mb", "N/A"),
                    cpu_memory.get("final_mb", "N/A"),
                    cpu_memory.get("change_mb", "N/A"),
                    cpu_memory.get("peak_mb", "N/A"),
                    "Yes" if analysis.get("memory_leak_detected", False) else "No"
                ])
        
        logger.info(f"Memory profiling summary reports saved to {self.memory_tracker.output_dir}")
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        logger.info("Cleaning up resources...")
        
        if self.processor:
            try:
                await self.processor.shutdown()
            except Exception as e:
                logger.warning(f"Error during processor shutdown: {e}")
        
        # Stop tracemalloc if it was started
        if tracemalloc.is_tracing():
            tracemalloc.stop()

async def main():
    """Main entry point for model memory tracking script."""
    parser = argparse.ArgumentParser(description="CasaLingua Model Memory Tracker")
    
    parser.add_argument(
        "--config", 
        type=str, 
        help="Path to configuration file"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="memory_tracking_results", 
        help="Directory for memory tracking results"
    )
    parser.add_argument(
        "--profile-all", 
        action="store_true", 
        help="Profile all available models"
    )
    parser.add_argument(
        "--model-id", 
        type=str, 
        help="Specific model ID to profile"
    )
    parser.add_argument(
        "--operation", 
        type=str, 
        choices=["translate", "simplify", "detect", "anonymize", "analyze", "summarize"], 
        default="translate", 
        help="Operation to profile"
    )
    parser.add_argument(
        "--batch-test", 
        action="store_true", 
        help="Test different batch sizes"
    )
    
    args = parser.parse_args()
    
    # Initialize profiler
    profiler = ModelMemoryProfiler(
        config_path=args.config,
        output_dir=args.output_dir
    )
    
    try:
        # Initialize components
        await profiler.initialize()
        
        if args.profile_all:
            # Profile all models
            await profiler.profile_all_models()
        elif args.model_id:
            if args.batch_test:
                # Profile with different batch sizes
                await profiler.profile_memory_with_batch_sizes(
                    model_id=args.model_id,
                    operation=args.operation,
                    batch_sizes=[1, 2, 4, 8, 16]
                )
            else:
                # Profile specific model
                sample_texts = [
                    "This is a test of the model memory profiling system.",
                    "The quick brown fox jumps over the lazy dog.",
                    "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
                    "Machine learning models can consume significant memory.",
                    "Memory profiling helps optimize resource allocation."
                ]
                
                # Profile loading
                await profiler.profile_model_loading([args.model_id])
                
                # Profile inference
                await profiler.profile_inference(
                    model_id=args.model_id,
                    sample_texts=sample_texts,
                    operation=args.operation
                )
        else:
            # Default to profiling all
            await profiler.profile_all_models()
        
    finally:
        # Clean up
        await profiler.cleanup()

if __name__ == "__main__":
    asyncio.run(main())