"""
CasaLingua - Language Processing & Translation Pipeline

This module serves as the primary entry point for the CasaLingua API application.
It initializes the FastAPI application, configures middleware, and establishes
routing for all API endpoints. The application is designed to process multiple
types of input (text, audio, documents) through specialized pipelines.

Author: Exygy Development Team
Version: 1.0.0
License: MIT
"""

import os
from dotenv import load_dotenv
load_dotenv()

# Set development mode during development and testing
os.environ["CASALINGUA_ENV"] = "development"
ENVIRONMENT = os.getenv("CASALINGUA_ENV", "production").lower()
print(f"🔧 Starting CasaLingua in {ENVIRONMENT} mode")
import sys
import time
import logging
import platform
import torch
import psutil
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Literal
from enum import Enum
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
from pathlib import Path

# Import rich Table and Panel for UI summaries
from rich.table import Table
from rich.panel import Panel

from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Import UI components first for proper initialization
from app.ui.colors import init_terminal_colors
from app.ui.banner import print_startup_banner
from app.ui.console import setup_console_logging
from app.ui.console import console

# Import model management
from app.services.models.loader import ModelLoader
from app.services.models.manager import EnhancedModelManager

# Import SimpleHardwareDetector if needed later
# from app.services.hardware.simple_detector import SimpleHardwareDetector

# Import core components
from app.core.pipeline.processor import UnifiedProcessor
# Import the tokenizer pipeline
from app.core.pipeline.tokenizer import TokenizerPipeline

# Import audit components
from app.audit.logger import AuditLogger
from app.audit.metrics import MetricsCollector

# Import utility functions
from app.utils.config import load_config
from app.utils.logging import configure_logging

# Import API routers
from app.api.routes.admin import router as admin_router
from app.api.routes.pipeline import router as pipeline_router
from app.api.routes.rag import router as rag_router
from app.api.routes.health import router as health_router

# Updated schema imports
from app.api.schemas.base import BaseResponse, StatusEnum, MetadataModel, ErrorDetail
from app.api.schemas.translation import TranslationResult, TranslationResponse, DocumentTranslationResult, DocumentTranslationResponse
from app.api.schemas.language import LanguageDetectionResult, LanguageDetectionResponse
from app.api.schemas.analysis import TextAnalysisResult, TextAnalysisResponse
from app.api.schemas.queue import QueueStatus, QueueStatusResponse
from app.api.schemas.verification import VerificationResult, VerificationResponse

# Initialize terminal colors
init_terminal_colors()

# Configure base logging
log_level = "DEBUG" if ENVIRONMENT == "development" else "INFO"
console_logger = setup_console_logging(log_level)
logging.getLogger().handlers.clear()
logging.getLogger().addHandler(console_logger.handlers[0])
if ENVIRONMENT == "development":
    print("🚧 Dev mode active: DEBUG logging enabled, hot reload expected.")

# Suppress Hugging Face model loading messages
import transformers
transformers.logging.set_verbosity_error()

# Model size options
class ModelSize(str, Enum):
    LARGE = "large"
    MEDIUM = "medium"
    SMALL = "small"

# Model types
class ModelType(str, Enum):
    TRANSLATION = "translation"
    MULTIPURPOSE = "multipurpose"
    VERIFICATION = "verification"

# Processor types
class ProcessorType(str, Enum):
    APPLE_SILICON = "apple_silicon"
    INTEL = "intel"
    NVIDIA = "nvidia"
    AMD = "amd"
    OTHER = "other"

@dataclass
class ModelConfig:
    size: ModelSize
    memory_required: int  # in bytes
    model_path: str
    quantization: int  # bits (4, 8, 16)

@dataclass
class GPUInfo:
    """Information about a single GPU device"""
    device_id: int
    name: str
    memory_total: int  # in bytes
    memory_available: int  # in bytes
    compute_capability: Optional[str] = None
    vendor: str = "unknown"
    
    def __str__(self) -> str:
        return f"{self.name} (ID: {self.device_id}, {self.memory_total / (1024**3):.1f} GB)"

@dataclass
class EnhancedHardwareInfo:
    """Comprehensive hardware information with multi-GPU support"""
    total_memory: int  # in bytes
    available_memory: int  # in bytes
    processor_type: ProcessorType
    has_gpu: bool
    cpu_cores: int = 0
    cpu_threads: int = 0
    system_name: Optional[str] = None
    gpu_count: int = 0
    
    # For backwards compatibility
    gpu_memory: Optional[int] = None  # Total memory of first GPU, None if no GPU
    gpu_name: Optional[str] = None  # Name of first GPU, None if no GPU
    
    # New multi-GPU support
    gpus: List[GPUInfo] = field(default_factory=list)  # List of all GPUs
    
    def get_best_gpu(self) -> Optional[GPUInfo]:
        """Returns the GPU with the most available memory, or None if no GPUs"""
        if not self.gpus:
            return None
        return max(self.gpus, key=lambda gpu: gpu.memory_available)
    
    def get_gpu_by_id(self, device_id: int) -> Optional[GPUInfo]:
        """Returns the GPU with the specified device ID, or None if not found"""
        for gpu in self.gpus:
            if gpu.device_id == device_id:
                return gpu
        return None
    
    def get_total_gpu_memory(self) -> int:
        """Returns the total memory across all GPUs in bytes"""
        return sum(gpu.memory_total for gpu in self.gpus)
        
    def get_available_gpu_memory(self) -> int:
        """Returns the available memory across all GPUs in bytes"""
        return sum(gpu.memory_available for gpu in self.gpus)

class ModelSizeConfig:
    """Configuration details for models of different sizes"""
    
    # Memory requirements in GB, converted to bytes
    MEMORY_REQUIREMENTS = {
        ModelType.TRANSLATION: {
            ModelSize.LARGE: 12 * 1024 * 1024 * 1024,
            ModelSize.MEDIUM: 6 * 1024 * 1024 * 1024,
            ModelSize.SMALL: 2 * 1024 * 1024 * 1024
        },
        ModelType.MULTIPURPOSE: {
            ModelSize.LARGE: 16 * 1024 * 1024 * 1024,
            ModelSize.MEDIUM: 8 * 1024 * 1024 * 1024,
            ModelSize.SMALL: 4 * 1024 * 1024 * 1024
        },
        ModelType.VERIFICATION: {
            ModelSize.LARGE: 10 * 1024 * 1024 * 1024,
            ModelSize.MEDIUM: 5 * 1024 * 1024 * 1024,
            ModelSize.SMALL: 2 * 1024 * 1024 * 1024
        }
    }
    
    # Model paths based on type and size
    MODEL_PATHS = {
        ModelType.TRANSLATION: {
            ModelSize.LARGE: "models/translation/large",
            ModelSize.MEDIUM: "models/translation/medium",
            ModelSize.SMALL: "models/translation/small"
        },
        ModelType.MULTIPURPOSE: {
            ModelSize.LARGE: "models/multipurpose/large",
            ModelSize.MEDIUM: "models/multipurpose/medium",
            ModelSize.SMALL: "models/multipurpose/small"
        },
        ModelType.VERIFICATION: {
            ModelSize.LARGE: "models/verification/large",
            ModelSize.MEDIUM: "models/verification/medium",
            ModelSize.SMALL: "models/verification/small"
        }
    }

class EnhancedHardwareDetector:
    """Enhanced hardware detector that provides detailed system information and model sizing recommendations."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger("casa_lingua.hardware")
    
    async def detect_all(self) -> EnhancedHardwareInfo:
        """
        Detect all hardware capabilities and return comprehensive information
        with multi-GPU support.
        """
        self.logger.info("Detecting hardware capabilities...")
        
        # Get memory information
        mem = psutil.virtual_memory()
        total_memory = mem.total
        available_memory = mem.available
        
        # Get CPU information
        cpu_cores = psutil.cpu_count(logical=False)
        cpu_threads = psutil.cpu_count(logical=True)
        
        # Detect processor type
        processor_type = self._detect_processor_type()
        system_name = platform.system()
        
        # Check for GPUs with the enhanced multi-GPU detection
        has_gpu, first_gpu_memory, first_gpu_name, gpu_list = self._detect_gpu()
        
        # Create enhanced hardware info with multi-GPU support
        hardware_info = EnhancedHardwareInfo(
            total_memory=total_memory,
            available_memory=available_memory,
            processor_type=processor_type,
            has_gpu=has_gpu,
            gpu_memory=first_gpu_memory,  # For backward compatibility
            gpu_name=first_gpu_name,      # For backward compatibility
            cpu_cores=cpu_cores,
            cpu_threads=cpu_threads,
            system_name=system_name,
            gpu_count=len(gpu_list),
            gpus=gpu_list
        )
        
        # Log hardware details
        self.logger.info(f"Hardware Detection Results:")
        self.logger.info(f"  System: {system_name}")
        self.logger.info(f"  Processor: {processor_type.value}")
        self.logger.info(f"  CPU Cores: {cpu_cores} (Physical), {cpu_threads} (Logical)")
        self.logger.info(f"  Memory: {total_memory / (1024**3):.1f} GB (Total), {available_memory / (1024**3):.1f} GB (Available)")
        
        if has_gpu:
            # Log summary of all GPUs
            self.logger.info(f"  GPUs: {len(gpu_list)} device(s) detected")
            
            # Log detailed information for each GPU
            for i, gpu in enumerate(gpu_list):
                self.logger.info(f"  GPU {i}: {gpu.name}")
                self.logger.info(f"    Memory: {gpu.memory_total / (1024**3):.1f} GB total, {gpu.memory_available / (1024**3):.1f} GB available")
                if gpu.compute_capability:
                    self.logger.info(f"    Compute Capability: {gpu.compute_capability}")
                self.logger.info(f"    Vendor: {gpu.vendor}")
                
            # Log total GPU memory
            if len(gpu_list) > 1:
                total_gpu_memory = hardware_info.get_total_gpu_memory()
                available_gpu_memory = hardware_info.get_available_gpu_memory()
                self.logger.info(f"  Total GPU Memory: {total_gpu_memory / (1024**3):.1f} GB total, {available_gpu_memory / (1024**3):.1f} GB available")
        else:
            self.logger.info("  GPU: None detected")
        
        return hardware_info
    
    def _detect_processor_type(self) -> ProcessorType:
        """Detect the type of processor."""
        system = platform.system()
        machine = platform.machine()
        
        if system == "Darwin" and machine in ["arm64", "arm"]:
            return ProcessorType.APPLE_SILICON
        elif "intel" in platform.processor().lower():
            return ProcessorType.INTEL
        elif system == "Linux":
            # Check for NVIDIA or AMD via GPU detection
            try:
                if torch.cuda.is_available():
                    # Check vendor name in device properties
                    device_name = torch.cuda.get_device_name(0).lower()
                    if "nvidia" in device_name:
                        return ProcessorType.NVIDIA
                    elif "amd" in device_name:
                        return ProcessorType.AMD
            except:
                pass
            
        return ProcessorType.OTHER
    
    def _detect_gpu(self) -> Tuple[bool, Optional[int], Optional[str], List[GPUInfo]]:
        """
        Detect all available GPUs and gather detailed information about each one.
        
        Returns:
            Tuple containing:
            - bool: Whether any GPU is available
            - Optional[int]: Memory of first GPU in bytes (for backward compat)
            - Optional[str]: Name of first GPU (for backward compat)
            - List[GPUInfo]: List of all GPU details
        """
        gpu_list = []
        has_gpu = torch.cuda.is_available()
        first_gpu_memory = None
        first_gpu_name = None
        
        if has_gpu:
            # Get GPU info via torch
            device_count = torch.cuda.device_count()
            self.logger.info(f"Detected {device_count} GPU devices")
            
            for device_id in range(device_count):
                try:
                    # Get properties for this GPU
                    device_props = torch.cuda.get_device_properties(device_id)
                    
                    # Get memory information
                    gpu_memory_total = device_props.total_memory
                    
                    # Try to estimate available memory
                    # This is an approximation since torch doesn't expose this directly
                    try:
                        torch.cuda.set_device(device_id)
                        torch.cuda.empty_cache()
                        # Reserve a small tensor to force memory allocation
                        temp = torch.zeros(1024, device=f'cuda:{device_id}')
                        # Get memory stats after allocation
                        gpu_memory_allocated = torch.cuda.memory_allocated(device_id)
                        gpu_memory_reserved = torch.cuda.memory_reserved(device_id)
                        # Calculate available as total minus reserved
                        gpu_memory_available = gpu_memory_total - gpu_memory_reserved
                        # Clean up
                        del temp
                        torch.cuda.empty_cache()
                    except Exception as e:
                        self.logger.warning(f"Could not determine available memory for GPU {device_id}: {e}")
                        # Assume 90% of total memory is available if we can't measure it
                        gpu_memory_available = int(gpu_memory_total * 0.9)
                    
                    # Determine vendor
                    gpu_name = device_props.name.lower()
                    if "nvidia" in gpu_name:
                        vendor = "nvidia"
                    elif "amd" in gpu_name or "radeon" in gpu_name:
                        vendor = "amd"
                    else:
                        vendor = "unknown"
                    
                    # Get compute capability for NVIDIA GPUs
                    compute_capability = None
                    if vendor == "nvidia":
                        compute_capability = f"{device_props.major}.{device_props.minor}"
                    
                    # Create GPU info object
                    gpu_info = GPUInfo(
                        device_id=device_id,
                        name=device_props.name,
                        memory_total=gpu_memory_total,
                        memory_available=gpu_memory_available,
                        compute_capability=compute_capability,
                        vendor=vendor
                    )
                    
                    # Add to list
                    gpu_list.append(gpu_info)
                    
                    # Store first GPU info for backward compatibility
                    if device_id == 0:
                        first_gpu_memory = gpu_memory_total
                        first_gpu_name = device_props.name
                    
                    # Log GPU details
                    self.logger.info(f"GPU {device_id}: {device_props.name}, "
                                   f"Memory: {gpu_memory_total / (1024**3):.1f} GB total, "
                                   f"{gpu_memory_available / (1024**3):.1f} GB available")
                    
                except Exception as e:
                    self.logger.error(f"Error detecting GPU {device_id}: {e}")
        
        return has_gpu, first_gpu_memory, first_gpu_name, gpu_list
    
    def recommend_config(self) -> Dict[ModelType, ModelSize]:
        """Determine appropriate model sizes based on hardware capabilities."""
        self.logger.info("Determining optimal model configuration based on hardware...")

        # Use enhanced detection for unified memory on Apple Silicon
        if self.config.get("model_size") in ["small", "medium", "large"]:
            override = self.config["model_size"]
            self.logger.info(f"Model size manually set to '{override}' in config.")
            return {
                ModelType.TRANSLATION: ModelSize(override),
                ModelType.MULTIPURPOSE: ModelSize(override),
                ModelType.VERIFICATION: ModelSize(override)
            }

        # Apple Silicon logic: use total unified memory
        if self.config.get("processor_type") == ProcessorType.APPLE_SILICON or self._detect_processor_type() == ProcessorType.APPLE_SILICON:
            effective_memory = psutil.virtual_memory().total
        else:
            mem = psutil.virtual_memory()
            gpu_mem = 0
            if torch.cuda.is_available():
                try:
                    gpu_mem = torch.cuda.get_device_properties(0).total_memory
                except:
                    pass
            effective_memory = min(mem.available, gpu_mem) if gpu_mem else mem.available

        if effective_memory >= 40 * 1024**3:
            self.logger.info("High-end environment detected: Using all large models")
            return {
                ModelType.TRANSLATION: ModelSize.LARGE,
                ModelType.MULTIPURPOSE: ModelSize.LARGE,
                ModelType.VERIFICATION: ModelSize.LARGE
            }
        elif effective_memory >= 25 * 1024**3:
            self.logger.info("Mid-range environment detected: Using mixed large/medium models")
            return {
                ModelType.TRANSLATION: ModelSize.LARGE,
                ModelType.MULTIPURPOSE: ModelSize.MEDIUM,
                ModelType.VERIFICATION: ModelSize.MEDIUM
            }
        elif effective_memory >= 15 * 1024**3:
            self.logger.info("Standard environment detected: Using medium/small models")
            return {
                ModelType.TRANSLATION: ModelSize.MEDIUM,
                ModelType.MULTIPURPOSE: ModelSize.SMALL,
                ModelType.VERIFICATION: ModelSize.SMALL
            }
        else:
            self.logger.info("Limited environment detected: Using all small models")
            return {
                ModelType.TRANSLATION: ModelSize.SMALL,
                ModelType.MULTIPURPOSE: ModelSize.SMALL,
                ModelType.VERIFICATION: ModelSize.SMALL
            }
    
    def apply_configuration(self, model_sizes: Dict[ModelType, ModelSize]) -> Dict[str, Any]:
        """Apply the model configuration to the system and return updated config."""
        # Create a configuration dictionary to integrate with existing configs
        model_config = {}
        for model_type, size in model_sizes.items():
            # Determine the quantization level based on processor type
            quantization = self._determine_quantization(model_type)
            
            # Store detailed configuration
            model_config[model_type.value] = {
                "size": size.value,
                "quantization": quantization,
                "model_path": ModelSizeConfig.MODEL_PATHS[model_type][size],
                "memory_required": ModelSizeConfig.MEMORY_REQUIREMENTS[model_type][size] * (quantization / 16)
            }
        
        # Log configuration details
        self.logger.info("Applied model configuration:")
        for model_type, config in model_config.items():
            self.logger.info(f"  {model_type}: {config['size']} size, {config['quantization']}-bit quantization, {config['memory_required'] / (1024**3):.2f} GB required")
        
        # Calculate total memory required
        total_memory_required = sum(config["memory_required"] for config in model_config.values())
        self.logger.info(f"Total memory required: {total_memory_required / (1024**3):.2f} GB")
        
        return model_config
    
    def _determine_quantization(self, model_type: ModelType) -> int:
        """
        Determine the most efficient quantization level based on hardware and memory constraints.
        Implements aggressive quantization for low-memory environments.
        
        Returns:
            int: Quantization level in bits (4, 8, or 16)
        """
        processor_type = self._detect_processor_type()
        model_sizes = self.recommend_config()
        
        # Get memory information
        mem = psutil.virtual_memory()
        available_mem_gb = mem.available / (1024**3)
        
        # Get CPU information for thread-count based decision making
        cpu_threads = psutil.cpu_count(logical=True)
        
        # Check for low memory environment (less than 8GB available)
        is_low_memory = available_mem_gb < 8.0
        # Check for very low memory environment (less than 4GB available)
        is_very_low_memory = available_mem_gb < 4.0
        # Check for extreme low memory environment (less than 2GB available)
        is_extreme_low_memory = available_mem_gb < 2.0
        
        # Log memory status for diagnostics
        self.logger.info(f"Memory status: {available_mem_gb:.2f}GB available, " +
                   f"Low memory: {is_low_memory}, Very low: {is_very_low_memory}, Extreme: {is_extreme_low_memory}")
        
        # Apply quantization based on hardware type and memory constraints
        if processor_type == ProcessorType.APPLE_SILICON:
            # Apple Silicon can efficiently handle 16-bit, but use lower in extreme cases
            if is_extreme_low_memory:
                return 4  # Use 4-bit in extreme low memory cases
            elif is_very_low_memory:
                return 8  # Use 8-bit in very low memory cases
            else:
                return 16  # Default 16-bit precision for Apple Silicon
                
        elif processor_type == ProcessorType.NVIDIA:
            # NVIDIA GPUs work well with 8-bit, but can use 4-bit for low memory
            if is_very_low_memory:
                return 4  # Use 4-bit in very low memory cases
            else:
                return 8  # Default 8-bit for NVIDIA
                
        elif processor_type == ProcessorType.INTEL:
            # For Intel, aggressively quantize based on model size and memory
            if is_extreme_low_memory:
                return 4  # Always use 4-bit in extreme low memory cases
            elif is_low_memory:
                return 4  # Use 4-bit in low memory cases
            else:
                # Use 8-bit for smaller models, 4-bit for larger models to conserve memory
                if model_sizes[model_type] == ModelSize.LARGE:
                    return 4
                elif model_sizes[model_type] == ModelSize.MEDIUM:
                    return 8
                else:
                    return 8
                    
        elif processor_type == ProcessorType.AMD:
            # AMD similar to Intel but may handle 8-bit better
            if is_very_low_memory:
                return 4
            else:
                return 8
                
        else:
            # Unknown processor - be conservative with quantization
            if is_low_memory:
                return 4  # Aggressive quantization for low memory
            else:
                return 8  # Default fallback
                
        # Note: This line was unreachable due to the returns above, removed to prevent errors

# Find the first EnhancedModelManager class definition

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan context manager for startup and shutdown events.
    
    Handles initialization during startup and cleanup during shutdown.
    """
    # Track server start time for uptime/health endpoint
    app.state.start_time = time.time()
    # Display startup banner
    print_startup_banner()

    # Startup
    try:
        # Load configuration
        config = load_config()
        console_logger.info(f"Environment: {config['environment']}")
        console_logger.info(f"Python version: {platform.python_version()}")
        console_logger.info(f"Server time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Configure application-wide logging
        app_logger = configure_logging(config)

        # Hardware detection phase - use simplified detector to avoid colorama issues
        console.rule("[bold green]🚀 PHASE 1: Detecting Hardware[/bold green]")
        app_logger.info("PHASE 1/5: Detecting hardware capabilities...")
        
        try:
            # Import SimpleHardwareDetector for startup stability
            from app.services.hardware.simple_detector import SimpleHardwareDetector
            hardware_detector = SimpleHardwareDetector(config)
            hardware_info = hardware_detector.detect_all()
            optimal_config = hardware_detector.recommend_config()
            
            # Use actual hardware detection results instead of hardcoded values
            # Create a simple object that mimics EnhancedHardwareInfo properties with detected attributes
            
            # Calculate accurate memory values from hardware_info
            total_memory_bytes = int(hardware_info.get("memory", {}).get("total_gb", 8) * 1024 * 1024 * 1024)
            available_memory_bytes = int(hardware_info.get("memory", {}).get("available_gb", 4) * 1024 * 1024 * 1024)
            
            # Determine processor type based on detected hardware
            detected_processor_type = ProcessorType.OTHER
            if hardware_info.get("system", {}).get("processor_type", "").startswith("apple_silicon"):
                detected_processor_type = ProcessorType.APPLE_SILICON
            elif "intel" in hardware_info.get("cpu", {}).get("brand", "").lower():
                detected_processor_type = ProcessorType.INTEL
            
            # Get CPU core counts
            cpu_cores = hardware_info.get("cpu", {}).get("count_physical", 4)
            cpu_threads = hardware_info.get("cpu", {}).get("count_logical", 8)
            
            # Check for GPU
            has_gpu = hardware_info.get("gpu", {}).get("has_gpu", False)
            gpu_name = hardware_info.get("gpu", {}).get("gpu_name", None)
            
            # Get GPU memory if available
            gpu_memory = None
            if has_gpu and "gpu_memory_gb" in hardware_info.get("gpu", {}):
                gpu_memory = int(hardware_info.get("gpu", {}).get("gpu_memory_gb", 0) * 1024 * 1024 * 1024)
            
            # Create GPU info list if available
            gpu_list = []
            if has_gpu and "devices" in hardware_info.get("gpu", {}):
                for device in hardware_info.get("gpu", {}).get("devices", []):
                    gpu_memory_gb = device.get("memory_gb", 0)
                    gpu_list.append(GPUInfo(
                        device_id=device.get("index", 0),
                        name=device.get("name", "Unknown GPU"),
                        memory_total=int(gpu_memory_gb * 1024 * 1024 * 1024),
                        memory_available=int(gpu_memory_gb * 0.9 * 1024 * 1024 * 1024),  # Estimate 90% available
                        vendor="apple" if device.get("apple_silicon", False) else "unknown"
                    ))
            
            # Create enhanced hardware info object with actual detected values
            enhanced_info = type('EnhancedHardwareInfo', (object,), {
                'processor_type': detected_processor_type,
                'total_memory': total_memory_bytes,
                'available_memory': available_memory_bytes,
                'cpu_cores': cpu_cores,
                'cpu_threads': cpu_threads,
                'has_gpu': has_gpu,
                'gpu_name': gpu_name,
                'gpu_memory': gpu_memory,
                'gpus': gpu_list,
                'system_name': platform.system(),
                'gpu_count': len(gpu_list),
                'get_best_gpu': lambda self: self.gpus[0] if self.gpus else None,
                'get_gpu_by_id': lambda self, device_id: next((gpu for gpu in self.gpus if gpu.device_id == device_id), None),
                'get_total_gpu_memory': lambda self: sum(gpu.memory_total for gpu in self.gpus),
                'get_available_gpu_memory': lambda self: sum(gpu.memory_available for gpu in self.gpus)
            })()
        except Exception as e:
            app_logger.error(f"Error during hardware detection: {e}")
            hardware_info = {}
            # Create minimal dummy hardware info
            enhanced_info = type('EnhancedHardwareInfo', (object,), {
                'processor_type': ProcessorType.OTHER,
                'total_memory': 4 * 1024 * 1024 * 1024,  # 4GB
                'available_memory': 2 * 1024 * 1024 * 1024,  # 2GB
                'cpu_cores': 2,
                'cpu_threads': 4,
                'has_gpu': False,
                'gpu_name': None,
                'gpu_memory': None,
                'gpus': [],
                'system_name': platform.system()
            })()
            optimal_config = {"device": "cpu", "memory": {"model_size": "small", "batch_size": 4}}
        # Skip model size overriding to avoid startup issues
        # We'll just use the simple detector's recommendations
        if optimal_config.get("memory", {}).get("model_size") == "large":
            config["model_size"] = "medium"
            app_logger.info("🧠 Auto-upgraded model_size to 'medium' based on hardware detection")
        
        # No enhanced detector in simplified startup
        model_config = optimal_config
        app_logger.info("✓ Hardware detection complete")

        # Model loader initialization (no fallback to mock logic)
        console.rule("[bold cyan]🔧 PHASE 2: Loading Models[/bold cyan]")
        app_logger.info("PHASE 2/5: Initializing model loader...")
        from app.services.models.loader import ModelLoader
        from app.services.models.loader import load_registry_config
        registry_config = load_registry_config(config)
        # Patch registry_config for language_detection to use papluca/xlm-roberta-base-language-detection
        if "language_detection" in registry_config:
            registry_config["language_detection"] = {
                "model_name": "papluca/xlm-roberta-base-language-detection",
                "model_type": "transformers",
                "tokenizer_name": "papluca/xlm-roberta-base-language-detection"
            }
        # Inject fallback config for "ner_detection" if missing
        if "ner_detection" not in registry_config:
            registry_config["ner_detection"] = {
                "model_name": "google/mt5-small",
                "tokenizer_name": "google/mt5-small",
                "task": "ner_detection",
                "type": "transformers",
                "framework": "transformers",
                "module": "app.services.models.wrappers.ner"
            }
            console_logger.info("✔️ Injected fallback 'ner_detection' model config.")

        # Inject fallback config for "translation" if missing
        if "translation" not in registry_config:
            registry_config["translation"] = {
                "model_name": "google/mt5-small",
                "tokenizer_name": "google/mt5-small",
                "task": "translation",
                "type": "transformers",
                "framework": "transformers",
                "module": "app.services.models.wrappers.translator"
            }
            console_logger.info("✔️ Injected fallback 'translation' model config.")

        # Inject fallback config for "simplifier" if missing
        if "simplifier" not in registry_config:
            registry_config["simplifier"] = {
                "model_name": "t5-small",
                "tokenizer_name": "t5-small",
                "task": "simplification",
                "type": "transformers",
                "model_class": "T5ForConditionalGeneration",
                "framework": "transformers",
                "module": "app.services.models.wrappers.simplification"
            }
            console_logger.info("✔️ Injected fallback 'simplifier' model config.")

        # Inject fallback config for "rag_generator" if missing
        if "rag_generator" not in registry_config:
            registry_config["rag_generator"] = {
                "model_name": "google/mt5-small",
                "tokenizer_name": "google/mt5-small",
                "task": "rag_generation",
                "type": "transformers",
                "framework": "transformers",
                "module": "app.services.models.wrappers.rag_generator"
            }
            console_logger.info("✔️ Injected fallback 'rag_generator' model config.")

        # Inject fallback config for "anonymizer" if missing
        if "anonymizer" not in registry_config:
            registry_config["anonymizer"] = {
                "model_name": "bert-base-cased",
                "tokenizer_name": "bert-base-cased",
                "task": "anonymization",
                "type": "transformers",
                "framework": "transformers",
                "module": "app.services.models.wrappers.anonymization"
            }
            console_logger.info("✔️ Injected fallback 'anonymizer' model config.")

        # Inject fallback config for "rag_retriever" if missing
        if "rag_retriever" not in registry_config:
            registry_config["rag_retriever"] = {
                "model_name": "facebook/dpr-question_encoder-single-nq-base",
                "tokenizer_name": "facebook/dpr-question_encoder-single-nq-base",
                "task": "rag_retrieval",
                "type": "transformers",
                "framework": "transformers",
                "module": "app.services.models.wrappers.rag_retriever"
            }
            console_logger.info("✔️ Injected fallback 'rag_retriever' model config.")
        model_loader = ModelLoader(config=config)
        model_loader.model_config = registry_config

        # Check if required models are available and download them if not
        app_logger.info("Checking for required models and downloading if necessary...")
        import asyncio
        import sys
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
        from scripts.download_models import download_model, DEFAULT_MODELS, ADVANCED_MODELS
        
        # Identify the needed models
        models_to_check = ["translation_model", "translation_small", "language_detection"]
        models_dir = Path(config.get("models", {}).get("models_dir", "models"))
        cache_dir = Path(config.get("models", {}).get("cache_dir", "cache/models"))
        
        # Create directories if they don't exist
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(cache_dir, exist_ok=True)
        
        # Gather the model definitions from the download script
        combined_models = {**DEFAULT_MODELS, **ADVANCED_MODELS}
        
        # Add the MBART models if they're not in the default definitions
        if "translation_model" not in combined_models:
            combined_models["translation_model"] = {
                "name": "Multilingual Translation",
                "model_name": "facebook/mbart-large-50-many-to-many-mmt",
                "model_type": "seq2seq",
                "tokenizer_name": "facebook/mbart-large-50-many-to-many-mmt",
                "size_gb": 2.3,
                "languages": ["en", "es", "fr", "de", "ru", "zh", "ja", "pt", "it", "ar", "hi", "vi"],
                "tasks": ["translation"],
                "description": "Multilingual translation model (MBART-50) for 50 languages",
                "requires_gpu": False,
                "memory_required": 4.0,
                "gpu_memory_required": 3.0,
                "model_format": "transformers"
            }
        
        if "translation_small" not in combined_models:
            combined_models["translation_small"] = {
                "name": "Multilingual Translation (Small)",
                "model_name": "facebook/mbart-large-50-one-to-many-mmt",
                "model_type": "seq2seq",
                "tokenizer_name": "facebook/mbart-large-50-one-to-many-mmt",
                "size_gb": 1.2,
                "languages": ["en", "es", "fr", "de", "ru", "zh", "ja", "pt", "it", "ar", "hi", "vi"],
                "tasks": ["translation"],
                "description": "Multilingual translation model (MBART-50) optimized for translating from English",
                "requires_gpu": False,
                "memory_required": 2.0,
                "gpu_memory_required": 1.5,
                "model_format": "transformers"
            }
        
        # Check and download each required model
        for model_id in models_to_check:
            model_dir = models_dir / model_id
            if not (model_dir / "config.json").exists():
                if model_id in combined_models:
                    app_logger.info(f"Model {model_id} not found, downloading...")
                    success = await download_model(
                        model_id=model_id,
                        model_info=combined_models[model_id],
                        output_dir=models_dir,
                        cache_dir=cache_dir,
                        force=False,
                        use_mlx=False
                    )
                    if success:
                        app_logger.info(f"✓ Successfully downloaded model {model_id}")
                    else:
                        app_logger.error(f"❌ Failed to download model {model_id}")
                else:
                    app_logger.warning(f"Model {model_id} not found in model definitions, will use fallback")
            else:
                app_logger.info(f"Model {model_id} already exists, skipping download")
                
        # Use enhanced model manager with real model loader
        app_logger.info("Initializing enhanced model manager...")
        model_manager = EnhancedModelManager(model_loader, hardware_info, config)
        app_logger.info("PHASE 2b: Bootstrapping all models before exposing API...")
        await model_manager.loader.bootstrap_models()
        app_logger.info("✓ Model system initialized")

        # Initialize shared tokenizer pipeline before processor init
        app_logger.info("Initializing shared tokenizer pipeline...")
        tokenizer_pipeline = TokenizerPipeline(model_name="google/mt5-small")
        app.state.tokenizer = tokenizer_pipeline

        # Audit system initialization
        console.rule("[bold yellow]🧾 PHASE 3: Audit & Metrics[/bold yellow]")
        app_logger.info("PHASE 3/5: Starting audit system...")
        audit_logger = AuditLogger(config)
        metrics = MetricsCollector(config)
        await audit_logger.initialize()
        app_logger.info("✓ Audit system initialized")

        # Pipeline processor initialization
        console.rule("[bold magenta]🧠 PHASE 4: Pipeline Init[/bold magenta]")
        app_logger.info("PHASE 4/5: Setting up processing pipeline...")
        processor = UnifiedProcessor(model_manager, audit_logger, metrics, config, registry_config=registry_config)
        await processor.initialize()
        app_logger.info("✓ Processing pipeline ready")

        # API initialization
        console.rule("[bold blue]🌐 PHASE 5: API Ready[/bold blue]")
        app_logger.info("PHASE 5/5: Initializing API endpoints...")
        # Will be handled by FastAPI after context manager yields

        # Initialize route cache if enabled in config
        app_logger.info("Initializing route cache...")
        route_cache_enabled = config.get("enable_route_cache", True)
        if route_cache_enabled:
            from app.services.storage.route_cache import RouteCacheManager
            # Initialize default cache instance
            default_cache = await RouteCacheManager.get_cache(
                name="default",
                max_size=config.get("route_cache_size", 1000),
                ttl_seconds=config.get("route_cache_ttl", 3600),
                bloom_compatible=True
            )
            # Initialize translation-specific cache
            translation_cache = await RouteCacheManager.get_cache(
                name="translation",
                max_size=config.get("translation_cache_size", 2000),
                ttl_seconds=config.get("translation_cache_ttl", 7200),
                bloom_compatible=True
            )
            app.state.route_cache = True  # Flag to indicate route cache is available
            app_logger.info(f"✓ Route cache initialized with {default_cache.max_size} entries default capacity")
        else:
            app_logger.info("Route cache is disabled in configuration")
        
        # Store components in application state
        app.state.config = config
        app.state.hardware_detector = hardware_detector
        # No enhanced detector in simplified startup
        app.state.enhanced_detector = None
        app.state.hardware_info = hardware_info
        app.state.enhanced_info = enhanced_info
        
        # Create and assign ModelRegistry to app.state
        from app.services.models.loader import ModelRegistry
        app.state.model_registry = model_loader.registry
        
        app.state.model_manager = model_manager
        app.state.processor = processor
        app.state.audit_logger = audit_logger
        app.state.metrics = metrics
        app.state.logger = app_logger
        # tokenizer_pipeline is now available as app.state.tokenizer for the app
        # Hardware Summary Table
        table = Table(title="🔍 Hardware Summary", style="bold blue")
        table.add_column("Component", style="cyan", no_wrap=True)
        table.add_column("Details", style="white")
        
        # Get more detailed CPU information if available
        cpu_info = ""
        if "cpu" in hardware_info and "brand" in hardware_info["cpu"]:
            cpu_brand = hardware_info["cpu"]["brand"]
            cpu_info = f"{cpu_brand} - "
        cpu_info += f"{enhanced_info.cpu_cores} cores / {enhanced_info.cpu_threads} threads"
        
        # Add additional Apple Silicon details if available
        if hardware_info.get("is_apple_silicon", False) or (
            "system" in hardware_info and 
            "processor_type" in hardware_info["system"] and 
            hardware_info["system"]["processor_type"].startswith("apple_silicon")
        ):
            # Check if apple_silicon_model is in the main hardware_info
            m_series = ""
            if "apple_silicon_model" in hardware_info:
                m_series = hardware_info["apple_silicon_model"]
            # If not, try to derive it from cpu info or system info
            elif "cpu" in hardware_info and "chip_type" in hardware_info["cpu"]:
                chip_type = hardware_info["cpu"]["chip_type"]
                if chip_type == "M4_Pro_Max":
                    m_series = "M4 Pro/Max"
                elif chip_type and chip_type.startswith("M"):
                    m_series = chip_type.replace("_", " ")
            elif "system" in hardware_info and "apple_silicon" in hardware_info["system"]:
                m_series = hardware_info["system"]["apple_silicon"]
                
            if m_series:
                cpu_info = f"Apple {m_series} - {enhanced_info.cpu_cores} cores / {enhanced_info.cpu_threads} threads"
        
        # Get memory details with proper formatting
        mem_total = enhanced_info.total_memory / (1024**3)
        mem_avail = enhanced_info.available_memory / (1024**3)
        memory_info = f"{mem_total:.1f} GB total / {mem_avail:.1f} GB available"
        
        # Get proper GPU information
        gpu_info = "None"
        if enhanced_info.has_gpu:
            if enhanced_info.gpu_name:
                gpu_info = enhanced_info.gpu_name
            elif enhanced_info.gpus and len(enhanced_info.gpus) > 0:
                gpu_info = enhanced_info.gpus[0].name
                
            # Add GPU memory information if available
            if enhanced_info.gpu_memory:
                gpu_memory_gb = enhanced_info.gpu_memory / (1024**3)
                gpu_info += f" ({gpu_memory_gb:.1f} GB)"
            elif enhanced_info.gpus and enhanced_info.gpus[0].memory_total:
                gpu_memory_gb = enhanced_info.gpus[0].memory_total / (1024**3)
                gpu_info += f" ({gpu_memory_gb:.1f} GB)"
        
        # Add table rows
        table.add_row("CPU", cpu_info)
        table.add_row("Memory", memory_info)
        table.add_row("GPU", gpu_info)
        
        # Add extra row for Apple Silicon if applicable
        is_apple_silicon = hardware_info.get("is_apple_silicon", False) or (
            "system" in hardware_info and 
            "processor_type" in hardware_info["system"] and 
            hardware_info["system"]["processor_type"].startswith("apple_silicon")
        )
        
        if is_apple_silicon:
            # Try to get the model information from different sources
            model_info = ""
            memory_info = ""
            
            # Check main hardware_info
            if "apple_silicon_model" in hardware_info:
                model_info = hardware_info["apple_silicon_model"]
            elif "apple_silicon_memory" in hardware_info:
                memory_info = hardware_info["apple_silicon_memory"]
                
            # Check cpu info
            if not model_info and "cpu" in hardware_info:
                cpu = hardware_info["cpu"]
                if "chip_type" in cpu:
                    chip_type = cpu["chip_type"]
                    if chip_type == "M4_Pro_Max":
                        model_info = "M4 Pro/Max"
                    elif chip_type and chip_type.startswith("M"):
                        model_info = chip_type.replace("_", " ")
                elif "apple_silicon" in cpu and cpu["apple_silicon"]:
                    if "chip_variant" in cpu:
                        model_info = cpu["chip_variant"]
                    elif "brand" in cpu and "Apple" in cpu["brand"]:
                        model_info = cpu["brand"].replace("Apple ", "")
            
            # Check system info
            if not model_info and "system" in hardware_info and "apple_silicon" in hardware_info["system"]:
                model_info = hardware_info["system"]["apple_silicon"]
            
            # Check memory info
            if not memory_info and "memory" in hardware_info:
                memory = hardware_info["memory"]
                if "apple_silicon_memory" in memory:
                    memory_info = memory["apple_silicon_memory"]
                elif "memory_configuration" in memory and memory["memory_configuration"] == "m4_pro_max_48gb":
                    memory_info = "48GB"
                elif "total_gb" in memory:
                    memory_info = f"{int(memory['total_gb'])}GB"
            
            # Format the row text
            if model_info and memory_info:
                apple_silicon_text = f"{model_info.replace('_', ' ')} with {memory_info}"
                table.add_row("Apple Silicon", apple_silicon_text)
            elif model_info:
                table.add_row("Apple Silicon", model_info.replace('_', ' '))
            elif is_apple_silicon:
                table.add_row("Apple Silicon", "Detected")
            
        console.print(table)
        # Final celebratory banner
        console.print(
            Panel.fit(
                "[bold green]🎉 CasaLingua Boot Complete[/bold green]\n"
                "[dim]System is ready and accepting requests[/dim]",
                border_style="green"
            )
        )
        app_logger.info("✓ CasaLingua system initialization complete")
        app_logger.info("✓ Server is ready to accept connections!")
        
    except Exception as e:
        # Enhanced: print exception with rich formatting
        console.print_exception()
        console_logger.error(f"FATAL ERROR during startup: {str(e)}", exc_info=True)
        raise
    
    # Yield control to FastAPI
    yield
    
    # Shutdown
    try:
        app_logger = app.state.logger
        app_logger.info("Initiating graceful shutdown...")
        console.rule("[bold red]Server Shutdown Initiated[/bold red]")

        # Flush audit logs
        if hasattr(app.state, "audit_logger") and app.state.audit_logger:
            app_logger.info("Flushing audit logs...")
            await app.state.audit_logger.flush()
            app_logger.info("✓ Audit logs flushed")

        # Save metrics
        if hasattr(app.state, "metrics") and app.state.metrics:
            app_logger.info("Saving performance metrics...")
            app.state.metrics.save_metrics()
            app_logger.info("✓ Performance metrics saved")

        # Clean up route cache if enabled
        if hasattr(app.state, "route_cache") and app.state.route_cache:
            app_logger.info("Cleaning up route cache...")
            from app.services.storage.route_cache import RouteCacheManager
            await RouteCacheManager.clear_all()
            app_logger.info("✓ Route cache cleared")
        
        # Unload models
        if hasattr(app.state, "model_manager") and app.state.model_manager:
            app_logger.info("Unloading models...")
            await app.state.model_manager.unload_all_models()
            app_logger.info("✓ Models unloaded")

        app_logger.info("✓ CasaLingua server shutdown complete")
        
    except Exception as e:
        # Enhanced: print exception with rich formatting
        console.print_exception()
        console_logger.error(f"Error during shutdown: {str(e)}", exc_info=True)


# Create FastAPI application instance with lifespan
app = FastAPI(
    title="CasaLingua API",
    description="Language Processing & Translation Pipeline for Housing Accessibility",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# Add CORS middleware with configurable origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Process-Time"]
)

# Add timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Middleware to measure and log request processing time."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    
    # If we have metrics initialized, record the request
    if hasattr(app.state, "metrics"):
        metrics = app.state.metrics
        route = request.url.path
        metrics.record_request(route, response.status_code < 400, process_time)
    
    return response

# Add security headers middleware
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    return response

# Import enhanced error handler
from app.utils.error_handler import APIError, ErrorCategory, ErrorResponse

# Exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled exceptions."""
    # Enhanced: print exception with rich formatting
    console.print_exception()
    if hasattr(app.state, "logger"):
        app.state.logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    else:
        console_logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    
    # Record in metrics if available
    if hasattr(app.state, "metrics"):
        route = request.url.path
        app.state.metrics.record_request(route, False, 0)
    
    # Generate request ID for tracking
    import uuid
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    
    # Use new error handling system
    if isinstance(exc, APIError):
        # If it's our custom APIError, use its response format
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.to_response().dict()
        )
    else:
        # For other exceptions, convert to our standard format
        error = ErrorResponse(
            status_code=500,
            error_code="internal_error",
            category=ErrorCategory.INTERNAL_ERROR,
            message="Internal server error",
            details={"error": str(exc)} if os.environ.get("DEBUG", "false").lower() == "true" else None,
            request_id=request_id
        )
        return JSONResponse(
            status_code=500,
            content=error.dict()
        )

# Include routers
app.include_router(health_router)
app.include_router(pipeline_router, prefix="/pipeline", tags=["Pipeline"])
app.include_router(admin_router, prefix="/admin", tags=["Admin"])
app.include_router(rag_router, prefix="/rag", tags=["RAG"])

# Include Bloom Housing compatibility router
from app.api.routes.bloom_housing import router as bloom_housing_router
app.include_router(bloom_housing_router)

# Include streaming API routes
from app.api.routes.streaming import router as streaming_router
app.include_router(streaming_router)

# Add new model management endpoint
from fastapi import APIRouter
model_router = APIRouter()

@model_router.get("/models/info")
async def get_model_info():
    """Get information about all loaded models and their configurations."""
    if not hasattr(app.state, "model_manager"):
        raise HTTPException(status_code=503, detail="Model manager not initialized")
    
    return {
        "models": app.state.model_manager.get_model_info(),
        "hardware": {
            "processor_type": app.state.enhanced_info.processor_type.value if hasattr(app.state, "enhanced_info") else "unknown",
            "total_memory_gb": app.state.hardware_info.get("total_memory", 0) / (1024**3),
            "available_memory_gb": app.state.hardware_info.get("available_memory", 0) / (1024**3),
            "has_gpu": app.state.hardware_info.get("has_gpu", False),
            "gpu_memory_gb": app.state.hardware_info.get("gpu_memory", 0) / (1024**3) if app.state.hardware_info.get("has_gpu", False) else 0,
        }
    }

@model_router.post("/models/unload/{model_type}")
async def unload_model(model_type: str):
    """Temporarily unload a model to free memory."""
    if not hasattr(app.state, "model_manager"):
        raise HTTPException(status_code=503, detail="Model manager not initialized")
    
    try:
        model_enum = ModelType(model_type)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid model type: {model_type}")
    
    await app.state.model_manager.unload_model(model_enum)
    return {"status": "success", "message": f"Model {model_type} unloaded successfully"}

@model_router.post("/models/reload/{model_type}")
async def reload_model(model_type: str):
    """Reload a previously unloaded model."""
    if not hasattr(app.state, "model_manager"):
        raise HTTPException(status_code=503, detail="Model manager not initialized")
    
    try:
        model_enum = ModelType(model_type)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid model type: {model_type}")
    
    await app.state.model_manager.reload_model(model_enum)
    return {"status": "success", "message": f"Model {model_type} reloaded successfully"}

@model_router.get("/cache/stats")
async def get_cache_stats():
    """Get statistics about the route cache."""
    if not hasattr(app.state, "route_cache") or not app.state.route_cache:
        raise HTTPException(status_code=404, detail="Route cache is not enabled")
    
    from app.services.storage.route_cache import RouteCacheManager
    cache_stats = await RouteCacheManager.get_all_stats()
    
    return {
        "status": "success",
        "data": cache_stats,
        "summary": {
            "total_cache_entries": sum(stats.get("size", 0) for stats in cache_stats.values()),
            "total_hit_rate": sum(stats.get("hit_rate", 0) * stats.get("size", 0) 
                               for stats in cache_stats.values()) / 
                           max(1, sum(stats.get("size", 0) for stats in cache_stats.values())),
            "cache_instances": list(cache_stats.keys())
        }
    }

@model_router.post("/cache/clear")
async def clear_cache(cache_name: Optional[str] = None):
    """Clear the route cache, either completely or for a specific instance."""
    if not hasattr(app.state, "route_cache") or not app.state.route_cache:
        raise HTTPException(status_code=404, detail="Route cache is not enabled")
    
    from app.services.storage.route_cache import RouteCacheManager
    
    if cache_name:
        # Check if the named cache exists
        stats = await RouteCacheManager.get_all_stats()
        if cache_name not in stats:
            raise HTTPException(status_code=404, detail=f"Cache instance '{cache_name}' not found")
        
        # Clear the specific cache
        cache = await RouteCacheManager.get_cache(name=cache_name)
        await cache.clear()
        return {"status": "success", "message": f"Cache '{cache_name}' cleared successfully"}
    else:
        # Clear all caches
        await RouteCacheManager.clear_all()
        return {"status": "success", "message": "All cache instances cleared successfully"}

app.include_router(model_router, prefix="/admin", tags=["Models"])

# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")
app.mount("/admin-ui", StaticFiles(directory="app/static/admin", html=True), name="admin-ui")

# Entry point for running directly
if __name__ == "__main__":
    import uvicorn

    # Load configuration
    config = load_config()

    uvicorn.run(
        "app.main:app",
        host=config.get("server_host", "0.0.0.0"),
        port=config.get("server_port", 8000),
        reload=ENVIRONMENT == "development",
        log_level="debug" if ENVIRONMENT == "development" else config.get("log_level", "info").lower(),
        access_log=True
    )