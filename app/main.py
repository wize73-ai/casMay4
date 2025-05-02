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
ENVIRONMENT = os.getenv("CASALINGUA_ENV", "production").lower()
import sys
import time
import logging
import platform
import torch
import psutil
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Literal
from enum import Enum
from dataclasses import dataclass
from contextlib import asynccontextmanager

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

# Import enhanced hardware detection and model management
from app.services.hardware.detector import HardwareDetector
from app.services.models.manager import ModelManager as EnhancedModelManager

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
class EnhancedHardwareInfo:
    total_memory: int  # in bytes
    available_memory: int  # in bytes
    processor_type: ProcessorType
    has_gpu: bool
    gpu_memory: Optional[int] = None  # in bytes, None if no GPU
    cpu_cores: int = 0
    cpu_threads: int = 0
    gpu_name: Optional[str] = None
    system_name: Optional[str] = None

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
        """Detect all hardware capabilities and return comprehensive information."""
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
        
        # Check for GPU
        has_gpu, gpu_memory, gpu_name = self._detect_gpu()
        
        hardware_info = EnhancedHardwareInfo(
            total_memory=total_memory,
            available_memory=available_memory,
            processor_type=processor_type,
            has_gpu=has_gpu,
            gpu_memory=gpu_memory,
            cpu_cores=cpu_cores,
            cpu_threads=cpu_threads,
            gpu_name=gpu_name,
            system_name=system_name
        )
        
        # Log hardware details
        self.logger.info(f"Hardware Detection Results:")
        self.logger.info(f"  System: {system_name}")
        self.logger.info(f"  Processor: {processor_type.value}")
        self.logger.info(f"  CPU Cores: {cpu_cores} (Physical), {cpu_threads} (Logical)")
        self.logger.info(f"  Memory: {total_memory / (1024**3):.1f} GB (Total), {available_memory / (1024**3):.1f} GB (Available)")
        
        if has_gpu:
            self.logger.info(f"  GPU: {gpu_name}")
            self.logger.info(f"  GPU Memory: {gpu_memory / (1024**3):.1f} GB")
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
    
    def _detect_gpu(self) -> Tuple[bool, Optional[int], Optional[str]]:
        """Detect if a GPU is available, its memory, and name."""
        has_gpu = torch.cuda.is_available()
        
        if has_gpu:
            # Get GPU info via torch
            device_count = torch.cuda.device_count()
            if device_count > 0:
                # Use the first GPU for simplicity
                device_props = torch.cuda.get_device_properties(0)
                gpu_memory = device_props.total_memory
                gpu_name = device_props.name
                return True, gpu_memory, gpu_name
        
        return False, None, None
    
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
        """Determine the quantization level based on hardware."""
        processor_type = self._detect_processor_type()
        model_sizes = self.recommend_config()
        
        # Apply different quantization based on processor type
        if processor_type == ProcessorType.APPLE_SILICON:
            return 16  # 16-bit precision for Apple Silicon
        elif processor_type == ProcessorType.INTEL:
            # For Intel, use 8-bit for larger models, 4-bit for smaller
            if model_sizes[model_type] == ModelSize.LARGE:
                return 8
            else:
                return 4
        else:
            # Default case
            return 8

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

        # Hardware detection phase
        console.rule("[bold green]🚀 PHASE 1: Detecting Hardware[/bold green]")
        app_logger.info("PHASE 1/5: Detecting hardware capabilities...")
        # Use original hardware detector for compatibility
        hardware_detector = HardwareDetector(config)
        hardware_info = hardware_detector.detect_all()

        # Add enhanced hardware detection
        enhanced_detector = EnhancedHardwareDetector(config)
        enhanced_info = await enhanced_detector.detect_all()
        optimal_config = enhanced_detector.recommend_config()
        # Override model_size globally if medium+ hardware is detected
        # This ensures CasaLingua selects medium models automatically for capable Apple Silicon machines.
        if (
            enhanced_info.processor_type == ProcessorType.APPLE_SILICON
            and enhanced_info.total_memory >= 32 * 1024 * 1024 * 1024
        ):
            config["model_size"] = "medium"
            app_logger.info("🧠 Auto-upgraded model_size to 'medium' for Apple M-series Mac with >=32GB RAM")
        model_config = enhanced_detector.apply_configuration(optimal_config)
        app_logger.info("✓ Hardware detection complete")

        # Model loader initialization (no fallback to mock logic)
        console.rule("[bold cyan]🔧 PHASE 2: Loading Models[/bold cyan]")
        app_logger.info("PHASE 2/5: Initializing model loader...")
        from app.services.models.loader import ModelLoader
        from app.services.models.registry import load_registry_config
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

        # Store components in application state
        app.state.config = config
        app.state.hardware_detector = hardware_detector
        app.state.enhanced_detector = enhanced_detector
        app.state.hardware_info = hardware_info
        app.state.enhanced_info = enhanced_info
        # app.state.model_registry = None  # ModelRegistry is not used in this flow
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
        table.add_row("CPU", f"{enhanced_info.cpu_cores} cores / {enhanced_info.cpu_threads} threads")
        table.add_row("Memory", f"{enhanced_info.total_memory / (1024**3):.1f} GB total / {enhanced_info.available_memory / (1024**3):.1f} GB available")
        table.add_row("GPU", enhanced_info.gpu_name or 'None')
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
    
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": "Internal server error",
            "detail": str(exc) if os.environ.get("DEBUG", "false").lower() == "true" else None
        }
    )

# Include routers
app.include_router(health_router)
app.include_router(pipeline_router, prefix="/pipeline", tags=["Pipeline"])
app.include_router(admin_router, prefix="/admin", tags=["Admin"])
app.include_router(rag_router, prefix="/rag", tags=["RAG"])

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