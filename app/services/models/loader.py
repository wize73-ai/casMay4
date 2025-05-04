"""
Model Loader Module for CasaLingua
Handles loading and management of ML models
"""

import os
import json
import logging
import torch
import gc
import time
import asyncio
from typing import Dict, Any, Tuple, Optional, List, Union, Callable
from pathlib import Path
from dataclasses import dataclass

# Rich console components
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text

# Initialize rich console
console = Console()

# Configure logging
logger = logging.getLogger(__name__)

# Try to import optional dependencies with fallbacks
try:
    import transformers
    from transformers import AutoModel, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification
    from transformers import MT5ForConditionalGeneration, T5ForConditionalGeneration 
    from transformers import T5Tokenizer, BertTokenizer, BertModel
    from transformers import AutoModelForTokenClassification
    HAVE_TRANSFORMERS = True
except ImportError:
    HAVE_TRANSFORMERS = False
    # Define empty classes to avoid NameError when referenced
    AutoModel = None
    AutoTokenizer = None
    AutoModelForSeq2SeqLM = None
    AutoModelForSequenceClassification = None
    AutoModelForTokenClassification = None
    MT5ForConditionalGeneration = None
    T5ForConditionalGeneration = None
    T5Tokenizer = None
    BertTokenizer = None
    BertModel = None
    logger.warning("transformers not available - model loading will be limited")

try:
    from sentence_transformers import SentenceTransformer
    HAVE_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAVE_SENTENCE_TRANSFORMERS = False
    logger.warning("sentence_transformers not available - embedding models will be limited")

try:
    import onnxruntime as ort
    HAVE_ONNX = True
except ImportError:
    HAVE_ONNX = False
    logger.warning("onnxruntime not available - ONNX model acceleration will be disabled")

# Configuration for model registry
DEFAULT_REGISTRY_PATH = "config/model_registry.json"
DEFAULT_CACHE_DIR = "./.cache/models"

# Singleton instance for global access
_MODEL_LOADER_INSTANCE = None

def get_model_loader():
    """Returns the global ModelLoader instance"""
    global _MODEL_LOADER_INSTANCE
    if _MODEL_LOADER_INSTANCE is None:
        _MODEL_LOADER_INSTANCE = ModelLoader()
    return _MODEL_LOADER_INSTANCE

@dataclass
class ModelConfig:
    """Configuration for a specific model"""
    model_name: str
    tokenizer_name: str = None
    task: str = None
    type: str = "transformers"
    framework: str = "transformers"
    module: str = None
    quantization: Optional[int] = None  # bits (4, 8, 16, 32)
    device: str = None  # "cpu", "cuda", "mps"
    batch_size: int = 16
    max_length: int = 512
    model_kwargs: Dict[str, Any] = None
    tokenizer_kwargs: Dict[str, Any] = None
    
    def __post_init__(self):
        """Initialize defaults after creation"""
        # If tokenizer_name is None, use model_name
        if self.tokenizer_name is None:
            self.tokenizer_name = self.model_name
            
        # Initialize dictionaries if None
        if self.model_kwargs is None:
            self.model_kwargs = {}
        if self.tokenizer_kwargs is None:
            self.tokenizer_kwargs = {}

def load_registry_config(config: Dict = None) -> Dict[str, Any]:
    """
    Load model registry configuration
    
    Args:
        config (Dict): Application configuration
        
    Returns:
        Dict[str, Any]: Model registry configuration
    """
    registry_path = DEFAULT_REGISTRY_PATH
    
    # Override with config if provided
    if config and "model_registry_path" in config:
        registry_path = config["model_registry_path"]
    
    try:
        # Check if file exists
        if os.path.exists(registry_path):
            with open(registry_path, 'r') as f:
                registry_config = json.load(f)
                logger.info(f"Loaded model registry from {registry_path}")
                return registry_config
        else:
            logger.warning(f"Model registry file {registry_path} not found, using defaults")
            # Return default registry
            return {
                "language_detection": {
                    "model_name": "papluca/xlm-roberta-base-language-detection",
                    "model_type": "transformers",
                    "tokenizer_name": "papluca/xlm-roberta-base-language-detection",
                    "task": "language_detection",
                    "framework": "transformers"
                },
                "translation": {
                    "model_name": "google/mt5-small",
                    "tokenizer_name": "google/mt5-small",
                    "task": "translation",
                    "type": "transformers",
                    "framework": "transformers"
                },
                "ner_detection": {
                    "model_name": "dslim/bert-base-NER",
                    "tokenizer_name": "dslim/bert-base-NER",
                    "task": "ner_detection",
                    "type": "transformers",
                    "framework": "transformers"
                },
                "simplifier": {
                    "model_name": "t5-small",
                    "tokenizer_name": "t5-small",
                    "task": "simplification",
                    "type": "transformers",
                    "framework": "transformers"
                },
                "rag_generator": {
                    "model_name": "google/mt5-small",
                    "tokenizer_name": "google/mt5-small",
                    "task": "rag_generation",
                    "type": "transformers",
                    "framework": "transformers"
                },
                "anonymizer": {
                    "model_name": "bert-base-cased",
                    "tokenizer_name": "bert-base-cased",
                    "task": "anonymization",
                    "type": "transformers",
                    "framework": "transformers"
                },
                "rag_retriever": {
                    "model_name": "sentence-transformers/all-MiniLM-L6-v2",
                    "tokenizer_name": "sentence-transformers/all-MiniLM-L6-v2",
                    "task": "embedding",
                    "type": "sentence-transformers",
                    "framework": "sentence-transformers"
                }
            }
    except Exception as e:
        logger.error(f"Error loading model registry: {e}")
        # Return minimal fallback registry
        return {
            "language_detection": {
                "model_name": "papluca/xlm-roberta-base-language-detection",
                "tokenizer_name": "papluca/xlm-roberta-base-language-detection"
            },
            "translation": {
                "model_name": "google/mt5-small",
                "tokenizer_name": "google/mt5-small"
            }
        }

class ModelRegistry:
    """Registry of available models and their configurations"""
    
    def __init__(self, registry_config: Dict[str, Any] = None):
        """
        Initialize model registry
        
        Args:
            registry_config (Dict[str, Any], optional): Registry configuration
        """
        # Initialize with empty registry
        self.registry = {}
        
        # Load registry if provided
        if registry_config:
            self.update_registry(registry_config)
    
    def update_registry(self, registry_config: Dict[str, Any]) -> None:
        """
        Update registry with new configuration
        
        Args:
            registry_config (Dict[str, Any]): Registry configuration
        """
        for model_type, config in registry_config.items():
            # Convert to ModelConfig if it's a dict
            if isinstance(config, dict):
                # Extract known fields for ModelConfig
                model_config_args = {
                    k: v for k, v in config.items() 
                    if k in [f.name for f in ModelConfig.__dataclass_fields__.values()]
                }
                
                # Create ModelConfig instance
                model_config = ModelConfig(**model_config_args)
                
                # Add additional fields
                for k, v in config.items():
                    if k not in model_config_args:
                        setattr(model_config, k, v)
                
                self.registry[model_type] = model_config
            else:
                # Assume it's already a ModelConfig
                self.registry[model_type] = config
                
        logger.info(f"Updated model registry with {len(registry_config)} entries")
        
        # Display registry as table
        table = Table(title="[bold]Model Registry[/bold]")
        table.add_column("Model Type", style="cyan")
        table.add_column("Model", style="yellow")
        table.add_column("Framework", style="green")
        
        for model_type, config in self.registry.items():
            table.add_row(
                model_type,
                config.model_name,
                config.framework
            )
        
        console.print(table)
    
    def get_model_config(self, model_type: str) -> Optional[ModelConfig]:
        """
        Get model configuration
        
        Args:
            model_type (str): Model type
            
        Returns:
            Optional[ModelConfig]: Model configuration or None if not found
        """
        return self.registry.get(model_type)
    
    def get_model_and_tokenizer(self, model_type: str) -> Tuple[str, str]:
        """
        Get model and tokenizer names for a given model type
        
        Args:
            model_type (str): Model type
            
        Returns:
            Tuple[str, str]: Model name and tokenizer name
        """
        # Default fallbacks for common model types
        defaults = {
            "translation": ("google/mt5-small", "google/mt5-small"),
            "multipurpose": ("google/mt5-small", "google/mt5-small"),
            "verification": ("google/mt5-small", "google/mt5-small"),
            "ner_detection": ("dslim/bert-base-NER", "dslim/bert-base-NER"),
            "language_detection": ("papluca/xlm-roberta-base-language-detection", 
                                  "papluca/xlm-roberta-base-language-detection"),
            "simplifier": ("t5-small", "t5-small"),
            "rag_generator": ("google/mt5-small", "google/mt5-small"),
            "rag_retriever": ("sentence-transformers/all-MiniLM-L6-v2", 
                             "sentence-transformers/all-MiniLM-L6-v2"),
            "anonymizer": ("bert-base-cased", "bert-base-cased")
        }
        
        if model_type in self.registry:
            config = self.registry[model_type]
            tokenizer_name = config.tokenizer_name if config.tokenizer_name else config.model_name
            return config.model_name, tokenizer_name
        
        # Return default fallbacks
        if model_type in defaults:
            return defaults[model_type]
        
        # Last resort fallback
        logger.warning(f"No model configuration found for '{model_type}', using generic fallback")
        return "google/mt5-small", "google/mt5-small"
    
    def list_models(self) -> List[str]:
        """
        List all available models
        
        Returns:
            List[str]: List of model types
        """
        return list(self.registry.keys())

class ModelLoader:
    """Handles loading and management of models with hardware awareness"""
    
    def __init__(self, config: Dict = None, hardware_info=None):
        """
        Initialize model loader with multi-GPU support
        
        Args:
            config (Dict, optional): Application configuration
            hardware_info (optional): Hardware information with GPU details
        """
        self.config = config or {}
        self.model_config = load_registry_config(config)
        self.registry = ModelRegistry(self.model_config)
        self.loaded_models = {}
        self.loaded_tokenizers = {}
        self.model_devices = {}  # Track which device each model is loaded on
        self.cache_dir = self.config.get("model_cache_dir", DEFAULT_CACHE_DIR)
        
        # Progressive loading support
        self.lazy_load_models = []  # Models that will be loaded on first use
        self.pending_lazy_loads = {}  # Track requested but not yet loaded models
        self.failed_loads = {}  # Track models that failed to load with error details
        self.load_locks = {}  # Locks to prevent concurrent loading of the same model
        
        # Multi-GPU support
        self.hardware_info = hardware_info
        self.available_devices = self._discover_available_devices()
        self.default_device = self.available_devices[0] if self.available_devices else "cpu"
        self.device = self.default_device  # Add device attribute
        
        # GPU memory tracking
        self.gpu_memory_used = {}  # Track memory usage per GPU
        for device in self.available_devices:
            if device != "cpu" and device != "mps":
                # Extract device ID from CUDA device string (e.g., "cuda:0" -> 0)
                if ":" in device:
                    device_id = int(device.split(":")[-1])
                    self.gpu_memory_used[device_id] = 0
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Log available devices
        devices_str = ", ".join(self.available_devices)
        logger.info(f"ModelLoader initialized with available devices: {devices_str}")
        console.print(f"[bold cyan]ModelLoader initialized[/bold cyan] - Available devices: [yellow]{devices_str}[/yellow]")
        console.print(f"[bold cyan]Default device:[/bold cyan] [yellow]{self.default_device}[/yellow]")
        
    def _discover_available_devices(self) -> List[str]:
        """
        Discover all available compute devices (CPU, CUDA GPUs, MPS)
        
        Returns:
            List[str]: List of available device strings
        """
        devices = []
        
        # Check if hardware_info was provided with GPU details
        if self.hardware_info and hasattr(self.hardware_info, "gpus") and self.hardware_info.gpus:
            # We have detailed GPU info, use it to create device strings
            for gpu in self.hardware_info.gpus:
                devices.append(f"cuda:{gpu.device_id}")
            logger.info(f"Discovered {len(devices)} GPUs from hardware info")
        else:
            # Fallback to basic CUDA detection
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                for i in range(device_count):
                    devices.append(f"cuda:{i}")
                logger.info(f"Discovered {device_count} CUDA devices")
        
        # Check for MPS (Apple Silicon)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            devices.append("mps")
            logger.info("Discovered MPS device (Apple Silicon)")
        
        # Always add CPU as fallback
        devices.append("cpu")
        
        return devices
        
    def _determine_device(self, model_type: str = None, model_size: str = None) -> str:
        """
        Determine the appropriate device for a specific model
        
        Args:
            model_type (str, optional): Type of model
            model_size (str, optional): Size of model (small, medium, large)
            
        Returns:
            str: Device string (e.g., "cuda:0", "mps", "cpu")
        """
        # If no GPUs, use CPU
        if not self.available_devices or len(self.available_devices) == 1 and self.available_devices[0] == "cpu":
            return "cpu"
            
        # If only one GPU, use it
        if len(self.available_devices) == 2 and "cpu" in self.available_devices:
            return [d for d in self.available_devices if d != "cpu"][0]
            
        # For multiple GPUs, implement load balancing
        # Strategy: put large models on GPUs with most available memory
        if model_type and model_size and self.hardware_info and hasattr(self.hardware_info, "gpus"):
            # Get memory usage estimate for this model
            memory_estimate = self._estimate_model_memory(model_type, model_size)
            
            # Find the GPU with the most available memory that can fit this model
            best_gpu = None
            best_available_memory = 0
            
            for gpu in self.hardware_info.gpus:
                # Calculate current available memory accounting for what we've already allocated
                used_memory = self.gpu_memory_used.get(gpu.device_id, 0)
                available_memory = gpu.memory_available - used_memory
                
                # If this GPU has more available memory than our current best and can fit the model
                if available_memory > best_available_memory and available_memory >= memory_estimate:
                    best_gpu = gpu
                    best_available_memory = available_memory
            
            # If we found a suitable GPU, use it
            if best_gpu:
                device = f"cuda:{best_gpu.device_id}"
                
                # Update memory tracking
                self.gpu_memory_used[best_gpu.device_id] = self.gpu_memory_used.get(best_gpu.device_id, 0) + memory_estimate
                
                logger.info(f"Assigned model {model_type} ({model_size}) to {device} "
                          f"with {best_available_memory / (1024**3):.1f}GB available memory")
                return device
        
        # Fallback to simple round-robin assignment
        # Skip CPU in round-robin since we only want to use it as a last resort
        gpu_devices = [d for d in self.available_devices if d != "cpu"]
        if gpu_devices:
            # Count models on each device and pick the one with fewest models
            device_counts = {}
            for device in self.model_devices.values():
                device_counts[device] = device_counts.get(device, 0) + 1
            
            # Find device with fewest models
            min_count = float('inf')
            best_device = gpu_devices[0]
            
            for device in gpu_devices:
                count = device_counts.get(device, 0)
                if count < min_count:
                    min_count = count
                    best_device = device
                    
            return best_device
            
        # Final fallback
        return self.default_device
        
    def _estimate_model_memory(self, model_type: str, model_size: str) -> int:
        """
        Estimate memory requirement for a model in bytes
        
        Args:
            model_type (str): Type of model
            model_size (str): Size of model (small, medium, large)
            
        Returns:
            int: Estimated memory requirement in bytes
        """
        # Default memory estimates by model type and size
        memory_estimates = {
            "translation": {
                "small": 2 * 1024**3,    # 2GB
                "medium": 6 * 1024**3,   # 6GB
                "large": 12 * 1024**3    # 12GB
            },
            "language_detection": {
                "small": 500 * 1024**2,  # 500MB
                "medium": 1 * 1024**3,   # 1GB
                "large": 2 * 1024**3     # 2GB
            },
            "ner_detection": {
                "small": 1 * 1024**3,    # 1GB
                "medium": 3 * 1024**3,   # 3GB
                "large": 6 * 1024**3     # 6GB
            },
            "simplifier": {
                "small": 1 * 1024**3,    # 1GB
                "medium": 3 * 1024**3,   # 3GB
                "large": 6 * 1024**3     # 6GB
            },
            "rag_generator": {
                "small": 2 * 1024**3,    # 2GB
                "medium": 6 * 1024**3,   # 6GB
                "large": 12 * 1024**3    # 12GB
            },
            "rag_retriever": {
                "small": 500 * 1024**2,  # 500MB
                "medium": 1 * 1024**3,   # 1GB
                "large": 2 * 1024**3     # 2GB
            },
            "anonymizer": {
                "small": 500 * 1024**2,  # 500MB
                "medium": 1 * 1024**3,   # 1GB
                "large": 2 * 1024**3     # 2GB
            }
        }
        
        # Get model-specific estimate or use a generic default
        type_estimates = memory_estimates.get(model_type, {
            "small": 1 * 1024**3,    # 1GB
            "medium": 4 * 1024**3,   # 4GB
            "large": 8 * 1024**3     # 8GB
        })
        
        return type_estimates.get(model_size, 4 * 1024**3)  # Default to 4GB if size unknown
    
    # The original _determine_device method has been replaced by the enhanced version above
    
    def get_model_and_tokenizer(self, model_type: str) -> Tuple[str, str]:
        """
        Get model and tokenizer names for a given model type
        
        Args:
            model_type (str): Model type
            
        Returns:
            Tuple[str, str]: Model name and tokenizer name
        """
        return self.registry.get_model_and_tokenizer(model_type)
    
    def get_cache_path(self, model_name: str) -> str:
        """
        Get the cache path for a model
        
        Args:
            model_name (str): Model name
            
        Returns:
            str: Cache path
        """
        # Replace slashes in model name with underscores
        safe_name = model_name.replace("/", "_")
        return os.path.join(self.cache_dir, safe_name)
    
    async def bootstrap_models(self) -> bool:
        """
        Initialize basic models needed for startup with progressive loading strategy
        to reduce initial memory footprint.
        
        Returns:
            bool: Success status
        """
        console.print("\n[bold magenta]Bootstrapping Essential Models[/bold magenta]")
        
        # Define model priority tiers
        # Tier 1: Critical models needed immediately at startup
        # Tier 2: Important models needed soon after startup
        # Tier 3: Optional models that can be lazy-loaded on first use
        model_tiers = {
            "tier1": ["language_detection"],
            "tier2": [],
            "tier3": []
        }
        
        # Add translation model to tier 1 if configured
        if "translation" in self.registry.list_models():
            model_tiers["tier1"].append("translation")
            
        # Add other models to appropriate tiers
        if "ner_detection" in self.registry.list_models():
            model_tiers["tier2"].append("ner_detection")
            
        if "simplifier" in self.registry.list_models():
            model_tiers["tier2"].append("simplifier")
            
        if "rag_generator" in self.registry.list_models():
            model_tiers["tier3"].append("rag_generator")
            
        if "rag_retriever" in self.registry.list_models():
            model_tiers["tier3"].append("rag_retriever")
            
        if "anonymizer" in self.registry.list_models():
            model_tiers["tier3"].append("anonymizer")
        
        # Track all models loaded as part of bootstrap
        all_bootstrap_models = []
        all_bootstrap_models.extend(model_tiers["tier1"])
        all_bootstrap_models.extend(model_tiers["tier2"])
        
        # Track load status
        success = True
        tier1_success = True
        
        # Load Tier 1 models (critical) immediately
        console.print("[bold cyan]Loading Tier 1 (Critical) Models...[/bold cyan]")
        for model_type in model_tiers["tier1"]:
            try:
                logger.info(f"Loading critical model: {model_type}...")
                console.print(f"[green]Loading {model_type}...[/green]")
                
                start_time = time.time()
                await self.load_model_async(model_type)
                load_time = time.time() - start_time
                
                logger.info(f"Bootstrapped critical model {model_type} in {load_time:.2f}s")
                console.print(f"[green]✓ Loaded {model_type} in {load_time:.2f}s[/green]")
            except Exception as e:
                console.print(f"[red]Failed to load {model_type}[/red]")
                logger.error(f"Failed to bootstrap critical model {model_type}: {e}", exc_info=True)
                tier1_success = False
                success = False
            
        # If critical models failed, stop the bootstrap process
        if not tier1_success:
            console.print(Panel(
                "[bold red]⚠ Critical models failed to bootstrap[/bold red]\n"
                "[yellow]The system cannot continue with minimum functionality[/yellow]",
                border_style="red"
            ))
            return False
            
        # Load Tier 2 models (important) in background after Tier 1 completes
        if model_tiers["tier2"]:
            console.print("[bold cyan]Loading Tier 2 (Important) Models...[/bold cyan]")
            
            for model_type in model_tiers["tier2"]:
                try:
                    logger.info(f"Loading important model: {model_type}...")
                    console.print(f"[green]Loading {model_type}...[/green]")
                    
                    start_time = time.time()
                    await self.load_model_async(model_type)
                    load_time = time.time() - start_time
                    
                    logger.info(f"Bootstrapped important model {model_type} in {load_time:.2f}s")
                    console.print(f"[green]✓ Loaded {model_type} in {load_time:.2f}s[/green]")
                except Exception as e:
                    console.print(f"[red]Failed to load {model_type}[/red]")
                    logger.error(f"Failed to bootstrap important model {model_type}: {e}", exc_info=True)
                    # Continue even if Tier 2 models fail
                    success = False
        
        # Store Tier 3 models for lazy loading
        if model_tiers["tier3"]:
            console.print("[bold cyan]Registering Tier 3 (On-demand) Models for lazy loading...[/bold cyan]")
            self.lazy_load_models = model_tiers["tier3"]
            logger.info(f"Registered {len(model_tiers['tier3'])} models for lazy loading: {', '.join(model_tiers['tier3'])}")
            
            # Create a dictionary to track which models have been requested but not loaded
            self.pending_lazy_loads = {}
                
        if success:
            console.print(Panel(
                "[bold green]✓ Progressive model bootstrap successful[/bold green]\n"
                f"[green]Loaded {len(model_tiers['tier1']) + len(model_tiers['tier2'])} models immediately[/green]\n"
                f"[blue]Registered {len(model_tiers['tier3'])} models for lazy loading[/blue]",
                border_style="green"
            ))
        else:
            console.print(Panel(
                "[bold yellow]⚠ Progressive model bootstrap partially successful[/bold yellow]\n"
                "[green]Critical models loaded successfully[/green]\n"
                "[yellow]Some non-critical models failed to load[/yellow]\n"
                "[blue]System will continue with reduced functionality[/blue]",
                border_style="yellow"
            ))
            
        return True
    
    async def load_model_async(self, model_type: str, **kwargs) -> Dict[str, Any]:
        """
        Asynchronously load a model of the specified type
        
        Args:
            model_type (str): Model type
            **kwargs: Additional arguments
            
        Returns:
            Dict[str, Any]: Model information
        """
        # Create a wrapper for async execution
        loop = asyncio.get_event_loop()
        
        # Execute load_model in a thread pool
        return await loop.run_in_executor(
            None, 
            lambda: self.load_model(model_type, **kwargs)
        )
    
    def load_model(self, model_type: str, **kwargs) -> Dict[str, Any]:
        """
        Load a model of the specified type with support for lazy loading and automatic recovery
        
        Args:
            model_type (str): Model type
            **kwargs: Additional arguments
            
        Returns:
            Dict[str, Any]: Model information
        """
        start_time = time.time()
        
        # Check if requesting a fallback/alternative model
        attempt_number = kwargs.pop("_attempt", 1)
        is_fallback = kwargs.pop("_is_fallback", False)
        original_model_type = kwargs.pop("_original_model_type", model_type)
        max_retry_attempts = kwargs.pop("max_retry_attempts", 3)
        allow_fallback = kwargs.pop("allow_fallback", True)
        
        # Check if this is a lazy-loadable model being requested for the first time
        is_lazy_load = model_type in self.lazy_load_models and model_type not in self.loaded_models
        force_load = kwargs.pop("force_load", False)  # Option to force load even for lazy models
        
        # If this is a lazy load model and we're not forcing, just record the request
        if is_lazy_load and not force_load and not is_fallback:
            # Check if model has previously failed to load
            if model_type in self.failed_loads:
                error_info = self.failed_loads[model_type]
                # Check if it's been a while since last failure - we could retry
                retry_after_seconds = 3600  # 1 hour
                time_since_failure = time.time() - error_info.get("timestamp", 0)
                
                if time_since_failure > retry_after_seconds:
                    # It's been long enough, let's retry
                    logger.info(f"Retrying previously failed model {model_type} after {time_since_failure/3600:.1f} hours")
                    # Clear the failed record
                    del self.failed_loads[model_type]
                else:
                    logger.warning(f"Skipping previously failed model: {model_type}. Error: {error_info['error']}")
                    # If we have a fallback strategy for this model type, try it instead
                    if allow_fallback and not is_fallback:
                        fallback_model = self._get_fallback_model(model_type)
                        if fallback_model:
                            logger.info(f"Using fallback model {fallback_model} instead of {model_type}")
                            # Call load_model with the fallback model type
                            return self.load_model(
                                fallback_model, 
                                _is_fallback=True,
                                _original_model_type=model_type,
                                **kwargs
                            )
                    
                    raise ValueError(f"Model {model_type} previously failed to load: {error_info['error']}")
            
            # Just track that this model was requested but don't actually load it yet
            if model_type not in self.pending_lazy_loads:
                self.pending_lazy_loads[model_type] = {
                    "first_requested": time.time(),
                    "request_count": 1
                }
                logger.info(f"Recorded first request for lazy-load model: {model_type}")
            else:
                self.pending_lazy_loads[model_type]["request_count"] += 1
                
                # If the model has been requested multiple times, consider loading it
                if self.pending_lazy_loads[model_type]["request_count"] >= 3:
                    logger.info(f"Model {model_type} has been requested {self.pending_lazy_loads[model_type]['request_count']} times. Loading now.")
                    # Continue with loading by setting force_load to True
                    force_load = True
                else:
                    logger.debug(f"Model {model_type} requested {self.pending_lazy_loads[model_type]['request_count']} times but not loading yet")
                    # Create a placeholder to represent the not-yet-loaded model
                    return {
                        "type": model_type,
                        "model": None,
                        "tokenizer": None,
                        "config": self.registry.get_model_config(model_type),
                        "status": "pending",
                        "message": f"Model scheduled for lazy loading (requested {self.pending_lazy_loads[model_type]['request_count']} times)"
                    }
        
        # Actual model loading process with lock to prevent concurrent loading of the same model
        if model_type in self.load_locks:
            # If there's already a loading process for this model, wait for it to complete
            logger.debug(f"Model {model_type} is currently being loaded by another process. Waiting...")
            return {
                "type": model_type,
                "model": None,
                "tokenizer": None,
                "config": self.registry.get_model_config(model_type),
                "status": "loading",
                "message": "Model is currently being loaded by another process"
            }
        
        # Show appropriate loading message
        if is_fallback:
            console.print(f"[bold yellow]Loading fallback model:[/bold yellow] [yellow]{model_type}[/yellow] (replacing {original_model_type})")
        elif attempt_number > 1:
            console.print(f"[bold cyan]Retry #{attempt_number} loading model:[/bold cyan] [yellow]{model_type}[/yellow]")
        else:
            console.print(f"[bold cyan]Loading model:[/bold cyan] [yellow]{model_type}[/yellow]")
        
        # Create a lock for this model
        self.load_locks[model_type] = True
        
        try:
            # Override config with kwargs
            model_kwargs = {**kwargs}
            
            # Check if model is already loaded
            if model_type in self.loaded_models:
                console.print(f"[green]✓[/green] Model [yellow]{model_type}[/yellow] already loaded from cache")
                result = {
                    "type": model_type,
                    "model": self.loaded_models[model_type],
                    "tokenizer": self.loaded_tokenizers.get(model_type),
                    "config": self.registry.get_model_config(model_type),
                    "status": "loaded"
                }
                return result
            
            # Get model configuration
            model_config = self.registry.get_model_config(model_type)
            
            # If we didn't find a configuration, try to use a more generic version
            if not model_config:
                # Get model and tokenizer names
                model_name, tokenizer_name = self.get_model_and_tokenizer(model_type)
                
                # Create default config
                model_config = ModelConfig(
                    model_name=model_name,
                    tokenizer_name=tokenizer_name,
                    task=model_type
                )
            
            # Override device if provided
            device = model_kwargs.get('device', self.device)
            
            # Determine appropriate loader function based on model type
            if not HAVE_TRANSFORMERS:
                raise ImportError("transformers library is required for model loading")
            
            # Log start of loading
            logger.info(f"Loading {model_type} ({model_config.model_name})")
            console.print(f"[bold blue]Loading {model_type} ({model_config.model_name})[/bold blue]")
            
            try:
                # Choose loader based on model configuration
                if model_config.type == "sentence-transformers":
                    # Load SentenceTransformer model
                    if not HAVE_SENTENCE_TRANSFORMERS:
                        raise ImportError("sentence_transformers library is required for embedding models")
                    
                    logger.info(f"Loading embedding model for {model_type}...")
                    console.print(f"[cyan]Loading embedding model for {model_type}...[/cyan]")
                    model = self._load_sentence_transformer(model_config, device)
                    tokenizer = None  # SentenceTransformer handles tokenization internally
                    
                elif model_config.type == "transformers":
                    # Load Hugging Face Transformers model
                    logger.info(f"Loading tokenizer for {model_type}...")
                    console.print(f"[cyan]Loading tokenizer for {model_type}...[/cyan]")
                    tokenizer = self._load_tokenizer(model_config)
                    
                    logger.info(f"Loading model for {model_type}...")
                    console.print(f"[cyan]Loading model for {model_type}...[/cyan]")
                    model = self._load_transformers_model(model_config, device)
                    
                elif model_config.type == "onnx":
                    # Load ONNX model
                    if not HAVE_ONNX:
                        raise ImportError("onnxruntime is required for ONNX models")
                    
                    logger.info(f"Loading tokenizer for {model_type}...")
                    console.print(f"[cyan]Loading tokenizer for {model_type}...[/cyan]")
                    tokenizer = self._load_tokenizer(model_config)
                    
                    logger.info(f"Loading ONNX model for {model_type}...")
                    console.print(f"[cyan]Loading ONNX model for {model_type}...[/cyan]")
                    model = self._load_onnx_model(model_config)
                    
                else:
                    # Unsupported model type
                    raise ValueError(f"Unsupported model type: {model_config.type}")
                
                # Log finalizing
                logger.info(f"Finalizing {model_type} model...")
                console.print(f"[cyan]Finalizing {model_type} model...[/cyan]")
                
                # Cache the model and tokenizer
                self.loaded_models[model_type] = model
                if tokenizer:
                    self.loaded_tokenizers[model_type] = tokenizer
                
                # If this was a fallback model for another model type, record the mapping
                if is_fallback and original_model_type != model_type:
                    self._register_fallback_mapping(original_model_type, model_type)
                
                # Log completion
                logger.info(f"Model {model_type} loaded successfully!")
                console.print(f"[green]Model {model_type} loaded successfully![/green]")
            
            except Exception as e:
                console.print(f"[red]Error loading {model_type}: {str(e)}[/red]")
                logger.error(f"Error loading model {model_type}: {e}", exc_info=True)
                
                # Record the failure
                self.failed_loads[model_type] = {
                    "error": str(e),
                    "timestamp": time.time(),
                    "attempts": self.failed_loads.get(model_type, {}).get("attempts", 0) + 1
                }
                
                # Implement retry mechanism for non-fatal errors
                if attempt_number < max_retry_attempts:
                    # Sleep with exponential backoff before retry
                    retry_delay = 2 ** (attempt_number - 1)  # 1, 2, 4, 8, etc. seconds
                    logger.info(f"Retrying model load for {model_type} in {retry_delay}s (attempt {attempt_number+1}/{max_retry_attempts})")
                    time.sleep(retry_delay)
                    
                    # Release the lock before retry
                    if model_type in self.load_locks:
                        del self.load_locks[model_type]
                    
                    # Retry with potentially different options
                    retry_options = {
                        # Adjust options based on the specific error
                        "device": "cpu" if "CUDA out of memory" in str(e) and device != "cpu" else device,
                        # Increment attempt counter
                        "_attempt": attempt_number + 1,
                        # Preserve other important options
                        "_is_fallback": is_fallback,
                        "_original_model_type": original_model_type,
                        "max_retry_attempts": max_retry_attempts,
                        "allow_fallback": allow_fallback
                    }
                    
                    # Merge with original kwargs, but let retry options take precedence
                    for k, v in kwargs.items():
                        if k not in retry_options:
                            retry_options[k] = v
                    
                    return self.load_model(model_type, **retry_options)
                
                # If we've exhausted retries, try fallback model if allowed
                if allow_fallback and not is_fallback:
                    fallback_model = self._get_fallback_model(model_type)
                    if fallback_model:
                        logger.info(f"Trying fallback model {fallback_model} after {max_retry_attempts} failed attempts to load {model_type}")
                        # Release the lock before trying fallback
                        if model_type in self.load_locks:
                            del self.load_locks[model_type]
                        
                        # Call load_model with the fallback model type
                        return self.load_model(
                            fallback_model, 
                            _is_fallback=True, 
                            _original_model_type=model_type,
                            **kwargs
                        )
                    
                    # If no more retries or fallbacks, re-raise the exception
                    raise
            
            elapsed_time = time.time() - start_time
            
            # Remove from lazy load tracking if it was a lazy model
            if model_type in self.lazy_load_models:
                # Remove from pending lazy loads
                if model_type in self.pending_lazy_loads:
                    request_count = self.pending_lazy_loads[model_type]["request_count"]
                    del self.pending_lazy_loads[model_type]
                    logger.info(f"Lazy-loaded model {model_type} after {request_count} requests")
            
            # If it was previously marked as failed, remove that record
            if model_type in self.failed_loads:
                del self.failed_loads[model_type]
                logger.info(f"Cleared failure record for {model_type} after successful load")
            
            # Show success message
            if is_fallback:
                console.print(Panel(
                    f"[bold yellow]✓ Successfully loaded fallback model:[/bold yellow] [yellow]{model_type}[/yellow]\n"
                    f"[dim]Replaces {original_model_type} • Loaded in {elapsed_time:.2f}s[/dim]",
                    border_style="yellow"
                ))
            else:
                console.print(Panel(
                    f"[bold green]✓ Successfully loaded model:[/bold green] [yellow]{model_type}[/yellow]\n"
                    f"[dim]Loaded in {elapsed_time:.2f}s[/dim]",
                    border_style="green"
                ))
            
            result = {
                "type": model_type,
                "model": model,
                "tokenizer": tokenizer,
                "config": model_config,
                "status": "loaded",
                "is_fallback": is_fallback,
                "original_type": original_model_type if is_fallback else model_type
            }
            return result
            
        finally:
            # Always release the lock when done, even if an exception occurred
            if model_type in self.load_locks:
                del self.load_locks[model_type]
                
    def _register_fallback_mapping(self, original_model_type: str, fallback_model_type: str) -> None:
        """
        Register a mapping between original model and fallback model for future reference
        
        Args:
            original_model_type (str): The original model type that failed
            fallback_model_type (str): The fallback model type that succeeded
        """
        if not hasattr(self, "fallback_mappings"):
            self.fallback_mappings = {}
            
        self.fallback_mappings[original_model_type] = fallback_model_type
        logger.info(f"Registered fallback mapping: {original_model_type} -> {fallback_model_type}")
        
    def _get_fallback_model(self, model_type: str) -> Optional[str]:
        """
        Get an appropriate fallback model type based on the original model type
        
        Args:
            model_type (str): The original model type
            
        Returns:
            Optional[str]: Fallback model type or None if no fallback available
        """
        # Check if we have a previously successful mapping
        if hasattr(self, "fallback_mappings") and model_type in self.fallback_mappings:
            return self.fallback_mappings[model_type]
            
        # Define fallback strategy based on model type
        fallback_strategies = {
            # Translation models
            "translation": ["google/mt5-small", "facebook/nllb-200-distilled-600M", "Helsinki-NLP/opus-mt-mul-en"],
            
            # Language detection
            "language_detection": ["papluca/xlm-roberta-base-language-detection", "xlm-roberta-base"],
            
            # NER detection
            "ner_detection": ["dslim/bert-base-NER", "flair/ner-english", "dbmdz/bert-large-cased-finetuned-conll03-english"],
            
            # RAG models
            "rag_generator": ["google/flan-t5-small", "facebook/bart-base"],
            "rag_retriever": ["sentence-transformers/all-MiniLM-L6-v2", "sentence-transformers/all-mpnet-base-v2"],
            
            # Other models
            "simplifier": ["t5-small", "facebook/bart-base"],
            "anonymizer": ["bert-base-cased", "distilbert-base-cased"]
        }
        
        # Get fallback options
        fallback_options = fallback_strategies.get(model_type, [])
        
        # Prefer smaller models as fallbacks
        smaller_model_types = {
            "translation": "language_detection",  # Fall back to simpler task
            "rag_generator": "simplifier",        # Fall back to simpler task
            "ner_detection": "language_detection" # Fall back to simpler task
        }
        
        if model_type in smaller_model_types:
            # Try loading a completely different model type as fallback
            different_type = smaller_model_types[model_type]
            if different_type in self.loaded_models:
                logger.info(f"Using already loaded model type {different_type} as fallback for {model_type}")
                return different_type
                
            # If the different type isn't loaded but is available in registry, add it as a fallback option
            if different_type in self.registry.list_models():
                fallback_options.append(different_type)
        
        # If we have a specific model name, try variants of that model
        current_config = self.registry.get_model_config(model_type)
        if current_config and current_config.model_name:
            model_name = current_config.model_name
            
            # If we're using a large model, try falling back to smaller variants
            if "large" in model_name:
                smaller_variant = model_name.replace("large", "base")
                fallback_options.append(smaller_variant)
                
                # Try small variant too
                smaller_variant = model_name.replace("large", "small")
                fallback_options.append(smaller_variant)
            elif "medium" in model_name or "base" in model_name:
                # Try small variant
                smaller_variant = model_name.replace("medium", "small").replace("base", "small")
                fallback_options.append(smaller_variant)
        
        # Filter out options that have already failed and the original model type
        filtered_options = []
        for option in fallback_options:
            # Skip if it's the same as original
            if option == model_type:
                continue
                
            # Skip if it's already failed
            if option in self.failed_loads:
                logger.debug(f"Skipping fallback option {option} because it previously failed")
                continue
                
            # Skip if it's not available in registry and not a model name
            if option not in self.registry.list_models() and "/" not in option:
                logger.debug(f"Skipping fallback option {option} because it's not in registry")
                continue
                
            filtered_options.append(option)
        
        # Return first valid fallback, or None if no valid options
        if filtered_options:
            return filtered_options[0]
            
        return None
    
    def _load_transformers_model(self, model_config: ModelConfig, device: str) -> Any:
        """
        Load a Hugging Face Transformers model with multi-GPU support
        
        Args:
            model_config (ModelConfig): Model configuration
            device (str): Device to load model on (e.g., "cuda:0", "cuda:1", "mps", "cpu")
            
        Returns:
            Any: Loaded model
        """
        # Import all needed transformers components here to ensure they're available
        import transformers
        from transformers import (
            AutoModel, 
            AutoModelForSeq2SeqLM,
            AutoModelForSequenceClassification,
            AutoModelForTokenClassification,
            T5ForConditionalGeneration,
            MT5ForConditionalGeneration,
            BertForTokenClassification,
            BertModel
        )
        
        # Extract model type and size from configuration for device selection
        model_type = model_config.task
        model_size = "medium"  # Default assumption
        
        # Try to determine model size from model name or config
        if "large" in model_config.model_name.lower():
            model_size = "large"
        elif "small" in model_config.model_name.lower():
            model_size = "small"
        elif "medium" in model_config.model_name.lower() or "base" in model_config.model_name.lower():
            model_size = "medium"
            
        # If device not specified, determine optimal device for this model
        if device in ["cuda", "auto"]:
            device = self._determine_device(model_type, model_size)
            logger.info(f"Auto-selected device {device} for {model_type} ({model_size})")
        
        logger.info(f"Loading Transformers model: {model_config.model_name} on {device}")
        
        # Get model kwargs
        model_kwargs = model_config.model_kwargs or {}
        
        # Add cache directory
        model_kwargs["cache_dir"] = self.get_cache_path(model_config.model_name)
        
        # Add quantization for low memory devices if needed
        if device.startswith("cuda"):
            # Extract device ID to check available memory
            device_id = int(device.split(":")[-1]) if ":" in device else 0
            
            # Check if we need to apply quantization
            if hasattr(self, "hardware_info") and self.hardware_info:
                # Try to find the GPU in hardware info
                gpu_info = next((g for g in self.hardware_info.gpus if g.device_id == device_id), None)
                
                if gpu_info:
                    # Track our model with its assigned device
                    self.model_devices[model_config.task] = device
                    
                    # Memory available on this GPU
                    mem_available = gpu_info.memory_available
                    mem_available_gb = mem_available / (1024**3)
                    
                    # Memory required for this model (estimate)
                    mem_required = self._estimate_model_memory(model_type, model_size)
                    mem_required_gb = mem_required / (1024**3)
                    
                    logger.info(f"Model {model_type} ({model_size}) requires ~{mem_required_gb:.1f}GB, "
                             f"GPU has {mem_available_gb:.1f}GB available")
                    
                    # If memory is tight, apply quantization
                    if mem_available < mem_required * 1.5:  # If less than 1.5x required memory available
                        if model_size == "large":
                            # For large models with tight memory, try to use 8-bit quantization
                            logger.info(f"Using 8-bit quantization for {model_type} due to memory constraints")
                            model_kwargs["load_in_8bit"] = True
                        elif mem_available < mem_required:
                            # For extreme cases, try more aggressive quantization
                            logger.info(f"Using 4-bit quantization for {model_type} due to severe memory constraints")
                            model_kwargs["load_in_4bit"] = True
                            model_kwargs["bnb_4bit_compute_dtype"] = torch.float16
                            model_kwargs["bnb_4bit_use_double_quant"] = True
                    
        # Determine appropriate model class based on task or explicit model_class
        try:
            # Check for explicit model_class in config first
            if hasattr(model_config, "model_class") and model_config.model_class:
                # Import the specified model class from transformers
                if model_config.model_class == "T5ForConditionalGeneration":
                    model = T5ForConditionalGeneration.from_pretrained(
                        model_config.model_name,
                        **model_kwargs
                    )
                else:
                    # Try to use the specified class
                    try:
                        model_class = getattr(transformers, model_config.model_class)
                        model = model_class.from_pretrained(
                            model_config.model_name,
                            **model_kwargs
                        )
                    except (ImportError, AttributeError) as e:
                        logger.error(f"Failed to load model class {model_config.model_class}: {e}")
                        raise ValueError(f"Unsupported model class: {model_config.model_class}")
            # Fall back to task-based selection
            elif model_config.task == "translation" or model_config.task == "rag_generation":
                # Use MT5 for translation or RAG generation
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_config.model_name, 
                    **model_kwargs
                )
            elif model_config.task == "language_detection":
                # Use sequence classification for language detection
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_config.model_name, 
                    **model_kwargs
                )
            elif model_config.task == "ner_detection":
                # Use token classification for NER
                model = AutoModelForTokenClassification.from_pretrained(
                    model_config.model_name, 
                    **model_kwargs
                )
            elif model_config.task == "simplification":
                # Use T5ForConditionalGeneration for simplification
                model = T5ForConditionalGeneration.from_pretrained(
                    model_config.model_name,
                    **model_kwargs
                )
            else:
                # Default to AutoModel
                model = AutoModel.from_pretrained(
                    model_config.model_name, 
                    **model_kwargs
                )
            
            # Move model to device
            model = model.to(device)
            
            # Set eval mode for inference
            model.eval()
            
            # Log successful load
            logger.info(f"Successfully loaded {model_config.model_name} on {device}")
            
            return model
            
        except Exception as e:
            # Handle out of memory errors specially
            if "CUDA out of memory" in str(e):
                logger.error(f"CUDA out of memory when loading {model_config.model_name} on {device}")
                logger.error(f"Trying with more aggressive memory optimization...")
                
                # Try again with more aggressive optimization
                try:
                    # Add more aggressive quantization
                    model_kwargs["load_in_8bit"] = True
                    model_kwargs["device_map"] = "auto"  # Let transformers handle device mapping
                    
                    # Check for explicit model_class in config first
                    if hasattr(model_config, "model_class") and model_config.model_class:
                        # Import the specified model class from transformers
                        if model_config.model_class == "T5ForConditionalGeneration":
                            from transformers import T5ForConditionalGeneration
                            model = T5ForConditionalGeneration.from_pretrained(
                                model_config.model_name,
                                **model_kwargs
                            )
                        else:
                            # Try to dynamically import the specified class
                            try:
                                from transformers import AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, AutoModelForTokenClassification, AutoModel
                                from transformers import T5ForConditionalGeneration, MT5ForConditionalGeneration, BertForTokenClassification
                                model_class = eval(model_config.model_class)
                                model = model_class.from_pretrained(
                                    model_config.model_name,
                                    **model_kwargs
                                )
                            except (ImportError, NameError) as e:
                                logger.error(f"Failed to import model class {model_config.model_class}: {e}")
                                raise ValueError(f"Unsupported model class: {model_config.model_class}")
                    elif model_config.task == "translation" or model_config.task == "rag_generation":
                        model = AutoModelForSeq2SeqLM.from_pretrained(
                            model_config.model_name, 
                            **model_kwargs
                        )
                    elif model_config.task == "language_detection":
                        model = AutoModelForSequenceClassification.from_pretrained(
                            model_config.model_name, 
                            **model_kwargs
                        )
                    elif model_config.task == "ner_detection":
                        model = AutoModelForTokenClassification.from_pretrained(
                            model_config.model_name, 
                            **model_kwargs
                        )
                    elif model_config.task == "simplification":
                        # Use T5ForConditionalGeneration for simplification
                        from transformers import T5ForConditionalGeneration
                        model = T5ForConditionalGeneration.from_pretrained(
                            model_config.model_name,
                            **model_kwargs
                        )
                    else:
                        model = AutoModel.from_pretrained(
                            model_config.model_name, 
                            **model_kwargs
                        )
                    
                    # Set eval mode for inference
                    model.eval()
                    
                    logger.info(f"Successfully loaded {model_config.model_name} with memory optimization")
                    return model
                    
                except Exception as e2:
                    # If that failed too, try on CPU as last resort
                    logger.error(f"Failed to load with optimization: {e2}")
                    logger.warning(f"Falling back to CPU for {model_config.model_name}")
                    
                    try:
                        # Remove quantization args, they might not work on CPU
                        if "load_in_8bit" in model_kwargs:
                            del model_kwargs["load_in_8bit"]
                        if "load_in_4bit" in model_kwargs:
                            del model_kwargs["load_in_4bit"]
                        if "bnb_4bit_compute_dtype" in model_kwargs:
                            del model_kwargs["bnb_4bit_compute_dtype"]
                        if "bnb_4bit_use_double_quant" in model_kwargs:
                            del model_kwargs["bnb_4bit_use_double_quant"]
                        if "device_map" in model_kwargs:
                            del model_kwargs["device_map"]
                            
                        # Try to load on CPU
                        if hasattr(model_config, "model_class") and model_config.model_class:
                            # Import the specified model class from transformers
                            if model_config.model_class == "T5ForConditionalGeneration":
                                from transformers import T5ForConditionalGeneration
                                model = T5ForConditionalGeneration.from_pretrained(
                                    model_config.model_name,
                                    **model_kwargs
                                )
                            else:
                                # Try to dynamically import the specified class
                                try:
                                    from transformers import AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, AutoModelForTokenClassification, AutoModel
                                    from transformers import T5ForConditionalGeneration, MT5ForConditionalGeneration, BertForTokenClassification
                                    model_class = eval(model_config.model_class)
                                    model = model_class.from_pretrained(
                                        model_config.model_name,
                                        **model_kwargs
                                    )
                                except (ImportError, NameError) as e:
                                    logger.error(f"Failed to import model class {model_config.model_class}: {e}")
                                    raise ValueError(f"Unsupported model class: {model_config.model_class}")
                        elif model_config.task == "translation" or model_config.task == "rag_generation":
                            model = AutoModelForSeq2SeqLM.from_pretrained(
                                model_config.model_name, 
                                **model_kwargs
                            )
                        elif model_config.task == "language_detection":
                            model = AutoModelForSequenceClassification.from_pretrained(
                                model_config.model_name, 
                                **model_kwargs
                            )
                        elif model_config.task == "ner_detection":
                            model = AutoModelForTokenClassification.from_pretrained(
                                model_config.model_name, 
                                **model_kwargs
                            )
                        elif model_config.task == "simplification":
                            # Use T5ForConditionalGeneration for simplification
                            from transformers import T5ForConditionalGeneration
                            model = T5ForConditionalGeneration.from_pretrained(
                                model_config.model_name,
                                **model_kwargs
                            )
                        else:
                            model = AutoModel.from_pretrained(
                                model_config.model_name, 
                                **model_kwargs
                            )
                        
                        # Move to CPU explicitly
                        model = model.to("cpu")
                        model.eval()
                        
                        # Update tracking to show this model is on CPU
                        self.model_devices[model_config.task] = "cpu"
                        
                        logger.warning(f"Loaded {model_config.model_name} on CPU as fallback")
                        return model
                        
                    except Exception as e3:
                        # If all attempts failed, re-raise the original error
                        logger.error(f"All loading attempts failed for {model_config.model_name}: {e3}")
                        raise e
            else:
                # For other errors, just re-raise
                raise
    
    def _load_tokenizer(self, model_config: ModelConfig) -> Any:
        """
        Load a tokenizer for a model
        
        Args:
            model_config (ModelConfig): Model configuration
            
        Returns:
            Any: Tokenizer
        """
        logger.info(f"Loading tokenizer: {model_config.tokenizer_name}")
        
        # Get tokenizer kwargs
        tokenizer_kwargs = model_config.tokenizer_kwargs or {}
        
        # Add cache directory
        tokenizer_kwargs["cache_dir"] = self.get_cache_path(model_config.tokenizer_name)
        
        # Load tokenizer
        return AutoTokenizer.from_pretrained(
            model_config.tokenizer_name, 
            **tokenizer_kwargs
        )
    
    def _load_sentence_transformer(self, model_config: ModelConfig, device: str) -> Any:
        """
        Load a SentenceTransformer model
        
        Args:
            model_config (ModelConfig): Model configuration
            device (str): Device to load model on
            
        Returns:
            Any: SentenceTransformer model
        """
        logger.info(f"Loading SentenceTransformer model: {model_config.model_name}")
        
        # Get model kwargs
        model_kwargs = model_config.model_kwargs or {}
        
        # Add cache directory
        model_kwargs["cache_folder"] = self.get_cache_path(model_config.model_name)
        
        # Load model
        model = SentenceTransformer(
            model_config.model_name, 
            device=device,
            **model_kwargs
        )
        
        return model
    
    def _load_onnx_model(self, model_config: ModelConfig) -> Any:
        """
        Load an ONNX model
        
        Args:
            model_config (ModelConfig): Model configuration
            
        Returns:
            Any: ONNX model
        """
        logger.info(f"Loading ONNX model: {model_config.model_name}")
        
        # Determine appropriate provider
        if self.device == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers():
            providers = ["CUDAExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
        
        # Load model
        model_path = model_config.model_name
        if not os.path.exists(model_path):
            # Try cache directory
            model_path = os.path.join(self.get_cache_path(model_config.model_name), "model.onnx")
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"ONNX model not found: {model_config.model_name}")
        
        # Create session
        session = ort.InferenceSession(model_path, providers=providers)
        
        return session
    
    def unload_model(self, model_type: str) -> bool:
        """
        Unload a model from memory
        
        Args:
            model_type (str): Model type
            
        Returns:
            bool: Success status
        """
        if model_type not in self.loaded_models:
            console.print(f"[yellow]⚠ Model {model_type} not loaded, nothing to unload[/yellow]")
            return False
        
        console.print(f"[cyan]Unloading model: [yellow]{model_type}[/yellow][/cyan]")
        
        # Get model and tokenizer
        model = self.loaded_models.pop(model_type)
        tokenizer = self.loaded_tokenizers.pop(model_type, None)
        
        # Free memory
        del model
        del tokenizer
        
        # Force garbage collection
        gc.collect()
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        console.print(f"[green]✓ Model {model_type} unloaded successfully[/green]")
        return True
    
    def unload_all_models(self) -> bool:
        """
        Unload all models from memory
        
        Returns:
            bool: Success status
        """
        console.print("[bold cyan]Unloading all models...[/bold cyan]")
        
        # Get list of loaded models
        model_types = list(self.loaded_models.keys())
        
        if not model_types:
            console.print("[yellow]No models currently loaded[/yellow]")
            return True
        
        # Log start of unloading
        logger.info(f"Unloading {len(model_types)} models")
        
        # Unload each model
        for model_type in model_types:
            logger.info(f"Unloading {model_type}...")
            console.print(f"[cyan]Unloading {model_type}...[/cyan]")
            self.unload_model(model_type)
        
        console.print(Panel("[bold green]✓ All models unloaded successfully[/bold green]", border_style="green"))
        return True
    
    def get_loaded_models(self) -> List[str]:
        """
        Get a list of loaded models
        
        Returns:
            List[str]: List of loaded model types
        """
        return list(self.loaded_models.keys())
    
    def get_model_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about loaded models
        
        Returns:
            Dict[str, Dict[str, Any]]: Model information
        """
        info = {}
        
        for model_type, model in self.loaded_models.items():
            model_config = self.registry.get_model_config(model_type)
            
            info[model_type] = {
                "loaded": True,
                "model_name": model_config.model_name if model_config else "unknown",
                "device": self.device
            }
        
        # Add information about registered but not loaded models
        for model_type in self.registry.list_models():
            if model_type not in info:
                model_config = self.registry.get_model_config(model_type)
                
                info[model_type] = {
                    "loaded": False,
                    "model_name": model_config.model_name,
                    "device": None
                }
        
        # Display model info as table
        table = Table(title="[bold]Model Information[/bold]")
        table.add_column("Model Type", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Model Name", style="yellow")
        table.add_column("Device", style="magenta")
        
        for model_type, model_info in info.items():
            status = "[green]✓ Loaded[/green]" if model_info.get("loaded", False) else "[dim]Not Loaded[/dim]"
            device = model_info.get("device", "-") or "-"
            
            table.add_row(
                model_type,
                status,
                model_info.get("model_name", "unknown"),
                device
            )
        
        console.print(table)
        return info