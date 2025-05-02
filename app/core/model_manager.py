"""
CasaLingua Model Manager
------------------------
Manages loading and configuration of multiple LLMs based on hardware capabilities.
Implements size-based model selection for translation, multi-purpose, and verification LLMs.
"""

import os
import sys
import platform
import psutil
import torch
import logging
import time
from typing import Dict, Literal, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich.logging import RichHandler
from rich.text import Text

from app.core.pipeline.tokenizer import TokenizerPipeline
# Import ModelRegistry for dynamic model/tokenizer selection
from app.services.models.loader import ModelRegistry

# PDF processor support check
try:
    import fitz  # PyMuPDF
    from fpdf import FPDF
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

# Configure rich console for pretty output
console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True, console=console)]
)
logger = logging.getLogger("casa_lingua")

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
class HardwareInfo:
    total_memory: int  # in bytes
    available_memory: int  # in bytes
    processor_type: ProcessorType
    has_gpu: bool
    gpu_memory: Optional[int] = None  # in bytes, None if no GPU

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

class HardwareDetector:
    """Detects and provides information about the hardware environment."""
    
    @staticmethod
    def get_hardware_info() -> HardwareInfo:
        """Collect information about the system hardware."""
        
        with console.status("[bold green]Detecting hardware...", spinner="dots"):
            # Get memory information
            mem = psutil.virtual_memory()
            total_memory = mem.total
            available_memory = mem.available
            
            # Detect processor type
            processor_type = HardwareDetector._detect_processor_type()
            
            # Check for GPU
            has_gpu, gpu_memory = HardwareDetector._detect_gpu()
            
            console.print(Panel(
                Text("Hardware Detection Complete", style="bold green"),
                subtitle=f"Detected: {processor_type.value.upper()} processor with {total_memory / (1024**3):.1f} GB RAM"
            ))
            
            return HardwareInfo(
                total_memory=total_memory,
                available_memory=available_memory,
                processor_type=processor_type,
                has_gpu=has_gpu,
                gpu_memory=gpu_memory
            )
    
    @staticmethod
    def _detect_processor_type() -> ProcessorType:
        """Detect the type of processor."""
        system = platform.system()
        machine = platform.machine()
        
        if system == "Darwin" and machine in ["arm64", "arm"]:
            return ProcessorType.APPLE_SILICON
        elif "intel" in platform.processor().lower():
            return ProcessorType.INTEL
        else:
            # Check for NVIDIA or AMD via more complex logic
            # This is simplified for this example
            return ProcessorType.OTHER
    
    @staticmethod
    def _detect_gpu() -> Tuple[bool, Optional[int]]:
        """Detect if a GPU is available and its memory."""
        has_gpu = torch.cuda.is_available()
        
        if has_gpu:
            # Get GPU info via torch
            device_count = torch.cuda.device_count()
            if device_count > 0:
                # Use the first GPU for simplicity
                gpu_memory = torch.cuda.get_device_properties(0).total_memory
                return True, gpu_memory
        
        return False, None

class ModelManager:
    """Manages the loading and configuration of LLMs based on hardware capabilities."""
    
    def __init__(self, model_loader):
        self.model_loader = model_loader
        self.hardware_info = HardwareDetector.get_hardware_info()
        self.model_sizes = self._determine_model_sizes()
        self.models = {}

        # PDF processor initialization/warning
        if not PDF_SUPPORT:
            logger.warning("⚠️ PyMuPDF (fitz) or FPDF not available. PDF processing will be limited.")
        else:
            logger.info("✓ PDF processor initialized with full support")
        
        # Display selected configuration
        self._display_model_configuration()
    
    def _determine_model_sizes(self) -> Dict[str, ModelSize]:
        """Determine appropriate model sizes based on hardware capabilities."""
        
        # Calculate effective memory (either RAM or GPU, whichever is smaller but not zero)
        if self.hardware_info.has_gpu and self.hardware_info.gpu_memory:
            effective_memory = min(
                self.hardware_info.available_memory, 
                self.hardware_info.gpu_memory
            )
        else:
            effective_memory = self.hardware_info.available_memory
        
        # Size configurations based on available memory
        if effective_memory >= 40 * 1024 * 1024 * 1024:  # 40GB+
            return {
                ModelType.TRANSLATION: ModelSize.LARGE,
                ModelType.MULTIPURPOSE: ModelSize.LARGE,
                ModelType.VERIFICATION: ModelSize.LARGE
            }
        elif effective_memory >= 25 * 1024 * 1024 * 1024:  # 25GB+
            return {
                ModelType.TRANSLATION: ModelSize.LARGE,
                ModelType.MULTIPURPOSE: ModelSize.MEDIUM,
                ModelType.VERIFICATION: ModelSize.MEDIUM
            }
        elif effective_memory >= 15 * 1024 * 1024 * 1024:  # 15GB+
            return {
                ModelType.TRANSLATION: ModelSize.MEDIUM,
                ModelType.MULTIPURPOSE: ModelSize.SMALL,
                ModelType.VERIFICATION: ModelSize.SMALL
            }
        else:  # Minimal configuration
            return {
                ModelType.TRANSLATION: ModelSize.SMALL,
                ModelType.MULTIPURPOSE: ModelSize.SMALL,
                ModelType.VERIFICATION: ModelSize.SMALL
            }
    
    def _determine_quantization(self, model_type: ModelType) -> int:
        """Determine the quantization level based on hardware."""
        
        # Apply different quantization based on processor type
        if self.hardware_info.processor_type == ProcessorType.APPLE_SILICON:
            return 16  # 16-bit precision for Apple Silicon
        elif self.hardware_info.processor_type == ProcessorType.INTEL:
            # For Intel, use 8-bit for larger models, 4-bit for smaller
            if self.model_sizes[model_type] == ModelSize.LARGE:
                return 8
            else:
                return 4
        else:
            # Default case
            return 8
    
    def _display_model_configuration(self):
        """Display the selected model configuration in a pretty table."""
        
        table = Table(title="CasaLingua Model Configuration")
        table.add_column("Model Type", style="cyan")
        table.add_column("Size", style="green")
        table.add_column("Memory Required", style="magenta")
        table.add_column("Quantization", style="yellow")
        
        # Calculate total memory required
        total_memory_required = 0
        
        for model_type in ModelType:
            size = self.model_sizes[model_type]
            memory_required = ModelSizeConfig.MEMORY_REQUIREMENTS[model_type][size]
            quantization = self._determine_quantization(model_type)
            
            # Adjust memory based on quantization (simplified calculation)
            adjusted_memory = memory_required * (quantization / 16)
            total_memory_required += adjusted_memory
            
            table.add_row(
                model_type.value.capitalize(),
                size.value.capitalize(),
                f"{adjusted_memory / (1024**3):.2f} GB",
                f"{quantization}-bit"
            )
        
        # Add total row
        table.add_row(
            "TOTAL", 
            "", 
            f"{total_memory_required / (1024**3):.2f} GB", 
            "",
            style="bold"
        )
        
        console.print(table)
        
        # Print memory status
        available_gb = self.hardware_info.available_memory / (1024**3)
        required_gb = total_memory_required / (1024**3)
        remaining_gb = available_gb - required_gb
        
        if remaining_gb > 0:
            console.print(f"[bold green]✓ Configuration fits within available memory "
                          f"({remaining_gb:.2f} GB remaining)[/bold green]")
        else:
            console.print(f"[bold red]⚠ Warning: Configuration requires more memory than available "
                          f"({abs(remaining_gb):.2f} GB deficit)[/bold red]")
    
    def load_models(self):
        """Load all models based on the determined configuration."""
        
        for model_type in ModelType:
            self._load_model(model_type)
    
    def _load_model(self, model_type: ModelType):
        """Load a specific model with progress indication."""
        
        size = self.model_sizes[model_type]
        model_path = ModelSizeConfig.MODEL_PATHS[model_type][size]
        quantization = self._determine_quantization(model_type)
        
        memory_required = ModelSizeConfig.MEMORY_REQUIREMENTS[model_type][size]
        adjusted_memory = memory_required * (quantization / 16)

        # Dynamically obtain model/tokenizer names from registry
        registry = ModelRegistry()
        model_name, tokenizer_name = registry.get_model_and_tokenizer(model_type.value)
        # Dynamically load the tokenizer for this model_type
        tokenizer = TokenizerPipeline(model_name=tokenizer_name, task_type=model_type.value)
        logger.debug(f"Loaded tokenizer '{tokenizer_name}' for model_type={model_type.value}")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("[bold green]{task.percentage:>3.0f}%"),
            TimeElapsedColumn()
        ) as progress:
            task = progress.add_task(
                f"Loading {model_type.value} model ({size.value}, {quantization}-bit)", 
                total=100
            )
            
            # Simulate loading progress
            # In a real implementation, this would be actual model loading code
            for i in range(100):
                # Simulate work
                time.sleep(0.05)
                progress.update(task, advance=1)
            
            # Instead of model_registry, use model_loader
            # Example: model = self.model_loader.load_model(model_path, quantization=quantization)
            # For demonstration, we just save ModelConfig as before
            self.models[model_type] = ModelConfig(
                size=size,
                memory_required=adjusted_memory,
                model_path=model_path,
                quantization=quantization
            )
            # Note: If you want to associate the tokenizer to the model, you can store it in self.models as a tuple or dict
            # For now, just log/validate the tokenizer object
        
        console.print(f"[bold green]✓ Loaded {model_type.value} model:[/bold green] "
                      f"{size.value} size, {quantization}-bit quantization")
    
    def get_model(self, model_type: ModelType):
        """Get a loaded model by type."""
        if model_type not in self.models:
            raise ValueError(f"Model {model_type.value} not loaded")
        
        # In a real implementation, this would return the actual model
        return self.models[model_type]
    
    def unload_model(self, model_type: ModelType):
        """Temporarily unload a model to free memory."""
        if model_type in self.models:
            console.print(f"[yellow]Unloading {model_type.value} model to free memory[/yellow]")
            del self.models[model_type]
            
            # Force garbage collection
            import gc
            gc.collect()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def reload_model(self, model_type: ModelType):
        """Reload a previously unloaded model."""
        if model_type not in self.models:
            self._load_model(model_type)

# Middleware for model operations
class ModelMiddleware:
    """Middleware that wraps model operations for logging, error handling, etc."""
    
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
    
    def execute(self, model_type: ModelType, operation: str, *args, **kwargs):
        """Execute an operation on a model with middleware processing."""
        
        start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        
        if start_time:
            start_time.record()
        
        operation_start = time.time()
        
        try:
            # Ensure model is loaded
            if model_type not in self.model_manager.models:
                logger.info(f"Model {model_type.value} not loaded, loading now...")
                self.model_manager.reload_model(model_type)
            
            # Execute operation (in real implementation, this would call model methods)
            result = self._simulate_model_operation(model_type, operation, *args, **kwargs)
            
            if end_time:
                end_time.record()
                torch.cuda.synchronize()
                gpu_time_ms = start_time.elapsed_time(end_time)
                logger.info(f"GPU time: {gpu_time_ms:.2f} ms")
            
            cpu_time = time.time() - operation_start
            logger.info(f"CPU time: {cpu_time*1000:.2f} ms")
            
            return result
            
        except Exception as e:
            logger.error(f"Error during {operation} with {model_type.value} model: {str(e)}")
            raise
    
    def _simulate_model_operation(self, model_type: ModelType, operation: str, *args, **kwargs):
        """Simulate a model operation (for demonstration)."""
        
        # In a real implementation, this would perform actual operations on the model
        console.print(f"[cyan]Executing {operation} on {model_type.value} model[/cyan]")
        
        # Simulate some work
        time.sleep(0.5)
        
        return {"status": "success", "operation": operation, "model": model_type.value}

# Example usage function
def run_casa_lingua():
    """Run the CasaLingua system with the model manager."""
    
    console.print(Panel.fit(
        Text("CasaLingua LLM System", style="bold cyan"),
        subtitle="Multilingual AI Assistant",
        border_style="green"
    ))
    
    try:
        # You must provide a model_loader instance here
        # For demonstration, we'll use a placeholder object
        class DummyModelLoader:
            def load_model(self, *args, **kwargs):
                return None
        model_loader = DummyModelLoader()
        # Initialize the model manager
        model_manager = ModelManager(model_loader)
        
        # Load the models
        model_manager.load_models()
        
        # Create middleware
        middleware = ModelMiddleware(model_manager)
        
        # Example operations
        console.print("\n[bold]Performing example operations:[/bold]\n")
        
        # Translation example
        console.print(Panel(
            "Translating: 'Hello, how are you?' to Spanish",
            title="Translation Task",
            border_style="blue"
        ))
        middleware.execute(ModelType.TRANSLATION, "translate", 
                          text="Hello, how are you?", target_language="es")
        
        # RAG example
        console.print(Panel(
            "Query: 'What are the benefits of learning multiple languages?'",
            title="RAG Query Task",
            border_style="magenta"
        ))
        middleware.execute(ModelType.MULTIPURPOSE, "rag_query", 
                          query="What are the benefits of learning multiple languages?")
        
        # Verification example
        console.print(Panel(
            "Verify fact: 'Spanish is the second most spoken language globally'",
            title="Fact Verification",
            border_style="yellow"
        ))
        middleware.execute(ModelType.VERIFICATION, "verify_fact",
                          statement="Spanish is the second most spoken language globally")
        
        # Demonstrate dynamic model unloading/reloading
        console.print("\n[bold]Demonstrating dynamic model management:[/bold]\n")
        
        # Unload a model to save memory
        model_manager.unload_model(ModelType.VERIFICATION)
        
        # Try an operation that requires the model, forcing a reload
        console.print("Attempting operation with unloaded model:")
        middleware.execute(ModelType.VERIFICATION, "verify_fact",
                          statement="Learning languages improves cognitive abilities")
        
        console.print("\n[bold green]All operations completed successfully![/bold green]")
        
    except Exception as e:
        console.print(f"[bold red]Error running CasaLingua system: {str(e)}[/bold red]")
        import traceback
        console.print(traceback.format_exc())
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(run_casa_lingua())