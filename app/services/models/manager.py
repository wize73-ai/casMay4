"""
Enhanced Model Manager for CasaLingua
Provides advanced model management with hardware-aware loading
"""

import os
import logging
import asyncio
import torch
import gc
import time
from typing import Dict, Any, List, Optional, Tuple, Union
from enum import Enum
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table

# Configure logging
logger = logging.getLogger(__name__)

# Initialize rich console
console = Console()

# Import model loader
from app.services.models.loader import ModelLoader

class ModelType(str, Enum):
    """Model types enum"""
    TRANSLATION = "translation"
    MULTIPURPOSE = "multipurpose" 
    VERIFICATION = "verification"
    LANGUAGE_DETECTION = "language_detection"
    NER_DETECTION = "ner_detection"
    SIMPLIFIER = "simplifier"
    RAG_GENERATOR = "rag_generator"
    RAG_RETRIEVER = "rag_retriever"
    ANONYMIZER = "anonymizer"

class EnhancedModelManager:
    """Enhanced model manager with hardware-aware capabilities"""
    
    def __init__(self, loader: ModelLoader, hardware_info: Dict[str, Any], config: Dict[str, Any] = None):
        """
        Initialize enhanced model manager
        
        Args:
            loader (ModelLoader): Model loader instance
            hardware_info (Dict[str, Any]): Hardware information
            config (Dict[str, Any], optional): Application configuration
        """
        self.loader = loader
        self.hardware_info = hardware_info
        self.config = config or {}
        self.loaded_models = {}
        self.model_metadata = {}
        
        # Device and precision configuration
        self.device = self._determine_device()
        self.precision = self._determine_precision()
        
        console.print(Panel(
            f"[bold cyan]Enhanced Model Manager Initialized[/bold cyan]\n"
            f"Device: [yellow]{self.device}[/yellow], Precision: [green]{self.precision}[/green]",
            border_style="blue"
        ))
    
    def _determine_device(self) -> str:
        """
        Determine the appropriate device for model execution
        
        Returns:
            str: Device string ("cuda", "mps", "cpu")
        """
        # Check if device is specified in config
        if self.config.get("device"):
            return self.config["device"]
        
        # Check hardware information
        gpu_info = self.hardware_info.get("gpu", {})
        has_gpu = gpu_info.get("has_gpu", False)
        
        if has_gpu:
            if gpu_info.get("cuda_available", False):
                return "cuda"
            elif gpu_info.get("mps_available", False):
                return "mps"
        
        # Default to CPU
        return "cpu"
    
    def _determine_precision(self) -> str:
        """
        Determine the appropriate precision for models
        
        Returns:
            str: Precision string ("float32", "float16", "int8")
        """
        # Check if precision is specified in config
        if self.config.get("precision"):
            return self.config["precision"]
        
        # Base decision on hardware
        if self.device == "cuda":
            # For CUDA, use float16 if enough memory
            gpu_memory = self.hardware_info.get("gpu", {}).get("gpu_memory", 0)
            if gpu_memory > 4 * 1024**3:  # > 4GB
                return "float16"
            else:
                return "int8"
        elif self.device == "mps":
            # For MPS (Apple Silicon), use float16
            return "float16"
        else:
            # For CPU, use int8 if AVX2 available, otherwise float32
            cpu_info = self.hardware_info.get("cpu", {})
            if cpu_info.get("supports_avx2", False):
                return "int8"
            else:
                return "float32"
    
    async def load_model(self, model_type: Union[ModelType, str]) -> Dict[str, Any]:
        """
        Load a model with the appropriate settings based on hardware
        
        Args:
            model_type (Union[ModelType, str]): Model type
            
        Returns:
            Dict[str, Any]: Model information
        """
        # Convert enum to string if needed
        if isinstance(model_type, ModelType):
            model_type_str = model_type.value
        else:
            model_type_str = model_type
            
        console.print(f"[bold cyan]Enhanced loading of model:[/bold cyan] [yellow]{model_type_str}[/yellow]")
        
        # Check if model is already loaded
        if model_type_str in self.loaded_models:
            console.print(f"[green]✓[/green] Model [yellow]{model_type_str}[/yellow] already loaded from cache")
            return {
                "model": self.loaded_models[model_type_str],
                "tokenizer": self.model_metadata.get(model_type_str, {}).get("tokenizer"),
                "type": model_type_str
            }
        
        # Prepare kwargs for loading
        kwargs = {
            "device": self.device,
            "precision": self.precision
        }
        
        # Load model using loader
        try:
            # Log start of loading instead of using Progress
            logger.info(f"Loading {model_type_str} model with {self.precision} precision...")
            console.print(f"[bold blue]Loading {model_type_str} model with {self.precision} precision...[/bold blue]")
            
            # Load model using async loader
            start_time = time.time()
            model_info = await self.loader.load_model_async(model_type_str, **kwargs)
            load_time = time.time() - start_time
            
            # Extract model and tokenizer
            model = model_info["model"]
            tokenizer = model_info.get("tokenizer")
            config = model_info.get("config")
            
            # Store model and metadata
            self.loaded_models[model_type_str] = model
            self.model_metadata[model_type_str] = {
                "tokenizer": tokenizer,
                "config": config,
                "loaded_at": asyncio.get_event_loop().time()
            }
            
            # Log completion
            logger.info(f"Model {model_type_str} loaded successfully in {load_time:.2f}s")
            console.print(Panel(f"[bold green]✓ Successfully loaded model:[/bold green] [yellow]{model_type_str}[/yellow] in {load_time:.2f}s", border_style="green"))
            
            return {
                "model": model,
                "tokenizer": tokenizer,
                "type": model_type_str
            }
        except Exception as e:
            console.print(Panel(f"[bold red]⚠ Error loading model:[/bold red] [yellow]{model_type_str}[/yellow]\n{str(e)}", border_style="red"))
            logger.error(f"Error loading model {model_type_str}: {e}")
            raise
    
    async def unload_all_models(self) -> None:
        """Unload all models to free memory"""
        console.print("[bold cyan]Unloading all models...[/bold cyan]")
        
        try:
            # Get list of loaded models
            model_types = list(self.loaded_models.keys())
            
            if not model_types:
                console.print("[yellow]No models currently loaded[/yellow]")
                return
            
            # Progress tracking
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                console=console
            ) as progress:
                # Create overall task
                task = progress.add_task(f"Unloading {len(model_types)} models", total=len(model_types))
                
                # Unload each model
                for model_type in model_types:
                    progress.update(task, description=f"Unloading {model_type}...")
                    # Use loader to unload model
                    self.loader.unload_model(model_type)
                    
                    # Remove from local cache
                    if model_type in self.loaded_models:
                        del self.loaded_models[model_type]
                    if model_type in self.model_metadata:
                        del self.model_metadata[model_type]
                    
                    progress.update(task, advance=1)
            
            # Force garbage collection
            gc.collect()
            
            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            console.print(Panel("[bold green]✓ All models unloaded successfully[/bold green]", border_style="green"))
        except Exception as e:
            console.print(Panel(f"[bold red]⚠ Error unloading models:[/bold red]\n{str(e)}", border_style="red"))
            logger.error(f"Error unloading models: {e}")
    
    async def unload_model(self, model_type: Union[ModelType, str]) -> None:
        """
        Unload a specific model to free memory
        
        Args:
            model_type (Union[ModelType, str]): Model type
        """
        # Convert enum to string if needed
        if isinstance(model_type, ModelType):
            model_type_str = model_type.value
        else:
            model_type_str = model_type
            
        console.print(f"[cyan]Unloading model: [yellow]{model_type_str}[/yellow][/cyan]")
        
        try:
            # Use loader to unload model
            success = self.loader.unload_model(model_type_str)
            
            if success:
                # Remove from local cache
                if model_type_str in self.loaded_models:
                    del self.loaded_models[model_type_str]
                if model_type_str in self.model_metadata:
                    del self.model_metadata[model_type_str]
                
                # Force garbage collection
                gc.collect()
                
                # Clear CUDA cache if available
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                console.print(f"[green]✓ Model {model_type_str} unloaded successfully[/green]")
            else:
                console.print(f"[yellow]⚠ Failed to unload model {model_type_str}[/yellow]")
        except Exception as e:
            console.print(f"[red]⚠ Error unloading model {model_type_str}: {str(e)}[/red]")
            logger.error(f"Error unloading model {model_type_str}: {e}")
    
    async def reload_model(self, model_type: Union[ModelType, str]) -> Dict[str, Any]:
        """
        Reload a previously unloaded model
        
        Args:
            model_type (Union[ModelType, str]): Model type
            
        Returns:
            Dict[str, Any]: Model information
        """
        # Convert enum to string if needed
        if isinstance(model_type, ModelType):
            model_type_str = model_type.value
        else:
            model_type_str = model_type
            
        console.print(f"[bold cyan]Reloading model: [yellow]{model_type_str}[/yellow][/bold cyan]")
        
        # Ensure model is unloaded first
        await self.unload_model(model_type_str)
        
        # Load the model
        return await self.load_model(model_type_str)
    
    async def run_model(self, model_type: str, method_name: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a model with the specified method and input data
        
        Args:
            model_type (str): Model type
            method_name (str): Method to call on the model wrapper
            input_data (Dict[str, Any]): Input data for the model
            
        Returns:
            Dict[str, Any]: Model output
        """
        logger.info(f"Running model {model_type}.{method_name}")
        
        # Ensure the model is loaded
        if model_type not in self.loaded_models:
            logger.info(f"Model {model_type} not loaded, loading now")
            model_info = await self.load_model(model_type)
            
            if not model_info.get("model"):
                raise ValueError(f"Failed to load model {model_type}")
        
        # Get the model and its wrapper
        model = self.loaded_models[model_type]
        tokenizer = self.model_metadata.get(model_type, {}).get("tokenizer")
        
        # Create input for the model wrapper
        from app.services.models.wrapper import ModelInput
        
        # Extract common fields from input_data
        text = input_data.get("text", "")
        source_language = input_data.get("source_language")
        target_language = input_data.get("target_language")
        context = input_data.get("context", [])
        parameters = input_data.get("parameters", {})
        
        # Create ModelInput instance
        model_input = ModelInput(
            text=text,
            source_language=source_language,
            target_language=target_language,
            context=context,
            parameters=parameters
        )
        
        # Import wrapper factory function
        from app.services.models.wrapper import create_model_wrapper
        
        # Create wrapper for the model
        wrapper = create_model_wrapper(
            model_type,
            model,
            tokenizer,
            {"task": model_type, "device": self.device, "precision": self.precision}
        )
        
        # Call the appropriate method
        if method_name == "process":
            # Synchronous processing
            result = wrapper.process(model_input)
            return {"result": result.result, "metadata": result.metadata, "metrics": result.metrics}
        elif method_name == "process_async":
            # Asynchronous processing
            result = await wrapper.process_async(model_input)
            return {"result": result.result, "metadata": result.metadata, "metrics": result.metrics}
        else:
            raise ValueError(f"Unknown method {method_name}")
    
    def get_model_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all loaded models
        
        Returns:
            Dict[str, Dict[str, Any]]: Model information
        """
        # Get information from loader
        loader_info = self.loader.get_model_info()
        
        # Enhance with local metadata
        info = {}
        for model_type, model_info in loader_info.items():
            metadata = self.model_metadata.get(model_type, {})
            model_config = metadata.get("config")
            
            info[model_type] = {
                "loaded": model_type in self.loaded_models,
                "model_name": model_config.model_name if model_config else model_info.get("model_name", "unknown"),
                "device": self.device,
                "precision": self.precision
            }
        
        # Create a table to display model information
        table = Table(title="[bold]Loaded Models[/bold]")
        table.add_column("Model Type", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Model Name", style="yellow")
        table.add_column("Device", style="magenta")
        table.add_column("Precision", style="blue")
        
        for model_type, model_info in info.items():
            status = "[green]✓ Loaded[/green]" if model_info.get("loaded", False) else "[dim]Not Loaded[/dim]"
            table.add_row(
                model_type,
                status,
                model_info.get("model_name", "unknown"),
                model_info.get("device", "unknown"),
                model_info.get("precision", "unknown")
            )
        
        console.print(table)
        return info