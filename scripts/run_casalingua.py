#!/usr/bin/env python3
"""
CasaLingua Runner Script
------------------------
Entry point for the CasaLingua standalone application that provides 
a command-line interface for interacting with the multilingual AI system.

This script serves as a standalone entry point separate from the FastAPI
application, allowing model usage directly from the command line.

Author: Exygy Development Team
Version: 1.0.0
License: MIT
"""

import os
import sys
import argparse
import time
import json
import logging
import platform
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional
from enum import Enum
from pathlib import Path

# Setup base path to allow imports from app directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

# Import UI components
from app.ui.colors import init_terminal_colors
from app.ui.banner import print_startup_banner
from app.ui.console import setup_console_logging, log_with_color, Console

# Import utility functions
from app.utils.config import load_config
from app.utils.logging import configure_logging

# Import core classes (these will be defined in main.py and other files)
from app.main import (
    ModelType, ModelSize, EnhancedHardwareDetector, 
    EnhancedModelManager, ModelRegistry, MetricsCollector
)

# Initialize terminal colors
init_terminal_colors()

# Configure console logging with color
console = Console()
logger = setup_console_logging()

# Initialize event loop for async operations
def get_event_loop():
    try:
        return asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop

class CasaLinguaRunner:
    """Main runner class for CasaLingua CLI application."""
    
    def __init__(self, config_path=None):
        """Initialize the runner with configuration."""
        self.config_path = config_path
        self.config = None
        self.hardware_detector = None
        self.hardware_info = None
        self.model_registry = None
        self.model_manager = None
        self.metrics = None
    
    async def initialize(self):
        """Initialize the system components."""
        # Load configuration
        self.config = self._load_config()
        logger.info(f"Environment: {self.config.get('environment', 'development')}")
        logger.info(f"Python version: {platform.python_version()}")
        
        # Configure detailed logging
        configure_logging(self.config)
        
        # Hardware detection
        logger.info("Detecting hardware capabilities...")
        self.hardware_detector = EnhancedHardwareDetector(self.config)
        self.hardware_info = await self.hardware_detector.detect_all()
        optimal_config = self.hardware_detector.recommend_config()
        model_config = self.hardware_detector.apply_configuration(optimal_config)
        
        # Initialize metrics collector
        self.metrics = MetricsCollector(self.config)
        
        # Initialize model components
        logger.info("Initializing model registry...")
        self.model_registry = ModelRegistry(self.config)
        await self.model_registry.initialize(self._convert_hardware_info())
        
        # Initialize model manager
        logger.info("Initializing model manager...")
        self.model_manager = EnhancedModelManager(
            self.model_registry, 
            self._convert_hardware_info(), 
            self.config
        )
        
        logger.info("✓ Initialization complete!")
    
    def _load_config(self):
        """Load configuration from file or default."""
        if self.config_path and os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                return json.load(f)
        
        # Use app's config loading function as fallback
        return load_config()
    
    def _convert_hardware_info(self):
        """Convert EnhancedHardwareInfo to dict for compatibility with existing code."""
        return {
            "total_memory": self.hardware_info.total_memory,
            "available_memory": self.hardware_info.available_memory,
            "processor_type": self.hardware_info.processor_type.value,
            "has_gpu": self.hardware_info.has_gpu,
            "gpu_memory": self.hardware_info.gpu_memory,
            "cpu_cores": self.hardware_info.cpu_cores,
            "cpu_threads": self.hardware_info.cpu_threads
        }
    
    async def load_models(self):
        """Load all models based on hardware configuration."""
        logger.info("Loading models...")
        await self.model_manager.load_base_models()
        logger.info("✓ All models loaded!")
    
    async def translate_text(self, text, target_lang, source_lang=None):
        """Translate text to the specified language."""
        logger.info(f"Translating: '{text}' to {target_lang}")
        
        # Ensure translation model is loaded
        translation_model = self.model_manager.get_model(ModelType.TRANSLATION)
        if not translation_model:
            logger.info("Translation model not loaded, loading now...")
            await self.model_manager.load_model(ModelType.TRANSLATION)
            translation_model = self.model_manager.get_model(ModelType.TRANSLATION)
        
        # Execute translation (this is a placeholder - replace with actual call)
        # In a real implementation, this would call the actual translation function
        console.print(f"\n[bold]Translating:[/bold] {text}")
        console.print(f"[bold]Target language:[/bold] {target_lang}")
        if source_lang:
            console.print(f"[bold]Source language:[/bold] {source_lang}")
        
        # Simulate processing
        with console.status("[bold green]Translating...", spinner="dots"):
            time.sleep(1)  # Simulate processing time
        
        # This is a placeholder - replace with actual model call
        # translated_text = await translation_model.translate(text, target_lang, source_lang)
        
        # Simulate response
        translated = {
            "es": "¡Hola! ¿Cómo estás?",
            "fr": "Salut! Comment ça va?",
            "de": "Hallo! Wie geht es dir?",
            "it": "Ciao! Come stai?",
            "pt": "Olá! Como você está?",
            "zh": "你好！你好吗？",
            "ja": "こんにちは！お元気ですか？",
            "ko": "안녕하세요! 어떻게 지내세요?",
            "ru": "Привет! Как дела?",
            "ar": "مرحبا! كيف حالك؟"
        }.get(target_lang.lower(), f"[Translation to {target_lang}]")
        
        console.print(f"\n[bold green]Translation result:[/bold green] {translated}")
        return translated
    
    async def process_query(self, query):
        """Process a query using the multipurpose model."""
        logger.info(f"Processing query: '{query}'")
        
        # Ensure multipurpose model is loaded
        multipurpose_model = self.model_manager.get_model(ModelType.MULTIPURPOSE)
        if not multipurpose_model:
            logger.info("Multipurpose model not loaded, loading now...")
            await self.model_manager.load_model(ModelType.MULTIPURPOSE)
            multipurpose_model = self.model_manager.get_model(ModelType.MULTIPURPOSE)
        
        # Execute query (this is a placeholder - replace with actual call)
        console.print(f"\n[bold]Query:[/bold] {query}")
        
        # Simulate processing
        with console.status("[bold green]Processing query...", spinner="dots"):
            time.sleep(1.5)  # Simulate processing time
        
        # This is a placeholder - replace with actual model call
        # response = await multipurpose_model.process_query(query)
        
        # Simulate response
        response = "Based on my knowledge, learning multiple languages enhances cognitive abilities, improves memory, and can lead to better decision-making. Studies have shown that multilingual individuals often demonstrate greater creativity and problem-solving skills."
        
        console.print(f"\n[bold green]Response:[/bold green] {response}")
        return response
    
    async def verify_fact(self, statement):
        """Verify a statement using the verification model."""
        logger.info(f"Verifying statement: '{statement}'")
        
        # Ensure verification model is loaded
        verification_model = self.model_manager.get_model(ModelType.VERIFICATION)
        if not verification_model:
            logger.info("Verification model not loaded, loading now...")
            await self.model_manager.load_model(ModelType.VERIFICATION)
            verification_model = self.model_manager.get_model(ModelType.VERIFICATION)
        
        # Execute verification (this is a placeholder - replace with actual call)
        console.print(f"\n[bold]Verifying statement:[/bold] {statement}")
        
        # Simulate processing
        with console.status("[bold yellow]Verifying...", spinner="dots"):
            time.sleep(1.2)  # Simulate processing time
        
        # This is a placeholder - replace with actual model call
        # verification_result = await verification_model.verify_statement(statement)
        
        # Simulate response
        if "spanish" in statement.lower() and "second" in statement.lower():
            verification = "PARTIALLY ACCURATE: Spanish is indeed one of the most widely spoken languages globally, but it ranks as the 4th most spoken language by total speakers. It is the 2nd most spoken language by native speakers."
        else:
            verification = "NEEDS CONTEXT: This statement requires additional context for full verification. Based on available research, there is evidence supporting the cognitive benefits of language learning, though the exact mechanisms and extent vary among individuals."
        
        console.print(f"\n[bold yellow]Verification result:[/bold yellow] {verification}")
        return verification
    
    async def show_model_info(self):
        """Display information about models and hardware."""
        if not self.model_manager:
            logger.error("Model manager not initialized")
            return
        
        console.print("\n[bold]CasaLingua Model Information[/bold]")
        console.print("=" * 50)
        
        # Display hardware info
        console.print("\n[bold cyan]Hardware Information:[/bold cyan]")
        console.print(f"Processor: {self.hardware_info.processor_type.value}")
        console.print(f"CPU Cores: {self.hardware_info.cpu_cores} (Physical), {self.hardware_info.cpu_threads} (Logical)")
        console.print(f"Total Memory: {self.hardware_info.total_memory / (1024**3):.1f} GB")
        console.print(f"Available Memory: {self.hardware_info.available_memory / (1024**3):.1f} GB")
        
        if self.hardware_info.has_gpu:
            console.print(f"GPU: {self.hardware_info.gpu_name}")
            console.print(f"GPU Memory: {self.hardware_info.gpu_memory / (1024**3):.1f} GB")
        else:
            console.print("GPU: None detected")
        
        # Display model configurations
        console.print("\n[bold green]Model Configurations:[/bold green]")
        
        model_info = self.model_manager.get_model_info()
        for model_type_name, info in model_info.items():
            status_color = "green" if info["status"] == "loaded" else "yellow"
            console.print(f"[bold]{model_type_name.capitalize()}[/bold]: [{status_color}]{info['status']}[/{status_color}]")
            console.print(f"  Size: {info['size']}")
            console.print(f"  Quantization: {info['quantization']}-bit")
            console.print(f"  Memory Required: {self.model_manager.model_config[model_type_name]['memory_required'] / (1024**3):.2f} GB")
            console.print("")
        
        # Calculate total memory usage
        total_memory_used = sum(
            self.model_manager.model_config[model_type_name]["memory_required"] 
            for model_type_name in model_info.keys()
        )
        console.print(f"[bold]Total Memory Usage:[/bold] {total_memory_used / (1024**3):.2f} GB")
    
    async def run_interactive_mode(self):
        """Run the application in interactive mode."""
        console.print("\n[bold cyan]CasaLingua Interactive Mode[/bold cyan]")
        console.print("Type 'help' for available commands, 'exit' to quit\n")
        
        while True:
            try:
                user_input = input("\nCasaLingua> ")
                
                if user_input.lower() in ["exit", "quit"]:
                    console.print("[yellow]Exiting CasaLingua...[/yellow]")
                    break
                
                elif user_input.lower() == "help":
                    console.print("\n[bold cyan]Available Commands:[/bold cyan]")
                    console.print("translate <text> to <language> - Translate text to the specified language")
                    console.print("query <text> - Ask a question to the AI assistant")
                    console.print("verify <statement> - Verify whether a statement is accurate")
                    console.print("info - Show model and hardware information")
                    console.print("help - Show this help message")
                    console.print("exit - Exit the program")
                
                elif user_input.lower().startswith("translate "):
                    parts = user_input[10:].split(" to ")
                    if len(parts) != 2:
                        console.print("[bold red]Invalid translation command. Format: translate [text] to [language][/bold red]")
                        continue
                    
                    text, target_lang = parts[0], parts[1]
                    await self.translate_text(text, target_lang)
                
                elif user_input.lower().startswith("query "):
                    query = user_input[6:]
                    await self.process_query(query)
                
                elif user_input.lower().startswith("verify "):
                    statement = user_input[7:]
                    await self.verify_fact(statement)
                
                elif user_input.lower() == "info":
                    await self.show_model_info()
                
                else:
                    console.print("[bold red]Unknown command. Type 'help' for available commands.[/bold red]")
            
            except KeyboardInterrupt:
                console.print("\n[yellow]Interrupted. Type 'exit' to quit.[/yellow]")
            
            except Exception as e:
                console.print(f"[bold red]Error: {str(e)}[/bold red]")
    
    async def shutdown(self):
        """Clean up resources and shut down."""
        logger.info("Shutting down CasaLingua...")
        
        if self.model_manager:
            logger.info("Unloading models...")
            await self.model_manager.unload_all_models()
        
        if self.metrics:
            logger.info("Saving metrics...")
            self.metrics.save_metrics()
        
        logger.info("Shutdown complete!")


async def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="CasaLingua - Multilingual AI Assistant")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--translate", "-t", help="Text to translate")
    parser.add_argument("--to", help="Target language for translation")
    parser.add_argument("--from", dest="from_lang", help="Source language for translation")
    parser.add_argument("--query", "-q", help="Query for the AI assistant")
    parser.add_argument("--verify", "-v", help="Statement to verify")
    parser.add_argument("--info", "-i", action="store_true", help="Show model information")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()
    
    # Enable debug mode if requested
    if args.debug:
        os.environ["DEBUG"] = "true"
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Print banner
    print_startup_banner(cli_mode=True)
    
    # Initialize the runner
    runner = CasaLinguaRunner(args.config)
    try:
        await runner.initialize()
        await runner.load_models()
        
        # Process commands
        if args.translate:
            target_lang = args.to or "es"  # Default to Spanish
            source_lang = args.from_lang
            await runner.translate_text(args.translate, target_lang, source_lang)
        
        elif args.query:
            await runner.process_query(args.query)
        
        elif args.verify:
            await runner.verify_fact(args.verify)
        
        elif args.info:
            await runner.show_model_info()
        
        elif args.interactive or not any([args.translate, args.query, args.verify, args.info]):
            await runner.run_interactive_mode()
        
        # Clean up
        await runner.shutdown()
        
    except KeyboardInterrupt:
        logger.info("Operation interrupted")
        await runner.shutdown()
    
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        return 1
    
    return 0

if __name__ == "__main__":
    loop = get_event_loop()
    exit_code = loop.run_until_complete(main())
    sys.exit(exit_code)