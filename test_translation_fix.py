#!/usr/bin/env python3
"""
Test script to verify translation model functionality after fixes.
Tests specifically for proper handling of MT5 models.
"""

import asyncio
import logging
import sys
import os
from pathlib import Path

# Add the app directory to the Python path
sys.path.append(str(Path(__file__).parent))

from app.services.models.manager import EnhancedModelManager
from app.services.models.loader import ModelLoader
from app.services.models.wrapper import TranslationModelWrapper
from app.utils.config import load_config
from app.ui.console import Console

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("translation_test")
console = Console()

async def test_translation():
    """Test the translation model with various inputs and language pairs."""
    console.print("[bold blue]Testing Translation Model After Fixes[/bold blue]")
    
    # Load configuration
    config = load_config()
    
    # First create the hardware detector
    from app.services.hardware.detector import HardwareDetector
    hardware_detector = HardwareDetector(config)
    hardware_info = hardware_detector.detect_all()
    
    # Create model loader
    model_loader = ModelLoader(config)
    
    # Create enhanced model manager
    model_manager = EnhancedModelManager(model_loader, hardware_info, config)
    
    # Create a model directly for testing
    # Let's manually create a translation model wrapper and test the fix
    console.print("[bold cyan]Loading a translation model directly for testing...[/bold cyan]")
    
    # Test texts
    test_cases = [
        {"text": "Hello, how are you?", "source": "en", "target": "es"},
        {"text": "This is a test of the translation system.", "source": "en", "target": "fr"},
        {"text": "I hope this works correctly now.", "source": "en", "target": "de"},
    ]
    
    try:
        # First check if we have any models loaded
        model_info = model_loader.get_model_info()
        console.print(f"[bold cyan]Available Models:[/bold cyan] {list(model_info.keys())}")
        
        # Try to load a translation model directly
        translation_models = [k for k in model_info.keys() if 'translation' in k]
        
        if not translation_models:
            console.print("[bold yellow]No translation models registered. Attempting to load one...[/bold yellow]")
            # Try to load a translation model
            model_data = await model_manager.load_model("translation")
            translation_models = ["translation"]
        else:
            console.print(f"[bold green]Found translation models:[/bold green] {translation_models}")
            # Load the first translation model
            model_name = translation_models[0]
            model_data = await model_manager.load_model(model_name)
        
        # If we got here, we have a model
        if model_data and model_data.get("model"):
            model = model_data["model"]
            model_name = model_data.get("type", "translation")
            
            console.print(f"\n[bold magenta]Testing model:[/bold magenta] {model_name}")
            console.print(f"Model type: {model.__class__.__name__}")
            
            # Check if it's an MT5 model by name
            is_mt5 = "mt5" in str(model).lower()
            console.print(f"Is MT5 model: {is_mt5}")
            
            # Create wrapper
            wrapper = TranslationModelWrapper(model, config.get("translation", {}))
            
            for test_case in test_cases:
                text = test_case["text"]
                source_lang = test_case["source"]
                target_lang = test_case["target"]
                
                console.print(f"\n[bold]Translating:[/bold] '{text}'")
                console.print(f"From {source_lang} to {target_lang}")
                
                # Test direct translation call
                try:
                    # Create ModelInput object for the wrapper
                    from app.services.models.wrapper import ModelInput
                    input_data = ModelInput(
                        text=text,
                        source_language=source_lang,
                        target_language=target_lang
                    )
                    
                    # Use the process method instead
                    output = wrapper.process(input_data)
                    result = output.result
                    
                    if isinstance(result, list) and result:
                        result = result[0]
                        
                    console.print(f"[bold green]Result:[/bold green] '{result}'")
                    
                    # Check if result contains unwanted tokens
                    if "<extra_id_0>" in str(result):
                        console.print("[bold red]WARNING: Result contains <extra_id_0> token![/bold red]")
                    
                except Exception as e:
                    console.print(f"[bold red]Error during translation: {e}[/bold red]")
                    import traceback
                    console.print(traceback.format_exc())
        else:
            console.print("[bold red]Could not load any translation model.[/bold red]")
            
    except Exception as e:
        console.print(f"[bold red]Error during test: {e}[/bold red]")
        import traceback
        console.print(traceback.format_exc())

if __name__ == "__main__":
    asyncio.run(test_translation())