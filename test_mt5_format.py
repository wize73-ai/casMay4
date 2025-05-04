#!/usr/bin/env python3
"""
Test script to specifically verify the MT5 prompt format fix.
This simplified test doesn't require loading the full model.
"""

import sys
from pathlib import Path

# Add the app directory to the Python path
sys.path.append(str(Path(__file__).parent))

from app.services.models.wrapper import ModelInput, TranslationModelWrapper
from app.ui.console import Console

# Mock MT5 model for testing
class MockMT5Model:
    """Mock MT5 model for testing the prompt format"""
    class Config:
        model_type = "mt5"
    
    def __init__(self):
        self.config = self.Config()
    
    def __str__(self):
        return "MT5ForConditionalGeneration"

    async def generate(self, **kwargs):
        # In a real model, this would use the inputs to generate translations
        # Here we just return the prompt format for inspection
        prompts = kwargs.get('texts', [])
        return prompts

# Setup console
console = Console()

def run_test():
    """Test the MT5 prompt format"""
    console.print("[bold blue]Testing MT5 Model Prompt Format Fix[/bold blue]")
    
    # Create mock MT5 model
    model = MockMT5Model()
    
    # Create wrapper
    wrapper = TranslationModelWrapper(model)
    
    # Create test input
    input_data = ModelInput(
        text="Hello, how are you?",
        source_language="en",
        target_language="es"
    )
    
    # Preprocess input
    preprocessed = wrapper._preprocess(input_data)
    
    # Test detection
    console.print(f"[bold cyan]Is MT5 model:[/bold cyan] {preprocessed.get('is_mt5', False)}")
    
    # Check prompt format
    console.print("[bold cyan]Checking prompt format:[/bold cyan]")
    if 'texts' in preprocessed['inputs']:
        prompts = preprocessed['inputs']['texts']
        for prompt in prompts:
            console.print(f"[green]Prompt: {prompt}[/green]")
            
            # Verify format is correct
            expected_format = f"translate {input_data.source_language} to {input_data.target_language}: {input_data.text}"
            if prompt == expected_format:
                console.print("[bold green]✓ Format is correct![/bold green]")
            else:
                console.print(f"[bold red]× Format is incorrect! Expected: {expected_format}[/bold red]")
    else:
        console.print("[yellow]No prompts found in input![/yellow]")
    
    console.print("\n[bold blue]Test Complete[/bold blue]")

if __name__ == "__main__":
    run_test()