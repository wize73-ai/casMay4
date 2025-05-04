#!/usr/bin/env python3
"""
Test script for translation model in CasaLingua

This script tests the translation model by directly accessing
the model manager and attempting to load and run the translation model.
"""

import asyncio
import sys
import time
import os
from pathlib import Path

# Add the root directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import necessary components
from app.utils.config import load_config
from app.services.models.loader import ModelLoader, load_registry_config
from app.services.hardware.detector import HardwareDetector
from app.services.models.manager import EnhancedModelManager

async def test_translation():
    """Test translation model loading and execution"""
    print("=== CasaLingua Translation Model Test ===")
    
    # Load configuration
    config = load_config()
    print(f"Environment: {config.get('environment', 'development')}")
    
    # Detect hardware
    print("\nDetecting hardware...")
    hardware_detector = HardwareDetector(config)
    hardware_info = hardware_detector.detect_all()
    
    print(f"Hardware detected: Device = {hardware_info.get('processor_type', 'unknown')}")
    print(f"GPU available: {hardware_info.get('has_gpu', False)}")
    if hardware_info.get('has_gpu', False):
        print(f"GPU: {hardware_info.get('gpu_name', 'unknown')}")
    
    # Load model registry
    print("\nLoading model registry...")
    registry_config = load_registry_config(config)
    
    # Add debugging output for registry
    print("Registry models:")
    for model_type, model_config in registry_config.items():
        print(f"  - {model_type}: {model_config.get('model_name', 'unknown')}")
    
    # Create model loader and manager
    print("\nInitializing model loader...")
    model_loader = ModelLoader(config)
    model_loader.model_config = registry_config
    
    print("Initializing model manager...")
    model_manager = EnhancedModelManager(model_loader, hardware_info, config)
    
    # Try loading language detection model first (usually succeeds)
    try:
        print("\nLoading language detection model...")
        start_time = time.time()
        lang_model = await model_manager.load_model("language_detection")
        load_time = time.time() - start_time
        print(f"Language detection model loaded in {load_time:.2f}s")
        
        # Test language detection
        print("\nTesting language detection...")
        input_data = {
            "text": "Hello, this is a test.",
            "parameters": {"detailed": True}
        }
        
        result = await model_manager.run_model(
            "language_detection",
            "process",
            input_data
        )
        
        print(f"Language detection result: {result}")
    except Exception as e:
        print(f"Error loading language detection model: {e}")
    
    # Now try loading the translation model
    try:
        print("\nLoading translation model...")
        start_time = time.time()
        translation_model = await model_manager.load_model("translation")
        load_time = time.time() - start_time
        print(f"Translation model loaded in {load_time:.2f}s")
        
        # Print model info
        model_info = model_manager.get_model_info()
        if "translation" in model_info:
            print(f"Translation model info: {model_info['translation']}")
        
        # Test translation
        print("\nTesting translation...")
        input_data = {
            "text": "Hello, how are you?",
            "source_language": "en",
            "target_language": "es",
            "parameters": {}
        }
        
        result = await model_manager.run_model(
            "translation",
            "process",
            input_data
        )
        
        print(f"Translation result: {result}")
    except Exception as e:
        print(f"Error with translation model: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    asyncio.run(test_translation())