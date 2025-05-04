#!/usr/bin/env python3
"""
Test script for MBART as primary translation model
"""

import os
import sys
import json
import asyncio
import logging
from typing import Dict, Any, List, Optional

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("mbart_primary_test")

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import necessary components
try:
    from app.core.pipeline.processor import UnifiedProcessor
    from app.services.models.manager import EnhancedModelManager
    from app.api.schemas.translation import TranslationRequest
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    logger.error("Make sure you're running this script from the project root directory")
    sys.exit(1)

async def main():
    logger.info("Testing MBART as primary translation model")
    
    # Create minimal configuration for testing
    config = {
        "environment": "development",
        "cache_enabled": True,
        "cache_ttl": 3600,
        "device": "cpu",
        "log_level": "INFO"
    }
    
    # Create dummy hardware info for the model manager
    hardware_info = {
        "has_gpu": False,
        "gpu_memory": None,
        "gpu_name": None,
        "processor_type": "other",
        "system_name": "test",
        "total_memory": 8 * 1024 * 1024 * 1024,  # 8GB
        "available_memory": 4 * 1024 * 1024 * 1024  # 4GB
    }
    
    # Create model manager
    logger.info("Initializing EnhancedModelManager")
    model_manager = EnhancedModelManager(config, hardware_info)
    
    # Create dummy metrics collector
    class DummyMetrics:
        def __init__(self):
            pass
            
        async def record_event(self, *args, **kwargs):
            pass
            
        async def record_request(self, *args, **kwargs):
            pass
            
    metrics = DummyMetrics()
    
    # Create processor
    logger.info("Initializing UnifiedProcessor")
    processor = UnifiedProcessor(model_manager, config, metrics)
    
    # Initialize processor
    logger.info("Initializing processor pipeline")
    await processor.initialize()
    
    # Test text and language pairs
    test_cases = [
        {
            "text": "The quick brown fox jumps over the lazy dog.",
            "source": "en", 
            "target": "es",
            "description": "English to Spanish"
        },
        {
            "text": "El zorro marrón rápido salta sobre el perro perezoso.",
            "source": "es", 
            "target": "en",
            "description": "Spanish to English"
        },
        {
            "text": "Machine learning is transforming how we approach translation problems.",
            "source": "en", 
            "target": "fr",
            "description": "English to French"
        }
    ]
    
    # Run translation tests
    results = []
    for i, test_case in enumerate(test_cases):
        logger.info(f"Test case {i+1}: {test_case['description']}")
        logger.info(f"Translating: {test_case['text']}")
        
        result = await processor.process_translation(
            text=test_case['text'],
            source_language=test_case['source'],
            target_language=test_case['target'],
            request_id=f"test_{i}",
            use_mbart=True  # Explicitly use MBART
        )
        
        # Log and store result
        logger.info(f"Translation result: {result['translated_text']}")
        
        test_result = {
            "test_case": test_case,
            "result": result
        }
        results.append(test_result)
    
    # Try with different models to test model selection
    logger.info("Testing with explicit model selection")
    
    # First with MT5 model (which should still trigger MBART fallback if poor quality)
    mt5_result = await processor.process_translation(
        text="The weather is nice today.",
        source_language="en",
        target_language="es",
        model_id="mt5_translation",  # Explicitly request MT5
        request_id="test_mt5"
    )
    
    logger.info(f"MT5 translation result: {mt5_result['translated_text']}")
    logger.info(f"Used fallback: {mt5_result.get('used_fallback', False)}")
    logger.info(f"Primary model: {mt5_result.get('model_used', 'unknown')}")
    
    results.append({
        "test_case": {
            "text": "The weather is nice today.",
            "source": "en",
            "target": "es",
            "description": "With MT5 model"
        },
        "result": mt5_result
    })
    
    # Now with MBART model explicitly
    mbart_result = await processor.process_translation(
        text="The weather is nice today.",
        source_language="en",
        target_language="es",
        model_id="mbart_translation",  # Explicitly request MBART
        request_id="test_mbart"
    )
    
    logger.info(f"MBART translation result: {mbart_result['translated_text']}")
    logger.info(f"Primary model: {mbart_result.get('model_used', 'unknown')}")
    
    results.append({
        "test_case": {
            "text": "The weather is nice today.",
            "source": "en",
            "target": "es",
            "description": "With MBART model"
        },
        "result": mbart_result
    })
    
    # Save test results
    with open("mbart_primary_test_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info("Test results saved to mbart_primary_test_results.json")
    
    # Cleanup
    await processor.cleanup()
    
    return 0

if __name__ == "__main__":
    asyncio.run(main())