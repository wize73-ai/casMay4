#!/usr/bin/env python3
"""
Test script for MT5 to MBART fallback mechanism
"""

import os
import sys
import json
import logging
from typing import Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("mt5_fallback_test")

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the necessary modules
try:
    from app.services.models.wrapper import TranslationModelWrapper, ModelInput, ModelType
    from transformers import MBartForConditionalGeneration, MBart50TokenizerFast, MT5ForConditionalGeneration, MT5TokenizerFast
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    logger.error("Make sure you're running this script from the project root directory")
    sys.exit(1)

def main():
    logger.info("Starting MT5 to MBART fallback testing")
    
    # Test text in various languages
    test_text = "The quick brown fox jumps over the lazy dog."
    src_lang = "en"
    tgt_lang = "es"
    
    try:
        # First load MT5 model
        logger.info("Loading MT5 model and tokenizer")
        mt5_model = MT5ForConditionalGeneration.from_pretrained("google/mt5-base")
        mt5_tokenizer = MT5TokenizerFast.from_pretrained("google/mt5-base")
        
        # Create MT5 wrapper
        mt5_wrapper = TranslationModelWrapper(
            model=mt5_model,
            tokenizer=mt5_tokenizer,
            config={"task": ModelType.TRANSLATION}
        )
        logger.info("MT5 model loaded successfully")
        
        # Now load MBART model for fallback
        logger.info("Loading MBART model and tokenizer")
        mbart_model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
        mbart_tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
        
        # Create MBART wrapper
        mbart_wrapper = TranslationModelWrapper(
            model=mbart_model,
            tokenizer=mbart_tokenizer,
            config={"task": ModelType.TRANSLATION}
        )
        logger.info("MBART model loaded successfully")
        
        # First test normal MT5 translation
        logger.info("Testing normal MT5 translation")
        mt5_input = ModelInput(
            text=test_text,
            source_language=src_lang,
            target_language=tgt_lang
        )
        
        mt5_result = mt5_wrapper.process(mt5_input)
        logger.info(f"MT5 result: {mt5_result.result}")
        
        # Now test fallback mechanism 
        logger.info("Testing fallback mechanism with deliberately poor MT5 output")
        
        # Create a corrupted MT5 output to trigger fallback
        # Replace the _postprocess method temporarily to simulate poor quality
        orig_process = mt5_wrapper.process
        
        def mock_process(input_data):
            result = orig_process(input_data)
            # Corrupt the result to trigger fallback
            result.result = "<extra_id_0> This is a poor quality translation"
            return result
        
        # Apply the mock
        mt5_wrapper.process = mock_process
        
        # Add the fallback capability to MT5 wrapper
        mt5_wrapper._get_mbart_fallback_translation = mbart_wrapper._get_mbart_fallback_translation
        
        # Test with artificially corrupted output
        mt5_poor_result = mt5_wrapper.process(mt5_input)
        
        # Try to use the fallback directly
        logger.info("Attempting direct fallback translation")
        fallback_result = mt5_wrapper._get_mbart_fallback_translation(
            test_text, src_lang, tgt_lang
        )
        
        logger.info(f"Direct fallback result: {fallback_result}")
        
        # Store results
        result = {
            "source_lang": src_lang,
            "target_lang": tgt_lang,
            "text": test_text,
            "mt5_translation": mt5_result.result,
            "mt5_corrupted": mt5_poor_result.result,
            "fallback_result": fallback_result
        }
        
        # Save results to file
        with open("mt5_fallback_test_results.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Test results saved to mt5_fallback_test_results.json")
        
    except Exception as e:
        logger.error(f"Error during MT5 fallback testing: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)