#!/usr/bin/env python3
"""
Simplified test script for MBART translation
"""

import os
import sys
import json
import logging
from typing import Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("simple_mbart_test")

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the necessary modules
try:
    from app.services.models.wrapper import TranslationModelWrapper, ModelInput, ModelType
    from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    logger.error("Make sure you're running this script from the project root directory")
    sys.exit(1)

def main():
    logger.info("Starting simplified MBART testing")
    
    # Test text in various languages
    test_text = "The quick brown fox jumps over the lazy dog."
    src_lang = "en"
    tgt_lang = "es"
    
    try:
        # Load MBART model
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
        
        # Test with MBART
        try:
            # Create ModelInput
            mbart_input = ModelInput(
                text=test_text,
                source_language=src_lang,
                target_language=tgt_lang,
                parameters={
                    "mbart_source_lang": mbart_wrapper._get_mbart_language_code(src_lang),
                    "mbart_target_lang": mbart_wrapper._get_mbart_language_code(tgt_lang)
                }
            )
            
            # Process with MBART
            mbart_result = mbart_wrapper.process(mbart_input)
            logger.info(f"MBART result: {mbart_result.result}")
            
            # Store results
            result = {
                "source_lang": src_lang,
                "target_lang": tgt_lang,
                "text": test_text,
                "mbart_translation": mbart_result.result
            }
            
            # Save results to file
            with open("simple_mbart_test_results.json", "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
                
            logger.info(f"Test results saved to simple_mbart_test_results.json")
            
        except Exception as e:
            logger.error(f"Error testing {src_lang} to {tgt_lang}: {e}")
            return 1
            
    except Exception as e:
        logger.error(f"Failed during model testing: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)