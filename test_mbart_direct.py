#!/usr/bin/env python3
"""
Direct test script for MBART translation without server

This script directly tests the translation models without needing
a server to be running, allowing for isolated testing of the MBART
fallback mechanism.
"""

import os
import sys
import json
import asyncio
import logging
from typing import Dict, Any, Optional, List

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("mbart_direct_test")

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

async def main():
    logger.info("Starting direct MBART and MT5 testing")
    
    # Test text in various languages
    test_texts = {
        "en": "The quick brown fox jumps over the lazy dog.",
        "es": "El zorro marrón rápido salta sobre el perro perezoso.",
        "fr": "Le rapide renard brun saute par-dessus le chien paresseux.",
        "de": "Der schnelle braune Fuchs springt über den faulen Hund."
    }
    
    # Language pairs to test
    language_pairs = [
        ("en", "es"),
        ("es", "en"),
        ("en", "fr"),
        ("fr", "en")
    ]
    
    logger.info("Loading models for testing...")
    
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
        
        # Try to load MT5 model (this might fail if model not available)
        logger.info("Loading MT5 model and tokenizer")
        try:
            mt5_model = MT5ForConditionalGeneration.from_pretrained("google/mt5-base")
            mt5_tokenizer = MT5TokenizerFast.from_pretrained("google/mt5-base")
            
            # Create MT5 wrapper
            mt5_wrapper = TranslationModelWrapper(
                model=mt5_model,
                tokenizer=mt5_tokenizer,
                config={"task": ModelType.TRANSLATION}
            )
            logger.info("MT5 model loaded successfully")
            has_mt5 = True
        except Exception as e:
            logger.warning(f"Failed to load MT5 model: {e}")
            logger.warning("Continuing with MBART testing only")
            has_mt5 = False
        
        # Test language pairs
        results = []
        for src_lang, tgt_lang in language_pairs:
            if src_lang not in test_texts:
                logger.warning(f"No test text available for {src_lang}")
                continue
                
            text = test_texts[src_lang]
            logger.info(f"Testing {src_lang} to {tgt_lang}: '{text}'")
            
            # Test with MBART
            try:
                # Create ModelInput
                mbart_input = ModelInput(
                    text=text,
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
                    "text": text,
                    "mbart_translation": mbart_result.result
                }
                
                # Test with MT5 if available
                if has_mt5:
                    try:
                        # Create ModelInput for MT5
                        mt5_input = ModelInput(
                            text=text,
                            source_language=src_lang,
                            target_language=tgt_lang
                        )
                        
                        # Process with MT5
                        mt5_result = mt5_wrapper.process(mt5_input)
                        logger.info(f"MT5 result: {mt5_result.result}")
                        
                        # Add to results
                        result["mt5_translation"] = mt5_result.result
                        
                        # Test fallback mechanism
                        logger.info("Testing fallback mechanism")
                        # Monkey-patch MT5 to produce poor quality
                        original_result = mt5_result.result
                        mt5_result.result = f"<extra_id_0> {src_lang} {tgt_lang} {text}"
                        
                        # Use the fallback method
                        fallback_result = mbart_wrapper._get_mbart_fallback_translation(
                            text, src_lang, tgt_lang
                        )
                        
                        # Add to results
                        result["mt5_bad_output"] = mt5_result.result
                        result["fallback_result"] = fallback_result
                        logger.info(f"Fallback result: {fallback_result}")
                        
                        # Restore original result
                        mt5_result.result = original_result
                    except Exception as e:
                        logger.error(f"Error testing MT5: {e}")
                        result["mt5_error"] = str(e)
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error testing {src_lang} to {tgt_lang}: {e}")
                results.append({
                    "source_lang": src_lang,
                    "target_lang": tgt_lang,
                    "text": text,
                    "error": str(e)
                })
        
        # Save results to file
        with open("mbart_test_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Test results saved to mbart_test_results.json")
        
    except Exception as e:
        logger.error(f"Failed during model testing: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)