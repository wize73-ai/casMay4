#!/usr/bin/env python3
"""
Simple test script for MBART translation
"""

import os
import sys
import torch
import logging
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("mbart_test")

# Test function
def translate_with_mbart(text, source_lang, target_lang):
    """Translate text using MBART model"""
    
    # MBART language code mapping
    mbart_lang_map = {
        "en": "en_XX", "es": "es_XX", "fr": "fr_XX", "de": "de_DE", "it": "it_IT", 
        "pt": "pt_XX", "nl": "nl_XX", "pl": "pl_PL", "ru": "ru_RU", "zh": "zh_CN", 
        "ja": "ja_XX", "ko": "ko_KR", "ar": "ar_AR", "hi": "hi_IN", "tr": "tr_TR",
        "vi": "vi_VN", "th": "th_TH", "id": "id_ID", "tl": "fil_PH", "ro": "ro_RO"
    }
    
    # Convert language codes to MBART format
    mbart_source_lang = mbart_lang_map.get(source_lang, f"{source_lang}_XX")
    mbart_target_lang = mbart_lang_map.get(target_lang, f"{target_lang}_XX")
    
    # Load model and tokenizer
    logger.info("Loading MBART model and tokenizer...")
    model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
    tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
    
    # Set source language
    tokenizer.src_lang = mbart_source_lang
    
    # Encode text
    logger.info(f"Translating from {source_lang} ({mbart_source_lang}) to {target_lang} ({mbart_target_lang})...")
    encoded = tokenizer(text, return_tensors="pt")
    
    # Generate translation
    with torch.no_grad():
        generated_tokens = model.generate(
            **encoded, 
            forced_bos_token_id=tokenizer.lang_code_to_id[mbart_target_lang]
        )
    
    # Decode translation
    translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    return translation

def main():
    # Test data
    test_cases = [
        {"text": "The quick brown fox jumps over the lazy dog.", "source": "en", "target": "es"},
        {"text": "El zorro marrón rápido salta sobre el perro perezoso.", "source": "es", "target": "en"},
        {"text": "Machine learning is transforming how we approach translation problems.", "source": "en", "target": "fr"}
    ]
    
    # Run tests
    results = []
    for i, test in enumerate(test_cases):
        logger.info(f"Test case {i+1}: {test['source']} to {test['target']}")
        logger.info(f"Source text: {test['text']}")
        
        translation = translate_with_mbart(
            test['text'], 
            test['source'], 
            test['target']
        )
        
        logger.info(f"Translation: {translation}")
        results.append({
            "source_text": test['text'],
            "source_lang": test['source'],
            "target_lang": test['target'],
            "translation": translation
        })
    
    # Print results summary
    print("\nTest Results Summary:")
    print("=====================")
    for i, result in enumerate(results):
        print(f"Test {i+1} ({result['source_lang']} → {result['target_lang']}):")
        print(f"  Source: {result['source_text']}")
        print(f"  Result: {result['translation']}\n")
    
    return 0

if __name__ == "__main__":
    main()