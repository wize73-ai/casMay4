#!/usr/bin/env python3
"""
Simple test script for MT5 translation
"""

import os
import sys
import torch
import logging
from transformers import MT5ForConditionalGeneration, MT5TokenizerFast

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("mt5_test")

# Test function
def translate_with_mt5(text, source_lang, target_lang):
    """Translate text using MT5 model"""
    
    # Load model and tokenizer
    logger.info("Loading MT5 model and tokenizer...")
    model = MT5ForConditionalGeneration.from_pretrained("google/mt5-base")
    tokenizer = MT5TokenizerFast.from_pretrained("google/mt5-base")
    
    # Create translation prompt
    prompt = f"translate {source_lang} to {target_lang}: {text}"
    logger.info(f"MT5 prompt: {prompt}")
    
    # Encode text
    logger.info(f"Translating from {source_lang} to {target_lang}...")
    encoded = tokenizer(prompt, return_tensors="pt")
    
    # Generate translation
    with torch.no_grad():
        generated_tokens = model.generate(
            **encoded, 
            max_length=128,
            num_beams=4,
            length_penalty=0.6
        )
    
    # Decode translation
    translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    
    # Clean up the output if needed (MT5 sometimes has formatting issues)
    import re
    
    # Remove any extra_id tokens
    translation = re.sub(r'<extra_id_\d+>', '', translation)
    
    # Clean up any remaining prompt text
    prefixes = [
        f"translate {source_lang} to {target_lang}:", 
        "translate to:", 
        f"translate {source_lang} to {target_lang}",
        f"translate {source_lang} to:",
        "translate from",
        "translation:",
        f"{source_lang} to {target_lang}:",
        f"{source_lang} {target_lang}"
    ]
    
    for prefix in prefixes:
        if translation.startswith(prefix):
            translation = translation[len(prefix):].strip()
        translation = translation.replace(prefix, " ")
    
    return translation.strip()

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
        
        translation = translate_with_mt5(
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