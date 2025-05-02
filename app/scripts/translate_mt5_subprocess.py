#!/usr/bin/env python

import argparse
import json
import torch
from app.services.models.loader import ModelRegistry
from app.core.pipeline.tokenizer import TokenizerPipeline
from transformers import AutoModelForSeq2SeqLM

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True, help="Text to translate")
    parser.add_argument("--source", default="en", help="Source language code")
    parser.add_argument("--target", default="es", help="Target language code")
    args = parser.parse_args()

    # Load model and tokenizer from registry
    registry = ModelRegistry()
    model_name, tokenizer_name = registry.get_model_and_tokenizer("translation")
    tokenizer = TokenizerPipeline(model_name=tokenizer_name, task_type="translation")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    inputs = tokenizer.prepare_translation_inputs(args.text, args.source, args.target)
    output_ids = model.generate(**inputs, max_new_tokens=128)
    translation = tokenizer.detokenize(output_ids[0])

    result = {
        "source_text": args.text,
        "translated_text": translation,
        "source_language": args.source,
        "target_language": args.target,
        "confidence": 1.0,
        "model_id": model_name
    }
    print(json.dumps(result, ensure_ascii=False))

if __name__ == "__main__":
    main()