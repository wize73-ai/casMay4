import os
import json
import logging
from pathlib import Path

# ------------------------------------------------------------------------
# Model Registry Fix Script
# This script ensures that the model registry JSON file contains
# all necessary model and task alias entries for the Casalingua pipeline.
# It is intended to be run manually or as part of a setup process.
# ------------------------------------------------------------------------

# Ladder Logic Flow:
# ┌─────────────────────────────────────────────┐
# │ 1. Load or create model registry JSON file   │
# ├─────────────────────────────────────────────┤
# │ 2. Define default model entries              │
# ├─────────────────────────────────────────────┤
# │ 3. Add models to registry                    │
# ├─────────────────────────────────────────────┤
# │ 4. Add task aliases to registry              │
# ├─────────────────────────────────────────────┤
# │ 5. Save updated registry to disk             │
# └─────────────────────────────────────────────┘

# Create a logger for debugging
logger = logging.getLogger("model_registry_fix")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

def fix_registry():
    """Force-add real models to the registry file."""
    # Path to registry file
    models_dir = Path("models")
    registry_file = models_dir / "registry.json"
    
    # Create models directory if it doesn't exist
    os.makedirs(models_dir, exist_ok=True)
    
    # Default model configurations to add
    default_models = {
        "facebook/nllb-200-distilled-600M": {
            "model_name": "facebook/nllb-200-distilled-600M",
            "tasks": ["translation"],
            "languages": ["en", "es", "fr", "de", "it", "zh", "ja", "ar", "ru", "pt"],
            "loader_module": "app.services.models.loaders.translation_loader",
            "loader_class": "TranslationModelLoader",
            "model_type": "seq2seq",
            "precision": "fp16",
            "size_gb": 2.0,
            "language_codes": {
                "en": "eng_Latn",
                "fr": "fra_Latn",
                "es": "spa_Latn",
                "de": "deu_Latn",
                "it": "ita_Latn",
                "zh": "zho_Hans",
                "ja": "jpn_Jpan",
                "ru": "rus_Cyrl",
                "ar": "ara_Arab",
                "pt": "por_Latn"
            }
        },
        "google/mt5-small": {
            "model_name": "google/mt5-small",
            "tasks": ["multipurpose", "summarization"],
            "languages": ["en", "es", "fr", "de", "it", "zh", "ja", "ar", "ru", "pt"],
            "loader_module": "app.services.models.loaders.huggingface_loader",
            "loader_class": "HuggingFaceLoader",
            "model_type": "seq2seq",
            "precision": "fp16",
            "size_gb": 1.2
        },
        "microsoft/deberta-v3-xsmall": {
            "model_name": "microsoft/deberta-v3-xsmall",
            "tasks": ["verification", "classification"],
            "languages": ["en"],
            "loader_module": "app.services.models.loaders.verification_loader",
            "loader_class": "VerificationModelLoader",
            "model_type": "classifier",
            "precision": "fp16",
            "size_gb": 0.5,
            "labels": ["false", "true"]
        },
        "language_detection": {
            "model_name": "fasttext/lid.176.bin",
            "tasks": ["language_detection"],
            "languages": ["all"],
            "loader_module": "app.services.models.loaders.language_detection_loader",
            "loader_class": "LanguageDetectionLoader",
            "model_type": "language_detection",
            "precision": "fp32",
            "size_gb": 0.5
        }
    }
    
    # Create a registry mapping task name directly to model
    task_mapping = {
        "translation": "facebook/nllb-200-distilled-600M",
        "multipurpose": "google/mt5-small",
        "verification": "microsoft/deberta-v3-xsmall",
        "language_detection": "language_detection"
    }
    
    # Load existing registry if available
    registry = {}
    if registry_file.exists():
        try:
            with open(registry_file, "r", encoding="utf-8") as f:
                registry = json.load(f)
            logger.info(f"Loaded existing registry with {len(registry)} models")
        except Exception as e:
            logger.error(f"Error loading registry: {str(e)}")
    
    # Add all default models to registry
    for model_id, model_info in default_models.items():
        registry[model_id] = model_info
        logger.info(f"Added/updated model: {model_id}")
    
    # Add direct task mappings as explicit entries
    for task, model_id in task_mapping.items():
        if model_id in default_models:
            # Create a special task-based entry that links to the model
            registry[task] = model_id
            logger.info(f"Added direct mapping: {task} -> {model_id}")
    
    # Save the updated registry
    try:
        with open(registry_file, "w", encoding="utf-8") as f:
            json.dump(registry, f, indent=2)
        logger.info(f"Saved registry with {len(registry)} entries to {registry_file}")
    except Exception as e:
        logger.error(f"Error saving registry: {str(e)}")

if __name__ == "__main__":
    fix_registry()
    print("✅ Model registry update complete. Ready for pipeline usage.")
    