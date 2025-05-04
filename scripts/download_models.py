"""
CasaLingua Model Downloader

This script downloads and sets up language models required for the
CasaLingua language processing platform. It handles model downloading,
verification, and registration in the model registry.

Author: Exygy Development Team
Version: 1.0.0
License: MIT
"""

import os
import sys
import json
import argparse
import logging
import asyncio
import hashlib
from typing import Dict, List, Any, Optional, Set
from pathlib import Path
from datetime import datetime
import shutil

import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM
from tqdm import tqdm

# Colorama for colored terminal output
from colorama import init as colorama_init, Fore, Style

# Add project root to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.utils.config import load_config
from app.utils.logging import configure_logging, get_logger

# Configure logging
logger = get_logger("tools.download_models")

# Model definitions with metadata
DEFAULT_MODELS = {
    "embedding_model": {
        "name": "Multilingual Sentence Embeddings",
        "model_name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "model_type": "encoder",
        "tokenizer_name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "size_gb": 0.12,
        "languages": ["en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko", "ar", "hi", "vi", "tl", "th"],
        "tasks": ["embedding"],
        "description": "Multilingual sentence embeddings for semantic similarity and search",
        "requires_gpu": False,
        "memory_required": 1.0,
        "gpu_memory_required": 0.5,
        "model_format": "transformers"
    },
    "translation_model": {
        "name": "Multilingual Translation",
        "model_name": "facebook/mbart-large-50-many-to-many-mmt",
        "model_type": "seq2seq",
        "tokenizer_name": "facebook/mbart-large-50-many-to-many-mmt",
        "size_gb": 2.3,
        "languages": ["en", "es", "fr", "de", "ru", "zh", "ja", "pt", "it", "ar", "hi", "vi"],
        "tasks": ["translation"],
        "description": "Multilingual translation model (MBART-50) for 50 languages",
        "requires_gpu": False,
        "memory_required": 4.0,
        "gpu_memory_required": 3.0,
        "model_format": "transformers"
    },
    "translation_small": {
        "name": "Multilingual Translation (Small)",
        "model_name": "facebook/mbart-large-50-one-to-many-mmt",
        "model_type": "seq2seq",
        "tokenizer_name": "facebook/mbart-large-50-one-to-many-mmt",
        "size_gb": 1.2,
        "languages": ["en", "es", "fr", "de", "ru", "zh", "ja", "pt", "it", "ar", "hi", "vi"],
        "tasks": ["translation"],
        "description": "Multilingual translation model (MBART-50) optimized for translating from English",
        "requires_gpu": False,
        "memory_required": 2.0,
        "gpu_memory_required": 1.5,
        "model_format": "transformers"
    },
    "language_detection": {
        "name": "Language Detection",
        "model_name": "papluca/xlm-roberta-base-language-detection",
        "model_type": "classifier",
        "tokenizer_name": "papluca/xlm-roberta-base-language-detection",
        "size_gb": 0.48,
        "languages": ["en", "es", "fr", "de", "it", "pt", "nl", "pl", "ru", "ja", "zh", "ar", "bg", "cs", "da", 
                     "el", "et", "fi", "hr", "hu", "id", "ko", "lt", "lv", "no", "ro", "sk", "sl", "sv", "tr", "uk"],
        "tasks": ["language_detection"],
        "description": "Language detection model supporting 30+ languages",
        "requires_gpu": False,
        "memory_required": 1.0,
        "gpu_memory_required": 0.75,
        "model_format": "transformers"
    }
}

# Optional advanced models
ADVANCED_MODELS = {
    "translation_en_es_large": {
        "name": "English-Spanish Large Translation",
        "model_name": "facebook/mbart-large-50-one-to-many-mmt",
        "model_type": "seq2seq",
        "tokenizer_name": "facebook/mbart-large-50-one-to-many-mmt",
        "size_gb": 2.3,
        "languages": ["en", "es"],
        "tasks": ["translation"],
        "description": "Large English to Spanish machine translation model with higher quality",
        "requires_gpu": True,
        "memory_required": 4.0,
        "gpu_memory_required": 3.0,
        "model_format": "transformers"
    },
    "text_analysis": {
        "name": "Multilingual Text Analysis",
        "model_name": "xlm-roberta-large",
        "model_type": "encoder",
        "tokenizer_name": "xlm-roberta-large",
        "size_gb": 1.6,
        "languages": ["en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ar"],
        "tasks": ["sentiment", "classification"],
        "description": "Multilingual text analysis model for sentiment and classification",
        "requires_gpu": True,
        "memory_required": 3.0,
        "gpu_memory_required": 2.0,
        "model_format": "transformers"
    }
}

def calculate_file_hash(file_path: str) -> str:
    """
    Calculate MD5 hash of a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        MD5 hash string
    """
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

async def download_model(
    model_id: str,
    model_info: Dict[str, Any],
    output_dir: Path,
    cache_dir: Path,
    force: bool = False,
    use_mlx: bool = False
) -> bool:
    """
    Download a model using the Hugging Face transformers library.
    
    Args:
        model_id: Model identifier
        model_info: Model metadata
        output_dir: Directory to save model files
        cache_dir: Cache directory for downloads
        force: Whether to force download even if files exist
        
    Returns:
        True if download successful
    """
    model_name = model_info["model_name"]
    model_type = model_info["model_type"]
    tokenizer_name = model_info.get("tokenizer_name", model_name)
    
    # Create output directory
    model_dir = output_dir / model_id
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    
    # Check if model already exists
    model_config_path = model_dir / "config.json"
    if model_config_path.exists() and not force:
        logger.info(f"Model {model_id} already exists at {model_dir}")
        return True
        
    logger.info(f"Downloading model {model_id} ({model_name})")
    
    try:
        # Download tokenizer
        logger.info(f"Downloading tokenizer: {tokenizer_name}")
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            cache_dir=cache_dir
        )
        
        # Download model based on type
        logger.info(f"Downloading model: {model_name} (type: {model_type})")
        if model_type == "encoder":
            model = AutoModel.from_pretrained(
                model_name,
                cache_dir=cache_dir
            )
        elif model_type == "seq2seq":
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                cache_dir=cache_dir
            )
        else:
            # For other model types, use AutoModel
            model = AutoModel.from_pretrained(
                model_name,
                cache_dir=cache_dir
            )
            
        # Save model and tokenizer
        logger.info(f"Saving model and tokenizer to {model_dir}")
        model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)
        
        # Save model metadata
        metadata = {
            "id": model_id,
            "downloaded_at": datetime.now().isoformat(),
            "downloaded_by": os.getenv("USER", "system"),
            **model_info
        }
        if use_mlx:
            metadata["optimized_for"] = "mlx"
        with open(model_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Model {model_id} downloaded successfully")
        print(f"{Fore.GREEN}✓ Downloaded {model_id} ({model_name}){Style.RESET_ALL}")
        return True

    except Exception as e:
        logger.error(f"Error downloading model {model_id}: {str(e)}")
        print(f"{Fore.RED}✗ Error downloading {model_id}: {str(e)}{Style.RESET_ALL}")
        # Clean up in case of partial download
        if model_dir.exists():
            shutil.rmtree(model_dir)
        return False

def get_model_size(model_dir: Path) -> float:
    """
    Calculate the size of a model directory in GB.
    
    Args:
        model_dir: Path to model directory
        
    Returns:
        Size in GB
    """
    total_size = 0
    for path in model_dir.glob("**/*"):
        if path.is_file():
            total_size += path.stat().st_size
    
    # Convert bytes to GB
    return total_size / (1024 * 1024 * 1024)

async def verify_model(
    model_id: str,
    model_dir: Path,
    expected_files: Optional[List[str]] = None
) -> bool:
    """
    Verify that a model was downloaded correctly.
    
    Args:
        model_id: Model identifier
        model_dir: Path to model directory
        expected_files: List of expected files
        
    Returns:
        True if verification successful
    """
    # Check if model directory exists
    if not model_dir.exists():
        logger.error(f"Model directory not found: {model_dir}")
        print(f"{Fore.RED}✗ Model directory not found: {model_dir}{Style.RESET_ALL}")
        return False

    # Check for essential files (support both pytorch_model.bin and model.safetensors)
    essential_files = ["config.json"] if not expected_files else expected_files
    # Accept either pytorch_model.bin or model.safetensors
    has_bin = (model_dir / "pytorch_model.bin").exists()
    has_safetensors = (model_dir / "model.safetensors").exists()
    if not expected_files:
        if not (has_bin or has_safetensors):
            essential_files += ["pytorch_model.bin/model.safetensors"]
    missing_files = []
    for file in essential_files:
        if file == "pytorch_model.bin/model.safetensors":
            if not (has_bin or has_safetensors):
                missing_files.append("pytorch_model.bin/model.safetensors")
        elif not (model_dir / file).exists():
            missing_files.append(file)

    if missing_files:
        logger.error(f"Model {model_id} is missing files: {', '.join(missing_files)}")
        print(f"{Fore.RED}✗ Model {model_id} is missing files: {', '.join(missing_files)}{Style.RESET_ALL}")
        return False

    # Load metadata
    metadata_path = model_dir / "metadata.json"
    if not metadata_path.exists():
        logger.error(f"Model {model_id} is missing metadata.json")
        print(f"{Fore.RED}✗ Model {model_id} is missing metadata.json{Style.RESET_ALL}")
        return False

    # Check model size
    actual_size = get_model_size(model_dir)

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    expected_size = metadata.get("size_gb", 0)

    # Allow some tolerance in size comparison
    size_tolerance = 0.1  # 10% tolerance
    if abs(actual_size - expected_size) / max(expected_size, 0.1) > size_tolerance:
        logger.warning(
            f"Model {model_id} size mismatch: expected {expected_size:.2f} GB, "
            f"got {actual_size:.2f} GB"
        )
        print(f"{Fore.YELLOW}! Model {model_id} size mismatch: expected {expected_size:.2f} GB, got {actual_size:.2f} GB{Style.RESET_ALL}")

    logger.info(f"Model {model_id} verified successfully")
    print(f"{Fore.GREEN}✓ Model {model_id} verified successfully{Style.RESET_ALL}")
    return True

async def update_registry(
    registry_path: Path,
    models_dir: Path,
    downloaded_models: Set[str],
    use_mlx: bool = False
) -> None:
    """
    Update model registry with downloaded models.
    
    Args:
        registry_path: Path to registry file
        models_dir: Path to models directory
        downloaded_models: Set of downloaded model IDs
    """
    # Load existing registry or create new one
    if registry_path.exists():
        with open(registry_path, "r") as f:
            registry = json.load(f)
    else:
        registry = {}
        
    # Update registry with downloaded models
    for model_id in downloaded_models:
        model_dir = models_dir / model_id
        metadata_path = model_dir / "metadata.json"

        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            reg_entry = {
                "id": model_id,
                "name": metadata.get("name", model_id),
                "model_type": metadata.get("model_type", "unknown"),
                "size_gb": get_model_size(model_dir),
                "languages": metadata.get("languages", []),
                "tasks": metadata.get("tasks", []),
                "description": metadata.get("description", ""),
                "requires_gpu": metadata.get("requires_gpu", False),
                "memory_required": metadata.get("memory_required", 1.0),
                "gpu_memory_required": metadata.get("gpu_memory_required", 0.0),
                "location": "local",
                "path": str(model_dir),
                "last_updated": datetime.now().isoformat()
            }
            if use_mlx:
                reg_entry["optimized_for"] = "mlx"
            elif metadata.get("optimized_for"):
                reg_entry["optimized_for"] = metadata["optimized_for"]
            registry[model_id] = reg_entry
    # Save updated registry
    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2)
        
    logger.info(f"Updated registry with {len(downloaded_models)} models")

async def main():
    """Main entry point for model downloader."""
    colorama_init(autoreset=True)
    parser = argparse.ArgumentParser(description="CasaLingua Model Downloader")

    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        help="Directory to store models"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        help="Cache directory for downloads"
    )
    parser.add_argument(
        "--model",
        type=str,
        action="append",
        help="Specific model(s) to download"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all default models"
    )
    parser.add_argument(
        "--advanced",
        action="store_true",
        help="Include advanced models"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download of existing models"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify downloaded models"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    parser.add_argument(
        "--use-mlx",
        action="store_true",
        help="Tag downloaded models as optimized for MLX"
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Load configuration
    config = load_config(args.config) if args.config else load_config()

    # Set up directories
    models_dir = Path(args.models_dir) if args.models_dir else Path(config.get("models", {}).get("models_dir", "models"))
    cache_dir = Path(args.cache_dir) if args.cache_dir else Path(config.get("models", {}).get("cache_dir", "cache/models"))
    registry_path = models_dir / "registry.json"

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    # Determine which models to download
    models_to_download = {}

    if args.model:
        # Download specific models
        for model_id in args.model:
            if model_id in DEFAULT_MODELS:
                models_to_download[model_id] = DEFAULT_MODELS[model_id]
            elif model_id in ADVANCED_MODELS:
                models_to_download[model_id] = ADVANCED_MODELS[model_id]
            else:
                logger.warning(f"Unknown model: {model_id}")
                print(f"{Fore.YELLOW}! Unknown model: {model_id}{Style.RESET_ALL}")
    elif args.all:
        # Download all default models
        models_to_download.update(DEFAULT_MODELS)
        if args.advanced:
            models_to_download.update(ADVANCED_MODELS)
    else:
        # Download models specified in config
        preload_models = config.get("models", {}).get("preload_models", [])
        for model_id in preload_models:
            if model_id in DEFAULT_MODELS:
                models_to_download[model_id] = DEFAULT_MODELS[model_id]
            elif model_id in ADVANCED_MODELS:
                models_to_download[model_id] = ADVANCED_MODELS[model_id]
            else:
                logger.warning(f"Unknown model in config: {model_id}")
                print(f"{Fore.YELLOW}! Unknown model in config: {model_id}{Style.RESET_ALL}")

        if not models_to_download:
            # If no models specified, download default models
            logger.info("No models specified, downloading default models")
            print(f"{Fore.YELLOW}No models specified, downloading default models{Style.RESET_ALL}")
            models_to_download.update(DEFAULT_MODELS)

    if not models_to_download:
        logger.error("No models to download")
        print(f"{Fore.RED}No models to download{Style.RESET_ALL}")
        return

    # Check if GPU is available
    gpu_available = torch.cuda.is_available()
    if not gpu_available:
        # Filter out models that require GPU
        models_requiring_gpu = [
            model_id for model_id, info in models_to_download.items()
            if info.get("requires_gpu", False)
        ]

        if models_requiring_gpu:
            logger.warning(
                f"No GPU available, skipping models that require GPU: "
                f"{', '.join(models_requiring_gpu)}"
            )
            print(f"{Fore.YELLOW}! No GPU available, skipping models that require GPU: {', '.join(models_requiring_gpu)}{Style.RESET_ALL}")

            models_to_download = {
                model_id: info for model_id, info in models_to_download.items()
                if not info.get("requires_gpu", False)
            }

    # Download models
    logger.info(f"Downloading {len(models_to_download)} models to {models_dir}")
    print(f"\nDownloading {len(models_to_download)} models to {models_dir}")

    downloaded_models = set()
    for model_id, model_info in models_to_download.items():
        success = await download_model(
            model_id=model_id,
            model_info=model_info,
            output_dir=models_dir,
            cache_dir=cache_dir,
            force=args.force,
            use_mlx=args.use_mlx
        )

        if success:
            downloaded_models.add(model_id)

            # Verify model if requested
            if args.verify:
                model_dir = models_dir / model_id
                verified = await verify_model(
                    model_id=model_id,
                    model_dir=model_dir
                )

                if not verified:
                    logger.error(f"Model {model_id} verification failed")
                    print(f"{Fore.RED}✗ Model {model_id} verification failed{Style.RESET_ALL}")

    # Update registry
    if downloaded_models:
        await update_registry(
            registry_path=registry_path,
            models_dir=models_dir,
            downloaded_models=downloaded_models,
            use_mlx=args.use_mlx
        )

    logger.info(f"Downloaded {len(downloaded_models)} models")

    # Print summary with colored model sizes
    if downloaded_models:
        print(f"\n{Style.BRIGHT}Downloaded models:{Style.RESET_ALL}")
        for model_id in downloaded_models:
            model_dir = models_dir / model_id
            size = get_model_size(model_dir)
            if size < 0.5:
                color = Fore.GREEN
            elif size < 1.0:
                color = Fore.YELLOW
            else:
                color = Fore.RED
            print(f"  - {model_id}: {color}{size:.2f} GB{Style.RESET_ALL}")

    skipped_models = set(models_to_download.keys()) - downloaded_models
    if skipped_models:
        print(f"\n{Style.BRIGHT}Skipped models:{Style.RESET_ALL}")
        for model_id in skipped_models:
            print(f"  - {Fore.YELLOW}{model_id}{Style.RESET_ALL}")

if __name__ == "__main__":
    asyncio.run(main())