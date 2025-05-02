#!/usr/bin/env python

# =============================================================================
# CasaLingua Registry Setup Script
# -----------------------------------------------------------------------------
# This script creates or updates the model registry with required models and
# their configurations. It also supports creating directory scaffolding for
# model loaders and updating existing entries.
#
# Ladder Logic Diagram:
# 
#         +-------------------------+
#         | Start setup_registry.py|
#         +-----------+-------------+
#                     |
#         +-----------v------------+
#         | Parse arguments        |
#         +-----------+------------+
#                     |
#         +-----------v------------+
#         | If --create-loaders    |
#         | → create_loader_dirs() |
#         +-----------+------------+
#                     |
#         +-----------v------------+
#         | Read existing registry |
#         +-----------+------------+
#                     |
#         +-----------v------------+
#         | Add/update models      |
#         +-----------+------------+
#                     |
#         +-----------v------------+
#         | Add task mappings      |
#         +-----------+------------+
#                     |
#         +-----------v------------+
#         | Save registry.json     |
#         +-----------+------------+
#                     |
#         +-----------v------------+
#         | Print completion msg   |
#         +------------------------+
#
# =============================================================================

import os
import json
import argparse
from pathlib import Path

def setup_registry(registry_path: str) -> None:
    """
    Setup the model registry with required models.
    
    Args:
        registry_path: Path to registry.json file
    """
    # Create models directory if it doesn't exist
    models_dir = os.path.dirname(registry_path)
    os.makedirs(models_dir, exist_ok=True)
    
    # Load existing registry if available
    registry = {}
    if os.path.exists(registry_path):
        try:
            with open(registry_path, 'r', encoding='utf-8') as f:
                registry = json.load(f)
            print(f"Loaded existing registry with {len(registry)} models")
        except Exception as e:
            print(f"Error loading registry, creating new one: {e}")
            registry = {}
    
    # Define required models with Facebook's NLLB for translation
    required_models = {
        "facebook-nllb-small": {
            "name": "Facebook NLLB Small",
            "description": "Facebook's No Language Left Behind (NLLB) small model for translation",
            "model_id": "facebook/nllb-200-distilled-600M",
            "version": "1.0.0",
            "tasks": ["translation", "language_detection"],
            "languages": ["en", "es", "fr", "de", "it", "pt", "ja", "zh", "ko", "ru"],
            "memory_required": 2.0,
            "disk_required": 1.0,
            "requires_gpu": False,
            "performance_score": 0.85,
            "size_gb": 2.0,
            "model_type": "transformer",
            "quantization": "16-bit",
            "location": "local",
            "loader_module": "app.services.models.loaders.translation_loader",
            "loader_class": "TranslationModelLoader",
            "locked": True  # Prevent filtering out during hardware constraints check
        },
        "google-mt5-small": {
            "name": "Google MT5 Small",
            "description": "Google's Multilingual T5 small model for multipurpose NLP tasks",
            "model_id": "google/mt5-small",
            "version": "1.0.0",
            "tasks": ["summarization", "question_answering", "text_generation"],
            "languages": ["en", "es", "fr", "de", "it", "pt", "ja", "zh", "ko", "ru"],
            "memory_required": 2.0,
            "disk_required": 1.0,
            "requires_gpu": False,
            "performance_score": 0.8,
            "size_gb": 1.5,
            "model_type": "transformer",
            "quantization": "16-bit",
            "location": "local",
            "loader_module": "app.services.models.loaders.multipurpose_loader",
            "loader_class": "MultipurposeModelLoader",
            "locked": True
        },
        "microsoft-deberta-small": {
            "name": "Microsoft DeBERTa Small",
            "description": "Microsoft's DeBERTa small model for verification tasks",
            "model_id": "microsoft/deberta-v3-xsmall",
            "version": "1.0.0",
            "tasks": ["verification", "error_detection"],
            "languages": ["en"],
            "memory_required": 1.0,
            "disk_required": 0.5,
            "requires_gpu": False,
            "performance_score": 0.75,
            "size_gb": 0.5,
            "model_type": "transformer",
            "quantization": "16-bit",
            "location": "local",
            "loader_module": "app.services.models.loaders.verification_loader",
            "loader_class": "VerificationModelLoader",
            "locked": True
        }
    }
    
    # Add task mappings for common model types
    task_mappings = {
        "translation": "facebook-nllb-small",
        "multipurpose": "google-mt5-small",
        "verification": "microsoft-deberta-small"
    }
    
    # Update required models in registry
    for model_id, model_info in required_models.items():
        if model_id not in registry:
            print(f"Adding required model: {model_id}")
            registry[model_id] = model_info
        else:
            print(f"Model {model_id} already exists in registry")
            
            # Check if model is complete
            missing_keys = [key for key in model_info if key not in registry[model_id]]
            if missing_keys:
                print(f"  Updating model with missing fields: {missing_keys}")
                for key in missing_keys:
                    registry[model_id][key] = model_info[key]
    
    # Add task mappings
    for task, model_id in task_mappings.items():
        registry[task] = model_id
        print(f"Added task mapping: {task} -> {model_id}")
    
    # Save updated registry
    try:
        with open(registry_path, 'w', encoding='utf-8') as f:
            json.dump(registry, f, indent=2)
        print(f"Saved registry with {len(registry)} models to {registry_path}")
    except Exception as e:
        print(f"Error saving registry: {e}")

def create_loader_directories():
    """Create the necessary loader directories and files."""
    # Create loaders directory
    loaders_dir = Path("app/services/models/loaders")
    if not loaders_dir.exists():
        os.makedirs(loaders_dir, exist_ok=True)
        print(f"Created directory: {loaders_dir}")
    
    # Create __init__.py
    init_path = loaders_dir / "__init__.py"
    if not init_path.exists():
        with open(init_path, "w") as f:
            f.write("""# Model loaders package for CasaLingua
"""
            )
        print(f"Created file: {init_path}")
    
    # Create base_loader.py
    base_loader_path = loaders_dir / "base_loader.py"
    if not base_loader_path.exists():
        with open(base_loader_path, "w") as f:
            f.write("""import os
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class ModelLoader(ABC):
    """
    Abstract base class for model loaders.
    """
    
    def __init__(self, model_path: Path, config: Dict[str, Any]):
        """
        Initialize the model loader.
        
        Args:
            model_path: Path to model directory
            config: Model configuration
        """
        self.model_path = model_path
        self.config = config
        self.model_type = config.get("model_type", "unknown")
        self.model_id = config.get("model_id", "")
        
    @abstractmethod
    async def load(self) -> Any:
        """
        Load the model.
        
        Returns:
            Loaded model instance
        """
        pass
"""
            )
        print(f"Created file: {base_loader_path}")
    
    # Create mock_loader.py
    mock_loader_path = loaders_dir / "mock_loader.py"
    if not mock_loader_path.exists():
        with open(mock_loader_path, "w") as f:
            f.write("""import logging
from typing import Dict, Any

from app.services.models.loaders.base_loader import ModelLoader

logger = logging.getLogger(__name__)

class MockModelLoader(ModelLoader):
    """
    Mock model loader for development and testing.
    """
    
    async def load(self) -> Any:
        """
        Load a mock model.
        
        Returns:
            Mock model instance
        """
        logger.info(f"Loading mock model for {self.model_type}...")
        
        # Create appropriate mock model based on tasks
        tasks = self.config.get("tasks", [])
        
        if "translation" in tasks:
            return MockTranslationModel(self.config)
        elif "verification" in tasks:
            return MockVerificationModel(self.config)
        else:
            return MockBaseModel(self.config)


class MockBaseModel:
    """Base class for mock models."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize mock model.
        
        Args:
            config: Model configuration
        """
        self.config = config
        self.name = config.get("name", "Mock Model")
        self.languages = config.get("languages", ["en"])
    
    async def process(self, text: str, **kwargs) -> Dict[str, Any]:
        """
        Process text with the mock model.
        """
        return {
            "input": text,
            "output": f"Mock output for: {text[:50]}...",
            "model": self.name
        }


class MockTranslationModel(MockBaseModel):
    """Mock translation model."""
    
    async def translate(self, text: str, source_lang: str, target_lang: str, **kwargs) -> Dict[str, Any]:
        """
        Perform mock translation.
        
        Args:
            text: Text to translate
            source_lang: Source language code
            target_lang: Target language code
            **kwargs: Additional arguments
            
        Returns:
            Mock translation result
        """
        # Check if languages are supported
        if source_lang not in self.languages or target_lang not in self.languages:
            return {
                "error": f"Language not supported: {source_lang} or {target_lang}",
                "input": text,
                "source_lang": source_lang,
                "target_lang": target_lang,
                "model": self.name
            }
        
        # Create mock translation
        if target_lang == "es":
            prefix = "¡"
            suffix = "!"
        elif target_lang == "fr":
            prefix = "Le "
            suffix = " (en français)"
        elif target_lang == "de":
            prefix = "Das "
            suffix = " (auf Deutsch)"
        else:
            prefix = ""
            suffix = f" (in {target_lang})"
        
        # Create mock translation
        translation = f"{prefix}{text}{suffix}"
        
        return {
            "input": text,
            "translation": translation,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "confidence": 0.95,
            "model": self.name
        }
    
    async def detect_language(self, text: str, **kwargs) -> Dict[str, Any]:
        """
        Perform mock language detection.
        
        Args:
            text: Text to analyze
            **kwargs: Additional arguments
            
        Returns:
            Mock language detection result
        """
        # Simple mock implementation
        detected_lang = "en"
        confidence = 0.9
        
        # Check for language indicators
        if "¡" in text or "¿" in text:
            detected_lang = "es"
            confidence = 0.95
        elif "ç" in text or "é" in text:
            detected_lang = "fr"
            confidence = 0.93
        elif "ß" in text or "ü" in text:
            detected_lang = "de"
            confidence = 0.94
        
        return {
            "input": text,
            "detected_language": detected_lang,
            "confidence": confidence,
            "model": self.name
        }


class MockVerificationModel(MockBaseModel):
    """Mock verification model."""
    
    async def verify_translation(self, source_text: str, translation: str, 
                              source_lang: str, target_lang: str, **kwargs) -> Dict[str, Any]:
        """
        Perform mock translation verification.
        
        Args:
            source_text: Original text
            translation: Translated text
            source_lang: Source language code
            target_lang: Target language code
            **kwargs: Additional arguments
            
        Returns:
            Mock verification result
        """
        # Simple mock implementation
        return {
            "source": source_text,
            "translation": translation,
            "accuracy": 0.92,
            "fluency": 0.89,
            "semantic_similarity": 0.95,
            "verified": True,
            "issues": [],
            "model": self.name
        }
"""
            )
        print(f"Created file: {mock_loader_path}")

def main():
    """
    =======================
    CasaLingua Registry Setup
    =======================

    This script initializes the model registry with required models and their configurations. 
    It supports creating directory scaffolding for model loaders and updating existing entries.

    Usage:
        python scripts/setup_registry.py --registry models/registry.json --create-loaders

    Key Features:
    - Ensures all mandatory models are registered (NLLB, mT5, DeBERTa)
    - Allows dynamic creation of loader scaffolding
    - Supports task-to-model mapping for the core system pipeline

    Ladder Logic Diagram:

            +-------------------------+
            | Start setup_registry.py|
            +-----------+-------------+
                        |
            +-----------v------------+
            | Parse arguments        |
            +-----------+------------+
                        |
            +-----------v------------+
            | If --create-loaders    |
            | → create_loader_dirs() |
            +-----------+------------+
                        |
            +-----------v------------+
            | Read existing registry |
            +-----------+------------+
                        |
            +-----------v------------+
            | Add/update models      |
            +-----------+------------+
                        |
            +-----------v------------+
            | Add task mappings      |
            +-----------+------------+
                        |
            +-----------v------------+
            | Save registry.json     |
            +-----------+------------+
                        |
            +-----------v------------+
            | Print completion msg   |
            +------------------------+

    """
    parser = argparse.ArgumentParser(description="Set up CasaLingua model registry")
    parser.add_argument("--registry", help="Path to registry.json file", default="models/registry.json")
    parser.add_argument("--create-loaders", help="Create loader directories and files", action="store_true")
    
    args = parser.parse_args()
    
    print(f"Setting up model registry at {args.registry}")
    
    # Create loader directories if requested
    if args.create_loaders:
        create_loader_directories()
    
    # Setup registry
    setup_registry(args.registry)
    
    print("Registry setup complete")

if __name__ == "__main__":
    main()