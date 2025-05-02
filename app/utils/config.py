# app/utils/config.py
"""
Configuration Utilities for CasaLingua

This module provides functions for loading, validating, and accessing
application configuration from various sources.
"""

from dotenv import load_dotenv
load_dotenv()

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger("casalingua.config")

def detect_macos_m4() -> bool:
    """
    Detect if running on an M4 Mac and allow GPU usage.
    
    Returns:
        bool: True if M4 Mac detected and torch.mps is available
    """
    import platform
    if platform.machine() == "arm64" and platform.system() == "Darwin":
        try:
            import torch
            return torch.backends.mps.is_available()
        except ImportError:
            return False
    return False

def load_config() -> Dict[str, Any]:
    """
    Load configuration from environment variables and config files.
    
    The configuration is loaded in the following priority order:
    1. Environment variables
    2. Environment-specific config file (development.json, production.json)
    3. Default config file (default.json)
    
    Returns:
        Dict[str, Any]: The merged configuration
    """
    # Start with default configuration
    config = _load_default_config()
    logger.debug(f"Loaded default config: {config}")
    
    # Override with environment-specific configuration
    env_config = _load_environment_config()
    logger.debug(f"Loaded environment-specific config: {env_config}")
    config.update(env_config)
    
    # Override with environment variables
    env_override = _load_environment_variables()
    logger.debug(f"Loaded environment variables config: {env_override}")
    config.update(env_override)
    
    config = validate_config(config)
    logger.debug(f"Validated final config: {config}")
    
    return config

def _load_default_config() -> Dict[str, Any]:
    """
    Load default configuration from default.json.
    
    Returns:
        Dict[str, Any]: Default configuration
    """
    config_path = Path("config/default.json")
    
    try:
        if config_path.exists():
            with open(config_path, "r") as f:
                return json.load(f)
        else:
            logger.warning(f"Default config file not found: {config_path}")
            # Create a basic default configuration
            return {
                "environment": "development",
                "debug": True,
                "log_level": "INFO",
                "server_host": "0.0.0.0",
                "server_port": 8000,
                "cors_origins": "*",
                "models_dir": "models",
                "log_dir": "logs",
                "use_gpu": detect_macos_m4(),
                "model_size": "medium",
                "worker_threads": 4,
                "batch_size": 8,
                "cache_size": 100,
                "audit_enabled": True
            }
    except Exception as e:
        logger.error(f"Error loading default config: {str(e)}")
        return {}

def _load_environment_config() -> Dict[str, Any]:
    """
    Load environment-specific configuration.
    
    Returns:
        Dict[str, Any]: Environment-specific configuration
    """
    environment = os.environ.get("ENVIRONMENT", "development")
    config_path = Path(f"config/{environment}.json")
    
    try:
        if config_path.exists():
            with open(config_path, "r") as f:
                return json.load(f)
        else:
            logger.debug(f"Environment config file not found: {config_path}")
            return {}
    except Exception as e:
        logger.error(f"Error loading environment config: {str(e)}")
        return {}

def _load_environment_variables() -> Dict[str, Any]:
    """
    Load configuration from environment variables.
    
    Environment variables prefixed with "CASALINGUA_" are loaded into
    the configuration with the prefix removed and the key converted to
    lowercase.
    
    Returns:
        Dict[str, Any]: Configuration from environment variables
    """
    env_config = {}
    prefix = "CASALINGUA_"
    
    for key, value in os.environ.items():
        if key.startswith(prefix):
            # Remove prefix and convert to lowercase
            config_key = key[len(prefix):].lower()
            
            # Convert value types
            if value.lower() in ("true", "yes", "1"):
                env_config[config_key] = True
            elif value.lower() in ("false", "no", "0"):
                env_config[config_key] = False
            elif value.isdigit():
                env_config[config_key] = int(value)
            elif value.replace(".", "", 1).isdigit() and value.count(".") <= 1:
                env_config[config_key] = float(value)
            else:
                env_config[config_key] = value
    
    return env_config

def get_config_value(config: Dict[str, Any], key: str, default: Any = None) -> Any:
    """
    Get a configuration value with fallback to default.
    
    Args:
        config: Configuration dictionary
        key: Configuration key
        default: Default value if key not found
        
    Returns:
        Any: Configuration value or default
    """
    return config.get(key, default)

def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate configuration and set defaults for missing values.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dict[str, Any]: Validated configuration
    """
    # Ensure required keys are present
    required_keys = {
        "environment": "development",
        "server_host": "0.0.0.0",
        "server_port": 8000,
        "log_level": "INFO",
        "models_dir": "models",
        "log_dir": "logs"
    }
    
    for key, default_value in required_keys.items():
        if key not in config:
            logger.warning(f"Missing required config key: {key}, using default: {default_value}")
            config[key] = default_value
    
    return config