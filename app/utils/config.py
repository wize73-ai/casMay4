# app/utils/config.py
"""
Configuration Utilities for CasaLingua with Dynamic Reloading Support

This module provides functions for loading, validating, and accessing
application configuration from various sources. It supports dynamic reloading
of configuration files and notification callbacks for configuration changes.
"""

from dotenv import load_dotenv
load_dotenv()

import os
import json
import time
import logging
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable, Union, Set

logger = logging.getLogger("casalingua.config")

# Global configuration and state
_config_cache = {}
_config_file_timestamps = {}
_config_callbacks = []
_config_watcher_thread = None
_watcher_lock = threading.RLock()
_watcher_active = False
_watcher_interval = 10  # seconds

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

def load_config(force_reload: bool = False) -> Dict[str, Any]:
    """
    Load configuration from environment variables and config files with caching and dynamic reloading.
    
    The configuration is loaded in the following priority order:
    1. Environment variables
    2. Environment-specific config file (development.json, production.json)
    3. Default config file (default.json)
    
    Args:
        force_reload: Whether to force reload even if cached
    
    Returns:
        Dict[str, Any]: The merged configuration
    """
    global _config_cache, _config_file_timestamps
    
    # Get the current environment
    environment = os.environ.get("ENVIRONMENT", "development")
    
    # Check if we should use the cached configuration
    if not force_reload and environment in _config_cache:
        # Check if configuration files have changed
        default_config_path = Path("config/default.json")
        env_config_path = Path(f"config/{environment}.json")
        needs_reload = False
        
        # Check default config file timestamp
        if default_config_path.exists():
            last_modified = default_config_path.stat().st_mtime
            if str(default_config_path) not in _config_file_timestamps or last_modified > _config_file_timestamps[str(default_config_path)]:
                needs_reload = True
        
        # Check environment config file timestamp
        if env_config_path.exists():
            last_modified = env_config_path.stat().st_mtime
            if str(env_config_path) not in _config_file_timestamps or last_modified > _config_file_timestamps[str(env_config_path)]:
                needs_reload = True
        
        # Return cached config if no reload needed
        if not needs_reload:
            logger.debug("Using cached configuration")
            return _config_cache[environment]
        else:
            logger.info("Configuration files changed, reloading")
    
    # Save the old config for change detection
    old_config = _config_cache.get(environment, {})
    
    # Start with default configuration
    config = _load_default_config()
    logger.debug(f"Loaded default config: {config}")
    
    # Save default config file timestamp
    default_config_path = Path("config/default.json")
    if default_config_path.exists():
        _config_file_timestamps[str(default_config_path)] = default_config_path.stat().st_mtime
    
    # Override with environment-specific configuration
    env_config = _load_environment_config()
    logger.debug(f"Loaded environment-specific config: {env_config}")
    config.update(env_config)
    
    # Save environment config file timestamp
    env_config_path = Path(f"config/{environment}.json")
    if env_config_path.exists():
        _config_file_timestamps[str(env_config_path)] = env_config_path.stat().st_mtime
    
    # Override with environment variables
    env_override = _load_environment_variables()
    logger.debug(f"Loaded environment variables config: {env_override}")
    config.update(env_override)
    
    # Validate the configuration
    config = validate_config(config)
    logger.debug(f"Validated final config: {config}")
    
    # Cache the configuration
    _config_cache[environment] = config
    
    # Notify callbacks of configuration changes if any
    if config != old_config:
        _notify_config_changes(old_config, config)
    
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
        key: Configuration key (supports dot notation for nested access)
        default: Default value if key not found
        
    Returns:
        Any: Configuration value or default
    """
    # Handle nested keys with dot notation
    if "." in key:
        parts = key.split(".")
        current = config
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default
        return current
        
    # Simple key lookup
    return config.get(key, default)


def get_nested_value(config: Dict[str, Any], key: str, default: Any = None) -> Any:
    """
    Get a nested value from a dictionary using dot notation.
    
    Args:
        config: Configuration dictionary
        key: Key in dot notation (e.g., "database.url")
        default: Default value if key not found
        
    Returns:
        Any: Value at specified key or default
    """
    return get_config_value(config, key, default)


class ConfigChangeCallback:
    """
    Callback for configuration changes with unique ID for management.
    """
    def __init__(self, callback_fn: Callable[[Dict[str, Any], Dict[str, Any]], None], 
                watch_keys: Optional[List[str]] = None):
        """
        Initialize a configuration change callback.
        
        Args:
            callback_fn: Function to call when config changes, with signature (old_config, new_config) -> None
            watch_keys: List of keys to watch for changes, or None to watch all keys
        """
        self.callback_fn = callback_fn
        self.watch_keys = watch_keys or []
        self.id = id(self)
        
    def __call__(self, old_config: Dict[str, Any], new_config: Dict[str, Any]) -> None:
        """
        Call the callback function if watched keys have changed.
        
        Args:
            old_config: Previous configuration
            new_config: New configuration
        """
        # If no specific keys to watch, always notify
        if not self.watch_keys:
            self.callback_fn(old_config, new_config)
            return
            
        # Check if any watched keys have changed
        for key in self.watch_keys:
            old_value = get_nested_value(old_config, key)
            new_value = get_nested_value(new_config, key)
            
            if old_value != new_value:
                self.callback_fn(old_config, new_config)
                return


def register_config_change_callback(callback_fn: Callable[[Dict[str, Any], Dict[str, Any]], None], 
                                   watch_keys: Optional[List[str]] = None) -> str:
    """
    Register a callback to be notified when configuration changes.
    
    Args:
        callback_fn: Function to call when config changes, with signature (old_config, new_config) -> None
        watch_keys: List of keys to watch for changes, or None to watch all keys
        
    Returns:
        str: Unique ID for the callback for later removal
    """
    global _config_callbacks
    
    # Create a new callback
    callback = ConfigChangeCallback(callback_fn, watch_keys)
    
    # Add to the list of callbacks
    _config_callbacks.append(callback)
    
    # Start the watcher thread if not already running
    start_config_watcher()
    
    logger.debug(f"Registered config change callback {callback.id}")
    
    return str(callback.id)


def unregister_config_change_callback(callback_id: str) -> bool:
    """
    Unregister a previously registered callback.
    
    Args:
        callback_id: ID of the callback to remove
        
    Returns:
        bool: True if the callback was found and removed, False otherwise
    """
    global _config_callbacks
    
    try:
        # Convert ID back to integer
        callback_id_int = int(callback_id)
        
        # Find and remove the callback
        for i, callback in enumerate(_config_callbacks):
            if callback.id == callback_id_int:
                _config_callbacks.pop(i)
                logger.debug(f"Unregistered config change callback {callback_id}")
                return True
    except ValueError:
        pass
        
    return False


def _notify_config_changes(old_config: Dict[str, Any], new_config: Dict[str, Any]) -> None:
    """
    Notify all registered callbacks of configuration changes.
    
    Args:
        old_config: Previous configuration
        new_config: New configuration
    """
    global _config_callbacks
    
    if old_config == new_config:
        return
        
    # Make a copy of the callbacks list to avoid issues if callbacks
    # register or unregister other callbacks during notification
    callbacks = _config_callbacks.copy()
    
    # Get environment
    environment = new_config.get('environment', 'development')
    
    logger.info(f"Configuration changed for environment '{environment}', notifying {len(callbacks)} callbacks")
    
    # Call each callback
    for callback in callbacks:
        try:
            callback(old_config, new_config)
        except Exception as e:
            logger.error(f"Error in config change callback: {e}", exc_info=True)

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


def start_config_watcher() -> None:
    """
    Start the configuration file watcher thread if not already running.
    The watcher periodically checks for changes in configuration files
    and triggers reloads when changes are detected.
    """
    global _config_watcher_thread, _watcher_active, _watcher_lock
    
    with _watcher_lock:
        # If already active, don't start another watcher
        if _watcher_active:
            return
            
        _watcher_active = True
        
        def watch_config_files():
            """Thread function that periodically checks for config changes"""
            global _watcher_active
            
            logger.info(f"Configuration watcher started, checking every {_watcher_interval} seconds")
            
            while _watcher_active:
                try:
                    # Check for changes by forcing a reload
                    load_config(force_reload=True)
                except Exception as e:
                    logger.error(f"Error in config watcher: {e}", exc_info=True)
                    
                # Sleep for the specified interval
                for _ in range(_watcher_interval * 10):  # Split into 0.1s checks for faster termination
                    if not _watcher_active:
                        break
                    time.sleep(0.1)
            
            logger.info("Configuration watcher stopped")
        
        # Create and start the thread
        _config_watcher_thread = threading.Thread(
            target=watch_config_files,
            name="ConfigWatcher",
            daemon=True
        )
        _config_watcher_thread.start()


def stop_config_watcher() -> None:
    """
    Stop the configuration file watcher thread if running.
    """
    global _watcher_active, _watcher_lock
    
    with _watcher_lock:
        if not _watcher_active:
            return
            
        # Signal the thread to stop
        _watcher_active = False
        
        # Wait for the thread to terminate (with timeout)
        if _config_watcher_thread and _config_watcher_thread.is_alive():
            _config_watcher_thread.join(timeout=2.0)
            
        logger.info("Configuration watcher thread stopped")


def set_config_watcher_interval(seconds: int) -> None:
    """
    Set the interval at which the configuration watcher checks for changes.
    
    Args:
        seconds: Interval in seconds between checks (minimum 1 second)
    """
    global _watcher_interval
    
    # Ensure minimum interval of 1 second
    _watcher_interval = max(1, seconds)
    logger.info(f"Set configuration watcher interval to {_watcher_interval} seconds")