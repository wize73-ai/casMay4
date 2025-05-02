"""
Enhanced Logging Module for CasaLingua

This module extends the basic logging utilities with structured logging,
custom formatters, routing-specific loggers, and specialized handlers
for the CasaLingua language processing platform.

Author: Exygy Development Team
Version: 1.0.0
License: MIT
"""

import os
import sys
import json
import logging
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import threading
import socket
import platform
from logging.handlers import RotatingFileHandler, QueueHandler, QueueListener
import queue

from app.utils.config import load_config, get_config_value

# Create local logger
_logger = logging.getLogger("casalingua.logging")

# Thread local storage for request context
_thread_local = threading.local()

class StructuredLogFormatter(logging.Formatter):
    """
    JSON formatter for structured logging with support for request context.
    """
    
    def __init__(self, include_context: bool = True):
        """
        Initialize the structured log formatter.
        
        Args:
            include_context: Whether to include request context in logs
        """
        super().__init__()
        self.include_context = include_context
        self.hostname = socket.gethostname()
        
    def format(self, record):
        """Format the log record as a JSON object."""
        # Create base log record
        log_record = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "message": record.getMessage(),
            "thread_id": record.thread,
            "process_id": record.process,
            "hostname": self.hostname
        }
        
        # Add exception info if present
        if record.exc_info:
            log_record["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": self.formatException(record.exc_info)
            }
            
        # Add request context if available and enabled
        if self.include_context and hasattr(_thread_local, "request_context"):
            log_record["request"] = _thread_local.request_context
            
        # Add extra fields if present
        for key, value in record.__dict__.items():
            if key.startswith("_") or key in log_record or key in {"args", "exc_info", "exc_text", "pathname", "stack_info", "msg"}:
                continue
                
            # Skip non-serializable objects
            try:
                json.dumps({key: value})
                log_record[key] = value
            except (TypeError, OverflowError):
                log_record[key] = str(value)
                
        return json.dumps(log_record)

class ConsoleFormatter(logging.Formatter):
    """
    Console formatter with color support and concise output.
    """
    
    # ANSI color codes
    COLORS = {
        "RESET": "\033[0m",
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",   # Green
        "WARNING": "\033[33m", # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[41m\033[37m"  # White on Red
    }
    
    def __init__(self, use_colors: bool = True, include_context: bool = True):
        """
        Initialize the console formatter.
        
        Args:
            use_colors: Whether to use ANSI colors
            include_context: Whether to include request context
        """
        super().__init__()
        self.use_colors = use_colors and sys.stdout.isatty()
        self.include_context = include_context
        
    def format(self, record):
        """Format the log record for console output."""
        # Format timestamp
        timestamp = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        
        # Format level with color if enabled
        level = record.levelname
        if self.use_colors:
            level = f"{self.COLORS.get(level, self.COLORS['RESET'])}{level}{self.COLORS['RESET']}"
            
        # Format logger name (shorten if too long)
        logger_name = record.name
        if logger_name.startswith("casalingua."):
            logger_name = logger_name[len("casalingua."):]
            
        # Format base message
        message = f"{timestamp} [{level}] [{logger_name}] {record.getMessage()}"
        
        # Add request context if available and enabled
        if self.include_context and hasattr(_thread_local, "request_context"):
            request_id = _thread_local.request_context.get("request_id", "")
            if request_id:
                message = f"{message} [req:{request_id}]"
                
        # Add exception info if present
        if record.exc_info:
            message = f"{message}\n{self.formatException(record.exc_info)}"
            
        return message

class AsyncQueueHandler(QueueHandler):
    """
    Queue handler with async support for FastAPI applications.
    """
    
    async def emit_async(self, record):
        """Emit a log record asynchronously."""
        # Format the record
        try:
            self.format(record)
            self.queue.put_nowait(record)
        except queue.Full:
            self.handleError(record)
        except Exception:
            self.handleError(record)

class RequestContextFilter(logging.Filter):
    """
    Filter that adds request context to log records.
    """
    
    def filter(self, record):
        """Add request context to the record if available."""
        if hasattr(_thread_local, "request_context"):
            for key, value in _thread_local.request_context.items():
                setattr(record, key, value)
        return True

def setup_logging(
    config: Optional[Dict[str, Any]] = None,
    log_level: Optional[str] = None,
    structured: bool = False
) -> logging.Logger:
    """
    Set up comprehensive logging for the application.
    
    Args:
        config: Application configuration
        log_level: Log level override
        structured: Whether to use structured JSON logging
        
    Returns:
        Root logger
    """
    # Load configuration if not provided
    if config is None:
        config = load_config()
        
    # Get logging configuration
    log_config = config.get("logging", {})
    log_level = log_level or log_config.get("level", "INFO")
    log_dir = Path(log_config.get("log_dir", "logs"))
    
    # Create log directory
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Get or create root logger
    logger = logging.getLogger("casalingua")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        
    # Configure console logging
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    
    # Use structured formatter if requested, otherwise use console formatter
    if structured:
        console_formatter = StructuredLogFormatter()
    else:
        use_colors = log_config.get("use_colors", True)
        console_formatter = ConsoleFormatter(use_colors=use_colors)
        
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # Configure file logging
    if log_config.get("file_logging", True):
        log_file = log_dir / "casalingua.log"
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=log_config.get("max_size", 10 * 1024 * 1024),  # 10 MB
            backupCount=log_config.get("backup_count", 5)
        )
        file_handler.setLevel(getattr(logging, log_level.upper()))
        
        # Use structured formatter for file logs
        file_formatter = StructuredLogFormatter()
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
    # Configure JSON logging for structured output
    if log_config.get("json_logging", structured):
        json_file = log_dir / "casalingua.json"
        json_handler = RotatingFileHandler(
            json_file,
            maxBytes=log_config.get("max_size", 10 * 1024 * 1024),  # 10 MB
            backupCount=log_config.get("backup_count", 5)
        )
        json_handler.setLevel(getattr(logging, log_level.upper()))
        json_handler.setFormatter(StructuredLogFormatter())
        logger.addHandler(json_handler)
        
    # Configure component-specific loggers
    component_config = log_config.get("components", {})
    for component, component_level in component_config.items():
        component_logger = logging.getLogger(f"casalingua.{component}")
        component_logger.setLevel(getattr(logging, component_level.upper()))
        
    # Set up request context filter
    context_filter = RequestContextFilter()
    logger.addFilter(context_filter)
    
    # Configure external library logging
    setup_external_loggers(log_config)
    
    # Log startup information
    logger.info(f"Logging initialized at level {log_level}")
    
    return logger

def setup_external_loggers(log_config: Dict[str, Any]) -> None:
    """
    Configure logging for external libraries.
    
    Args:
        log_config: Logging configuration
    """
    # Configure common libraries
    libraries = {
        "uvicorn": "INFO",
        "fastapi": "INFO",
        "sqlalchemy": "WARNING",
        "urllib3": "WARNING",
        "httpx": "WARNING",
        "pydantic": "WARNING",
        "transformers": "WARNING",
        "torch": "WARNING",
    }
    
    # Override with config if provided
    libraries.update(log_config.get("libraries", {}))
    
    # Set log levels
    for lib, level in libraries.items():
        lib_logger = logging.getLogger(lib)
        lib_logger.setLevel(getattr(logging, level.upper()))
        
        # Don't propagate if set to higher level than root
        root_level = logging.getLogger("casalingua").level
        lib_level = getattr(logging, level.upper())
        if lib_level > root_level:
            lib_logger.propagate = False
            
            # Add handler if not propagating
            if not lib_logger.handlers:
                handler = logging.StreamHandler()
                handler.setFormatter(ConsoleFormatter())
                lib_logger.addHandler(handler)

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a specific module.
    
    Args:
        name: Module name or path
        
    Returns:
        Logger for the module
    """
    # Handle relative module names
    if name.startswith("__"):
        # Get calling frame
        import inspect
        frame = inspect.currentframe().f_back
        module = inspect.getmodule(frame)
        name = module.__name__
        
    # Prepend casalingua prefix if not already present
    if not name.startswith("casalingua.") and not name == "casalingua":
        name = f"casalingua.{name}"
        
    return logging.getLogger(name)

def set_request_context(
    request_id: str, 
    user_id: Optional[str] = None,
    endpoint: Optional[str] = None,
    ip_address: Optional[str] = None,
    **additional_context
) -> None:
    """
    Set the current request context in thread-local storage.
    
    Args:
        request_id: Unique request identifier
        user_id: User identifier (if authenticated)
        endpoint: API endpoint path
        ip_address: Client IP address
        additional_context: Additional context values
    """
    _thread_local.request_context = {
        "request_id": request_id,
        "user_id": user_id,
        "endpoint": endpoint,
        "ip_address": ip_address,
        **additional_context
    }

def clear_request_context() -> None:
    """Clear the current request context from thread-local storage."""
    if hasattr(_thread_local, "request_context"):
        delattr(_thread_local, "request_context")

class RequestContextMiddleware:
    """
    Middleware for setting request context in logs.
    
    This middleware sets request context information in thread-local
    storage for use by the logging system.
    """
    
    async def __call__(self, request, call_next):
        """
        Process a request and set logging context.
        
        Args:
            request: FastAPI request
            call_next: Next middleware handler
            
        Returns:
            FastAPI response
        """
        # Generate request ID if not set
        if not hasattr(request.state, "request_id"):
            import uuid
            request.state.request_id = str(uuid.uuid4())
            
        # Get user ID if available
        user_id = None
        if hasattr(request.state, "user"):
            user_id = request.state.user.get("id")
            
        # Set request context
        set_request_context(
            request_id=request.state.request_id,
            user_id=user_id,
            endpoint=request.url.path,
            ip_address=request.client.host if request.client else None,
            method=request.method
        )
        
        try:
            # Process request
            response = await call_next(request)
            return response
        finally:
            # Clear request context
            clear_request_context()

class LogCapture:
    """
    Context manager for capturing logs during a specific operation.
    
    Example:
        ```
        with LogCapture("operation_name") as logs:
            # Do something
            
        # Access captured logs
        for log in logs.records:
            print(log.message)
        ```
    """
    
    def __init__(self, operation_name: str, level: int = logging.DEBUG):
        """
        Initialize log capture.
        
        Args:
            operation_name: Name of the operation
            level: Minimum log level to capture
        """
        self.operation_name = operation_name
        self.level = level
        self.records = []
        self.handler = None
        
    def __enter__(self):
        """Start capturing logs."""
        # Create handler that captures logs
        class CapturingHandler(logging.Handler):
            def __init__(self, records):
                super().__init__()
                self.records = records
                
            def emit(self, record):
                self.records.append(record)
                
        # Set up handler
        self.handler = CapturingHandler(self.records)
        self.handler.setLevel(self.level)
        
        # Add handler to root logger
        logger = logging.getLogger("casalingua")
        logger.addHandler(self.handler)
        
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop capturing logs."""
        # Remove handler
        if self.handler:
            logger = logging.getLogger("casalingua")
            logger.removeHandler(self.handler)
            
    def get_messages(self) -> List[str]:
        """
        Get list of captured log messages.
        
        Returns:
            List of log messages
        """
        return [record.getMessage() for record in self.records]
        
    def get_formatted_logs(self) -> str:
        """
        Get formatted string of all captured logs.
        
        Returns:
            Formatted log string
        """
        formatter = ConsoleFormatter(use_colors=False)
        return "\n".join(formatter.format(record) for record in self.records)

class PerformanceLogger:
    """
    Utility for logging performance metrics.
    
    Example:
        ```
        perf_logger = PerformanceLogger("database_query")
        with perf_logger:
            # Perform database query
            
        # Access timing
        print(f"Query took {perf_logger.duration:.2f}s")
        ```
    """
    
    def __init__(
        self,
        operation_name: str,
        logger: Optional[logging.Logger] = None,
        threshold: Optional[float] = None
    ):
        """
        Initialize performance logger.
        
        Args:
            operation_name: Name of the operation
            logger: Logger to use (defaults to module logger)
            threshold: Log level threshold in seconds
        """
        self.operation_name = operation_name
        self.logger = logger or _logger
        self.threshold = threshold
        self.start_time = None
        self.end_time = None
        self.duration = None
        
    def __enter__(self):
        """Start timing."""
        self.start_time = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """End timing and log."""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        
        # Log based on threshold
        if self.threshold is not None:
            if self.duration >= self.threshold:
                self.logger.warning(
                    f"Performance: {self.operation_name} took {self.duration:.4f}s "
                    f"(exceeded threshold of {self.threshold:.4f}s)"
                )
            else:
                self.logger.debug(
                    f"Performance: {self.operation_name} took {self.duration:.4f}s"
                )
        else:
            self.logger.info(f"Performance: {self.operation_name} took {self.duration:.4f}s")

def initialize_logging(app=None):
    """
    Initialize logging for the application.
    
    Args:
        app: FastAPI application (optional)
    """
    # Get configuration
    config = load_config()
    
    # Set up logging
    logger = setup_logging(config)
    
    # Set up application state if provided
    if app:
        # Store logger in app state
        app.state.logger = logger
        
        # Set up middleware
        from fastapi import FastAPI
        if isinstance(app, FastAPI):
            middleware_config = config.get("logging", {}).get("middleware", {})
            
            # Add middleware for request context
            if middleware_config.get("request_context", True):
                app.middleware("http")(RequestContextMiddleware())
                
    return logger