# app/ui/console.py
"""
Console Logging Utilities for CasaLingua

This module provides console logging setup with colorful output
and proper formatting for terminal display.
"""

import sys
import logging
import platform
from logging.handlers import RotatingFileHandler
from datetime import datetime
from typing import Optional, Any, List, Dict, Union, Callable

from colorama import Fore, Back, Style
from colorama import init
init(autoreset=True)
from app.ui.colors import colors

class ColoredFormatter(logging.Formatter):
    """Custom formatter for colored terminal output."""
    
    LEVEL_COLORS = {
        'DEBUG': colors.INFO,
        'INFO': colors.SUCCESS,
        'WARNING': colors.WARNING,
        'ERROR': colors.ERROR,
        'CRITICAL': colors.CRITICAL,
    }
    
    LEVEL_ICONS = {
        'DEBUG': '🔍',
        'INFO': '✓',
        'WARNING': '⚠️',
        'ERROR': '❌',
        'CRITICAL': '🚨'
    }
    
    def format(self, record):
        orig_levelname = record.levelname
        orig_module = getattr(record, "module", "unknown")

        levelname = record.levelname
        color = self.LEVEL_COLORS.get(levelname, "")
        icon = self.LEVEL_ICONS.get(levelname, "")
        reset = Style.RESET_ALL

        record.levelname = f"{icon} {levelname}"
        record.module = f"{colors.MODULE}{orig_module}{reset}"

        # Format the full log line
        line = super().format(record)

        # Apply color to the whole line
        result = f"{color}{line}{reset}"

        # Restore original values
        record.levelname = orig_levelname
        record.module = orig_module

        return result

class Console:
    """
    Enhanced console interface for CasaLingua.
    
    This class provides methods for displaying formatted text
    in the terminal with consistent styling and colors.
    """
    
    def __init__(self, width: int = 80):
        """
        Initialize the console interface.
        
        Args:
            width: Terminal width in characters
        """
        self.width = width
        self.logger = None
        
        # Try to detect terminal size
        try:
            import shutil
            terminal_size = shutil.get_terminal_size((width, 24))
            self.width = terminal_size.columns
        except:
            # Use default if detection fails
            pass
    
    def set_logger(self, logger: logging.Logger) -> None:
        """
        Set the logger for this console instance.
        
        Args:
            logger: Logger instance to use
        """
        self.logger = logger
    
    def print(self, message: str, color: str = None) -> None:
        """
        Print a message to the console with optional color.
        
        Args:
            message: Message to print
            color: Optional color from colors module
        """
        if color:
            message = f"{color}{message}{colors.RESET}"
        
        if self.logger:
            self.logger.info(message)
        else:
            sys.stdout.write(message + "\n")
            sys.stdout.flush()
    
    def print_header(self, title: str, subtitle: Optional[str] = None) -> None:
        """
        Print a section header with optional subtitle.
        
        Args:
            title: Header title
            subtitle: Optional subtitle
        """
        header = f"\n{colors.BORDER_LIGHT}{'═' * self.width}{colors.RESET}"
        header += f"\n{colors.HEADER}{title}{colors.RESET}"
        
        if subtitle:
            header += f"\n{colors.SUBTITLE}{subtitle}{colors.RESET}"
            
        header += f"\n{colors.BORDER_LIGHT}{'═' * self.width}{colors.RESET}\n"
        
        self.print(header)
    
    def print_success(self, message: str) -> None:
        """
        Print a success message.
        
        Args:
            message: Success message
        """
        self.print(f"{colors.SUCCESS}✓ {message}{colors.RESET}")
    
    def print_error(self, message: str) -> None:
        """
        Print an error message.
        
        Args:
            message: Error message
        """
        self.print(f"{colors.ERROR}❌ {message}{colors.RESET}")
    
    def print_warning(self, message: str) -> None:
        """
        Print a warning message.
        
        Args:
            message: Warning message
        """
        self.print(f"{colors.WARNING}⚠️ {message}{colors.RESET}")
    
    def print_info(self, message: str) -> None:
        """
        Print an info message.
        
        Args:
            message: Info message
        """
        self.print(f"{colors.INFO}ℹ️ {message}{colors.RESET}")
    
    def print_table(self, headers: List[str], rows: List[List[str]]) -> None:
        """
        Print a formatted table.
        
        Args:
            headers: List of column headers
            rows: List of rows, each a list of column values
        """
        # Calculate column widths
        col_widths = [len(h) for h in headers]
        for row in rows:
            if len(row) != len(headers):
                continue
            for i, cell in enumerate(row):
                if i < len(col_widths):
                    col_widths[i] = max(col_widths[i], len(str(cell)))
        
        # Add padding
        col_widths = [w + 2 for w in col_widths]
        
        # Create line separator
        separator = f"{colors.BORDER}+"
        for width in col_widths:
            separator += f"{'-' * width}+"
        separator += f"{colors.RESET}"
        
        # Print header row
        self.print(separator)
        header_row = f"{colors.BORDER}|{colors.RESET}"
        for i, header in enumerate(headers):
            header_row += f"{colors.HEADER}{header.ljust(col_widths[i])}{colors.RESET}{colors.BORDER}|{colors.RESET}"
        self.print(header_row)
        self.print(separator)
        
        # Print data rows
        for row in rows:
            if len(row) != len(headers):
                continue
            data_row = f"{colors.BORDER}|{colors.RESET}"
            for i, cell in enumerate(row):
                if i < len(col_widths):
                    data_row += f"{colors.VALUE}{str(cell).ljust(col_widths[i])}{colors.RESET}{colors.BORDER}|{colors.RESET}"
            self.print(data_row)
        
        self.print(separator)
    
    def print_progress_bar(self, current: int, total: int, prefix: str = '', suffix: str = '', 
                         length: int = 50, fill: str = '█') -> None:
        """
        Print a progress bar.
        
        Args:
            current: Current progress value
            total: Total value
            prefix: Text before the bar
            suffix: Text after the bar
            length: Bar length in characters
            fill: Character to use for filled portion
        """
        percent = f"{100 * (current / float(total)):.1f}"
        filled_length = int(length * current // total)
        bar = colors.PROGRESS + fill * filled_length + colors.RESET + '-' * (length - filled_length)
        
        progress_bar = f"\r{prefix} |{bar}| {percent}% {suffix}"
        sys.stdout.write(progress_bar)
        sys.stdout.flush()
        
        # Print new line if complete
        if current == total:
            print()
    
    def print_dictionary(self, data: Dict[str, Any], title: Optional[str] = None) -> None:
        """
        Print a dictionary with nice formatting.
        
        Args:
            data: Dictionary to print
            title: Optional title
        """
        if title:
            self.print(f"\n{colors.HEADER}{title}{colors.RESET}")
            self.print(f"{colors.BORDER_LIGHT}{'-' * len(title)}{colors.RESET}\n")
        
        for key, value in data.items():
            key_str = f"{colors.MODULE}{key}{colors.RESET}"
            
            # Format value based on type
            if isinstance(value, dict):
                value_str = "{...}"  # Indicate nested dictionary
            elif isinstance(value, list):
                value_str = f"[{len(value)} items]"  # Indicate list length
            else:
                value_str = f"{colors.VALUE}{value}{colors.RESET}"
                
            self.print(f"  {key_str}: {value_str}")


def setup_console_logging(level: str = "INFO") -> logging.Logger:
    """
    Set up console logging with colored output.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Returns:
        logging.Logger: Configured logger
    """
    # Set up the logger
    logger = logging.getLogger("casalingua")
    if logger.hasHandlers():
        logger.handlers.clear()
    # Ensure level is a string and valid for logging
    safe_level = getattr(logging, str(level).upper(), logging.INFO)
    logger.setLevel(safe_level)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(safe_level)
    
    # Create formatter
    console_format = "%(asctime)s - %(module)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    formatter = ColoredFormatter(console_format, datefmt=date_format)
    
    # Add formatter to handler
    console_handler.setFormatter(formatter)
    
    # Clean up and assign our handler to all major loggers
    for name in ("", "root", "casalingua", "uvicorn", "uvicorn.error", "uvicorn.access"):
        target_logger = logging.getLogger(name)
        target_logger.handlers.clear()
        target_logger.setLevel(safe_level)
        target_logger.propagate = False
        target_logger.addHandler(console_handler)
    return logger

def setup_file_logging(logger: logging.Logger, filename: str, level: str = "INFO", 
                      max_bytes: int = 10485760, backup_count: int = 5) -> None:
    """
    Add file logging to an existing logger.
    
    Args:
        logger: The logger to add file logging to
        filename: Path to log file
        level: Logging level
        max_bytes: Maximum log file size before rotation
        backup_count: Number of backup files to keep
    """
    # Create file handler
    file_handler = RotatingFileHandler(
        filename,
        maxBytes=max_bytes,
        backupCount=backup_count
    )
    file_handler.setLevel(getattr(logging, level))
    
    # Create formatter (without colors for file output)
    file_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(file_format, datefmt=date_format)
    
    # Add formatter to handler
    file_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(file_handler)

def log_section_start(logger: logging.Logger, section_name: str) -> None:
    """
    Log the start of a logical section with visual delimiter.
    
    Args:
        logger: Logger to use
        section_name: Name of the section
    """
    logger.info(f"{colors.BORDER_LIGHT}{'═' * 80}{colors.RESET}")
    logger.info(f"{colors.HEADER}▶ STARTING SECTION: {section_name}{colors.RESET}")
    logger.info(f"{colors.BORDER_LIGHT}{'═' * 80}{colors.RESET}")

def log_section_end(logger: logging.Logger, section_name: str) -> None:
    """
    Log the end of a logical section with visual delimiter.
    
    Args:
        logger: Logger to use
        section_name: Name of the section
    """
    logger.info(f"{colors.BORDER_LIGHT}{'─' * 80}{colors.RESET}")
    logger.info(f"{colors.HEADER}✓ COMPLETED SECTION: {section_name}{colors.RESET}")
    logger.info(f"{colors.BORDER_LIGHT}{'─' * 80}{colors.RESET}")

def log_processing_start(logger: logging.Logger, process_name: str, request_id: Optional[str] = None) -> None:
    """
    Log the start of a processing operation.
    
    Args:
        logger: Logger to use
        process_name: Name of the process
        request_id: Optional request ID
    """
    request_info = f" [Request: {request_id}]" if request_id else ""
    logger.info(f"{colors.PIPELINE}▶ Processing{request_info}: {process_name}{colors.RESET}")

def log_processing_complete(logger: logging.Logger, process_name: str, 
                          time_taken: float, request_id: Optional[str] = None) -> None:
    """
    Log the completion of a processing operation.
    
    Args:
        logger: Logger to use
        process_name: Name of the process
        time_taken: Time taken in seconds
        request_id: Optional request ID
    """
    request_info = f" [Request: {request_id}]" if request_id else ""
    logger.info(f"{colors.PIPELINE}✓ Completed{request_info}: {process_name} ({time_taken:.3f}s){colors.RESET}")

def log_with_color(message: str, color: str = None, logger: Optional[logging.Logger] = None) -> None:
    """
    Log a message with the specified color.
    
    Args:
        message: The message to log
        color: Color code from colors module (defaults to colors.INFO)
        logger: Optional logger to use (if None, prints to stdout)
    """
    # Use the specified color or default to INFO color
    color_code = color if color is not None else colors.INFO
    
    # Format the message with color
    colored_message = f"{color_code}{message}{colors.RESET}"
    
    # If logger is provided, use it
    if logger:
        logger.info(colored_message)
    else:
        # Otherwise, print directly to stdout
        print(colored_message)
from rich.console import Console as RichConsole

# Export a global rich.console.Console instance for global usage
console = RichConsole()