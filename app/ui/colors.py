# app/ui/colors.py

"""
Terminal Color Utilities for CasaLingua

This module provides color utilities for terminal output, ensuring consistent
and visually appealing console presentation across the application.
"""

import os
import platform
from colorama import Fore, Back, Style, init

# Define color constants for use throughout the application
class TerminalColors:
    """Terminal color constants."""
    # Message levels
    INFO = Fore.BLUE
    SUCCESS = Fore.GREEN
    WARNING = Fore.YELLOW
    ERROR = Fore.RED
    CRITICAL = Fore.RED + Style.BRIGHT

    # UI sections
    HEADER = Fore.CYAN + Style.BRIGHT
    TITLE = Fore.GREEN + Style.BRIGHT
    SUBTITLE = Fore.CYAN
    VALUE = Fore.WHITE
    HIGHLIGHT = Fore.YELLOW + Style.BRIGHT
    MUTED = Style.DIM

    # Progress indicators
    PROGRESS = Fore.MAGENTA
    PROGRESS_BAR = Fore.GREEN
    PROGRESS_BAR_BG = Back.BLACK

    # Components
    MODULE = Fore.BLUE
    PIPELINE = Fore.MAGENTA
    MODEL = Fore.CYAN

    # Borders
    BORDER = Fore.CYAN
    BORDER_LIGHT = Fore.BLUE
    BORDER_ACCENT = Fore.GREEN

    # Reset
    RESET = Style.RESET_ALL

# Create global instance
colors = TerminalColors()

def supports_color() -> bool:
    """
    Determine if the current terminal supports color output.
    Returns:
        bool: True if color is supported, False otherwise.
    """
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True

    plat = platform.system().lower()

    if plat == "windows":
        if int(platform.release()) >= 10 and int(platform.version().split('.')[-1]) >= 14931:
            return True
        if os.environ.get("WT_SESSION") or os.environ.get("TERM_PROGRAM"):
            return True

    if os.environ.get("TERM"):
        return True

    return True

def init_terminal_colors() -> None:
    """
    Initialize terminal color support and configure colorama accordingly.
    """
    # Check environment variable that can force color support
    if os.environ.get("FORCE_COLOR"):
        strip_colors = False
        os.environ.pop('NO_COLOR', None)  # Clear any NO_COLOR setting
        print("[CasaLingua UI] Terminal color output forced by FORCE_COLOR")
    else:
        strip_colors = not supports_color()
    
    # Initialize colorama with appropriate settings
    init(autoreset=True, strip=strip_colors, convert=True)
    
    if strip_colors and not os.environ.get("FORCE_COLOR"):
        print("[CasaLingua UI] Terminal color output disabled (NO_COLOR detected)")

def colored_text(text: str, color_code: str, style: str = None) -> str:
    """
    Apply color and style to text.
    Args:
        text (str): The text to color.
        color_code (str): A colorama.Fore or Back code.
        style (str): An optional colorama.Style.
    Returns:
        str: Formatted string with color codes applied.
    """
    style_code = style or ""
    return f"{style_code}{color_code}{text}{Style.RESET_ALL}"

def info(text: str) -> str:
    """Format text as info message."""
    return colored_text(text, colors.INFO)

def success(text: str) -> str:
    """Format text as success message."""
    return colored_text(text, colors.SUCCESS)

def warning(text: str) -> str:
    """Format text as warning message."""
    return colored_text(text, colors.WARNING)

def error(text: str) -> str:
    """Format text as error message."""
    return colored_text(text, colors.ERROR, Style.BRIGHT)

def highlight(text: str) -> str:
    """Format text as highlighted output."""
    return colored_text(text, colors.HIGHLIGHT, Style.BRIGHT)