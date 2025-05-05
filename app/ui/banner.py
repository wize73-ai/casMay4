# app/ui/banner.py
"""
Banner and Status Display for CasaLingua

This module provides functions for displaying attractive banners
and status information in the terminal.
"""

import platform
import psutil
import shutil
from datetime import datetime
from typing import Dict, Any, Optional
from colorama import Fore, Back, Style

from app.ui.colors import colors

def get_theme_chars(theme: str = "default"):
    return {
        "tl": "╔" if theme == "default" else "+",
        "tr": "╗" if theme == "default" else "+",
        "bl": "╚" if theme == "default" else "+",
        "br": "╝" if theme == "default" else "+",
        "horiz": "═" if theme == "default" else "-",
        "vert": "║" if theme == "default" else "|"
    }

def print_startup_banner(version: str = "1.0.0", cli_mode: bool = False, theme: str = "default") -> None:
    """
    Print a beautiful startup banner for CasaLingua.
    
    Args:
        version: The version string to display
        cli_mode: Whether the application is running in CLI mode
        theme: The theme style to use ("default" or "ascii")
    """
    term_width = min(shutil.get_terminal_size().columns, 100)
    chars = get_theme_chars(theme)
    banner_width = term_width - 4  # Allow for consistent padding
    horiz_line = chars["horiz"] * banner_width

    # Create the banner with proper alignment
    banner = f"\n{chars['tl']}{horiz_line}{chars['tr']}\n"
    empty_line = f"{chars['vert']}{' ' * banner_width}{chars['vert']}\n"
    
    # ASCII logo with consistent padding
    banner += empty_line
    banner += f"{chars['vert']}   {colors.TITLE}   _____                _      _                          {colors.RESET}                                   {chars['vert']}\n"
    banner += f"{chars['vert']}   {colors.TITLE}  / ____|              | |    (_)                         {colors.RESET}                                   {chars['vert']}\n"
    banner += f"{chars['vert']}   {colors.TITLE} | |     __ _ ___  __ _| |     _ _ __   __ _ _   _  __ _  {colors.RESET}                                   {chars['vert']}\n"
    banner += f"{chars['vert']}   {colors.TITLE} | |    / _` / __|/ _` | |    | | '_ \\ / _` | | | |/ _` | {colors.RESET}                                   {chars['vert']}\n"
    banner += f"{chars['vert']}   {colors.TITLE} | |___| (_| \\__ \\ (_| | |____| | | | | (_| | |_| | (_| | {colors.RESET}                                   {chars['vert']}\n"
    banner += f"{chars['vert']}   {colors.TITLE}  \\_____\\__,_|___/\\__,_|______|_|_| |_|\\__, |\\__,_|\\__,_| {colors.RESET}                                   {chars['vert']}\n"                             
    banner += f"{chars['vert']}   {colors.TITLE}                                        |___/             {colors.RESET}                                   {chars['vert']}\n"
    banner += empty_line
    
    # Additional information with consistent padding
    subtitle = "Language Processing & Translation Pipeline"
    subtitle_padding = banner_width - len(subtitle) - 6  # 6 for the padding (3 on each side)
    banner += f"{chars['vert']}   {colors.SUBTITLE}{subtitle}{' ' * subtitle_padding}{colors.RESET}   {chars['vert']}\n"
    
    version_str = f"Version: {version}"
    version_padding = banner_width - len(version_str) - 6
    banner += f"{chars['vert']}   {colors.VALUE}{version_str}{' ' * version_padding}{colors.RESET}   {chars['vert']}\n"

    # Add mode indicator if in CLI mode
    if cli_mode:
        mode_str = "Mode: CLI Interactive"
        mode_padding = banner_width - len(mode_str) - 6
        banner += f"{chars['vert']}   {colors.HIGHLIGHT}{mode_str}{' ' * mode_padding}{colors.RESET}   {chars['vert']}\n"
    
    banner += empty_line
    banner += f"{chars['bl']}{horiz_line}{chars['br']}\n"
    
    print(banner)

def print_system_info(hardware_info: Dict[str, Any], theme: str = "default") -> None:
    """
    Print system information in a visually appealing format.
    
    Args:
        hardware_info: Hardware information dictionary
        theme: The theme style to use ("default" or "ascii")
    """
    chars = get_theme_chars(theme)
    term_width = min(shutil.get_terminal_size().columns, 100)
    box_width = term_width - 2
    horiz_line = chars["horiz"] * (box_width - 2)
    empty_line = f"{chars['vert']}{' ' * (box_width - 2)}{chars['vert']}"

    # Extract relevant hardware information
    cpu_info = hardware_info.get("cpu", {})
    gpu_info = hardware_info.get("gpu", {})
    memory_info = hardware_info.get("memory", {})
    
    system_info = f"\n{chars['tl']}─ {colors.HEADER}System Information{colors.RESET} {chars['horiz']}───────────────────────────────────────────────────{chars['tr']}\n"
    system_info += empty_line + "\n"
    platform_str = f"Platform:      {platform.system()} {platform.release()}"
    system_info += f"{chars['vert']}  {colors.MODULE}{platform_str.ljust(box_width - 4)}{colors.RESET}{chars['vert']}\n"
    python_str = f"Python:        {platform.python_version()}"
    system_info += f"{chars['vert']}  {colors.MODULE}{python_str.ljust(box_width - 4)}{colors.RESET}{chars['vert']}\n"
    cpu_str = f"CPU:           {cpu_info.get('model', 'Unknown')} ({cpu_info.get('cores', 0)} cores)"
    system_info += f"{chars['vert']}  {colors.MODULE}{cpu_str.ljust(box_width - 4)}{colors.RESET}{chars['vert']}\n"
    mem_str = f"Memory:        {memory_info.get('total_gb', 0):.1f} GB total, {memory_info.get('available_gb', 0):.1f} GB available"
    system_info += f"{chars['vert']}  {colors.MODULE}{mem_str.ljust(box_width - 4)}{colors.RESET}{chars['vert']}\n"

    # Add GPU information if available
    if gpu_info.get("available", False):
        gpu_str = f"GPU:           {gpu_info.get('model', 'Unknown GPU')}"
        system_info += f"{chars['vert']}  {colors.MODULE}{gpu_str.ljust(box_width - 4)}{colors.RESET}{chars['vert']}\n"
        gpu_mem_str = f"GPU Memory:    {gpu_info.get('memory_gb', 0):.1f} GB"
        system_info += f"{chars['vert']}  {colors.MODULE}{gpu_mem_str.ljust(box_width - 4)}{colors.RESET}{chars['vert']}\n"
    else:
        gpu_str = f"GPU:           {colors.MUTED}Not available{colors.RESET}"
        system_info += f"{chars['vert']}  {colors.MODULE}{gpu_str.ljust(box_width - 4)}{colors.RESET}{chars['vert']}\n"

    system_info += empty_line + "\n"
    system_info += f"{chars['bl']}{chars['horiz'] * (box_width - 2)}{chars['br']}\n"
    
    print(system_info)

def print_model_status(model_info: Dict[str, Any], theme: str = "default") -> None:
    """
    Print model status in a visually appealing format.
    
    Args:
        model_info: Model information dictionary
        theme: The theme style to use ("default" or "ascii")
    """
    chars = get_theme_chars(theme)
    term_width = min(shutil.get_terminal_size().columns, 100)
    box_width = term_width - 2
    horiz_line = chars["horiz"] * (box_width - 2)
    empty_line = f"{chars['vert']}{' ' * (box_width - 2)}{chars['vert']}"

    model_status = f"\n{chars['tl']}─ {colors.HEADER}Model Status{colors.RESET} {chars['horiz']}──────────────────────────────────────────────────────{chars['tr']}\n"
    model_status += empty_line + "\n"

    # Display model information
    for model_type, model_data in model_info.items():
        status_icon = f"{colors.SUCCESS}✓{colors.RESET}" if model_data.get("loaded", False) else f"{colors.ERROR}✗{colors.RESET}"
        model_name = model_data.get("name", "Unknown")
        model_device = model_data.get("device", "cpu")

        left_part = f"{model_type.ljust(15)} {status_icon} {model_name.ljust(30)}"
        right_part = f"[{model_device}]"
        line_content = f"  {colors.MODEL}{left_part}{colors.RESET} {colors.VALUE}{right_part}{colors.RESET}"
        # Pad line_content to box_width - 2
        line_padded = line_content.ljust(box_width - 2)
        model_status += f"{chars['vert']}{line_padded}{chars['vert']}\n"
    
    model_status += empty_line + "\n"
    model_status += f"{chars['bl']}{chars['horiz'] * (box_width - 2)}{chars['br']}\n"
    
    print(model_status)

def print_ready_message(theme: str = "default") -> None:
    """Print an attractive 'ready' message."""
    chars = get_theme_chars(theme)
    term_width = min(shutil.get_terminal_size().columns, 100)
    box_width = term_width - 2
    horiz_line = chars["horiz"] * (box_width)
    empty_line = f"{chars['vert']}{' ' * (box_width)}{chars['vert']}"

    ready_msg = f"\n{chars['tl']}{horiz_line}{chars['tr']}\n"
    ready_msg += empty_line + "\n"
    msg = "✓ CasaLingua API is ready to accept connections!"
    content_line = f"{chars['vert']}  {colors.SUCCESS}{msg}{colors.RESET}{' ' * (box_width - len(msg) - 2)}{chars['vert']}\n"
    ready_msg += content_line
    ready_msg += empty_line + "\n"
    ready_msg += f"{chars['bl']}{horiz_line}{chars['br']}\n"
    print(ready_msg)

def print_endpoint_list(endpoints: Dict[str, str], theme: str = "default") -> None:
    """
    Print list of available endpoints.
    
    Args:
        endpoints: Dictionary mapping endpoint paths to descriptions
        theme: The theme style to use ("default" or "ascii")
    """
    chars = get_theme_chars(theme)
    term_width = min(shutil.get_terminal_size().columns, 100)
    box_width = term_width - 2
    horiz_line = chars["horiz"] * (box_width - 2)
    empty_line = f"{chars['vert']}{' ' * (box_width - 2)}{chars['vert']}"

    header_title = "Available Endpoints"
    endpoint_list = f"\n{chars['tl']}─ {colors.HEADER}{header_title}{colors.RESET} {chars['horiz']}──────────────────────────────────────────────{chars['tr']}\n"
    endpoint_list += empty_line + "\n"

    for path, description in endpoints.items():
        left = f"{path.ljust(25)}"
        right = f"{description}"
        content = f"  {colors.SUCCESS}{left}{colors.RESET} {colors.VALUE}{right}{colors.RESET}"
        content_line = content.ljust(box_width - 2)
        endpoint_list += f"{chars['vert']}{content_line}{chars['vert']}\n"
    
    endpoint_list += empty_line + "\n"
    endpoint_list += f"{chars['bl']}{chars['horiz'] * (box_width - 2)}{chars['br']}\n"
    
    print(endpoint_list)

def print_section_header(title: str) -> None:
    """
    Print a section header with consistent styling.
    
    Args:
        title: The section title
    """
    header = f"{colors.HEADER}{title}{colors.RESET}"
    separator = f"{colors.BORDER_LIGHT}{'─' * (len(title) + 10)}{colors.RESET}"
    
    print(f"\n{separator}")
    print(f"{colors.BORDER_LIGHT}──┤{colors.RESET} {header} {colors.BORDER_LIGHT}├──{colors.RESET}")
    print(f"{separator}\n")