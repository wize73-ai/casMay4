# app/ui/teacher.py
"""
Enhanced Terminal Output for Teaching with CasaLingua

This module provides advanced terminal visualization capabilities
designed to present information in a highly readable, colorful format
that's excellent for teaching and demonstration purposes.

Author: Exygy Development Team
Version: 1.0.0
License: MIT
"""

import os
import sys
import time
from typing import Dict, List, Any, Optional, Union, Tuple
from colorama import Fore, Back, Style

from app.ui.colors import colors

class TerminalTeacher:
    """
    Enhanced terminal output for teaching and demonstrations.
    
    Features:
    - Multi-column text display
    - Animated text reveals
    - Syntax highlighting for multiple languages
    - Side-by-side comparisons
    - Progress visualization
    - Interactive examples
    """
    
    def __init__(self, width: int = 80, height: int = 24, animation_speed: float = 0.01):
        """
        Initialize the terminal teacher.
        
        Args:
            width: Terminal width in characters
            height: Terminal height in characters
            animation_speed: Animation speed in seconds per step
        """
        self.width = width
        self.height = height
        self.animation_speed = animation_speed
        
        # Try to detect terminal size
        try:
            import shutil
            terminal_size = shutil.get_terminal_size((width, height))
            self.width = terminal_size.columns
            self.height = terminal_size.lines
        except Exception as e:
            # Use default values if detection fails
            import logging
            logging.warning(f"Failed to detect terminal size, using defaults: {e}")
            
    def print_header(self, title: str, subtitle: Optional[str] = None) -> None:
        """
        Print a colorful header with optional subtitle.
        
        Args:
            title: Header title
            subtitle: Optional subtitle
        """
        title_display = f"  {title}  "
        padding = max(0, (self.width - len(title_display)) // 2)
        
        # Print top border
        print(f"{colors.BORDER}{'═' * self.width}{colors.RESET}")
        
        # Print title
        print(f"{colors.BORDER}║{' ' * padding}{colors.TITLE}{title_display}{colors.RESET}{' ' * padding}{colors.BORDER}║{colors.RESET}")
        
        # Print subtitle if provided
        if subtitle:
            subtitle_padding = max(0, (self.width - len(subtitle) - 4) // 2)
            print(f"{colors.BORDER}║{' ' * subtitle_padding}{colors.SUBTITLE}{subtitle}{colors.RESET}{' ' * subtitle_padding}{colors.BORDER}║{colors.RESET}")
        
        # Print bottom border
        print(f"{colors.BORDER}{'═' * self.width}{colors.RESET}")
        
    def print_comparison(self, 
                       original: str, 
                       modified: str, 
                       title_left: str = "Original", 
                       title_right: str = "Modified") -> None:
        """
        Print a side-by-side comparison of two texts.
        
        Args:
            original: Original text
            modified: Modified text
            title_left: Title for left column
            title_right: Title for right column
        """
        # Split texts into lines
        original_lines = original.split('\n')
        modified_lines = modified.split('\n')
        
        # Calculate column width (accounting for separator)
        col_width = (self.width - 3) // 2
        
        # Print header
        print(f"{colors.BORDER}╔{'═' * col_width}╦{'═' * col_width}╗{colors.RESET}")
        print(f"{colors.BORDER}║{colors.HEADER}{title_left.center(col_width)}{colors.RESET}{colors.BORDER}║{colors.HEADER}{title_right.center(col_width)}{colors.RESET}{colors.BORDER}║{colors.RESET}")
        print(f"{colors.BORDER}╠{'═' * col_width}╬{'═' * col_width}╣{colors.RESET}")
        
        # Determine number of lines to display
        max_lines = max(len(original_lines), len(modified_lines))
        
        # Print content
        for i in range(max_lines):
            left = original_lines[i] if i < len(original_lines) else ""
            right = modified_lines[i] if i < len(modified_lines) else ""
            
            # Truncate if longer than column width
            if len(left) > col_width:
                left = left[:col_width-3] + "..."
            if len(right) > col_width:
                right = right[:col_width-3] + "..."
                
            # Pad to fill column width
            left = left.ljust(col_width)
            right = right.ljust(col_width)
            
            print(f"{colors.BORDER}║{colors.RESET}{left}{colors.BORDER}║{colors.RESET}{right}{colors.BORDER}║{colors.RESET}")
        
        # Print footer
        print(f"{colors.BORDER}╚{'═' * col_width}╩{'═' * col_width}╝{colors.RESET}")
        
    def print_syntax_highlighted(self, code: str, language: str) -> None:
        """
        Print code with syntax highlighting.
        
        Args:
            code: Code to highlight
            language: Programming language
        """
        # Check if pygments is available for syntax highlighting
        try:
            from pygments import highlight
            from pygments.lexers import get_lexer_by_name
            from pygments.formatters import Terminal256Formatter
            
            lexer = get_lexer_by_name(language, stripall=True)
            formatter = Terminal256Formatter(style='monokai')
            highlighted = highlight(code, lexer, formatter)
            
            print(highlighted)
            
        except ImportError:
            # Fall back to simple highlighting if pygments isn't available
            print(f"{colors.MODULE}# {language.upper()} CODE{colors.RESET}")
            print(f"{colors.VALUE}{code}{colors.RESET}")
            
    def print_animated_reveal(self, text: str, delay: float = None) -> None:
        """
        Print text with a typewriter-style animation.
        
        Args:
            text: Text to reveal
            delay: Delay between characters in seconds
        """
        delay = delay or self.animation_speed
        
        for char in text:
            sys.stdout.write(char)
            sys.stdout.flush()
            time.sleep(delay)
        
        print()
        
    def print_progress_steps(self, 
                          steps: List[str], 
                          current_step: int, 
                          completed_steps: List[int] = None) -> None:
        """
        Print progress through a series of steps.
        
        Args:
            steps: List of step descriptions
            current_step: Current step index (0-based)
            completed_steps: List of completed step indices
        """
        completed_steps = completed_steps or []
        max_step_length = max(len(step) for step in steps)
        
        for i, step in enumerate(steps):
            if i < current_step or i in completed_steps:
                # Completed step
                status = f"{colors.SUCCESS}✓{colors.RESET}"
                step_text = f"{colors.SUCCESS}{step}{colors.RESET}"
            elif i == current_step:
                # Current step
                status = f"{colors.HIGHLIGHT}▶{colors.RESET}"
                step_text = f"{colors.HIGHLIGHT}{step}{colors.RESET}"
            else:
                # Future step
                status = f"{colors.MUTED}○{colors.RESET}"
                step_text = f"{colors.MUTED}{step}{colors.RESET}"
                
            # Print step with padding
            padding = max_step_length - len(step)
            print(f"  {status} {step_text}{' ' * padding}")
            
            # Add connector line between steps
            if i < len(steps) - 1:
                if i < current_step or i in completed_steps:
                    print(f"  {colors.SUCCESS}│{colors.RESET}")
                else:
                    print(f"  {colors.MUTED}│{colors.RESET}")
                    
    def print_language_example(self, 
                            text: str, 
                            translation: str = None, 
                            pronunciation: str = None,
                            notes: List[Tuple[int, str]] = None) -> None:
        """
        Print a language example with translation and pronunciation.
        
        Args:
            text: Example text
            translation: Optional translation
            pronunciation: Optional pronunciation guide
            notes: Optional notes as (position, note) tuples
        """
        # Print example box
        print(f"{colors.BORDER_ACCENT}┌─ {colors.HEADER}Language Example{colors.RESET} {colors.BORDER_ACCENT}{'─' * (self.width - 20)}┐{colors.RESET}")
        print(f"{colors.BORDER_ACCENT}│{colors.RESET}")
        
        # Print example text
        print(f"{colors.BORDER_ACCENT}│  {colors.HIGHLIGHT}{text}{colors.RESET}")
        
        # Print pronunciation if provided
        if pronunciation:
            print(f"{colors.BORDER_ACCENT}│  {colors.MUTED}[{pronunciation}]{colors.RESET}")
            
        # Print translation if provided
        if translation:
            print(f"{colors.BORDER_ACCENT}│{colors.RESET}")
            print(f"{colors.BORDER_ACCENT}│  {colors.VALUE}{translation}{colors.RESET}")
            
        # Print notes if provided
        if notes:
            print(f"{colors.BORDER_ACCENT}│{colors.RESET}")
            print(f"{colors.BORDER_ACCENT}│  {colors.MODULE}Notes:{colors.RESET}")
            
            for position, note in notes:
                note_line = f"{position}. {note}"
                if len(note_line) > self.width - 6:
                    note_line = note_line[:self.width - 9] + "..."
                print(f"{colors.BORDER_ACCENT}│  {colors.VALUE}{note_line}{colors.RESET}")
                
        print(f"{colors.BORDER_ACCENT}│{colors.RESET}")
        print(f"{colors.BORDER_ACCENT}└{'─' * (self.width - 2)}┘{colors.RESET}")
        
    def print_multi_column(self, columns: List[List[str]], headers: List[str] = None) -> None:
        """
        Print text in multiple columns.
        
        Args:
            columns: List of column content lists
            headers: Optional column headers
        """
        # Determine number of columns and their content
        num_columns = len(columns)
        col_width = (self.width - (num_columns + 1)) // num_columns
        
        # Determine number of rows (max length of any column)
        num_rows = max(len(column) for column in columns)
        
        # Print headers if provided
        if headers:
            header_row = ""
            for i, header in enumerate(headers):
                header_text = header[:col_width].center(col_width)
                header_row += f"{colors.HEADER}{header_text}{colors.RESET}"
                if i < num_columns - 1:
                    header_row += f"{colors.BORDER}│{colors.RESET}"
            
            print(header_row)
            separator = f"{colors.BORDER}{'─' * col_width}{'┼' + '─' * col_width}{'─' * (col_width * (num_columns - 2))}{colors.RESET}"
            print(separator)
            
        # Print rows
        for row_idx in range(num_rows):
            row = ""
            for col_idx, column in enumerate(columns):
                # Get cell content or empty string if row doesn't exist in this column
                cell = column[row_idx] if row_idx < len(column) else ""
                cell = cell[:col_width].ljust(col_width)
                row += f"{colors.VALUE}{cell}{colors.RESET}"
                
                # Add column separator
                if col_idx < num_columns - 1:
                    row += f"{colors.BORDER}│{colors.RESET}"
                    
            print(row)
            
    def highlight_differences(self, original: str, modified: str) -> Tuple[str, str]:
        """
        Highlight differences between two texts.
        
        Args:
            original: Original text
            modified: Modified text
            
        Returns:
            Tuple of original and modified texts with highlighted differences
        """
        import difflib
        
        # Split into lines
        original_lines = original.split('\n')
        modified_lines = modified.split('\n')
        
        # Generate diff
        diff = difflib.ndiff(original_lines, modified_lines)
        
        # Process diff output
        highlighted_original = []
        highlighted_modified = []
        
        for line in diff:
            if line.startswith('- '):
                # Line only in original
                highlighted_original.append(f"{colors.ERROR}{line[2:]}{colors.RESET}")
            elif line.startswith('+ '):
                # Line only in modified
                highlighted_modified.append(f"{colors.SUCCESS}{line[2:]}{colors.RESET}")
            elif line.startswith('  '):
                # Line in both
                highlighted_original.append(line[2:])
                highlighted_modified.append(line[2:])
                
        return ('\n'.join(highlighted_original), '\n'.join(highlighted_modified))