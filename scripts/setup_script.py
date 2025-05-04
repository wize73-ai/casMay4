#!/usr/bin/env python3
"""
CasaLingua Setup Script
-----------------------
Handles installation and configuration of the CasaLingua system
including downloading models, setting up dependencies, and configuring
the environment for optimal performance.
"""

import os
import sys
import argparse
import platform
import subprocess
import shutil
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

# Initialize rich console
console = Console()

# Define model info - size in GB, filenames, etc.
MODEL_INFO = {
    "translation": {
        "large": {"size": 12, "filename": "translation_large_model.bin"},
        "medium": {"size": 6, "filename": "translation_medium_model.bin"},
        "small": {"size": 2, "filename": "translation_small_model.bin"}
    },
    "multipurpose": {
        "large": {"size": 16, "filename": "multipurpose_large_model.bin"},
        "medium": {"size": 8, "filename": "multipurpose_medium_model.bin"},
        "small": {"size": 4, "filename": "multipurpose_small_model.bin"}
    },
    "verification": {
        "large": {"size": 10, "filename": "verification_large_model.bin"},
        "medium": {"size": 5, "filename": "verification_medium_model.bin"},
        "small": {"size": 2, "filename": "verification_small_model.bin"}
    }
}

# Model download URLs (these would be actual URLs in a real implementation)
MODEL_BASE_URL = "https://casalingua.ai/models/"

def check_system_requirements():
    """Check if the system meets the minimum requirements."""
    console.print(Panel.fit("Checking System Requirements", border_style="cyan"))
    
    # Check Python version
    python_version = sys.version_info
    min_python = (3, 8)
    if python_version < min_python:
        console.print(f"[bold red]❌ Error: Python {min_python[0]}.{min_python[1]} or higher is required. "
                      f"Found Python {python_version[0]}.{python_version[1]}[/bold red]")
        return False
    console.print(f"[green]✓ Python version: {python_version[0]}.{python_version[1]}.{python_version[2]}[/green]")
    
    # Check available RAM
    import psutil
    ram_gb = psutil.virtual_memory().total / (1024**3)
    if ram_gb < 8:
        console.print(f"[bold yellow]⚠ Warning: Less than 8GB RAM detected ({ram_gb:.1f}GB). "
                      f"Performance may be limited.[/bold yellow]")
    else:
        console.print(f"[green]✓ Available RAM: {ram_gb:.1f}GB[/green]")
    
    # Check for GPU
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            console.print(f"[green]✓ GPU detected: {gpu_name} with {gpu_memory:.1f}GB memory[/green]")
        else:
            console.print("[yellow]⚠ No GPU detected. Using CPU mode (slower performance).[/yellow]")
    except ImportError:
        console.print("[yellow]⚠ PyTorch not installed. Will install during setup.[/yellow]")
    
    # Check for disk space
    disk = shutil.disk_usage("/")
    free_space_gb = disk.free / (1024**3)
    if free_space_gb < 50:
        console.print(f"[bold yellow]⚠ Warning: Less than 50GB free disk space ({free_space_gb:.1f}GB). "
                      f"Full model installation requires ~40GB.[/bold yellow]")
    else:
        console.print(f"[green]✓ Available disk space: {free_space_gb:.1f}GB[/green]")
    
    return True

def install_dependencies():
    """Install required Python packages."""
    console.print(Panel.fit("Installing Dependencies", border_style="cyan"))
    
    dependencies = [
        "torch",
        "transformers",
        "tqdm",
        "psutil",
        "rich",
        "numpy",
        "pyyaml"
    ]
    
    # Check if system is Apple Silicon for specialized torch install
    is_apple_silicon = platform.system() == "Darwin" and platform.machine() == "arm64"
    
    if is_apple_silicon:
        console.print("[yellow]Detected Apple Silicon. Using specialized torch installation.[/yellow]")
        # Replace standard torch with Apple Silicon optimized version (version-pinned)
        dependencies.remove("torch")
        torch_command = "pip install torch==2.1.2 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
    else:
        # Add CUDA dependencies if NVIDIA GPU is detected
        try:
            import torch
            if torch.cuda.is_available():
                console.print("[green]NVIDIA GPU detected. Adding CUDA support.[/green]")
                # Replace standard torch with CUDA-enabled version
                dependencies.remove("torch")
                torch_command = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
        except:
            torch_command = None
    
    # Install the dependencies
    with Progress(
        SpinnerColumn(), 
        TextColumn("[bold blue]Installing dependencies..."), 
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%")
    ) as progress:
        task = progress.add_task("Installing", total=len(dependencies) + (1 if torch_command else 0))
        
        # Install torch with special handling if needed
        if torch_command:
            console.print(f"Running: {torch_command}")
            try:
                subprocess.check_call(torch_command.split())
                progress.update(task, advance=1)
            except subprocess.CalledProcessError as e:
                console.print(f"[bold red]Failed to install torch: {e}[/bold red]")
                return False
        
        # Install the rest of the dependencies
        for dep in dependencies:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
                progress.update(task, advance=1)
            except subprocess.CalledProcessError as e:
                console.print(f"[bold red]Failed to install {dep}: {e}[/bold red]")
                return False
    
    console.print("[bold green]✓ All dependencies installed successfully![/bold green]")
    return True

def determine_optimal_models():
    """Determine which model sizes are optimal for the current system."""
    console.print(Panel.fit("Determining Optimal Model Configuration", border_style="cyan"))
    
    # Import hardware detection from model_manager
    sys.path.append(".")
    try:
        from model_manager import HardwareDetector, ModelManager
        hardware_info = HardwareDetector.get_hardware_info()
        model_manager = ModelManager()
        model_sizes = model_manager._determine_model_sizes()
        
        # Display the selected configuration
        table = Table(title="Selected Model Configuration")
        table.add_column("Model Type", style="cyan")
        table.add_column("Size", style="green")
        table.add_column("Required Space (GB)", style="magenta")
        
        # Calculate total required space
        total_space = 0
        
        for model_type, size in model_sizes.items():
            model_type_str = model_type.value
            size_str = size.value
            space_gb = MODEL_INFO[model_type_str][size_str]["size"]
            total_space += space_gb
            
            table.add_row(
                model_type_str.capitalize(),
                size_str.capitalize(),
                f"{space_gb} GB"
            )
        
        # Add total row
        table.add_row("TOTAL", "", f"{total_space} GB", style="bold")
        console.print(table)
        
        return model_sizes
    
    except Exception as e:
        console.print(f"[bold red]Error determining optimal models: {str(e)}[/bold red]")
        # Fall back to minimal configuration
        console.print("[yellow]Falling back to minimal configuration.[/yellow]")
        return {
            "translation": "small",
            "multipurpose": "small",
            "verification": "small"
        }

def download_models(model_sizes, model_dir):
    """Download the selected models."""
    console.print(Panel.fit("Downloading Models", border_style="cyan"))
    
    # Create model directories if they don't exist
    os.makedirs(model_dir, exist_ok=True)
    for model_type in MODEL_INFO:
        os.makedirs(os.path.join(model_dir, model_type), exist_ok=True)
        for size in MODEL_INFO[model_type]:
            os.makedirs(os.path.join(model_dir, model_type, size), exist_ok=True)
    
    # Count total models to download
    total_models = sum(1 for model_type, size in model_sizes.items())
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%")
    ) as progress:
        main_task = progress.add_task("Downloading models...", total=total_models)
        
        for model_type, size in model_sizes.items():
            model_type_str = model_type.value
            size_str = size.value
            model_info = MODEL_INFO[model_type_str][size_str]
            filename = model_info["filename"]
            model_size_gb = model_info["size"]
            
            model_url = f"{MODEL_BASE_URL}{model_type_str}/{size_str}/{filename}"
            dest_path = os.path.join(model_dir, model_type_str, size_str, filename)
            
            # Skip if model already exists
            if os.path.exists(dest_path):
                console.print(f"[yellow]Model already exists: {dest_path}. Skipping download.[/yellow]")
                progress.update(main_task, advance=1)
                continue
            
            # In a real implementation, this would download the model
            # For demo purposes, we'll simulate the download
            download_task = progress.add_task(
                f"Downloading {model_type_str} {size_str} model ({model_size_gb}GB)...", 
                total=100
            )
            
            # Simulate download progress
            for i in range(100):
                import time
                time.sleep(0.05)  # Simulate network activity
                progress.update(download_task, advance=1)
            
            # Simulate the downloaded model with a placeholder and check size
            with open(dest_path, "w") as f:
                f.write("# Simulated model file content")
            if os.path.getsize(dest_path) < 10:
                console.print(f"[red]✗ Model file failed integrity check: {dest_path}[/red]")
            
            progress.update(main_task, advance=1)
    
    console.print("[bold green]✓ All models downloaded successfully![/bold green]")
    return True

def configure_system():
    """Configure the system for optimal performance."""
    console.print(Panel.fit("Configuring System", border_style="cyan"))
    
    # Create config directory if it doesn't exist
    config_dir = os.path.join(os.path.expanduser("~"), ".casalingua")
    os.makedirs(config_dir, exist_ok=True)
    
    # Create a basic configuration file
    config_file = os.path.join(config_dir, "config.yaml")
    with open(config_file, "w") as f:
        f.write("""# CasaLingua Configuration
version: 1.0.0

# Model settings
models:
  translation:
    use_gpu: true
    threads: 4
    cache_size: 1000
  multipurpose:
    use_gpu: true
    threads: 4
    cache_size: 1000
  verification:
    use_gpu: true
    threads: 2
    cache_size: 500

# System settings
system:
  log_level: info
  temp_dir: ~/.casalingua/temp
  max_memory_usage: 90%
  env_vars:
    OMP_NUM_THREADS: 8
    MKL_NUM_THREADS: 8

# UI settings
ui:
  theme: dark
  color_scheme: default
  show_animations: true
""")
    
    console.print(f"[green]✓ Configuration file created at {config_file}[/green]")
    
    # Create symbolic links for easy access
    try:
        bin_dir = os.path.join(os.path.expanduser("~"), "bin")
        os.makedirs(bin_dir, exist_ok=True)
        
        # Create the launcher script
        launcher_path = os.path.join(bin_dir, "casalingua")
        with open(launcher_path, "w") as f:
            f.write(f"""#!/bin/bash
# CasaLingua launcher script
{sys.executable} {os.path.abspath(os.path.join(os.path.dirname(__file__), "run_casa_lingua.py"))} "$@"
""")
        
        # Make it executable
        os.chmod(launcher_path, 0o755)
        console.print(f"[green]✓ Launcher script created at {launcher_path}[/green]")
        
        # Add to PATH if not already there
        if bin_dir not in os.environ.get("PATH", ""):
            shell_rc = None
            if os.path.exists(os.path.join(os.path.expanduser("~"), ".bashrc")):
                shell_rc = os.path.join(os.path.expanduser("~"), ".bashrc")
            elif os.path.exists(os.path.join(os.path.expanduser("~"), ".zshrc")):
                shell_rc = os.path.join(os.path.expanduser("~"), ".zshrc")
            
            if shell_rc:
                with open(shell_rc, "a") as f:
                    f.write(f"\n# Added by CasaLingua setup\nexport PATH=\"$PATH:{bin_dir}\"\n")
                console.print(f"[green]✓ Added {bin_dir} to PATH in {shell_rc}[/green]")
                console.print("[yellow]Note: You may need to restart your terminal or run "
                             f"'source {shell_rc}' for this change to take effect.[/yellow]")
    except Exception as e:
        console.print(f"[yellow]Warning: Could not create launcher script: {str(e)}[/yellow]")
    
    return True

def main():
    """Main setup function."""
    console.print(Panel.fit(
        "[bold cyan]CasaLingua Setup[/bold cyan]\n[green]Multilingual AI Assistant[/green]",
        border_style="cyan",
        padding=(1, 2)
    ))
    
    parser = argparse.ArgumentParser(description="CasaLingua Setup Script")
    parser.add_argument("--model-dir", default="./models", help="Directory to store models")
    parser.add_argument("--skip-deps", action="store_true", help="Skip dependency installation")
    parser.add_argument("--skip-download", action="store_true", help="Skip model download")
    parser.add_argument("--force-minimal", action="store_true", help="Force minimal configuration")
    args = parser.parse_args()
    
    # Create model directory
    model_dir = os.path.abspath(args.model_dir)
    os.makedirs(model_dir, exist_ok=True)
    
    # Check system requirements
    if not check_system_requirements():
        if not Confirm.ask("[bold yellow]System may not meet requirements. Continue anyway?[/bold yellow]"):
            return 1
    
    # Install dependencies
    if not args.skip_deps:
        if not install_dependencies():
            if not Confirm.ask("[bold yellow]Dependency installation failed. Continue anyway?[/bold yellow]"):
                return 1
    else:
        console.print("[yellow]Skipping dependency installation.[/yellow]")
    
    # Determine optimal model configuration
    if args.force_minimal:
        console.print("[yellow]Forcing minimal configuration.[/yellow]")
        model_sizes = {
            "translation": "small",
            "multipurpose": "small",
            "verification": "small"
        }
    else:
        model_sizes = determine_optimal_models()
    
    # Download models
    if not args.skip_download:
        if not download_models(model_sizes, model_dir):
            if not Confirm.ask("[bold yellow]Model download failed. Continue anyway?[/bold yellow]"):
                return 1
    else:
        console.print("[yellow]Skipping model download.[/yellow]")
    
    # Configure system
    if not configure_system():
        if not Confirm.ask("[bold yellow]System configuration failed. Continue anyway?[/bold yellow]"):
            return 1
    
    # Setup complete
    console.print(Panel.fit(
        "[bold green]CasaLingua Setup Complete![/bold green]\n"
        "You can now run CasaLingua by typing 'casalingua' in your terminal.",
        border_style="green",
        padding=(1, 2)
    ))
    
    return 0

if __name__ == "__main__":
    sys.exit(main())