#!/usr/bin/env bash

echo
echo "========================================"
echo "      CasaLingua Installer Script       "
echo "========================================"
echo

# Check Python version
echo "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1)
echo "$PYTHON_VERSION found."

# Setup virtual environment
echo "Setting up virtual environment..."
if [ ! -d "venv" ]; then
  python3 -m venv venv
  echo "Virtual environment created."
else
  echo "Virtual environment already exists. Using existing environment."
fi

echo "To activate the virtual environment, run: source venv/bin/activate"

# Activate venv and install requirements
source venv/bin/activate

echo "Installing dependencies..."
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

# Install PyTorch for M1/M2/M3/M4 Mac with Metal
echo "Installing PyTorch with Metal support for macOS..."
python3 -m pip install torch torchvision torchaudio

# Create necessary folders
echo "Creating necessary directories..."
mkdir -p logs/app logs/audit logs/metrics logs/metrics/time_series models indexes knowledge_base

# Setup .env if not present
if [ ! -f ".env" ]; then
  cp .env.example .env
  echo ".env created from .env.example"
else
  echo ".env already exists. Keeping existing file."
fi

# Download models
echo "Downloading language models..."
python3 scripts/download_models.py

# Initialize database if needed
echo "Initializing database..."
python3 scripts/setup_registry.py

echo
echo "✅ Installation complete. To start CasaLingua, run:"
echo "source venv/bin/activate && ./scripts/startdev.sh"