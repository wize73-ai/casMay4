#!/bin/bash
# CasaLingua Launch Script
# This script provides a convenient way to run CasaLingua from anywhere

# Get the absolute path of the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Set environment variables
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Define color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to check if Python is installed
check_python() {
    if ! command -v python3 &> /dev/null; then
        echo -e "${YELLOW}Python 3 not found. Please install Python 3.8 or higher.${NC}"
        exit 1
    fi
    
    # Check Python version - fixed comparison
    PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    PY_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
    PY_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
    
    if [ "$PY_MAJOR" -lt 3 ] || ([ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 8 ]); then
        echo -e "${YELLOW}Python 3.8 or higher is required. Found Python $PYTHON_VERSION${NC}"
        exit 1
    else
        echo -e "${GREEN}Using Python $PYTHON_VERSION${NC}"
    fi
}

# Function to check if virtual environment exists
check_venv() {
    if [ ! -d "$PROJECT_ROOT/venv" ]; then
        echo -e "${YELLOW}Virtual environment not found. Creating one...${NC}"
        python3 -m venv "$PROJECT_ROOT/venv"
        source "$PROJECT_ROOT/venv/bin/activate"
        pip install -r "$PROJECT_ROOT/requirements.txt"
    else
        source "$PROJECT_ROOT/venv/bin/activate"
    fi
}

# Function to run the application
run_app() {
    # Execute the Python script with all arguments passed to this shell script
    python3 "$SCRIPT_DIR/run_casalingua.py" "$@"
}

# Main execution
check_python
check_venv

echo -e "${CYAN}Launching CasaLingua...${NC}"
run_app "$@"

# Deactivate virtual environment when done
deactivate