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
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to check if Python is installed
check_python() {
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}Python 3 not found. Please install Python 3.10 or higher.${NC}"
        exit 1
    fi
    
    # Check Python version
    PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    PY_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
    PY_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
    
    # Require Python 3.10+
    if [ "$PY_MAJOR" -lt 3 ] || ([ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]); then
        echo -e "${RED}Python 3.10 or higher is required. Found Python $PYTHON_VERSION${NC}"
        exit 1
    else
        echo -e "${GREEN}Using Python $PYTHON_VERSION${NC}"
    fi
}

# Function to check if virtual environment exists
check_venv() {
    # First check for .venv directory (from new install script)
    if [ -d "$PROJECT_ROOT/.venv" ]; then
        echo -e "${GREEN}Found virtual environment at .venv${NC}"
        source "$PROJECT_ROOT/.venv/bin/activate"
        return
    fi
    
    # Fall back to venv directory (from older installs)
    if [ -d "$PROJECT_ROOT/venv" ]; then
        echo -e "${GREEN}Found virtual environment at venv${NC}"
        source "$PROJECT_ROOT/venv/bin/activate"
        return
    fi
    
    # No virtual environment found, create one
    echo -e "${YELLOW}Virtual environment not found. Creating one...${NC}"
    python3 -m venv "$PROJECT_ROOT/.venv"
    source "$PROJECT_ROOT/.venv/bin/activate"
    
    echo -e "${BLUE}Installing dependencies...${NC}"
    pip install --upgrade pip
    pip install -r "$PROJECT_ROOT/requirements.txt"
    
    # Install optimization dependencies
    echo -e "${CYAN}Installing optimization dependencies...${NC}"
    pip install nest_asyncio httpx pytest-cov pytest-asyncio
}

# Function to configure optimizations
configure_optimizations() {
    # Load from .env if available
    if [ -f "$PROJECT_ROOT/.env" ]; then
        echo -e "${CYAN}Loading configuration from .env...${NC}"
        # shellcheck disable=SC1090
        source "$PROJECT_ROOT/.env"
    fi
    
    # Set optimization environment variables with defaults
    export ENVIRONMENT=${ENVIRONMENT:-"development"}
    export ROUTE_CACHE_SIZE=${ROUTE_CACHE_SIZE:-1000}
    export TRANSLATION_CACHE_SIZE=${TRANSLATION_CACHE_SIZE:-2000}
    export MAX_BATCH_SIZE=${MAX_BATCH_SIZE:-5}
    export ENABLE_STREAMING=${ENABLE_STREAMING:-true}
    
    # Create necessary directories
    mkdir -p "$PROJECT_ROOT/cache/models" "$PROJECT_ROOT/cache/api"
    
    echo -e "${GREEN}API optimizations configured${NC}"
}

# Function to run the application
run_app() {
    echo -e "${CYAN}Launching CasaLingua...${NC}"
    
    # Execute the Python script with all arguments passed to this shell script
    python3 "$SCRIPT_DIR/run_casalingua.py" "$@"
    
    # Capture exit code
    EXIT_CODE=$?
    
    echo -e "${CYAN}CasaLingua session completed${NC}"
    return $EXIT_CODE
}

# Main execution
check_python
check_venv
configure_optimizations

run_app "$@"
EXIT_CODE=$?

# Deactivate virtual environment when done
deactivate

# Return the exit code from the Python application
exit $EXIT_CODE