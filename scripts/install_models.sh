#!/bin/bash
# CasaLingua Model Installation Script
# Creates the necessary directory structure for models and prepares the environment

# Get the absolute path of the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Define color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Print banner
echo -e "${CYAN}"
echo "====================================================="
echo "   CasaLingua Model Installation and Setup Script    "
echo "====================================================="
echo -e "${NC}"

# Function to create model directories
create_model_directories() {
    echo -e "${CYAN}Creating model directories...${NC}"
    
    # Define model types and sizes
    MODEL_TYPES=("translation" "multipurpose" "verification")
    MODEL_SIZES=("large" "medium" "small")
    
    # Create the base models directory if it doesn't exist
    if [ ! -d "$PROJECT_ROOT/models" ]; then
        mkdir -p "$PROJECT_ROOT/models"
        echo -e "${GREEN}Created base models directory${NC}"
    fi
    
    # Create directories for each model type and size
    for type in "${MODEL_TYPES[@]}"; do
        for size in "${MODEL_SIZES[@]}"; do
            MODEL_DIR="$PROJECT_ROOT/models/$type/$size"
            if [ ! -d "$MODEL_DIR" ]; then
                mkdir -p "$MODEL_DIR"
                echo -e "${GREEN}Created directory: models/$type/$size${NC}"
            else
                echo -e "${YELLOW}Directory already exists: models/$type/$size${NC}"
            fi
        done
    done
    
    echo -e "${GREEN}✓ Model directories created successfully${NC}"
}

# Function to check required software
check_requirements() {
    echo -e "${CYAN}Checking system requirements...${NC}"
    
    # Check Python version
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}Python 3 not found. Please install Python 3.8 or higher.${NC}"
        exit 1
    fi
    
    # Get Python version as major.minor (e.g., 3.10)
    PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    echo -e "${GREEN}Python version: $PYTHON_VERSION${NC}"
    
    # Get major and minor versions as integers
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
    
    # Check if version meets requirements (3.8 or higher)
    if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]); then
        echo -e "${RED}Python 3.8 or higher is required. Found Python $PYTHON_VERSION${NC}"
        exit 1
    else
        echo -e "${GREEN}✓ Python version requirement satisfied${NC}"
    fi
    
    # Check pip
    if ! command -v pip3 &> /dev/null; then
        echo -e "${RED}pip3 not found. Please install pip for Python 3.${NC}"
        exit 1
    fi
    
    # Check torch
    if ! python3 -c "import torch" &> /dev/null; then
        echo -e "${YELLOW}PyTorch not found. Installing...${NC}"
        # Detect Apple Silicon and install optimized torch
        if [[ $(uname -m) == 'arm64' ]]; then
            echo -e "${CYAN}Installing Apple Silicon-optimized PyTorch...${NC}"
            pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
        else
            echo -e "${CYAN}Installing standard PyTorch...${NC}"
            pip3 install torch
        fi
        echo -e "${GREEN}✓ Optimized PyTorch installed for $(uname -m) architecture${NC}"
    else
        TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)")
        echo -e "${GREEN}PyTorch version: $TORCH_VERSION${NC}"
    fi
    
    # Check rich library for console output
    if ! python3 -c "import rich" &> /dev/null; then
        echo -e "${YELLOW}Rich library not found. Installing...${NC}"
        pip3 install rich
    fi
    
    # Check other dependencies
    echo -e "${CYAN}Installing required dependencies...${NC}"
    pip3 install -r "$PROJECT_ROOT/requirements.txt"
    
    echo -e "${GREEN}✓ All requirements satisfied${NC}"
}

# Function to check available disk space
check_disk_space() {
    echo -e "${CYAN}Checking available disk space...${NC}"
    
    # Required space in GB
    REQUIRED_SPACE_GB=40
    
    # Check available space
    if command -v df &> /dev/null; then
        # Get available space in KB and convert to GB
        AVAILABLE_SPACE_KB=$(df -k "$PROJECT_ROOT" | awk 'NR==2 {print $4}')
        AVAILABLE_SPACE_GB=$(echo "scale=2; $AVAILABLE_SPACE_KB / 1024 / 1024" | bc)
        
        echo -e "${GREEN}Available disk space: $AVAILABLE_SPACE_GB GB${NC}"
        
        if (( $(echo "$AVAILABLE_SPACE_GB < $REQUIRED_SPACE_GB" | bc -l) )); then
            echo -e "${RED}Warning: Less than $REQUIRED_SPACE_GB GB available. Full model installation requires approximately $REQUIRED_SPACE_GB GB.${NC}"
            read -p "Do you want to continue anyway? (y/n) " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                echo -e "${YELLOW}Installation aborted.${NC}"
                exit 1
            fi
        else
            echo -e "${GREEN}✓ Sufficient disk space available${NC}"
        fi
    else
        echo -e "${YELLOW}Could not determine available disk space. Please ensure you have at least $REQUIRED_SPACE_GB GB available.${NC}"
        read -p "Do you want to continue? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo -e "${YELLOW}Installation aborted.${NC}"
            exit 1
        fi
    fi
}

# Function to create registry.json file
create_model_registry() {
    echo -e "${CYAN}Creating model registry configuration...${NC}"
    
    REGISTRY_FILE="$PROJECT_ROOT/models/registry.json"
    
    # Define model registry content
    cat > "$REGISTRY_FILE" << EOL
{
    "version": "1.0.0",
    "models": {
        "translation": {
            "large": {
                "path": "models/translation/large",
                "memory_required_gb": 12,
                "files": ["model.bin", "tokenizer.json", "config.json"],
                "quantization_options": [16, 8, 4]
            },
            "medium": {
                "path": "models/translation/medium",
                "memory_required_gb": 6,
                "files": ["model.bin", "tokenizer.json", "config.json"],
                "quantization_options": [16, 8, 4]
            },
            "small": {
                "path": "models/translation/small",
                "memory_required_gb": 2,
                "files": ["model.bin", "tokenizer.json", "config.json"],
                "quantization_options": [16, 8, 4]
            }
        },
        "multipurpose": {
            "large": {
                "path": "models/multipurpose/large",
                "memory_required_gb": 16,
                "files": ["model.bin", "tokenizer.json", "config.json"],
                "quantization_options": [16, 8, 4]
            },
            "medium": {
                "path": "models/multipurpose/medium",
                "memory_required_gb": 8,
                "files": ["model.bin", "tokenizer.json", "config.json"],
                "quantization_options": [16, 8, 4]
            },
            "small": {
                "path": "models/multipurpose/small",
                "memory_required_gb": 4,
                "files": ["model.bin", "tokenizer.json", "config.json"],
                "quantization_options": [16, 8, 4]
            }
        },
        "verification": {
            "large": {
                "path": "models/verification/large",
                "memory_required_gb": 10,
                "files": ["model.bin", "tokenizer.json", "config.json"],
                "quantization_options": [16, 8, 4]
            },
            "medium": {
                "path": "models/verification/medium",
                "memory_required_gb": 5,
                "files": ["model.bin", "tokenizer.json", "config.json"],
                "quantization_options": [16, 8, 4]
            },
            "small": {
                "path": "models/verification/small",
                "memory_required_gb": 2,
                "files": ["model.bin", "tokenizer.json", "config.json"],
                "quantization_options": [16, 8, 4]
            }
        }
    }
}
EOL
    
    echo -e "${GREEN}✓ Model registry created at: $REGISTRY_FILE${NC}"
}

# Function to set up the launch script
setup_launcher() {
    echo -e "${CYAN}Setting up launcher script...${NC}"
    
    LAUNCHER_SCRIPT="$PROJECT_ROOT/scripts/casalingua.sh"
    
    # Make sure the script is executable
    chmod +x "$LAUNCHER_SCRIPT"
    chmod +x "$PROJECT_ROOT/scripts/run_casalingua.py"
    
    # Create a symlink in /usr/local/bin if running as root
    if [ "$EUID" -eq 0 ]; then
        ln -sf "$LAUNCHER_SCRIPT" /usr/local/bin/casalingua
        echo -e "${GREEN}✓ Created symlink in /usr/local/bin/casalingua${NC}"
    else
        echo -e "${YELLOW}Not running as root. To create a system-wide launcher, run:${NC}"
        echo -e "sudo ln -sf \"$LAUNCHER_SCRIPT\" /usr/local/bin/casalingua"
        
        # Alternative: create in user's bin directory
        if [ -d "$HOME/bin" ]; then
            ln -sf "$LAUNCHER_SCRIPT" "$HOME/bin/casalingua"
            echo -e "${GREEN}✓ Created symlink in $HOME/bin/casalingua${NC}"
        elif [ -d "$HOME/.local/bin" ]; then
            ln -sf "$LAUNCHER_SCRIPT" "$HOME/.local/bin/casalingua"
            echo -e "${GREEN}✓ Created symlink in $HOME/.local/bin/casalingua${NC}"
        else
            mkdir -p "$HOME/bin"
            ln -sf "$LAUNCHER_SCRIPT" "$HOME/bin/casalingua"
            echo -e "${GREEN}✓ Created $HOME/bin directory and symlink${NC}"
            echo -e "${YELLOW}Add $HOME/bin to your PATH to use the 'casalingua' command${NC}"
        fi
    fi
}

# Function to download model placeholders (for testing)
create_model_placeholders() {
    echo -e "${CYAN}Creating model placeholders for testing...${NC}"
    
    # Define model types and sizes
    MODEL_TYPES=("translation" "multipurpose" "verification")
    MODEL_SIZES=("large" "medium" "small")
    
    # Create placeholder files
    for type in "${MODEL_TYPES[@]}"; do
        for size in "${MODEL_SIZES[@]}"; do
            MODEL_DIR="$PROJECT_ROOT/models/$type/$size"
            
            # Create placeholder files
            touch "$MODEL_DIR/model.bin"
            
            # Create a minimal configuration file
            cat > "$MODEL_DIR/config.json" << EOL
{
    "model_type": "$type",
    "model_size": "$size",
    "version": "1.0.0",
    "created_at": "$(date -Iseconds)",
    "description": "Placeholder model for testing"
}
EOL
            
            # Create a minimal tokenizer file
            cat > "$MODEL_DIR/tokenizer.json" << EOL
{
    "model_type": "$type",
    "version": "1.0.0",
    "vocabulary_size": 50000
}
EOL
            
            echo -e "${GREEN}Created placeholders for $type/$size model${NC}"
        done
    done
    
    echo -e "${GREEN}✓ Model placeholders created successfully${NC}"
}

# Main execution
check_requirements
check_disk_space
create_model_directories
create_model_registry
create_model_placeholders
setup_launcher

echo -e "${GREEN}"
echo "====================================================="
echo "           CasaLingua Setup Complete!                "
echo "====================================================="
echo -e "${NC}"
echo -e "${CYAN}You can now run CasaLingua using:${NC}"
echo -e "${YELLOW}casalingua --interactive${NC}"
echo -e "${CYAN}or${NC}"
echo -e "${YELLOW}$PROJECT_ROOT/scripts/casalingua.sh --interactive${NC}"
echo ""