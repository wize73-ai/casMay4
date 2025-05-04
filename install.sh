#!/bin/bash
set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# ASCII Banner
clear
echo -e "${CYAN}"
cat << "EOF"
   _____                _      _                         
  / ____|              | |    (_)                        
 | |     __ _ ___  __ _| |     _ _ __   __ _ _   _  __ _ 
 | |    / _` / __|/ _` | |    | | '_ \ / _` | | | |/ _` |
 | |___| (_| \__ \ (_| | |____| | | | | (_| | |_| | (_| |
  \_____\__,_|___/\__,_|______|_|_| |_|\__, |\__,_|\__,_|
                                         __/ |           
                                        |___/            
EOF
echo -e "${NC}"

echo -e "${BOLD}${GREEN}🚀 Welcome to CasaLingua - Installation Wizard${NC}"
echo -e "${YELLOW}--------------------------------------------------------${NC}"

# Check Python 3.10+
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python3 is not installed. Please install Python 3.10 or newer.${NC}"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
REQUIRED_VERSION="3.10"
if [[ "$PYTHON_VERSION" < "$REQUIRED_VERSION" ]]; then
    echo -e "${RED}❌ Python 3.10 or higher is required. Found: $PYTHON_VERSION${NC}"
    exit 1
fi

# Set up installation directory
INSTALL_DIR=$(pwd)
echo -e "${BLUE}📂 Installation directory: ${INSTALL_DIR}${NC}"

# Check if already installed
if [ -d ".venv" ]; then
    echo -e "${CYAN}🔍 Existing virtual environment detected.${NC}"
    read -p "Would you like to reinstall everything? [y/N]: " REINSTALL
    if [[ "$REINSTALL" != "y" && "$REINSTALL" != "Y" ]]; then
        echo -e "${GREEN}✅ Setup skipped. To activate, run: source .venv/bin/activate${NC}"
        exit 0
    fi
    rm -rf .venv
fi

# Create virtual environment
echo -e "${BLUE}🛠️  Creating virtual environment...${NC}"
python3 -m venv .venv

# Activate environment
source .venv/bin/activate
echo -e "${GREEN}✅ Virtual environment activated${NC}"

# Upgrade pip
echo -e "${BLUE}⬆️  Upgrading pip...${NC}"
pip install --upgrade pip

# Install dependencies
echo -e "${BLUE}📦 Installing project requirements...${NC}"
pip install -r requirements.txt

# Install dev tools
echo -e "${CYAN}🔧 Installing dev utilities (black, mypy, rich, loguru)...${NC}"
pip install black mypy rich loguru pytest pytest-asyncio

# Install additional requirements for optimized API
echo -e "${CYAN}📦 Installing additional packages for API optimizations...${NC}"
pip install nest_asyncio httpx pytest-cov

# Create necessary directories
echo -e "${BLUE}📂 Creating necessary directories...${NC}"
mkdir -p logs/app logs/audit logs/metrics models/translation models/multipurpose models/verification cache/models cache/api

# Set up environment configuration
echo -e "${CYAN}🌍 Choose environment mode:${NC}"
select env in "Development" "Production"; do
    case $env in
        Development ) 
            MODE="dev"
            # Set default development cache sizes
            ROUTE_CACHE_SIZE=1000
            TRANSLATION_CACHE_SIZE=2000
            BATCH_SIZE=5
            break;;
        Production ) 
            MODE="prod"
            # Set larger production cache sizes
            ROUTE_CACHE_SIZE=5000
            TRANSLATION_CACHE_SIZE=10000
            BATCH_SIZE=20
            break;;
    esac
done

# Create or update .env file with optimization settings
if [ -f ".env" ]; then
    echo -e "${CYAN}📝 Updating .env file...${NC}"
    # Add optimization settings without overwriting existing values
    grep -q "ROUTE_CACHE_SIZE" .env || echo "ROUTE_CACHE_SIZE=$ROUTE_CACHE_SIZE" >> .env
    grep -q "TRANSLATION_CACHE_SIZE" .env || echo "TRANSLATION_CACHE_SIZE=$TRANSLATION_CACHE_SIZE" >> .env
    grep -q "MAX_BATCH_SIZE" .env || echo "MAX_BATCH_SIZE=$BATCH_SIZE" >> .env
    grep -q "ENABLE_STREAMING" .env || echo "ENABLE_STREAMING=true" >> .env
else
    echo -e "${CYAN}📝 Creating .env file...${NC}"
    cat > .env << EOL
# CasaLingua Environment Configuration
ENVIRONMENT=$MODE
PYTHONPATH=$INSTALL_DIR

# Cache settings
ROUTE_CACHE_SIZE=$ROUTE_CACHE_SIZE
TRANSLATION_CACHE_SIZE=$TRANSLATION_CACHE_SIZE
MAX_BATCH_SIZE=$BATCH_SIZE
ENABLE_STREAMING=true

# Database settings (update as needed)
DATABASE_URL=sqlite:///app.db
EOL
fi

# Make scripts executable
echo -e "${BLUE}🔧 Making scripts executable...${NC}"
chmod +x scripts/*.sh
chmod +x scripts/*.py

# Download models if requested
echo -e "${CYAN}🧠 Would you like to download language models now? [y/N]:${NC}"
read -p "" DOWNLOAD_MODELS
if [[ "$DOWNLOAD_MODELS" == "y" || "$DOWNLOAD_MODELS" == "Y" ]]; then
    echo -e "${BLUE}🔄 Downloading models...${NC}"
    python scripts/download_models.py
fi

# Launch suggestion
echo -e "${GREEN}✅ Installation complete! To start CasaLingua, run:${NC}"
echo ""
echo -e "${BOLD}    source .venv/bin/activate${NC}"
echo -e "${BOLD}    ./scripts/start$MODE.sh${NC}"
echo ""
echo -e "${YELLOW}💡 For the command-line interface, run:${NC}"
echo -e "${BOLD}    ./scripts/casalingua.sh --interactive${NC}"
echo ""
echo -e "${GREEN}Enjoy building with CasaLingua! 🚀${NC}"