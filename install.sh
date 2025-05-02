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
     ___           __    _           
    /   |  ____   / /_  (_)___  ____ 
   / /| | / __ \ / __ \/ / __ \/ __ \
  / ___ |/ /_/ // / / / / /_/ / /_/ /
 /_/  |_|\____//_/ /_/_/ .___/\____/ 
                     /_/     Installer
EOF
echo -e "${NC}"

echo -e "${BOLD}${GREEN}🚀 Welcome to CasaLingua on Apple M4 - Installation Wizard${NC}"
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

# Upgrade pip
echo -e "${BLUE}⬆️  Upgrading pip...${NC}"
pip install --upgrade pip

# Install dependencies
echo -e "${BLUE}📦 Installing project requirements...${NC}"
pip install -r requirements.txt

# Optional dev tools
echo -e "${CYAN}🔧 Installing dev utilities (black, mypy, rich, loguru)...${NC}"
pip install black mypy rich loguru

# Prompt for environment
echo -e "${CYAN}🌍 Choose environment mode:${NC}"
select env in "Development" "Production"; do
    case $env in
        Development ) MODE="dev"; break;;
        Production ) MODE="prod"; break;;
    esac
done

# Launch suggestion
echo -e "${GREEN}✅ Installation complete! To begin hacking, run:${NC}"
echo ""
echo -e "${BOLD}    source .venv/bin/activate${NC}"
echo -e "${BOLD}    ./scripts/start$MODE.sh${NC}"
echo ""
echo -e "${GREEN}Enjoy building with CasaLingua on your M4 Mac! 🍏${NC}"