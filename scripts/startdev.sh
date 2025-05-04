#!/bin/bash

# ───────────────────────────────────────────────
#   🚀 CasaLingua Dev Launcher
# ───────────────────────────────────────────────

clear
echo ""
echo " ██████╗ █████╗ ███████╗ █████╗ ██╗     ██╗███╗   ██╗ ██████╗ ██╗   ██╗  "
echo "██╔════╝██╔══██╗╚══███╔╝██╔══██╗██║     ██║████╗  ██║██╔═══██╗██║   ██║  "
echo "██║     ███████║  ███╔╝ ███████║██║     ██║██╔██╗ ██║██║   ██║██║   ██║  "
echo "██║     ██╔══██║ ███╔╝  ██╔══██║██║     ██║██║╚██╗██║██║   ██║██║   ██║  "
echo "╚██████╗██║  ██║███████╗██║  ██║███████╗██║██║ ╚████║╚██████╔╝╚██████╔╝  "
echo " ╚═════╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚══════╝╚═╝╚═╝  ╚═══╝ ╚═════╝  ╚═════╝   "
echo "  🧠 CasaLingua - DEVELOPMENT Mode with Optimized API"
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Check for venv
if [ ! -d ".venv" ]; then
  echo -e "${YELLOW}🔍 Python virtual environment not found.${NC}"
  read -p "Would you like to create and install dependencies now? (y/n): " confirm
  if [[ "$confirm" == "y" ]]; then
    echo -e "${BLUE}🔧 Creating virtual environment...${NC}"
    python3 -m venv .venv
    source .venv/bin/activate
    echo -e "${BLUE}📦 Installing dependencies...${NC}"
    pip install -r requirements.txt
    
    # Install additional packages for API optimizations
    echo -e "${CYAN}📦 Installing packages for API optimizations...${NC}"
    pip install nest_asyncio httpx pytest-cov
  else
    echo -e "${RED}❌ Aborting. Please set up the environment manually.${NC}"
    exit 1
  fi
else
  echo -e "${GREEN}✅ Virtual environment found. Activating...${NC}"
  source .venv/bin/activate
fi

# Set optimization environment variables
echo -e "${BLUE}⚙️ Configuring API optimizations...${NC}"

# Set environment variables if not already in .env
if [ -f ".env" ]; then
  echo -e "${CYAN}📝 Loading configuration from .env file...${NC}"
  # shellcheck disable=SC1090
  source .env
fi

# Set default optimization values if not already set
export ENVIRONMENT="development"
export CASALINGUA_ENV="development"  # Explicitly set for auth bypass in development
export ROUTE_CACHE_SIZE=${ROUTE_CACHE_SIZE:-1000}
export TRANSLATION_CACHE_SIZE=${TRANSLATION_CACHE_SIZE:-2000}
export MAX_BATCH_SIZE=${MAX_BATCH_SIZE:-5}
export ENABLE_STREAMING=${ENABLE_STREAMING:-true}

# Create necessary cache directories
mkdir -p cache/models cache/api

# Display configuration
echo -e "${CYAN}┌─────────────── API CONFIGURATION ───────────────┐${NC}"
echo -e "${CYAN}│${NC} Environment:          ${GREEN}development${NC}"
echo -e "${CYAN}│${NC} Route Cache Size:     ${GREEN}${ROUTE_CACHE_SIZE}${NC}"
echo -e "${CYAN}│${NC} Translation Cache:    ${GREEN}${TRANSLATION_CACHE_SIZE}${NC}"
echo -e "${CYAN}│${NC} Max Batch Size:       ${GREEN}${MAX_BATCH_SIZE}${NC}"
echo -e "${CYAN}│${NC} Streaming Enabled:    ${GREEN}${ENABLE_STREAMING}${NC}"
echo -e "${CYAN}└─────────────────────────────────────────────────┘${NC}"

echo -e "${BLUE}🚀 Launching development server using Uvicorn...${NC}"
echo ""
echo -e "${GREEN}🧪 Dev Mode | Local Debug UI at http://127.0.0.1:8000${NC}"
echo -e "${CYAN}📡 Watching for file changes...${NC}"
echo ""

# Run with optimizations enabled
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000