#!/bin/bash

# ───────────────────────────────────────────────
#   🚀 CasaLingua Dev Launcher
# ───────────────────────────────────────────────

# Force color output
export FORCE_COLOR=1

# Colors (using tput for better compatibility)
GREEN=$(tput setaf 2 2>/dev/null || echo '')
BLUE=$(tput setaf 4 2>/dev/null || echo '')
RED=$(tput setaf 1 2>/dev/null || echo '')
YELLOW=$(tput setaf 3 2>/dev/null || echo '')
CYAN=$(tput setaf 6 2>/dev/null || echo '')
NC=$(tput sgr0 2>/dev/null || echo '') # Reset

# Styled Banner with Box Drawing Characters
clear

# Box drawing chars
TL="╔"
TR="╗"
BL="╚"
BR="╝"
HORIZ="═"
VERT="║"

# Get terminal width
term_width=$(tput cols)
if [ $term_width -gt 100 ]; then
    term_width=100
fi

# Calculate banner width
banner_width=$((term_width - 4))

# Create horizontal line
horiz_line=""
for ((i=0; i<banner_width; i++)); do
    horiz_line="${horiz_line}${HORIZ}"
done

# Create empty line
empty_line="${VERT}$(printf "%${banner_width}s")${VERT}"

# Print banner top border
echo -e "\n${TL}${horiz_line}${TR}"

# Print empty line
echo -e "${empty_line}"

# Print logo
echo -e "${VERT}   ${CYAN}   _____                _      _                          ${NC}                                   ${VERT}"
echo -e "${VERT}   ${CYAN}  / ____|              | |    (_)                         ${NC}                                   ${VERT}"
echo -e "${VERT}   ${CYAN} | |     __ _ ___  __ _| |     _ _ __   __ _ _   _  __ _  ${NC}                                   ${VERT}"
echo -e "${VERT}   ${CYAN} | |    / _\` / __| / _\` | |    | | '_ \\ / _\` | | | |/ _\` | ${NC}                                   ${VERT}"
echo -e "${VERT}   ${CYAN} | |___| (_| \\__ \\ (_| | |____| | | | | (_| | |_| | (_| | ${NC}                                   ${VERT}"
echo -e "${VERT}   ${CYAN}  \\_____\\__,_|___/\\__,_|______|_|_| |_|\\__, |\\__,_|\\__,_| ${NC}                                   ${VERT}"
echo -e "${VERT}   ${CYAN}                                        |___/             ${NC}                                   ${VERT}"

# Print empty line
echo -e "${empty_line}"

# Print subtitle
subtitle="DEVELOPMENT Mode with Optimized API"
subtitle_len=${#subtitle}
padding=$((banner_width - subtitle_len - 6))
padding_spaces=$(printf "%${padding}s")
echo -e "${VERT}   ${YELLOW}${subtitle}${padding_spaces}${NC}   ${VERT}"

# Print empty line
echo -e "${empty_line}"

# Print bottom border
echo -e "${BL}${horiz_line}${BR}\n"

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