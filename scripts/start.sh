#!/bin/bash

# CasaLingua Start Script
# Flushes caches and starts the application
# Author: Exygy Development Team
# Version: 1.0.0
# License: MIT

set -e  # Exit on error

# Force color output
export FORCE_COLOR=1

# Color codes for pretty output using tput (more compatible)
GREEN=$(tput setaf 2 2>/dev/null || echo '')
BLUE=$(tput setaf 4 2>/dev/null || echo '')
RED=$(tput setaf 1 2>/dev/null || echo '')
YELLOW=$(tput setaf 3 2>/dev/null || echo '')
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
echo -e "${BLUE}"
echo -e "${VERT}   ${BLUE}   _____                _      _                          ${NC}                                   ${VERT}"
echo -e "${VERT}   ${BLUE}  / ____|              | |    (_)                         ${NC}                                   ${VERT}"
echo -e "${VERT}   ${BLUE} | |     __ _ ___  __ _| |     _ _ __   __ _ _   _  __ _  ${NC}                                   ${VERT}"
echo -e "${VERT}   ${BLUE} | |    / _\` / __| / _\` | |    | | '_ \\ / _\` | | | |/ _\` | ${NC}                                   ${VERT}"
echo -e "${VERT}   ${BLUE} | |___| (_| \\__ \\ (_| | |____| | | | | (_| | |_| | (_| | ${NC}                                   ${VERT}"
echo -e "${VERT}   ${BLUE}  \\_____\\__,_|___/\\__,_|______|_|_| |_|\\__, |\\__,_|\\__,_| ${NC}                                   ${VERT}"
echo -e "${VERT}   ${BLUE}                                        |___/             ${NC}                                   ${VERT}"

# Print empty line
echo -e "${empty_line}"

# Print subtitle
subtitle="Startup Script"
subtitle_len=${#subtitle}
padding=$((banner_width - subtitle_len - 6))
padding_spaces=$(printf "%${padding}s")
echo -e "${VERT}   ${YELLOW}${subtitle}${padding_spaces}${NC}   ${VERT}"

# Print empty line
echo -e "${empty_line}"

# Print bottom border
echo -e "${BL}${horiz_line}${BR}\n"

# Default values
ENV="development"
PORT=8000
HOST="0.0.0.0"
WORKERS=4
RELOAD=true
FLUSH_CACHE=true

# ------------------------
# Parse command line arguments
# ------------------------
# This section processes input flags and options provided when invoking the script.
# It allows overriding defaults such as environment, port, host, number of workers,
# and toggling reload or cache flush behavior.
while [[ $# -gt 0 ]]; do
  case $1 in
    --env)
      ENV="$2"
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    --host)
      HOST="$2"
      shift 2
      ;;
    --workers)
      WORKERS="$2"
      shift 2
      ;;
    --no-reload)
      RELOAD=false
      shift
      ;;
    --no-flush)
      FLUSH_CACHE=false
      shift
      ;;
    *)
      echo -e "${RED}Unknown argument: $1${NC}"
      exit 1
      ;;
  esac
done

# ------------------------
# Set environment variable
# ------------------------
# Export the selected environment so that the application and dependencies can
# adjust their behavior accordingly (e.g., dev, staging, production).
export ENVIRONMENT=$ENV
echo -e "${BLUE}Setting environment to: ${YELLOW}$ENV${NC}"

# ------------------------
# Activate virtual environment
# ------------------------
# If a Python virtual environment is not already activated, attempt to activate it.
# This ensures the application runs with the correct dependencies isolated from the system.
if [ -z "$VIRTUAL_ENV" ]; then
  echo -e "${BLUE}Activating virtual environment...${NC}"
  if [ -d "venv" ]; then
    source venv/bin/activate
  else
    echo -e "${YELLOW}Virtual environment not found. Continuing without it...${NC}"
  fi
fi

# ------------------------
# Flush caches and clean temporary files (optional)
# ------------------------
# If enabled, this step clears model caches, Python bytecode, pytest cache, coverage reports,
# and archives large log files to prevent disk bloat and ensure fresh application state.
if [ "$FLUSH_CACHE" = true ]; then
  echo -e "${BLUE}Flushing caches...${NC}"
  
  # Clear model cache
  if [ -d "cache/models" ]; then
    echo -e "${BLUE}Clearing model cache...${NC}"
    rm -rf cache/models/* || true
    echo -e "${GREEN}Model cache cleared.${NC}"
  fi
  
  # Clear temp files
  echo -e "${BLUE}Clearing temporary files...${NC}"
  find . -name "*.pyc" -delete
  find . -name "__pycache__" -type d -exec rm -rf {} +
  find . -name ".pytest_cache" -type d -exec rm -rf {} +
  rm -rf .coverage .coverage.* htmlcov/ .pytest_cache/
  echo -e "${GREEN}Temporary files cleared.${NC}"
  
  # Clear log files if they're too large
  echo -e "${BLUE}Checking log files...${NC}"
  MAX_LOG_SIZE=10485760  # 10MB
  
  for log_file in logs/*.log logs/audit/*.jsonl logs/metrics/*.json; do
    if [ -f "$log_file" ]; then
      file_size=$(stat -f%z "$log_file" 2>/dev/null || stat -c%s "$log_file" 2>/dev/null)
      if [ $? -eq 0 ] && [ $file_size -gt $MAX_LOG_SIZE ]; then
        echo -e "${YELLOW}Log file $log_file is large ($file_size bytes). Archiving...${NC}"
        mv "$log_file" "${log_file}.$(date +%Y%m%d%H%M%S).bak"
      fi
    fi
  done
  
  echo -e "${GREEN}Cache flush complete.${NC}"
fi

# ------------------------
# Check if language models are available
# ------------------------
# Warn the user if no language models are found, as this may impact application features.
# Provides instruction on how to download models if missing.
echo -e "${BLUE}Checking for language models...${NC}"
if [ ! -d "models" ] || [ -z "$(ls -A models 2>/dev/null)" ]; then
  echo -e "${YELLOW}Warning: No language models found. Some features may not work.${NC}"
  echo -e "${YELLOW}Run 'python -m app.tools.download_models --all' to download models.${NC}"
fi

# ------------------------
# Start the CasaLingua application server
# ------------------------
# Launches the Uvicorn ASGI server with the specified host, port, number of worker processes,
# and optional auto-reload for development convenience.
echo -e "${BLUE}Starting CasaLingua application...${NC}"
echo -e "${BLUE}Host: ${YELLOW}$HOST${BLUE}, Port: ${YELLOW}$PORT${BLUE}, Environment: ${YELLOW}$ENV${BLUE}, Workers: ${YELLOW}$WORKERS${NC}"

RELOAD_FLAG=""
if [ "$RELOAD" = true ]; then
  RELOAD_FLAG="--reload"
  echo -e "${BLUE}Auto-reload is enabled.${NC}"
fi

echo -e "${GREEN}Starting server...${NC}"
echo -e "${BLUE}Press Ctrl+C to stop the server.${NC}"
echo ""

# Start the server
uvicorn app.main:app --host $HOST --port $PORT --workers $WORKERS $RELOAD_FLAG

# This part will only execute if uvicorn is stopped
echo -e "${YELLOW}Server stopped.${NC}"

# Ladder Logic Diagram documenting startup steps
# ┌───────────────────────────────┐
# │      CasaLingua Startup       │
# └───────────────────────────────┘
#          │
#          ▼
# ┌───────────────────────┐
# │ Parse Command Line    │
# └───────────────────────┘
#          │
#          ▼
# ┌───────────────────────┐
# │ Set Environment       │
# └───────────────────────┘
#          │
#          ▼
# ┌───────────────────────┐
# │ Activate venv         │
# └───────────────────────┘
#          │
#          ▼
# ┌─────────────────────────────┐
# │ Flush Cache (optional)      │
# └─────────────────────────────┘
#          │
#          ▼
# ┌─────────────────────────────┐
# │ Check Models Exist          │
# └─────────────────────────────┘
#          │
#          ▼
# ┌─────────────────────────────┐
# │ Start Uvicorn Web Server    │
# └─────────────────────────────┘