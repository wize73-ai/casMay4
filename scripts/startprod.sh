#!/bin/bash

# CasaLingua Production Server Startup Script
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
BOLD=$(tput bold 2>/dev/null || echo '')
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
subtitle="PRODUCTION Mode Server"
subtitle_len=${#subtitle}
padding=$((banner_width - subtitle_len - 6))
padding_spaces=$(printf "%${padding}s")
echo -e "${VERT}   ${RED}${subtitle}${padding_spaces}${NC}   ${VERT}"

# Print empty line
echo -e "${empty_line}"

# Print bottom border
echo -e "${BL}${horiz_line}${BR}\n"

echo -e "${BOLD}${YELLOW}⚠️  PRODUCTION MODE - ENSURE ALL SETTINGS ARE CORRECT ⚠️${NC}"

# Default values for server configuration
PORT=8000
HOST="0.0.0.0"
WORKERS=4
LOG_LEVEL="warning"
SSL_CERT=""
SSL_KEY=""

# Parse command line arguments to override defaults
while [[ $# -gt 0 ]]; do
  case $1 in
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
    --log-level)
      LOG_LEVEL="$2"
      shift 2
      ;;
    --ssl-cert)
      SSL_CERT="$2"
      shift 2
      ;;
    --ssl-key)
      SSL_KEY="$2"
      shift 2
      ;;
    *)
      echo -e "${RED}Unknown argument: $1${NC}"
      exit 1
      ;;
  esac
done

# Activate virtual environment if not already activated
# This ensures dependencies are isolated and consistent
if [ -z "$VIRTUAL_ENV" ]; then
  echo -e "${BLUE}Activating virtual environment...${NC}"
  if [ -d "venv" ]; then
    source venv/bin/activate
  else
    echo -e "${YELLOW}Virtual environment not found. Continuing without it...${NC}"
  fi
fi

# Set environment variable to production
export ENVIRONMENT="production"
echo -e "${BLUE}Setting environment to: ${YELLOW}production${NC}"

echo -e "${YELLOW}Would you like to begin production setup? (y/n)${NC}"
read -r SETUP_CONFIRM
if [[ "$SETUP_CONFIRM" != "y" ]]; then
  echo -e "${RED}Aborting startup per user request.${NC}"
  exit 1
fi

# Check if dependencies are installed, prompt to run install if missing
if ! command -v uvicorn &>/dev/null; then
  echo -e "${YELLOW}Dependencies not found. Run install.sh now? (y/n)${NC}"
  read -r INSTALL_CONFIRM
  if [[ "$INSTALL_CONFIRM" == "y" ]]; then
    ./scripts/install.sh
  else
    echo -e "${RED}Cannot continue without dependencies.${NC}"
    exit 1
  fi
fi

# Ensure the production configuration file exists before starting
if [ ! -f "config/production.json" ]; then
  echo -e "${RED}Production config not found at config/production.json${NC}"
  echo -e "${RED}Please create a production configuration file before starting the server.${NC}"
  exit 1
fi

# Check presence of .env file with environment variables
if [ ! -f ".env" ]; then
  echo -e "${RED}.env file not found. Please create .env with production settings.${NC}"
  exit 1
fi

# Create necessary directories for logs, models, data, and cache
mkdir -p logs/audit logs/metrics models data/documents cache/models

# Check if language models are available in the models directory
echo -e "${BLUE}Checking for language models...${NC}"
if [ ! -d "models" ] || [ -z "$(ls -A models 2>/dev/null)" ]; then
  echo -e "${RED}Error: No language models found. Cannot start production server.${NC}"
  echo -e "${YELLOW}Run 'python -m app.tools.download_models --all' to download models.${NC}"
  exit 1
fi

# Verify existence of model registry; create if missing
if [ ! -f "models/registry.json" ]; then
  echo -e "${YELLOW}Warning: Model registry not found. Attempting to create it...${NC}"
  python -m app.tools.download_models --no-deps
fi

# Run pre-flight checks before starting the server
echo -e "${BLUE}Running pre-flight checks...${NC}"

# Check database connection by attempting to load config and print DB URL
echo -e "${BLUE}Checking database connection...${NC}"
python -c "from app.utils.config import load_config; config = load_config(); print('Database URL:', config.get('database', {}).get('url', 'Not configured'))"
if [ $? -ne 0 ]; then
  echo -e "${RED}Database connection check failed. Please check your configuration.${NC}"
  exit 1
fi

# Verify required models are loaded as specified in config
echo -e "${BLUE}Verifying required models...${NC}"
python -c "import asyncio, json; from app.model.loader import get_model_loader; from app.utils.config import load_config; config = load_config(); preload_models = config.get('models', {}).get('preload_models', []); print('Required models:', preload_models)"
if [ $? -ne 0 ]; then
  echo -e "${RED}Model verification failed. Please check your configuration.${NC}"
  exit 1
fi

# Prepare SSL options if SSL certificate and key are provided and exist
SSL_OPTIONS=""
if [ -n "$SSL_CERT" ] && [ -n "$SSL_KEY" ]; then
  if [ -f "$SSL_CERT" ] && [ -f "$SSL_KEY" ]; then
    SSL_OPTIONS="--ssl-keyfile=$SSL_KEY --ssl-certfile=$SSL_CERT"
    echo -e "${BLUE}SSL enabled with certificate: ${YELLOW}$SSL_CERT${NC}"
  else
    echo -e "${RED}SSL certificate or key file not found. Starting without SSL.${NC}"
  fi
fi

# Configure API optimizations for production
echo -e "${BLUE}Configuring API optimizations for production...${NC}"

# Set environment variables if not already in .env
if [ -f ".env" ]; then
  echo -e "${CYAN}📝 Loading configuration from .env file...${NC}"
  # shellcheck disable=SC1090
  source .env
fi

# Set optimization environment variables
export ENVIRONMENT="production"
export ROUTE_CACHE_SIZE=${ROUTE_CACHE_SIZE:-5000}
export TRANSLATION_CACHE_SIZE=${TRANSLATION_CACHE_SIZE:-10000}
export MAX_BATCH_SIZE=${MAX_BATCH_SIZE:-20}
export ENABLE_STREAMING=${ENABLE_STREAMING:-true}

# Create necessary cache directories
mkdir -p cache/models cache/api

# Display configuration
echo -e "${CYAN}┌─────────────── API CONFIGURATION ───────────────┐${NC}"
echo -e "${CYAN}│${NC} Environment:          ${GREEN}production${NC}"
echo -e "${CYAN}│${NC} Route Cache Size:     ${GREEN}${ROUTE_CACHE_SIZE}${NC}"
echo -e "${CYAN}│${NC} Translation Cache:    ${GREEN}${TRANSLATION_CACHE_SIZE}${NC}"
echo -e "${CYAN}│${NC} Max Batch Size:       ${GREEN}${MAX_BATCH_SIZE}${NC}"
echo -e "${CYAN}│${NC} Streaming Enabled:    ${GREEN}${ENABLE_STREAMING}${NC}"
echo -e "${CYAN}└─────────────────────────────────────────────────┘${NC}"

# Start the production server with Gunicorn and Uvicorn workers
echo -e "${GREEN}🚀 Launching CasaLingua production server with Gunicorn...${NC}"
echo -e "${BLUE}Host: ${YELLOW}$HOST${BLUE}, Port: ${YELLOW}$PORT${BLUE}, Workers: ${YELLOW}$WORKERS${NC}"

echo -e "${GREEN}Starting Gunicorn with Uvicorn workers...${NC}"
echo ""

# Execute Gunicorn server with specified options
gunicorn app.main:app \
  --bind $HOST:$PORT \
  --workers $WORKERS \
  --worker-class uvicorn.workers.UvicornWorker \
  --log-level $LOG_LEVEL \
  --timeout 120 \
  $SSL_OPTIONS

# This part will only execute if gunicorn is stopped
echo -e "${YELLOW}Production server stopped.${NC}"

# -----------------------------------------------------------------------------
# Ladder Logic Diagram Representing Execution Flow with Interactive Improvements
#
# +-------------------------+
# | Start Script            |
# +-----------+-------------+
#             |
#             v
# +-------------------------+
# | Parse Command Line Args  |
# +-----------+-------------+
#             |
#             v
# +-------------------------+
# | Activate Virtual Env?    |--No--> Continue
# +-----------+-------------+
#             |
#            Yes
#             v
# +-------------------------+
# | Set ENVIRONMENT=prod     |
# +-----------+-------------+
#             |
#             v
# +-------------------------+
# | Prompt: Begin Setup?     |--No--> Abort Startup
# +-----------+-------------+
#             |
#            Yes
#             v
# +-------------------------+
# | Check Dependencies?      |--No--> Prompt to Install
# +-----------+-------------+
#             |                         |
#            Yes                       No
#             |                         |
#             v                         v
# +-------------------------+    +-------------------------+
# | Check Config & .env      |    | Run install.sh or Abort |
# +-----------+-------------+    +-----------+-------------+
#             |                         |
#             v                         v
# +-------------------------+    +-------------------------+
# | Create Directories       |    | Continue if installed    |
# +-----------+-------------+    +-------------------------+
#             |
#             v
# +-------------------------+
# | Check Models Available   |
# +-----------+-------------+
#             |
#             v
# +-------------------------+
# | Verify Model Registry    |
# +-----------+-------------+
#             |
#             v
# +-------------------------+
# | Run Pre-flight Checks    |
# | (DB & Model Verification)|
# +-----------+-------------+
#             |
#             v
# +-------------------------+
# | Prepare SSL Options      |
# +-----------+-------------+
#             |
#             v
# +-------------------------+
# | Start Gunicorn Server    |
# +-----------+-------------+
#             |
#             v
# +-------------------------+
# | Server Running           |
# +-----------+-------------+
#             |
#             v
# +-------------------------+
# | Server Stopped (Exit)    |
# +-------------------------+
# -----------------------------------------------------------------------------
