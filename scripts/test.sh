#!/bin/bash

# CasaLingua Test Runner Script
# Author: Exygy Development Team
# Version: 1.1.0
# License: MIT

set -e  # Exit on error

# Color codes for pretty output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Banner
echo -e "${BLUE}"
echo "========================================"
echo "      CasaLingua Test Runner Script     "
echo "========================================"
echo -e "${NC}"

# Default values
ENV="test"
TEST_PATH="tests/"
COVERAGE=true
VERBOSE=false
PARALLEL=false
SKIP_SLOW=false
MARK=""
COLLECT_ONLY=false
FAIL_FAST=false
HTML_REPORT=true
TEST_OPTIMIZATIONS=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --path)
      TEST_PATH="$2"
      shift 2
      ;;
    --no-coverage)
      COVERAGE=false
      shift
      ;;
    --verbose|-v)
      VERBOSE=true
      shift
      ;;
    --parallel|-xvs)
      PARALLEL=true
      shift
      ;;
    --skip-slow)
      SKIP_SLOW=true
      shift
      ;;
    --mark|-m)
      MARK="$2"
      shift 2
      ;;
    --collect-only)
      COLLECT_ONLY=true
      shift
      ;;
    --fail-fast)
      FAIL_FAST=true
      shift
      ;;
    --no-html)
      HTML_REPORT=false
      shift
      ;;
    --unit)
      TEST_PATH="tests/unit"
      shift
      ;;
    --integration)
      TEST_PATH="tests/integration"
      shift
      ;;
    --api)
      TEST_PATH="tests/api"
      shift
      ;;
    --models)
      TEST_PATH="tests/models"
      shift
      ;;
    --optimizations)
      TEST_OPTIMIZATIONS=true
      TEST_PATH="app/tests/"
      shift
      ;;
    *)
      echo -e "${RED}Unknown argument: $1${NC}"
      echo -e "${YELLOW}Available options:${NC}"
      echo -e "  --path PATH          Specify test path"
      echo -e "  --no-coverage        Disable coverage reporting"
      echo -e "  --verbose, -v        Enable verbose output"
      echo -e "  --parallel, -xvs     Run tests in parallel"
      echo -e "  --skip-slow          Skip slow tests"
      echo -e "  --mark, -m MARK      Only run tests with given mark"
      echo -e "  --collect-only       Only collect tests, don't run them"
      echo -e "  --fail-fast          Stop after first failure"
      echo -e "  --no-html            Disable HTML coverage report"
      echo -e "  --unit               Run only unit tests"
      echo -e "  --integration        Run only integration tests"
      echo -e "  --api                Run only API tests"
      echo -e "  --models             Run only model tests"
      echo -e "  --optimizations      Run API optimization tests (route cache, batch optimizer)"
      exit 1
      ;;
  esac
done

# Activate virtual environment if not already activated
if [ -z "$VIRTUAL_ENV" ]; then
  echo -e "${BLUE}Activating virtual environment...${NC}"
  
  # First try .venv (from new install script)
  if [ -d ".venv" ]; then
    echo -e "${GREEN}Found virtual environment at .venv${NC}"
    source .venv/bin/activate
  # Then try venv (from older installations)
  elif [ -d "venv" ]; then
    echo -e "${GREEN}Found virtual environment at venv${NC}"
    source venv/bin/activate
  else
    echo -e "${YELLOW}Virtual environment not found. Continuing without it...${NC}"
  fi
fi

# Set environment
export ENVIRONMENT=$ENV
echo -e "${BLUE}Setting environment to: ${YELLOW}$ENV${NC}"

# Make sure test dependencies are installed
echo -e "${BLUE}Checking test dependencies...${NC}"
pip list | grep -q pytest || pip install pytest pytest-cov pytest-asyncio pytest-xdist

# Install optimization test dependencies if needed
if [ "$TEST_OPTIMIZATIONS" = true ]; then
  echo -e "${BLUE}Checking optimization test dependencies...${NC}"
  pip list | grep -q nest-asyncio || pip install nest-asyncio
  pip list | grep -q aiohttp || pip install aiohttp
  pip list | grep -q httpx || pip install httpx
  
  # Check for route cache and batch optimizer packages
  echo -e "${BLUE}Verifying optimization packages...${NC}"
  # If any packages are missing, inform the user but continue (they may be included in the app)
  pip list | grep -q "cachetools" || echo -e "${YELLOW}Warning: cachetools not installed (may be required for route cache)${NC}"
  pip list | grep -q "aiomultiprocess" || echo -e "${YELLOW}Warning: aiomultiprocess not installed (may be required for parallel processing)${NC}"
fi

# Prepare pytest command
PYTEST_CMD="python -m pytest"

# Add path
PYTEST_CMD="$PYTEST_CMD $TEST_PATH"

# Add options
if [ "$VERBOSE" = true ]; then
  PYTEST_CMD="$PYTEST_CMD -v"
fi

if [ "$PARALLEL" = true ]; then
  PYTEST_CMD="$PYTEST_CMD -xvs"
fi

if [ "$SKIP_SLOW" = true ]; then
  PYTEST_CMD="$PYTEST_CMD -m 'not slow'"
elif [ -n "$MARK" ]; then
  PYTEST_CMD="$PYTEST_CMD -m '$MARK'"
fi

if [ "$COLLECT_ONLY" = true ]; then
  PYTEST_CMD="$PYTEST_CMD --collect-only"
fi

if [ "$FAIL_FAST" = true ]; then
  PYTEST_CMD="$PYTEST_CMD -x"
fi

# Add coverage options
if [ "$COVERAGE" = true ]; then
  PYTEST_CMD="$PYTEST_CMD --cov=app --cov-report=term"
  if [ "$HTML_REPORT" = true ]; then
    PYTEST_CMD="$PYTEST_CMD --cov-report=html"
  fi
fi

# Display the command
echo -e "${BLUE}Running tests with command:${NC}"
echo -e "${CYAN}$PYTEST_CMD${NC}"
echo

# Set up test database if needed
if [ -f "app/tools/db_init.py" ]; then
  echo -e "${BLUE}Setting up test database...${NC}"
  TEST_DB=true python -m app.tools.db_init
fi

# Run the tests
echo -e "${BLUE}Running tests...${NC}"
echo -e "${YELLOW}=======================================${NC}"

if [ "$TEST_OPTIMIZATIONS" = true ]; then
  echo -e "${BLUE}Running API optimization tests...${NC}"
  
  # Set environment variables for optimizations
  export ROUTE_CACHE_SIZE=500
  export TRANSLATION_CACHE_SIZE=1000
  export MAX_BATCH_SIZE=10
  export ENABLE_STREAMING=true
  
  # Run the custom test runner
  python -m app.tests.run_tests
  TEST_RESULT=$?
else
  # Run standard tests
  eval $PYTEST_CMD
  TEST_RESULT=$?
fi

echo -e "${YELLOW}=======================================${NC}"

# Check test result
if [ $TEST_RESULT -eq 0 ]; then
  echo -e "${GREEN}All tests passed successfully!${NC}"
else
  echo -e "${RED}Tests failed with exit code $TEST_RESULT.${NC}"
fi

# Display coverage report location if generated
if [ "$COVERAGE" = true ] && [ "$HTML_REPORT" = true ]; then
  echo -e "${BLUE}Coverage HTML report generated in ${YELLOW}htmlcov/index.html${NC}"
  # Optional: auto-open HTML coverage report in browser (macOS only)
  if command -v open &>/dev/null; then
    open htmlcov/index.html
  fi
fi

# Run additional checks
if [ $TEST_RESULT -eq 0 ]; then
  echo -e "${BLUE}Running additional checks...${NC}"
  
  # Check for flake8
  if command -v flake8 &>/dev/null; then
    echo -e "${BLUE}Running flake8 code style checks...${NC}"
    flake8 app/ || true
  fi
  
  # Check for mypy
  if command -v mypy &>/dev/null; then
    echo -e "${BLUE}Running mypy type checks...${NC}"
    mypy app/ || true
  fi
fi

# Function to generate test report
generate_test_report() {
  REPORT_FILE="test_report_$(date +%Y%m%d_%H%M%S).txt"
  echo -e "${BLUE}Generating test report: ${YELLOW}$REPORT_FILE${NC}"
  
  echo "CasaLingua Test Report - $(date)" > $REPORT_FILE
  echo "===================================" >> $REPORT_FILE
  echo "Test Path: $TEST_PATH" >> $REPORT_FILE
  echo "Coverage: $COVERAGE" >> $REPORT_FILE
  echo "Environment: $ENV" >> $REPORT_FILE
  
  # Include optimization test information if applicable
  if [ "$TEST_OPTIMIZATIONS" = true ]; then
    echo "API Optimizations: Yes" >> $REPORT_FILE
    echo "  - Route Cache Size: $ROUTE_CACHE_SIZE" >> $REPORT_FILE
    echo "  - Translation Cache Size: $TRANSLATION_CACHE_SIZE" >> $REPORT_FILE
    echo "  - Max Batch Size: $MAX_BATCH_SIZE" >> $REPORT_FILE
    echo "  - Streaming Enabled: $ENABLE_STREAMING" >> $REPORT_FILE
  else
    echo "API Optimizations: No" >> $REPORT_FILE
  fi
  
  echo "===================================" >> $REPORT_FILE
  echo "" >> $REPORT_FILE
  
  if [ "$COVERAGE" = true ]; then
    echo "Coverage Summary:" >> $REPORT_FILE
    coverage report >> $REPORT_FILE
    echo "" >> $REPORT_FILE
  fi
  
  echo "Test Results:" >> $REPORT_FILE
  if [ $TEST_RESULT -eq 0 ]; then
    echo "All tests passed successfully." >> $REPORT_FILE
  else
    echo "Tests failed with exit code $TEST_RESULT." >> $REPORT_FILE
  fi
  
  # Add optimization-specific notes if applicable
  if [ "$TEST_OPTIMIZATIONS" = true ]; then
    echo "" >> $REPORT_FILE
    echo "API Optimization Notes:" >> $REPORT_FILE
    echo "- Route Cache: Thread-safe in-memory caching system for API responses" >> $REPORT_FILE
    echo "- Batch Optimizer: Automatic grouping of similar small requests" >> $REPORT_FILE
    echo "- Streaming: Chunk-based processing for large documents" >> $REPORT_FILE
    echo "- See OPTIMIZATIONS.md for detailed information" >> $REPORT_FILE
  fi
  
  echo -e "${GREEN}Test report generated: ${YELLOW}$REPORT_FILE${NC}"
}

# Ask if user wants to generate a report
if [ "$COLLECT_ONLY" = false ]; then
  echo -e "${BLUE}Do you want to generate a test report? [y/N]${NC}"
  read -r GENERATE_REPORT
  if [[ $GENERATE_REPORT =~ ^[Yy]$ ]]; then
    generate_test_report
  fi
fi

exit $TEST_RESULT