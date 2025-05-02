#!/bin/bash

# CasaLingua Test Runner Script
# Author: Exygy Development Team
# Version: 1.0.0
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
      exit 1
      ;;
  esac
done

# Activate virtual environment if not already activated
if [ -z "$VIRTUAL_ENV" ]; then
  echo -e "${BLUE}Activating virtual environment...${NC}"
  if [ -d "venv" ]; then
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
eval $PYTEST_CMD
TEST_RESULT=$?
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