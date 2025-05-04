#!/bin/bash

# Run Health Check Test Script
# This script helps run the health check test scripts against a running CasaLingua instance

# Colors for better readability
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== CasaLingua Health Check Test Runner ===${NC}"
echo

# Check if CasaLingua is running
echo -e "${YELLOW}Checking if CasaLingua server is running...${NC}"
if ! nc -z localhost 8000 >/dev/null 2>&1; then
    echo -e "${RED}Error: CasaLingua server does not appear to be running on port 8000${NC}"
    echo
    echo -e "${YELLOW}Would you like to start the server? (y/n)${NC}"
    read -r start_server
    
    if [[ "$start_server" == "y" || "$start_server" == "Y" ]]; then
        echo -e "${BLUE}Starting CasaLingua server...${NC}"
        
        # Kill any existing processes on port 8000
        lsof -ti:8000 | xargs kill -9 2>/dev/null
        
        # Start the server in the background
        cd "$(dirname "$0")/.." || exit
        # Use Development server for faster startup during testing
        python -m app.main &
        
        # Wait for the server to start
        echo -e "${YELLOW}Waiting for server to start...${NC}"
        for i in {1..30}; do
            if nc -z localhost 8000 >/dev/null 2>&1; then
                echo -e "${GREEN}Server started successfully${NC}"
                break
            fi
            echo -n "."
            sleep 1
            
            if [ $i -eq 30 ]; then
                echo -e "\n${RED}Error: Server failed to start within 30 seconds${NC}"
                exit 1
            fi
        done
        echo
    else
        echo -e "${RED}Aborting test run. Please start the server manually.${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}CasaLingua server is running${NC}"
fi

# Run the health check test script
echo
echo -e "${BLUE}Running health check tests...${NC}"
python "$(dirname "$0")/test_health_simple.py" "$@"

echo
echo -e "${GREEN}Health check tests completed${NC}"

# Check if we started the server and prompt to shut it down
if [[ "$start_server" == "y" || "$start_server" == "Y" ]]; then
    echo
    echo -e "${YELLOW}Would you like to shut down the server? (y/n)${NC}"
    read -r shutdown_server
    
    if [[ "$shutdown_server" == "y" || "$shutdown_server" == "Y" ]]; then
        echo -e "${BLUE}Shutting down CasaLingua server...${NC}"
        lsof -ti:8000 | xargs kill -9 2>/dev/null
        echo -e "${GREEN}Server shut down successfully${NC}"
    else
        echo -e "${YELLOW}Server will continue running in the background${NC}"
    fi
fi