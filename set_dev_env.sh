#!/bin/bash

# Print header
echo -e "\033[1;36m========================================\033[0m"
echo -e "\033[1;36m  CASALINGUA DEVELOPMENT ENVIRONMENT   \033[0m"
echo -e "\033[1;36m========================================\033[0m"

# Set development environment variables
export CASALINGUA_ENV="development"
export LOG_LEVEL="DEBUG"
export DEBUG="true"

# Report settings
echo -e "\033[1;32mEnvironment variables set:\033[0m"
echo -e "  CASALINGUA_ENV = \033[1;33m$CASALINGUA_ENV\033[0m"
echo -e "  LOG_LEVEL      = \033[1;33m$LOG_LEVEL\033[0m"
echo -e "  DEBUG          = \033[1;33m$DEBUG\033[0m"

# Provide instructions
echo -e "\033[1;32mRun the server with:\033[0m"
echo -e "  python -m app.main"
echo 
echo -e "\033[1;32mTest the API with auth bypass:\033[0m"
echo -e "  curl -s -X POST http://localhost:8000/pipeline/translate \\
    -H \"Content-Type: application/json\" \\
    -d '{\"text\": \"Hello world\", \"source_language\": \"en\", \"target_language\": \"es\"}' | jq"
echo
echo -e "\033[1;32mReturn to production mode:\033[0m"
echo -e "  unset CASALINGUA_ENV LOG_LEVEL DEBUG"
echo -e "\033[1;36m========================================\033[0m"

# Export the variables in the current shell
echo "Environmental variables are now set in this shell session."