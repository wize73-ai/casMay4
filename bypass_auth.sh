#!/bin/bash

# Set environment variable for development mode
export CASALINGUA_ENV=development

# Run a test to see if auth bypass works
echo "Testing auth bypass with CASALINGUA_ENV=development set"
curl -s -X POST http://localhost:8000/pipeline/detect \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, this is a test of the language detection system."}' \
  | jq

# Restart the server with debug logging for auth
echo "Restarting server with debug logging for auth..."
cd /Users/jameswilson/Desktop/PRODUCTION/may4/wip-may30
LOG_LEVEL=DEBUG python -m app.main