#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Set development mode for auth bypass
export CASALINGUA_ENV="development"
export LOG_LEVEL="INFO"
export DEBUG="true"

# Base URL for API
BASE_URL="http://localhost:8000"

# Function to test a specific model
test_model() {
    local model_type="$1"
    local endpoint="$2"
    local payload="$3"
    local description="$4"
    
    echo -e "${BLUE}===== Testing ${model_type} Model =====${NC}"
    echo -e "${YELLOW}Endpoint: ${endpoint}${NC}"
    echo -e "${YELLOW}Description: ${description}${NC}"
    echo -e "${YELLOW}Payload: ${payload}${NC}"
    
    # Make API request
    response=$(curl -s -X POST "${BASE_URL}${endpoint}" \
                  -H "Content-Type: application/json" \
                  -d "${payload}")
    
    # Check if response contains error
    if echo "$response" | grep -q '"status":"error"' || echo "$response" | grep -q '"detail":'; then
        echo -e "${RED}✗ Test failed for ${model_type}${NC}"
        echo -e "${RED}Response: ${response}${NC}"
        return 1
    else
        echo -e "${GREEN}✓ Test passed for ${model_type}${NC}"
        echo -e "${GREEN}Response (truncated): ${response:0:300}...${NC}"
        return 0
    fi
}

# Header
echo -e "${PURPLE}======================================================${NC}"
echo -e "${PURPLE}         CASALINGUA MODEL FUNCTIONALITY TEST          ${NC}"
echo -e "${PURPLE}======================================================${NC}"
echo -e "${YELLOW}Setting CASALINGUA_ENV=development for auth bypass${NC}"
echo ""

# Test all models

# 1. Language Detection Model
test_model "Language Detection" "/pipeline/detect" \
    '{"text": "Hello, this is a test of the language detection system."}' \
    "Tests the language detection model with English text"

# 2. Translation Model
test_model "Translation" "/pipeline/translate" \
    '{"text": "Hello, this is a test of the translation system.", "source_language": "en", "target_language": "es"}' \
    "Tests the translation model from English to Spanish"

# 3. NER Detection Model
test_model "NER Detection" "/pipeline/analyze" \
    '{"text": "John Smith works at Microsoft in Seattle.", "language": "en", "analyses": ["entities"]}' \
    "Tests the named entity recognition model with people and organization names"

# 4. Simplifier Model
test_model "Simplifier" "/pipeline/simplify" \
    '{"text": "The intricate mechanisms of quantum physics elude comprehension by many individuals.", "language": "en", "target_level": "simple"}' \
    "Tests the text simplification model with complex English text"

# 5. RAG Generator Model
test_model "RAG Generator" "/rag/query" \
    '{"query": "What is machine learning?", "max_results": 1}' \
    "Tests the RAG generator with a simple query"

# 6. Anonymizer Model
test_model "Anonymizer" "/pipeline/anonymize" \
    '{"text": "My name is John Smith and my email is john.smith@example.com.", "language": "en"}' \
    "Tests the anonymizer model with PII data"

# Summary
echo ""
echo -e "${PURPLE}======================================================${NC}"
echo -e "${PURPLE}                 TEST SUMMARY                         ${NC}"
echo -e "${PURPLE}======================================================${NC}"
echo -e "${YELLOW}Note: Some endpoints may not be implemented or may have implementation issues.${NC}"
echo -e "${YELLOW}The test focuses on basic functionality of each model type.${NC}"
echo -e "${YELLOW}Even if some tests fail, the models themselves may be loaded correctly.${NC}"

# Final reminder
echo ""
echo -e "${YELLOW}To check if models are loaded correctly:${NC}"
echo -e "curl -s ${BASE_URL}/health/models | grep 'loaded_models'"

# Reset environment variables
unset CASALINGUA_ENV LOG_LEVEL DEBUG