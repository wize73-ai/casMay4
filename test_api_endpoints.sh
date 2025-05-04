#!/bin/bash
# API Endpoint Testing Script (Shell Version)
#
# This script tests all available API endpoints in the CasaLingua application using curl.
# It provides a simple alternative to the Python-based test script.
#
# Usage:
#   ./test_api_endpoints.sh [BASE_URL]
#
# Options:
#   BASE_URL    Base URL of the API (default: http://localhost:8000)

# Set base URL - default to localhost:8000 if not provided
BASE_URL=${1:-http://localhost:8000}
echo "Testing API endpoints at $BASE_URL"

# Enable auth bypass for development environment
export CASALINGUA_ENV=development
echo "Set CASALINGUA_ENV=development for auth bypass"

# Colors for better readability
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
HEADER='\033[0;35m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Counters for summary
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# Function to print section headers
print_header() {
  echo -e "\n${HEADER}${BOLD}=== $1 ===${NC}"
}

# Function to make a GET request and display results
test_get_endpoint() {
  local endpoint=$1
  local description=$2
  local expected_status=${3:-200}
  
  TOTAL_TESTS=$((TOTAL_TESTS + 1))
  echo -e "\n${BLUE}Testing GET $endpoint - $description${NC}"
  
  # Make the request and capture status code
  response=$(curl -s -w "%{http_code}" -X GET "$BASE_URL$endpoint")
  status_code=${response: -3}
  response_body=${response:0:${#response}-3}
  
  # Check if status code matches expected
  if [ "$status_code" == "$expected_status" ]; then
    echo -e "${GREEN}✓ Status: $status_code (Expected: $expected_status)${NC}"
    PASSED_TESTS=$((PASSED_TESTS + 1))
  else
    echo -e "${RED}✗ Status: $status_code (Expected: $expected_status)${NC}"
    FAILED_TESTS=$((FAILED_TESTS + 1))
  fi
  
  # Print a truncated response for viewing
  if [ ${#response_body} -gt 300 ]; then
    echo -e "${YELLOW}Response (truncated): ${response_body:0:300}...${NC}"
  else
    echo -e "${YELLOW}Response: ${response_body}${NC}"
  fi
}

# Function to make a POST request and display results
test_post_endpoint() {
  local endpoint=$1
  local description=$2
  local data=$3
  local expected_status=${4:-200}
  
  TOTAL_TESTS=$((TOTAL_TESTS + 1))
  echo -e "\n${BLUE}Testing POST $endpoint - $description${NC}"
  
  # Make the request and capture status code
  response=$(curl -s -w "%{http_code}" -X POST -H "Content-Type: application/json" -d "$data" "$BASE_URL$endpoint")
  status_code=${response: -3}
  response_body=${response:0:${#response}-3}
  
  # Check if status code matches expected
  if [ "$status_code" == "$expected_status" ]; then
    echo -e "${GREEN}✓ Status: $status_code (Expected: $expected_status)${NC}"
    PASSED_TESTS=$((PASSED_TESTS + 1))
  else
    echo -e "${RED}✗ Status: $status_code (Expected: $expected_status)${NC}"
    FAILED_TESTS=$((FAILED_TESTS + 1))
  fi
  
  # Try to format JSON if possible
  if command -v jq &> /dev/null; then
    echo -e "${YELLOW}Response:${NC}"
    echo "$response_body" | jq '.' 2>/dev/null || echo -e "${YELLOW}$response_body${NC}"
  else
    # Print a truncated response for viewing if jq is not available
    if [ ${#response_body} -gt 300 ]; then
      echo -e "${YELLOW}Response (truncated): ${response_body:0:300}...${NC}"
    else
      echo -e "${YELLOW}Response: ${response_body}${NC}"
    fi
  fi
}

# Test Health and Status Endpoints
print_header "Testing Health & Status Endpoints"
test_get_endpoint "/health" "Basic health check"
test_get_endpoint "/health/detailed" "Detailed health check"
test_get_endpoint "/health/models" "Model health check"
test_get_endpoint "/readiness" "Readiness probe"
test_get_endpoint "/liveness" "Liveness probe"

# Test Translation Endpoints
print_header "Testing Translation Endpoints"
test_post_endpoint "/pipeline/translate" "Basic text translation (EN to ES)" '{
  "text": "Hello, how are you?",
  "source_language": "en",
  "target_language": "es"
}'

test_post_endpoint "/pipeline/translate" "Translation with auto-detection (FR to EN)" '{
  "text": "Bonjour, comment ça va?",
  "source_language": "auto",
  "target_language": "en"
}'

test_post_endpoint "/pipeline/translate/batch" "Batch translation (EN to ES)" '{
  "texts": ["Hello, how are you?", "The weather is nice today"],
  "source_language": "en",
  "target_language": "es"
}'

# Test Language Detection Endpoints
print_header "Testing Language Detection Endpoints"
test_post_endpoint "/pipeline/detect" "Language detection (English)" '{
  "text": "Hello, how are you?"
}'

test_post_endpoint "/pipeline/detect" "Detailed language detection (Spanish)" '{
  "text": "Hola, ¿cómo estás?",
  "detailed": true
}'

test_post_endpoint "/pipeline/detect-language" "Language detection alias endpoint (German)" '{
  "text": "Guten Tag, wie geht es Ihnen?"
}'

# Test Text Simplification Endpoint
print_header "Testing Text Simplification Endpoint"
test_post_endpoint "/pipeline/simplify" "Text simplification (English)" '{
  "text": "The mitochondrion is a double membrane-bound organelle found in most eukaryotic organisms. Mitochondria use aerobic respiration to generate most of the cells supply of adenosine triphosphate (ATP), which is used as a source of chemical energy.",
  "language": "en",
  "target_level": "simple"
}'

# Test Text Anonymization Endpoint
print_header "Testing Text Anonymization Endpoint"
test_post_endpoint "/pipeline/anonymize" "Text anonymization (English)" '{
  "text": "John Smith lives at 123 Main St, New York and his email is john.smith@example.com. His phone number is (555) 123-4567.",
  "language": "en",
  "strategy": "mask"
}'

# Test Text Analysis Endpoints
print_header "Testing Text Analysis Endpoints"
test_post_endpoint "/pipeline/analyze" "Sentiment analysis (Positive)" '{
  "text": "I love this product! It is amazing and works really well.",
  "language": "en",
  "analyses": ["sentiment"]
}'

test_post_endpoint "/pipeline/analyze" "Entity recognition" '{
  "text": "Apple Inc. is headquartered in Cupertino, California and was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne.",
  "language": "en",
  "analyses": ["entities"]
}'

test_post_endpoint "/pipeline/analyze" "Multiple analyses" '{
  "text": "Google announced a new partnership with Microsoft yesterday. The tech giants will collaborate on AI research.",
  "language": "en",
  "analyses": ["sentiment", "entities", "topics"]
}'

# Test Text Summarization Endpoint
print_header "Testing Text Summarization Endpoint"
test_post_endpoint "/pipeline/summarize" "Text summarization (English)" '{
  "text": "Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to natural intelligence displayed by animals including humans. AI research has been defined as the field of study of intelligent agents, which refers to any system that perceives its environment and takes actions that maximize its chance of achieving its goals. The term artificial intelligence had previously been used to describe machines that mimic and display human cognitive skills that are associated with the human mind, such as learning and problem-solving. This definition has since been rejected by major AI researchers who now describe AI in terms of rationality and acting rationally, which does not limit how intelligence can be articulated. AI applications include advanced web search engines (Google Search), recommendation systems (YouTube, Amazon, and Netflix), understanding human speech (Siri and Alexa), self-driving cars (Tesla), generative or creative tools (ChatGPT and AI art), automated decision-making, and competing at the highest level in strategic game systems (chess and Go).",
  "language": "en"
}'

# Print test summary
print_header "Test Summary"
PASS_RATE=$(echo "scale=2; ($PASSED_TESTS * 100) / $TOTAL_TESTS" | bc)

echo -e "${CYAN}${BOLD}Total Tests: $TOTAL_TESTS${NC}"
echo -e "${GREEN}${BOLD}Passed: $PASSED_TESTS${NC}"
echo -e "${RED}${BOLD}Failed: $FAILED_TESTS${NC}"
echo -e "${CYAN}${BOLD}Pass Rate: ${PASS_RATE}%${NC}"

# Reset environment variable
unset CASALINGUA_ENV
echo -e "\nUnset CASALINGUA_ENV environment variable"

# Exit with status code based on pass rate
if [ "$FAILED_TESTS" -eq 0 ]; then
  echo -e "${GREEN}${BOLD}All tests passed!${NC}"
  exit 0
else
  echo -e "${RED}${BOLD}Some tests failed.${NC}"
  exit 1
fi