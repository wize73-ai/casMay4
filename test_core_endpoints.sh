#!/bin/bash
# Test core functionality endpoints

# Set base URL - default to localhost:8001 if not provided
BASE_URL=${1:-http://localhost:8001}
echo "Testing core functionality endpoints at $BASE_URL"

# Function to make a POST request and display results
test_post_endpoint() {
  local endpoint=$1
  local description=$2
  local data=$3
  
  echo -e "\n=== Testing $description ($endpoint) ==="
  echo "Request: $data"
  echo "Response:"
  curl -s -X POST -H "Content-Type: application/json" -d "$data" "$BASE_URL$endpoint" | jq '.' 2>/dev/null || echo "Failed to parse response as JSON"
}

# Test translation endpoint with a simple text
test_post_endpoint "/pipeline/translate" "Text translation" '{
  "text": "Hello, how are you?",
  "source_language": "en",
  "target_language": "es"
}'

# Test batch translation
test_post_endpoint "/pipeline/translate/batch" "Batch translation" '{
  "texts": ["Hello, how are you?", "The weather is nice today"],
  "source_language": "en",
  "target_language": "es"
}'

# Test language detection
test_post_endpoint "/pipeline/detect" "Language detection" '{
  "text": "Hello, how are you?"
}'

# Test language detection with another language
test_post_endpoint "/pipeline/detect" "Language detection (Spanish)" '{
  "text": "Hola, ¿cómo estás?"
}'

# Test text analysis
test_post_endpoint "/pipeline/analyze" "Text analysis" '{
  "text": "Hello, my name is John. I live in New York and work at Google.",
  "analyses": ["sentiment", "entities"]
}'

# Test RAG-enhanced translation
test_post_endpoint "/rag/translate" "RAG-enhanced translation" '{
  "text": "The housing application needs to be submitted by next Friday.",
  "source_language": "en",
  "target_language": "es"
}'

echo -e "\n=== Core functionality tests completed ==="