#!/bin/bash
# Verify MT5 translation fix

# Set environment variables to bypass authentication
export CASALINGUA_ENV=development

# Set base URL - default to localhost:8002 if not provided
BASE_URL=${1:-http://localhost:8002}
echo "Testing MT5 translation fix at $BASE_URL"

# Test translation endpoint with different languages
languages=("es" "fr" "de" "it" "zh")
text="Hello, how are you?"

for lang in "${languages[@]}"; do
  echo -e "\n=== Testing translation to $lang ==="
  echo "Request: { \"text\": \"$text\", \"source_language\": \"en\", \"target_language\": \"$lang\" }"
  
  response=$(curl -s -X POST -H "Content-Type: application/json" -d "{
    \"text\": \"$text\",
    \"source_language\": \"en\",
    \"target_language\": \"$lang\"
  }" "$BASE_URL/pipeline/translate")
  
  # Check if the response contains "<extra_id_0>"
  if echo "$response" | grep -q "<extra_id_0>"; then
    echo "MT5 FIX NOT WORKING: Response contains <extra_id_0>"
    echo "$response" | jq '.' 2>/dev/null || echo "$response"
  else
    echo "MT5 FIX WORKING: Response does not contain <extra_id_0>"
    echo "$response" | jq '.' 2>/dev/null || echo "$response"
  fi
done

echo -e "\n=== MT5 translation fix test completed ==="