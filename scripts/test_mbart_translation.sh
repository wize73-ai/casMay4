#!/bin/bash

# Script to test MBART translation model functionality
# This script sends translation requests directly to MBART model
# instead of using MT5 with fallback

# Configuration
API_URL=${1:-"http://localhost:8002"}  # Default to localhost:8002
VERBOSE=${2:-"true"}                   # Whether to show detailed output

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Testing MBART Translation Model ===${NC}"
echo "API URL: $API_URL"
echo

# Function to perform translation
translate() {
    local source_lang=$1
    local target_lang=$2
    local text=$3
    local description=$4
    
    echo -e "${BLUE}Test: ${description}${NC}"
    echo "Translating from $source_lang to $target_lang: '$text'"
    
    # Send request to translation API - specifying mbart_translation model
    response=$(curl -s -X POST "$API_URL/pipeline/translate" \
        -H "Content-Type: application/json" \
        -d "{\"text\": \"$text\", \"source_language\": \"$source_lang\", \"target_language\": \"$target_lang\", \"model_name\": \"mbart_translation\"}")
    
    # Extract translated text
    translated_text=$(echo $response | grep -o '"translated_text":"[^"]*"' | sed 's/"translated_text":"//;s/"$//')
    model_used=$(echo $response | grep -o '"model_used":"[^"]*"' | sed 's/"model_used":"//;s/"$//')
    primary_model=$(echo $response | grep -o '"primary_model":"[^"]*"' | sed 's/"primary_model":"//;s/"$//')
    
    echo "Translated text: '$translated_text'"
    
    # Check if MBART was used
    if [ -n "$primary_model" ] && [ "$primary_model" = "mbart_translation" ]; then
        echo -e "${GREEN}Using MBART as primary model: YES${NC}"
    else
        echo -e "${YELLOW}Using MBART as primary model: NO${NC}"
    fi
    
    # Check if translation was successful
    if [ -z "$translated_text" ] || [ "$translated_text" = "None" ]; then
        echo -e "${RED}Translation failed!${NC}"
    else
        echo -e "${GREEN}Translation successful!${NC}"
    fi
    
    # Show full response if verbose
    if [ "$VERBOSE" = "true" ]; then
        echo "Full response:"
        echo "$response" | python -m json.tool
    fi
    
    echo
}

# Now test a variety of language pairs and text types

# Case 1: Simple greetings
translate "en" "es" "Hello, how are you today?" "Simple English to Spanish greeting"
translate "es" "en" "Hola, ¿cómo estás hoy?" "Simple Spanish to English greeting"

# Case 2: Complex technical text
translate "en" "de" "The quantum computing architecture leverages superconducting qubits to perform operations in superposition." "Complex technical text (EN to DE)"
translate "en" "fr" "Machine learning algorithms analyze patterns in data to make predictions without explicit programming." "Technical text (EN to FR)"

# Case 3: Text with idiomatic expressions
translate "en" "it" "It's raining cats and dogs outside." "Idiomatic expression (EN to IT)"
translate "fr" "en" "C'est la goutte d'eau qui fait déborder le vase." "Idiomatic expression (FR to EN)"

# Case 4: Long text
translate "en" "pt" "The integration of artificial intelligence and machine learning technologies into everyday applications continues to transform how we interact with digital systems. From recommendation engines to voice assistants, these technologies leverage vast amounts of data to provide increasingly personalized experiences." "Long text (EN to PT)"

# Case 5: Non-Latin script languages
translate "en" "zh" "Please translate this text to Chinese accurately and faithfully." "Translation to non-Latin script (EN to ZH)"
translate "zh" "en" "请准确地将这段文字翻译成英文。" "Translation from non-Latin script (ZH to EN)"
translate "en" "ja" "This is an example of text translation from English to Japanese." "Translation to Japanese (EN to JA)"
translate "ja" "en" "これは日本語から英語へのテキスト翻訳の例です。" "Translation from Japanese (JA to EN)"

# Case 6: Check if bad input is handled gracefully
translate "invalid" "es" "This should fail gracefully" "Invalid language code"
translate "en" "invalid" "This should also fail gracefully" "Invalid target language code"

# Case 7: Same language translation (should just return input)
translate "en" "en" "This text should be returned as-is." "Same language translation"

echo -e "${BLUE}=== Test Complete ===${NC}"