# CasaLingua Examples

This section provides examples and code snippets for using CasaLingua's API endpoints and features.

## Contents

1. [Translation Examples](./translation-examples.md)
2. [Simplification Examples](./simplification-examples.md)
3. [Language Detection Examples](./language-detection-examples.md)
4. [Verification Examples](./verification-examples.md)
5. [Auto-fixing Examples](./auto-fixing-examples.md)
6. [Legal Housing Examples](./legal-housing-examples.md)
7. [Batch Processing Examples](./batch-processing-examples.md)

## Basic Examples

### Translation Example

```python
import requests
import json

url = "http://localhost:8000/pipeline/translate"
headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer YOUR_API_KEY"
}
data = {
    "text": "The lease agreement requires tenants to maintain the property in good condition.",
    "source_language": "en",
    "target_language": "es",
    "preserve_formatting": True
}

response = requests.post(url, headers=headers, data=json.dumps(data))
result = response.json()

print(f"Original: {data['text']}")
print(f"Translation: {result['data']['translated_text']}")
```

### Simplification Example

```python
import requests
import json

url = "http://localhost:8000/pipeline/simplify"
headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer YOUR_API_KEY"
}
data = {
    "text": "The tenant shall indemnify and hold harmless the landlord from and against any and all claims, actions, suits, judgments and demands brought or recovered against the landlord by reason of any negligent or willful act or omission of the tenant.",
    "language": "en",
    "target_level": "simple",
    "domain": "legal-housing",
    "verify_output": True,
    "auto_fix": True
}

response = requests.post(url, headers=headers, data=json.dumps(data))
result = response.json()

print(f"Original: {data['text']}")
print(f"Simplified: {result['data']['simplified_text']}")
```

### Language Detection Example

```python
import requests
import json

url = "http://localhost:8000/pipeline/detect"
headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer YOUR_API_KEY"
}
data = {
    "text": "The quick brown fox jumps over the lazy dog.",
    "detailed": True
}

response = requests.post(url, headers=headers, data=json.dumps(data))
result = response.json()

print(f"Detected language: {result['data']['detected_language']}")
print(f"Confidence: {result['data']['confidence']}")
```

## Using the Examples

Each example in this directory demonstrates a specific feature or capability of CasaLingua:

1. **Code Snippets**: Ready-to-use code in Python, JavaScript, and cURL
2. **Sample Data**: Realistic data for testing each endpoint
3. **Expected Responses**: What to expect from the API
4. **Error Handling**: How to handle common errors

## API Keys

For the examples to work, you'll need to replace `YOUR_API_KEY` with a valid API key. You can generate one using:

```bash
curl -X POST http://localhost:8000/admin/api-keys \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ADMIN_KEY" \
  -d '{
    "name": "Example API Key",
    "scopes": ["translation:read", "translation:write", "simplification:read", "simplification:write"]
  }'
```

## Demo Environment

If you don't have a local installation, you can use our demo environment for testing:

```
https://demo.casalingua.example.com/
```

Note: The demo environment has rate limits and may not include all features.

## Additional Resources

- [API Reference](../api/README.md): Comprehensive API documentation
- [Getting Started Guide](../getting-started.md): Setup and basic usage
- [Legal Housing Guide](../guides/housing-legal.md): Specialized handling for housing legal documents