# Translation API

The Translation API allows you to translate text between supported languages with advanced options for verification, domain-specific terminology, and formatting preservation.

## Endpoint

```
POST /pipeline/translate
```

## Request Format

```json
{
  "text": "Text to translate",
  "source_language": "en",     // Optional, auto-detected if not provided
  "target_language": "es",     // Required
  "preserve_formatting": true, // Optional, defaults to true
  "model_name": "mbart_translation", // Optional, specific model to use
  "domain": "legal-housing",   // Optional, specify domain for specialized handling
  "glossary_id": "custom-glossary-123", // Optional, custom terminology
  "verify": false,            // Optional, enable veracity auditing
  "formality": "neutral"      // Optional, formality level (formal/neutral/informal)
}
```

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `text` | string | Yes | The text to translate |
| `source_language` | string | No | ISO language code of the source text (auto-detected if not provided) |
| `target_language` | string | Yes | ISO language code of the desired translation |
| `preserve_formatting` | boolean | No | Whether to preserve original formatting (default: true) |
| `model_name` | string | No | Specific model to use (e.g., "mbart_translation", "mt5") |
| `domain` | string | No | Domain specialization (e.g., "legal-housing", "medical", "general") |
| `glossary_id` | string | No | ID of a custom glossary to use for the translation |
| `verify` | boolean | No | Enable veracity auditing to verify translation quality (default: false) |
| `formality` | string | No | Desired formality level: "formal", "neutral", or "informal" (default: "neutral") |

## Response Format

```json
{
  "status": "success",
  "message": "Translation completed successfully",
  "data": {
    "source_text": "Text to translate",
    "translated_text": "Texto para traducir",
    "source_language": "en",
    "target_language": "es",
    "confidence": 0.95,
    "model_id": "mbart_translation",
    "model_used": "facebook/mbart-large-50-many-to-many-mmt",
    "word_count": 3,
    "character_count": 16,
    "detected_language": "en",
    "verified": true,
    "verification_score": 0.89,
    "process_time": 0.354
  },
  "metadata": {
    "request_id": "uuid-here",
    "timestamp": 1620160000,
    "version": "1.0.0",
    "process_time": 0.354,
    "cached": false
  }
}
```

### Response Data Fields

| Field | Type | Description |
|-------|------|-------------|
| `source_text` | string | The original text submitted for translation |
| `translated_text` | string | The translated text |
| `source_language` | string | The source language code (detected if not provided in request) |
| `target_language` | string | The target language code |
| `confidence` | number | Confidence score of the translation (0-1) |
| `model_id` | string | Identifier of the model used |
| `model_used` | string | Full name/path of the model used |
| `word_count` | number | Number of words in the source text |
| `character_count` | number | Number of characters in the source text |
| `detected_language` | string | Detected language if auto-detection was used |
| `verified` | boolean | Whether the translation passed verification (only if verify=true) |
| `verification_score` | number | Verification quality score (0-1, only if verify=true) |
| `process_time` | number | Time taken to process the translation (seconds) |

## Verification Features

When `verify=true`, the translation undergoes quality verification with the following checks:

1. **Semantic Equivalence**: Verifies that the meaning is preserved using embedding similarity
2. **Format Preservation**: Checks that formatting elements are maintained
3. **Number and Entity Preservation**: Ensures numbers and named entities are correctly translated
4. **Length Validation**: Confirms the translation length is appropriate for the language pair

Verification results are included in the response when enabled.

## Fallback Mechanism

The system may use fallback models if the primary model fails. In such cases, the response includes:

```json
"used_fallback": true,
"fallback_model": "mt5",
"primary_model": "mbart_translation"
```

## Errors

### Common Error Codes

| Status Code | Error Code | Description |
|-------------|------------|-------------|
| 400 | INVALID_LANGUAGE | Unsupported language specified |
| 400 | TEXT_TOO_LONG | Input text exceeds maximum length |
| 400 | MISSING_TARGET_LANGUAGE | Target language not specified |
| 500 | TRANSLATION_FAILED | Translation operation failed |
| 503 | MODEL_UNAVAILABLE | Required model is not available |

### Example Error Response

```json
{
  "status": "error",
  "message": "Unsupported target language: xyz",
  "error": {
    "code": "INVALID_LANGUAGE",
    "details": "The language code 'xyz' is not supported. Supported languages are: en, es, fr, de, ..."
  },
  "metadata": {
    "request_id": "uuid-here",
    "timestamp": 1620160000
  }
}
```

## Examples

### Basic Translation

```bash
curl -X POST http://localhost:8000/pipeline/translate \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "text": "Hello, how are you?",
    "source_language": "en",
    "target_language": "es"
  }'
```

### Legal Document Translation with Verification

```bash
curl -X POST http://localhost:8000/pipeline/translate \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "text": "The tenant shall indemnify and hold harmless the landlord from any claims.",
    "source_language": "en",
    "target_language": "es",
    "domain": "legal-housing",
    "verify": true,
    "formality": "formal"
  }'
```

## Supported Languages

The translation API supports the following language pairs:

| Language | Code | Translation From | Translation To |
|----------|------|-----------------|----------------|
| English | en | Yes | Yes |
| Spanish | es | Yes | Yes |
| French | fr | Yes | Yes |
| German | de | Yes | Yes |
| Chinese (Simplified) | zh | Yes | Yes |
| Italian | it | Yes | Yes |
| Japanese | ja | Yes | Yes |
| Korean | ko | Yes | Yes |
| Portuguese | pt | Yes | Yes |
| Russian | ru | Yes | Yes |
| Arabic | ar | Yes | Yes |
| Hindi | hi | Yes | Yes |
| ... | ... | ... | ... |

*For a complete list of supported languages, use the `/admin/languages` endpoint.*