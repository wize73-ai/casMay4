# Language Detection API

The Language Detection API identifies the language of provided text with high accuracy. It supports multiple languages and can provide confidence scores and alternative language predictions.

## Endpoints

```
POST /pipeline/detect
POST /pipeline/detect-language  # Alternative endpoint
```

## Request Format

```json
{
  "text": "Text to detect language",
  "detailed": false,
  "model_id": null
}
```

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `text` | string | Yes | The text to analyze for language detection |
| `detailed` | boolean | No | Include detailed results with multiple language predictions (default: false) |
| `model_id` | string | No | Specific model to use for detection (if null, uses the default) |

## Response Format

### Basic Response (detailed=false)

```json
{
  "status": "success",
  "message": "Language detection completed successfully",
  "data": {
    "text": "Text to detect language",
    "detected_language": "en",
    "confidence": 0.98,
    "process_time": 0.123
  },
  "metadata": {
    "request_id": "uuid-here",
    "timestamp": 1620160000,
    "version": "1.0.0",
    "process_time": 0.123,
    "model_used": "papluca/xlm-roberta-base-language-detection"
  }
}
```

### Detailed Response (detailed=true)

```json
{
  "status": "success",
  "message": "Language detection completed successfully",
  "data": {
    "text": "Text to detect language",
    "detected_language": "en",
    "confidence": 0.98,
    "alternatives": [
      {"language": "en", "confidence": 0.98},
      {"language": "de", "confidence": 0.01},
      {"language": "fr", "confidence": 0.005},
      {"language": "es", "confidence": 0.003},
      {"language": "it", "confidence": 0.002}
    ],
    "process_time": 0.123
  },
  "metadata": {
    "request_id": "uuid-here",
    "timestamp": 1620160000,
    "version": "1.0.0",
    "process_time": 0.123,
    "model_used": "papluca/xlm-roberta-base-language-detection"
  }
}
```

### Response Data Fields

| Field | Type | Description |
|-------|------|-------------|
| `text` | string | The text submitted for language detection |
| `detected_language` | string | The detected language code (ISO 639-1) |
| `confidence` | number | Confidence score for the detection (0-1) |
| `alternatives` | array | Alternative language predictions (only if detailed=true) |
| `alternatives[].language` | string | Language code for alternative prediction |
| `alternatives[].confidence` | number | Confidence score for alternative prediction (0-1) |
| `process_time` | number | Time taken to process the detection (seconds) |

## Errors

### Common Error Codes

| Status Code | Error Code | Description |
|-------------|------------|-------------|
| 400 | EMPTY_TEXT | Input text is empty or contains only whitespace |
| 400 | TEXT_TOO_SHORT | Input text is too short for reliable detection |
| 400 | TEXT_TOO_LONG | Input text exceeds maximum length |
| 500 | DETECTION_FAILED | Language detection operation failed |
| 503 | MODEL_UNAVAILABLE | Required model is not available |

### Example Error Response

```json
{
  "status": "error",
  "message": "Input text is too short for reliable language detection",
  "error": {
    "code": "TEXT_TOO_SHORT",
    "details": "Please provide at least 10 characters for reliable language detection"
  },
  "metadata": {
    "request_id": "uuid-here",
    "timestamp": 1620160000
  }
}
```

## Examples

### Basic Language Detection

```bash
curl -X POST http://localhost:8000/pipeline/detect \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "text": "The quick brown fox jumps over the lazy dog."
  }'
```

### Detailed Language Detection

```bash
curl -X POST http://localhost:8000/pipeline/detect \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "text": "The quick brown fox jumps over the lazy dog.",
    "detailed": true
  }'
```

## Supported Languages

The language detection API supports identifying over 100 languages, including:

| Language | Code |
|----------|------|
| Arabic | ar |
| Bengali | bn |
| Chinese (Simplified) | zh |
| Czech | cs |
| Danish | da |
| Dutch | nl |
| English | en |
| Finnish | fi |
| French | fr |
| German | de |
| Greek | el |
| Hebrew | he |
| Hindi | hi |
| Hungarian | hu |
| Indonesian | id |
| Italian | it |
| Japanese | ja |
| Korean | ko |
| Norwegian | no |
| Persian | fa |
| Polish | pl |
| Portuguese | pt |
| Romanian | ro |
| Russian | ru |
| Spanish | es |
| Swedish | sv |
| Thai | th |
| Turkish | tr |
| Ukrainian | uk |
| Vietnamese | vi |
| ... | ... |

*For a complete list of supported languages, use the `/admin/languages` endpoint.*

## Additional Notes

1. For more accurate detection, provide at least 20-50 characters of text.
2. The service works best with natural language text rather than codes, abbreviations, or lists.
3. For mixed-language text, the API returns the predominant language.
4. When the confidence score is low (<0.5), consider using the detailed option to see alternative predictions.