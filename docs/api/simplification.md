# Text Simplification API

The Text Simplification API enables you to convert complex text into simpler, more accessible language while preserving the original meaning. It's particularly effective for legal documents, technical content, and educational materials.

## Endpoint

```
POST /pipeline/simplify
```

## Request Format

```json
{
  "text": "The tenant shall indemnify and hold harmless the landlord from and against any and all claims, actions, suits, judgments and demands brought or recovered against the landlord by reason of any negligent or willful act or omission of the tenant.",
  "language": "en",
  "target_level": "simple",
  "domain": "legal-housing",
  "verify_output": true,
  "auto_fix": true,
  "model_id": null
}
```

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `text` | string | Yes | The text to simplify |
| `language` | string | No | ISO language code of the text (default: "en") |
| `target_level` | string or number | No | Simplification level: "simple" or numeric grade level (1-12) (default: "simple") |
| `domain` | string | No | Domain specialization (e.g., "legal-housing", "medical", "general") |
| `verify_output` | boolean | No | Enable veracity auditing to verify simplification quality (default: false) |
| `auto_fix` | boolean | No | Automatically fix issues detected during verification (default: false) |
| `model_id` | string | No | Specific model to use for simplification (if null, uses the default) |

## Response Format

```json
{
  "status": "success",
  "message": "Text simplification completed successfully",
  "data": {
    "original_text": "The tenant shall indemnify and hold harmless the landlord from and against any and all claims, actions, suits, judgments and demands brought or recovered against the landlord by reason of any negligent or willful act or omission of the tenant.",
    "simplified_text": "The tenant must protect the landlord from all claims, lawsuits, and demands caused by the tenant's careless or deliberate actions.",
    "language": "en",
    "target_level": "simple",
    "process_time": 0.456,
    "verification": {
      "verified": true,
      "score": 0.82,
      "confidence": 0.78,
      "metrics": {
        "word_count_original": 42,
        "word_count_simplified": 19,
        "word_ratio": 0.45,
        "semantic_similarity": 0.88,
        "avg_word_length_original": 5.8,
        "avg_word_length_simplified": 4.2,
        "avg_sentence_length_original": 42,
        "avg_sentence_length_simplified": 19,
        "readability_improvement": 0.65
      }
    }
  },
  "metadata": {
    "request_id": "uuid-here",
    "timestamp": 1620160000,
    "version": "1.0.0",
    "process_time": 0.456,
    "domain": "legal-housing",
    "model_name": "facebook/bart-large-cnn"
  }
}
```

### Response Data Fields

| Field | Type | Description |
|-------|------|-------------|
| `original_text` | string | The original text submitted for simplification |
| `simplified_text` | string | The simplified text |
| `language` | string | The language code of the text |
| `target_level` | string or number | The requested simplification level |
| `process_time` | number | Time taken to process the simplification (seconds) |
| `verification` | object | Verification results (only if verify_output=true) |
| `verification.verified` | boolean | Whether the simplification passed verification |
| `verification.score` | number | Overall verification score (0-1) |
| `verification.confidence` | number | Confidence in the verification result (0-1) |
| `verification.metrics` | object | Detailed metrics about the simplification |
| `verification.metrics.word_count_*` | number | Word counts for original and simplified texts |
| `verification.metrics.word_ratio` | number | Ratio of simplified to original word count |
| `verification.metrics.semantic_similarity` | number | Semantic similarity score (0-1) |
| `verification.metrics.avg_word_length_*` | number | Average word lengths |
| `verification.metrics.avg_sentence_length_*` | number | Average sentence lengths |
| `verification.metrics.readability_improvement` | number | Overall readability improvement score (0-1) |

## Verification and Auto-Fixing

When `verify_output=true`, the simplification undergoes quality verification with the following checks:

1. **Semantic Preservation**: Ensures the meaning hasn't been altered
2. **Lexical Simplification**: Verifies complex words have been replaced with simpler alternatives
3. **Syntactic Simplification**: Checks that sentence structures have been simplified
4. **Length Reduction**: Confirms the text has been made more concise
5. **Readability Improvement**: Measures overall readability enhancement

When `auto_fix=true`, the system automatically addresses issues detected during verification:

1. **Complex Word Replacement**: Replaces legal/technical terminology with simpler alternatives
2. **Sentence Splitting**: Breaks down complex sentences into shorter ones
3. **Redundancy Removal**: Eliminates unnecessary words and phrases
4. **Meaning Preservation**: Applies conservative fixes for meaning alterations

## Domain-Specific Simplification

The `domain` parameter enables specialized handling for different content types:

| Domain | Description | Specialized Handling |
|--------|-------------|----------------------|
| `legal-housing` | Housing legal documents | Preserves legal terms like "Landlord", "Tenant"; handles clause structure |
| `medical` | Medical content | Preserves medical terminology; explains complex terms |
| `educational` | Educational materials | Adjusts to appropriate grade levels; maintains instructional structure |
| `general` | General content | General-purpose simplification for any content |

## Errors

### Common Error Codes

| Status Code | Error Code | Description |
|-------------|------------|-------------|
| 400 | INVALID_LANGUAGE | Unsupported language specified |
| 400 | TEXT_TOO_LONG | Input text exceeds maximum length |
| 400 | INVALID_TARGET_LEVEL | Invalid simplification level specified |
| 500 | SIMPLIFICATION_FAILED | Simplification operation failed |
| 503 | MODEL_UNAVAILABLE | Required model is not available |

### Example Error Response

```json
{
  "status": "error",
  "message": "Invalid target level specified",
  "error": {
    "code": "INVALID_TARGET_LEVEL",
    "details": "Target level must be 'simple' or a number between 1 and 12"
  },
  "metadata": {
    "request_id": "uuid-here",
    "timestamp": 1620160000
  }
}
```

## Examples

### Basic Simplification

```bash
curl -X POST http://localhost:8000/pipeline/simplify \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "text": "The utilization of simplified language enhances comprehension for individuals across diverse educational backgrounds.",
    "language": "en",
    "target_level": "simple"
  }'
```

### Housing Legal Document Simplification with Verification

```bash
curl -X POST http://localhost:8000/pipeline/simplify \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "text": "Lessor reserves the right to access the premises for inspection purposes with 24 hours advance notice provided to the lessee, except in cases of emergency wherein immediate access may be required.",
    "language": "en",
    "target_level": "simple",
    "domain": "legal-housing",
    "verify_output": true,
    "auto_fix": true
  }'
```

## Supported Languages

The simplification API currently supports the following languages:

| Language | Code | Simplification Supported |
|----------|------|--------------------------|
| English | en | Full support |
| Spanish | es | Limited support |
| French | fr | Limited support |
| Chinese (Simplified) | zh | Experimental |
| German | de | Experimental |

*For the most up-to-date language support, use the `/admin/languages` endpoint.*