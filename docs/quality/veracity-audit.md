# Veracity Auditing System

The Veracity Auditing System is CasaLingua's quality assurance mechanism that verifies and validates language processing outputs to ensure accuracy, meaning preservation, and appropriateness for the target audience.

## Overview

CasaLingua's veracity auditing provides comprehensive verification for:

1. **Translation Verification**: Ensures translations preserve meaning, format, and content across languages
2. **Simplification Verification**: Validates simplifications maintain original meaning while reducing complexity
3. **Domain-Specific Validation**: Specialized verification for domains like legal housing documents

## Architecture

The veracity auditing system consists of these core components:

1. **VeracityAuditor**: Central class that manages verification processes
2. **Verification Methods**: Specialized algorithms for different content types
3. **Metrics Collection**: Quantitative measures of quality
4. **Auto-Fixing System**: Automatic remediation of detected issues

## Verification Process

### Translation Verification

For translations, the system performs:

1. **Basic Validation**:
   - Checks for empty translations
   - Detects untranslated content (identical to source)
   - Verifies the target text is in the correct language
   - Validates appropriate length ratio between languages

2. **Semantic Verification**:
   - Uses embeddings to measure semantic similarity
   - Compares against reference translations when available
   - Calculates confidence scores based on similarity distribution

3. **Content Integrity Checks**:
   - Verifies numbers are preserved correctly
   - Ensures named entities are handled appropriately
   - Detects potential hallucinations or omissions

### Simplification Verification

For text simplification, the system verifies:

1. **Meaning Preservation**:
   - Calculates semantic similarity between original and simplified text
   - Detects significant meaning alterations
   - Ensures critical information isn't lost

2. **Readability Improvement**:
   - Measures lexical simplification (word complexity reduction)
   - Evaluates syntactic simplification (sentence structure)
   - Calculates overall readability improvement

3. **Domain-Specific Requirements**:
   - For legal texts: preserves critical legal terms
   - For housing documents: maintains required disclosures and conditions
   - For educational content: ensures grade-level appropriateness

## Metrics

The veracity system calculates and reports these key metrics:

### Translation Metrics

- **Semantic Similarity**: Score between 0-1 measuring meaning preservation
- **Length Ratio**: Ratio of target to source text length
- **Entity Preservation**: Percentage of entities correctly preserved
- **Missing Numbers**: Count of numbers missing in the translation
- **Language Confidence**: Confidence that output is in the target language

### Simplification Metrics

- **Word Count Ratio**: Ratio of simplified to original word count
- **Average Word Length**: Changes in average word length
- **Average Sentence Length**: Changes in average sentence length
- **Semantic Similarity**: Score between 0-1 measuring meaning preservation
- **Readability Improvement**: Overall score for readability enhancement

## Auto-Fixing Capabilities

When enabled, the auto-fixing system can remediate issues including:

1. **Lexical Simplification**: Replacing complex words with simpler alternatives using an extensive dictionary
2. **Syntactic Simplification**: Breaking down long sentences into shorter ones
3. **Meaning Preservation**: Applying conservative fixes when meaning has been altered
4. **Conciseness**: Removing redundancies and wordiness from simplifications

## API Integration

### Enabling Verification

Verification can be enabled for any processing request:

```json
{
  "text": "Text to process",
  "source_language": "en",
  "target_language": "es",
  "verify": true  // For translations
}
```

Or for simplification:

```json
{
  "text": "Text to simplify",
  "language": "en",
  "target_level": "simple",
  "verify_output": true,
  "auto_fix": true  // Enable auto-fixing
}
```

### Verification Results

Verification results are included in the API response:

```json
{
  "status": "success",
  "data": {
    "processed_text": "...",
    "verification": {
      "verified": true,
      "score": 0.87,
      "confidence": 0.92,
      "metrics": {
        "semantic_similarity": 0.89,
        "word_ratio": 0.65,
        "...": "..."
      },
      "issues": [
        {
          "type": "slight_meaning_change",
          "severity": "warning",
          "message": "Slight semantic divergence detected"
        }
      ]
    }
  }
}
```

## Issue Severity Levels

The system reports issues with these severity levels:

1. **Critical**: Significant issues that require attention (e.g., meaning alteration)
2. **Warning**: Minor issues that may affect quality but aren't critical
3. **Info**: Informational notes about the processing

## Configuration

The veracity auditing system can be configured in the application settings:

```json
{
  "veracity": {
    "enabled": true,
    "threshold": 0.75,
    "max_sample_size": 1000,
    "min_confidence": 0.7,
    "reference_embeddings_path": "data/reference_embeddings.json",
    "language_pairs": [
      ["en", "es"],
      ["es", "en"]
    ]
  }
}
```

## References

- The veracity auditing system uses the `app/audit/veracity.py` class as its core implementation
- Auto-fixing capabilities are integrated in `app/services/models/wrapper.py`
- Configuration settings are defined in `config/default.json`

## Examples

### Sample Verification Call

```python
from app.audit.veracity import VeracityAuditor

auditor = VeracityAuditor()
result = await auditor.check(
    original_text,
    processed_text,
    {
        "operation": "simplification",
        "source_language": "en",
        "domain": "legal-housing"
    }
)
```

### Sample Verification Result

```json
{
  "verified": true,
  "score": 0.82,
  "confidence": 0.78,
  "issues": [],
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
```