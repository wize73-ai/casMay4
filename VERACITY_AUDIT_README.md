# Veracity Auditing System for CasaLingua

## Overview

The veracity auditing system provides quality assessment and verification for both translations and text simplifications in CasaLingua. This enhancement ensures that our language processing outputs meet high standards of quality and accuracy, particularly for housing legal text simplification.

## Key Features

### 1. Comprehensive Verification

- **Translation Verification**: Validates semantic equivalence, content integrity, and formatting preservation between source and target texts.
- **Simplification Verification**: Ensures simplified text maintains the original meaning while reducing complexity.
- **Housing Legal Domain Support**: Specialized verification rules for legal housing documents.

### 2. Multi-layer Assessment

- **Basic Validation**: Checks for empty outputs, untranslated content, and appropriate language.
- **Semantic Verification**: Uses embeddings to verify meaning preservation.
- **Content Integrity**: Ensures important elements like numbers, entities, and key terms are preserved.
- **Readability Metrics**: Evaluates the effectiveness of simplification.

### 3. Auto-Fixing Capability

The system provides automatic fixing for common simplification issues:

- **Lexical Simplification**: Replaces complex legal terms with simpler alternatives.
- **Syntactic Simplification**: Breaks down long sentences into shorter ones.
- **Meaning Preservation**: Applies conservative fixes when meaning has been altered.
- **Conciseness**: Removes redundancies and wordiness from simplifications.

### 4. Integration Points

- **Translation Pipeline**: Integrated directly into the translation workflow.
- **Simplification Pipeline**: Implemented in the SimplifierWrapper for real-time verification.
- **API Support**: Accessible through API endpoints with verify_output and auto_fix parameters.

## Usage

### For Translations

```python
# Verify translation quality
options = {
    "verify_output": True,
    "source_language": "en",
    "target_language": "es"
}

result = await processor.process_translation(
    text="The lease agreement requires tenants to maintain the property.",
    source_language="en",
    target_language="es",
    verify=True
)

# Check verification results
verification = result.get("metadata", {}).get("verification", {})
is_verified = verification.get("verified", False)
score = verification.get("score", 0.0)
issues = verification.get("issues", [])
```

### For Simplifications

```python
# Verify and auto-fix simplification
options = {
    "verify_output": True,
    "auto_fix": True,
    "domain": "legal-housing",
    "language": "en"
}

result = await processor.simplification_pipeline.simplify(
    text="Lessor reserves the right to access the premises for inspection purposes with 24 hours advance notice provided to the lessee.",
    language="en",
    level=1,
    options=options
)

# Check results
simplified_text = result.get("simplified_text")
verification = result.get("metadata", {}).get("verification", {})
```

## Housing Legal Text Simplification

The system is specifically enhanced for housing legal documents with:

1. **Legal Term Handling**: Preserves critical legal terms like "Landlord", "Tenant", "Security Deposit".
2. **Complex Term Replacement**: Simplifies legal jargon while maintaining meaning.
3. **Meaning Preservation**: Ensures critical clauses like notice periods and conditions aren't lost.
4. **Readability Enhancement**: Targets a grade 6-8 reading level for better accessibility.

## Implementation Details

### Key Components

1. **VeracityAuditor Class**: Core verification engine located in `/app/audit/veracity.py`.
2. **SimplifierWrapper**: Enhanced with verification and auto-fixing in `/app/services/models/wrapper.py`.
3. **UnifiedProcessor**: Integration point in `/app/core/pipeline/processor.py`.

### Verification Pipeline

1. Input text is processed through the standard pipeline.
2. Processed output is verified against the original using multiple metrics.
3. If issues are identified and auto-fix is enabled, appropriate fixes are applied.
4. Verification results are included in the response metadata.

## Test Results

The system has been tested with various housing legal texts, showing:

- **High Accuracy**: Correctly identifies simplified texts that preserve or alter original meaning.
- **Effective Auto-fixing**: Successfully corrects common simplification issues.
- **Domain Awareness**: Properly handles legal terminology and clauses.

## Future Enhancements

1. **Extended Legal Dictionary**: Expand the repository of legal terms and their simplified alternatives.
2. **Language-Specific Rules**: Add tailored verification for non-English languages.
3. **Reference-Based Learning**: Develop reference datasets of good/poor simplifications.
4. **Readability Formula Integration**: Incorporate established readability metrics like Flesch-Kincaid.

## Conclusion

The veracity auditing system significantly enhances the quality and reliability of CasaLingua's language processing capabilities, particularly for housing legal text simplification. By providing automated verification and fixing, it ensures that simplified texts remain accurate, accessible, and legally sound.