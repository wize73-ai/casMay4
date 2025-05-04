#!/bin/bash
# Run gh auth login first if not authenticated

cd "$(dirname "$0")"

# Create PR
gh pr create \
  --title "Implement MBART translation model support and enhance anonymization" \
  --body "## Summary
- Implement proper support for Facebook MBART translation models with appropriate language code handling
- Enhance anonymization pipeline with better pattern detection and deterministic fake data generation
- Fix metrics collection for better performance tracking
- Add tests to verify the functionality of MBART language codes and anonymization

## Implementation Details
### MBART Translation Support
- Added comprehensive language code mapping for MBART's special format (e.g., en → en_XX, es → es_XX)
- Added detection for MBART models to apply appropriate pre/post-processing
- Implemented specialized handling for MBART source and target language tokens
- Enhanced the translator pipeline to ensure consistent translations

### Anonymization Improvements
- Made the anonymization deterministic to ensure consistent replacements
- Enhanced pattern detection across various languages (EN, ES, FR, DE)
- Added comprehensive fake data generation for personally identifiable information
- Integrated previously orphaned utility functions for better pattern detection

### Testing
- Added test script to verify MBART language code conversion
- Implemented anonymization tests with different strategies
- Verified consistency of anonymized results

## Testing Done
- Tested MBART language code conversion for various languages
- Verified anonymization with different strategies (mask, redact, pseudonymize)
- Confirmed anonymization produces consistent results for the same inputs

## Next Steps
- Full end-to-end testing with actual MBART models
- Performance testing for anonymization with large datasets
- Expanding language support for both translation and anonymization" \
  --base main \
  --head mbart-translation-implementation
  
echo "PR creation command ready! Run this script after authenticating with 'gh auth login'"