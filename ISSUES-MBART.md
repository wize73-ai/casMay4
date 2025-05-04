# MBART Translation and Anonymization Issues

The following issues relate to our MBART translation implementation and anonymization enhancements:

## Issue 1: Add support for MBART language codes
**Title**: Add support for MBART language codes
**Labels**: enhancement, translation, bug
**Body**:
CasaLingua's translation pipeline currently only supports Helsinki-NLP models, but we need to add support for Facebook's MBART models which use a different language code format.

- Current Behavior: Language codes are used in their ISO 639-1 format (e.g., "en", "es").
- Expected Behavior: Add mapping dictionary for MBART language codes and detect when to use them.
- Solution: Implement `_get_mbart_language_code` method and update TranslationPipeline.

**Status**: Closed ✅

## Issue 2: Implement deterministic anonymization for consistent entity replacement
**Title**: Implement deterministic anonymization for consistent entity replacement
**Labels**: enhancement, anonymization, bug
**Body**:
The current anonymization pipeline uses random UUIDs for entity replacement, resulting in inconsistent anonymization of the same entities across multiple runs.

- Current Behavior: Entity replacement is random and changes each time the anonymization is run.
- Expected Behavior: Entities should be anonymized consistently between runs and within the same text.
- Solution: Use hash-based deterministic generation with entity text as seed.

**Status**: Closed ✅

## Issue 3: Fix pseudonymize strategy in anonymization pipeline
**Title**: Fix pseudonymize strategy in anonymization pipeline
**Labels**: bug, anonymization, quick-fix
**Body**:
The anonymization pipeline supports various strategies, but "pseudonymize" is not properly registered as a valid strategy.

- Current Behavior: When using "pseudonymize" strategy, the system falls back to the default strategy.
- Expected Behavior: The "pseudonymize" strategy should be recognized as valid.
- Solution: Add "pseudonymize" to AnonymizationStrategy.ALL_STRATEGIES.

**Status**: Closed ✅

## Issue 4: Improve language pattern detection for anonymization
**Title**: Improve language pattern detection for anonymization
**Labels**: enhancement, anonymization, internationalization
**Body**:
The language-specific pattern detection for anonymization is not fully integrated with the main anonymization pipeline.

- Current Behavior: `_get_patterns_for_language` method exists but isn't properly connected to the pattern loading system.
- Expected Behavior: Support for multiple languages (EN, ES, FR, DE) pattern detection.
- Solution: Integrate method with the main pipeline and add language-specific patterns.

**Status**: Closed ✅

## Issue 5: Update TranslationModelWrapper for MBART support
**Title**: Update TranslationModelWrapper for MBART support
**Labels**: enhancement, translation, bug
**Body**:
The TranslationModelWrapper does not properly handle MBART's special tokenization requirements.

- Current Behavior: No special handling for MBART models.
- Expected Behavior: Detect MBART models and apply appropriate preprocessing.
- Solution: Add detection, source language token setting, and target language forcing.

**Status**: Closed ✅

## Issue 6: Create comprehensive testing for MBART and anonymization
**Title**: Create comprehensive testing for MBART and anonymization
**Labels**: testing, enhancement
**Body**:
Need comprehensive tests for both MBART language code conversion and enhanced anonymization with deterministic replacement.

- Current Behavior: Limited test coverage for new features.
- Expected Behavior: Comprehensive tests for all new functionality, including edge cases.
- Solution: Create test_changes.py with detailed validation and reporting.

**Status**: Closed ✅

## Issue 7: Set up GitHub Actions workflows
**Title**: Set up GitHub Actions workflows
**Labels**: ci, enhancement
**Body**:
Need to set up GitHub Actions for testing, documentation, and PR automation.

- Current Behavior: No automated workflows for testing and documentation.
- Expected Behavior: Comprehensive CI/CD pipeline for the project.
- Solution: Create workflows for development history, testing, documentation, and PR analysis.

**Status**: Closed ✅

## Running Issue Creation Tool

To create GitHub issues for all the bugs and enhancements listed in this document:

```bash
# View issues that will be created
python scripts/create_issues.py

# Create all issues in GitHub
python scripts/create_issues.py --create-all
```

This will use the GitHub CLI to create properly labeled issues for all the fixed items.

**Note:** Make sure you are authenticated with the GitHub CLI before running this script.