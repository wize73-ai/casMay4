#!/usr/bin/env python
"""
Script to create GitHub issues for tracking bugs and fixes in the MBART implementation.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

def run_command(command, check=True):
    """Run a shell command and return its output."""
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=True,
            check=check
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {command}")
        print(f"Error: {e.stderr}")
        if check:
            sys.exit(1)
        return ""

def create_issue(title, body, labels=None):
    """Create a GitHub issue using the GitHub CLI."""
    # Check if gh CLI is installed
    if not run_command("which gh", check=False):
        print("GitHub CLI (gh) is not installed. Please install it to create issues.")
        print("See: https://github.com/cli/cli#installation")
        sys.exit(1)
    
    # Check if authenticated with GitHub
    auth_status = run_command("gh auth status", check=False)
    if "Logged in to" not in auth_status:
        print("Not authenticated with GitHub. Please run 'gh auth login' first.")
        sys.exit(1)
    
    # Create the issue
    print(f"Creating issue: {title}")
    
    # Build the command
    issue_command = f'gh issue create --title "{title}" --body "{body}"'
    
    # Add labels if provided
    if labels:
        labels_str = ','.join(labels)
        issue_command += f' --label "{labels_str}"'
    
    # Execute the command
    result = run_command(issue_command)
    
    print(f"Issue created: {result}")
    return result

def main():
    parser = argparse.ArgumentParser(description="Create GitHub issues for bugs and fixes")
    parser.add_argument("--create-all", action="store_true", help="Create all predefined issues")
    args = parser.parse_args()
    
    # Define issues to create
    issues = [
        {
            "title": "Add support for MBART language codes",
            "body": """## Issue Description
CasaLingua's translation pipeline currently only supports Helsinki-NLP models, but we need to add support for Facebook's MBART models which use a different language code format.

## Current Behavior
- Language codes are used in their ISO 639-1 format (e.g., "en", "es").
- No mapping exists for MBART's special language code format (e.g., "en_XX", "es_XX").

## Expected Behavior
- Add a comprehensive mapping dictionary for all languages supported by MBART.
- Implement a method to convert ISO 639-1 codes to MBART format.
- Detect when MBART models are being used and apply the correct language codes.

## Technical Details
MBART uses special language code formats like:
- English: `en_XX`
- Spanish: `es_XX`
- French: `fr_XX`
- German: `de_DE`

## Acceptance Criteria
- [x] Mapping dictionary for all MBART languages
- [x] Conversion method implemented
- [x] MBART model detection
- [x] Tests for language code conversion
""",
            "labels": ["enhancement", "translation", "bug"]
        },
        {
            "title": "Implement deterministic anonymization for consistent entity replacement",
            "body": """## Issue Description
The current anonymization pipeline uses random UUIDs for entity replacement, resulting in inconsistent anonymization of the same entities across multiple runs.

## Current Behavior
- Entity replacement is random and changes each time the anonymization is run.
- The same entity appearing multiple times in a text may be replaced with different values.

## Expected Behavior
- Entities should be anonymized consistently between runs.
- The same entity appearing multiple times should receive the same replacement.
- Anonymization should be deterministic based on the entity text.

## Technical Details
We need to:
1. Replace the random UUID generation with a deterministic hash-based approach
2. Maintain a mapping of original entities to their replacements
3. Support a "consistency" option in the anonymization parameters

## Acceptance Criteria
- [x] Implement deterministic random number generation
- [x] Create consistent entity replacement mechanism
- [x] Ensure same entities get same replacements
- [x] Add tests for consistency
""",
            "labels": ["enhancement", "anonymization", "bug"]
        },
        {
            "title": "Fix pseudonymize strategy in anonymization pipeline",
            "body": """## Issue Description
The anonymization pipeline supports various strategies, but "pseudonymize" is not properly registered as a valid strategy.

## Current Behavior
- When using "pseudonymize" strategy, the system falls back to the default strategy.
- Warning message appears: "Unknown anonymization strategy: pseudonymize, using default"

## Expected Behavior
- The "pseudonymize" strategy should be recognized as valid.
- No warning should be displayed when using this strategy.

## Technical Details
The `AnonymizationStrategy.ALL_STRATEGIES` set doesn't include "pseudonymize" as a valid strategy.

## Acceptance Criteria
- [x] Add "pseudonymize" to AnonymizationStrategy.ALL_STRATEGIES
- [x] Tests pass with pseudonymize strategy without warnings
""",
            "labels": ["bug", "anonymization", "quick-fix"]
        },
        {
            "title": "Improve language pattern detection for anonymization",
            "body": """## Issue Description
The language-specific pattern detection for anonymization is not fully integrated with the main anonymization pipeline.

## Current Behavior
- `_get_patterns_for_language` method exists but isn't properly connected to the pattern loading system.
- Limited patterns available for non-English languages.

## Expected Behavior
- Method should be integrated with the main pipeline.
- Support for multiple languages (EN, ES, FR, DE) pattern detection.

## Technical Details
The `_get_patterns_for_language` method needs to be called as part of the anonymization process, and language-specific patterns need to be defined for all supported languages.

## Acceptance Criteria
- [x] Integrated `_get_patterns_for_language` with pattern loading
- [x] Added language-specific patterns for EN, ES, FR
- [x] Tests for pattern loading in multiple languages
""",
            "labels": ["enhancement", "anonymization", "internationalization"]
        },
        {
            "title": "Update TranslationModelWrapper for MBART support",
            "body": """## Issue Description
The TranslationModelWrapper does not properly handle MBART's special tokenization requirements.

## Current Behavior
- No special handling for MBART models.
- MBART tokenization and generation parameters are not properly set.

## Expected Behavior
- Detect MBART models and apply appropriate preprocessing.
- Set source language tokens for MBART tokenizer.
- Force target language token for MBART generation.

## Technical Details
MBART requires:
1. Setting source language tokens via `set_src_lang_special_tokens`
2. Setting `forced_bos_token_id` for target language during generation

## Acceptance Criteria
- [x] Add MBART model detection
- [x] Implement source language token setting
- [x] Implement target language forcing
- [x] Tests for MBART-specific processing
""",
            "labels": ["enhancement", "translation", "bug"]
        },
        {
            "title": "Create comprehensive testing for MBART and anonymization",
            "body": """## Issue Description
Need comprehensive tests for both MBART language code conversion and enhanced anonymization with deterministic replacement.

## Current Behavior
- Limited test coverage for the new features.
- No specific tests for edge cases or multilingual support.

## Expected Behavior
- Comprehensive tests for MBART language codes, including edge cases.
- Tests for deterministic anonymization with different strategies.
- Tests for consistency across multiple runs.
- Tests for language-specific pattern detection.

## Technical Details
Create a test script that covers:
1. MBART language code conversion for all supported languages
2. Edge cases like invalid language codes
3. Anonymization with different strategies
4. Deterministic replacement consistency
5. Pattern detection in multiple languages

## Acceptance Criteria
- [x] Complete test script with comprehensive coverage
- [x] Tests for all major functionality
- [x] Edge case testing
- [x] Detailed test reporting
""",
            "labels": ["testing", "enhancement"]
        }
    ]
    
    # Create all issues
    if args.create_all:
        for issue_data in issues:
            create_issue(
                title=issue_data["title"],
                body=issue_data["body"],
                labels=issue_data["labels"]
            )
    else:
        # Print summary of issues that would be created
        print("Issues to be created:")
        for i, issue_data in enumerate(issues, 1):
            print(f"{i}. {issue_data['title']} [{', '.join(issue_data['labels'])}]")
        
        print("\nTo create these issues, run with --create-all")

if __name__ == "__main__":
    main()