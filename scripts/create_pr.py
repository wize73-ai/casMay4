#!/usr/bin/env python
"""
Script to create a GitHub pull request for the MBART implementation.
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

def create_pull_request(title, body, base_branch="main", head_branch="mbart-translation-implementation"):
    """Create a pull request using the GitHub CLI."""
    # Check if gh CLI is installed
    if not run_command("which gh", check=False):
        print("GitHub CLI (gh) is not installed. Please install it to create pull requests.")
        print("See: https://github.com/cli/cli#installation")
        sys.exit(1)
    
    # Check if authenticated with GitHub
    auth_status = run_command("gh auth status", check=False)
    if "Logged in to" not in auth_status:
        print("Not authenticated with GitHub. Please run 'gh auth login' first.")
        sys.exit(1)
    
    # Create the pull request
    print(f"Creating pull request: {title}")
    print(f"From: {head_branch} → {base_branch}")
    
    # Use the GitHub CLI to create a pull request
    pr_command = f'gh pr create --title "{title}" --body "{body}" --base {base_branch} --head {head_branch}'
    result = run_command(pr_command)
    
    print(f"Pull request created: {result}")
    return result

def main():
    parser = argparse.ArgumentParser(description="Create GitHub PR for MBART implementation")
    parser.add_argument("--title", default="Implement MBART Translation and Enhanced Anonymization", help="PR title")
    parser.add_argument("--base", default="main", help="Base branch")
    parser.add_argument("--head", default="mbart-translation-implementation", help="Head branch")
    args = parser.parse_args()
    
    # Prepare the PR body
    pr_body = """## Summary
- Replaced Helsinki-NLP translation models with Facebook MBART models
- Enhanced anonymization pipeline with deterministic entity replacement
- Added comprehensive language code handling for MBART models
- Added tests for MBART language codes and anonymization functionality
- Set up GitHub Actions for development documentation and testing

## Technical Implementation
- Added mapping dictionary for MBART language codes in TranslationPipeline
- Implemented language code conversion helper method
- Enhanced the anonymization pipeline with deterministic replacement
- Fixed the orphaned pattern detection method
- Created a comprehensive test suite

## Test Plan
- Run the test_changes.py script to verify MBART language code conversion
- Test anonymization with different strategies
- Verify consistency of anonymization results
- Run the GitHub Actions workflows to ensure CI/CD pipeline works correctly

### Related Issues
This PR addresses the requirement to improve translation quality and consistency.
"""
    
    # Create the pull request
    pr_url = create_pull_request(
        title=args.title,
        body=pr_body,
        base_branch=args.base,
        head_branch=args.head
    )
    
    # Open the PR URL
    if pr_url:
        print(f"Opening PR in browser: {pr_url}")
        run_command(f"open {pr_url}")

if __name__ == "__main__":
    main()