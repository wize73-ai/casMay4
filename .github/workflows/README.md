# CasaLingua GitHub Actions Workflows

This directory contains GitHub Actions workflows that automate testing, documentation, and other processes for the CasaLingua project.

## Available Workflows

### Development History

**File:** `development-history.yml`

Generates comprehensive documentation of the development history for the CasaLingua project, focusing on recent improvements to translation and anonymization functionality.

**Triggers:**
- Manual trigger (workflow_dispatch)

**Key Features:**
- Generates a markdown document of development history from git commits
- Documents major features implemented
- Documents technical architecture
- Optionally runs tests

### Publish Documentation

**File:** `publish-docs.yml`

Creates and publishes documentation to GitHub Pages, including implementation details and API reference.

**Triggers:**
- Push to main branch
- Completion of the Development History workflow
- Manual trigger (workflow_dispatch)

**Key Features:**
- Builds documentation using MkDocs with the Material theme
- Incorporates the development history document
- Creates comprehensive documentation of implementation details
- Provides API reference information
- Deploys to GitHub Pages

### Run Tests

**File:** `tests.yml`

Runs automated tests for the CasaLingua project.

**Triggers:**
- Push to main or mbart-translation-implementation branches
- Pull requests to main branch
- Manual trigger (workflow_dispatch)

**Key Features:**
- Tests on multiple Python versions (3.8, 3.9, 3.10)
- Tests MBART language code conversion
- Tests anonymization pattern loading
- Tests anonymization with different strategies
- Generates test results reports
- Optional model downloading for more comprehensive tests
- Model caching for faster test runs

### PR Analysis

**File:** `pr-analysis.yml`

Analyzes pull requests and adds an informative comment.

**Triggers:**
- New pull requests to main branch
- Updates to existing pull requests

**Key Features:**
- Detects PR categories based on keywords in title and description
- Determines PR size based on number of changes
- Adds a detailed analysis comment to the PR
- Provides information about changes and automated test status

## Usage

Most workflows can be triggered manually from the Actions tab in GitHub. Some workflows also run automatically on specific events like push or pull request.

To manually trigger a workflow:
1. Go to the Actions tab in GitHub
2. Select the workflow you want to run
3. Click "Run workflow"
4. Select the branch and any input parameters
5. Click "Run workflow"