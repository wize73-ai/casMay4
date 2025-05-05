# Git Repository Structure

This document explains how the CasaLingua project manages large files and directory structures in Git.

## Directory Structure

CasaLingua uses the following directory structure for files that are typically too large for version control:

```
casalingua/
├── models/              # Language model files
│   ├── translation/     # Translation models (mbart, mt5, etc.)
│   ├── multipurpose/    # Multipurpose language models
│   ├── verification/    # Verification models
│   └── tokenizers/      # Tokenizer models
├── cache/               # Cache files
│   ├── models/          # Model cache
│   └── api/             # API request/response cache
├── data/                # Persistent data storage
│   ├── backups/         # Database backups
│   └── documents/       # Document files for processing
├── logs/                # Log files
│   ├── app/             # Application logs
│   ├── audit/           # Audit logs
│   └── metrics/         # Performance metrics
├── temp/                # Temporary files
├── indexes/             # Search indexes
└── knowledge_base/      # Knowledge base data
```

## Git Ignore Rules

The `.gitignore` file is configured to exclude large binary files and generated content while preserving the directory structure. The key principles are:

1. All content in large directories is ignored with patterns like `models/*`
2. The directory structure is preserved with `.gitkeep` files
3. All exclusions have a corresponding negation rule: `!models/.gitkeep`

## Working with Large Files

When working with this codebase:

1. **First-time setup**: Run the installation script to download required models
   ```bash
   ./install.sh
   ```

2. **Model downloads**: Use the provided script to download required models
   ```bash
   python scripts/download_models.py
   ```

3. **Before committing**: Verify that you're not accidentally committing large files
   ```bash
   git status
   ```

## Adding New Files to Repository

When adding new files to the repository:

1. Consider if the file should be version controlled:
   - Code, configurations, documentation: YES
   - Generated models, large binary files, logs: NO

2. If creating a new directory that should be preserved but its contents ignored:
   - Update the `.gitignore` with appropriate patterns
   - Add a `.gitkeep` file to the directory with an explanatory comment
   - Update this documentation if needed

## File Size Limits

As a general rule:
- Files under 100KB are fine to commit
- Files between 100KB-1MB should be carefully considered
- Files over 1MB should usually be excluded from version control
- Text-based configuration files can be larger if needed

## Model Registry

The model registry configuration is version controlled, but the actual models are excluded. This allows the application to download the correct models at runtime without storing them in the repository.