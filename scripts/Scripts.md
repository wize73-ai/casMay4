# CasaLingua Scripts Documentation

## Overview

This document provides detailed information about the utility scripts included with the CasaLingua platform. These scripts help with installation, development, testing, and production deployment tasks.

## Table of Contents

1. [Installation Script](#installation-script)
2. [Development Server Script](#development-server-script)
3. [Production Server Script](#production-server-script)
4. [Test Runner Script](#test-runner-script)
5. [Cache Management Script](#cache-management-script)
6. [Model Download Script](#model-download-script)
7. [Benchmark Script](#benchmark-script)

---

## Installation Script

The `install.sh` script automates the initial setup and configuration of the CasaLingua platform.

### Usage

```bash
./install.sh [options]
```

### Options

| Option | Description |
|--------|-------------|
| `--with-models` | Download language models during installation |
| `--no-deps` | Skip dependency installation |
| `--cpu-only` | Install PyTorch without GPU support |
| `--dev` | Install development dependencies |
| `--env-file FILE` | Specify a custom .env file (default: .env) |
| `--env ENV` | Specify the environment configuration (default: development) |

### Examples

```bash
# Full installation with models
./install.sh --with-models

# Development installation
./install.sh --dev

# CPU-only installation for servers without GPUs
./install.sh --cpu-only --env production
```

### What It Does

1. Checks for appropriate Python version (3.10+)
2. Creates and configures a virtual environment
3. Installs dependencies with proper PyTorch installation
4. Sets up project directories and configuration files
5. Initializes the database
6. Optionally downloads language models

---

## Development Server Script

The `start-dev.sh` script configures and starts CasaLingua in development mode with hot-reloading enabled.

### Usage

```bash
./start-dev.sh [options]
```

### Options

| Option | Description |
|--------|-------------|
| `--port PORT` | Specify the server port (default: 8000) |
| `--host HOST` | Specify the server host (default: 0.0.0.0) |
| `--no-reload` | Disable auto-reload |
| `--log-level LEVEL` | Set the log level (default: info) |

### Examples

```bash
# Start development server on port 8080
./start-dev.sh --port 8080

# Start development server with debug logging
./start-dev.sh --log-level debug

# Start development server on localhost only
./start-dev.sh --host 127.0.0.1
```

### What It Does

1. Sets the environment to "development"
2. Creates necessary directories and config files if missing
3. Starts the server using Uvicorn with hot-reloading
4. Provides warnings if models aren't available

---

## Production Server Script

The `start-prod.sh` script configures and starts CasaLingua in production mode, optimized for performance and reliability.

### Usage

```bash
./start-prod.sh [options]
```

### Options

| Option | Description |
|--------|-------------|
| `--port PORT` | Specify the server port (default: 8000) |
| `--host HOST` | Specify the server host (default: 0.0.0.0) |
| `--workers N` | Set number of worker processes (default: 4) |
| `--log-level LEVEL` | Set the log level (default: warning) |
| `--ssl-cert FILE` | Path to SSL certificate file |
| `--ssl-key FILE` | Path to SSL key file |

### Examples

```bash
# Start production server with 8 workers
./start-prod.sh --workers 8

# Start production server with SSL enabled
./start-prod.sh --ssl-cert /path/to/cert.pem --ssl-key /path/to/key.pem

# Start production server on custom port
./start-prod.sh --port 443 --workers 8
```

### What It Does

1. Sets the environment to "production"
2. Performs pre-flight checks to ensure the system is ready
3. Verifies that required models are available
4. Starts the server using Gunicorn with Uvicorn workers
5. Configures SSL if certificates are provided

---

## Test Runner Script

The `test.sh` script runs the test suite for the CasaLingua platform, with various options for specific testing needs.

### Usage

```bash
./test.sh [options]
```

### Options

| Option | Description |
|--------|-------------|
| `--path PATH` | Specify test path (default: tests/) |
| `--no-coverage` | Disable coverage reporting |
| `--verbose`, `-v` | Enable verbose output |
| `--parallel`, `-xvs` | Run tests in parallel |
| `--skip-slow` | Skip slow tests |
| `--mark`, `-m MARK` | Only run tests with given mark |
| `--collect-only` | Only collect tests, don't run them |
| `--fail-fast` | Stop after first failure |
| `--no-html` | Disable HTML coverage report |
| `--unit` | Run only unit tests |
| `--integration` | Run only integration tests |
| `--api` | Run only API tests |
| `--models` | Run only model tests |

### Examples

```bash
# Run all tests with coverage reporting
./test.sh

# Run only unit tests
./test.sh --unit

# Run tests in parallel for faster execution
./test.sh --parallel

# Run only tests marked as 'model' tests
./test.sh --mark "model"

# Run tests with verbose output and stop on first failure
./test.sh --verbose --fail-fast
```

### What It Does

1. Sets up the test environment
2. Runs the specified tests with pytest
3. Generates coverage reports if enabled
4. Runs code style and type checks if available
5. Optionally generates a test report

---

## Cache Management Script

The `flush-cache.sh` script manages caches and starts the application.

### Usage

```bash
./flush-cache.sh [options]
```

### Options

| Option | Description |
|--------|-------------|
| `--env ENV` | Set the environment (default: development) |
| `--port PORT` | Set the server port (default: 8000) |
| `--host HOST` | Set the server host (default: 0.0.0.0) |
| `--workers N` | Set the number of worker processes (default: 4) |
| `--no-reload` | Disable auto-reload |
| `--no-flush` | Skip cache flushing |

### Examples

```bash
# Flush caches and start in development mode
./flush-cache.sh

# Flush caches and start in production mode
./flush-cache.sh --env production --workers 8

# Start without flushing caches
./flush-cache.sh --no-flush
```

### What It Does

1. Clears model cache directory
2. Removes Python bytecode files and cache directories
3. Archives large log files
4. Checks if language models are available
5. Starts the CasaLingua application

---

## Model Download Script

The `download_models.py` script handles downloading and setting up language models for the CasaLingua platform.

### Usage

```bash
python -m app.tools.download_models [options]
```

### Options

| Option | Description |
|--------|-------------|
| `--config FILE` | Path to configuration file |
| `--models-dir DIR` | Directory to store models |
| `--cache-dir DIR` | Cache directory for downloads |
| `--model MODEL` | Specific model(s) to download (can be used multiple times) |
| `--all` | Download all default models |
| `--advanced` | Include advanced models |
| `--force` | Force re-download of existing models |
| `--verify` | Verify downloaded models |
| `--debug` | Enable debug logging |

### Examples

```bash
# Download all default models
python -m app.tools.download_models --all

# Download specific models
python -m app.tools.download_models --model embedding_model --model translation_en_es

# Download all models, including advanced ones
python -m app.tools.download_models --all --advanced

# Force re-download and verify models
python -m app.tools.download_models --all --force --verify
```

### What It Does

1. Downloads models from the Hugging Face Transformers library
2. Verifies successful downloads
3. Updates the model registry
4. Provides a summary of downloaded models

---

## Benchmark Script

The `benchmark.py` script tests the performance of different translation models and configurations.

### Usage

```bash
python -m app.tools.benchmark [options]
```

### Options

| Option | Description |
|--------|-------------|
| `--config FILE` | Path to configuration file |
| `--output-dir DIR` | Directory for benchmark results |
| `--device DEVICE` | Device to use (cpu, cuda) |
| `--model MODEL` | Model to benchmark |
| `--test-data FILE` | Path to test data file |
| `--source-lang LANG` | Source language code |
| `--target-lang LANG` | Target language code |
| `--batch-size N` | Batch size for processing |
| `--verify` | Verify translations |
| `--compare-models MODELS` | Comma-separated list of models to compare |
| `--batch-sizes SIZES` | Comma-separated list of batch sizes to test |

### Examples

```bash
# Benchmark a single model
python -m app.tools.benchmark --model translation_en_es --test-data tests/data/en_samples.json --source-lang en --target-lang es

# Compare multiple models
python -m app.tools.benchmark --compare-models "translation_en_es,translation_en_es_large" --test-data tests/data/en_samples.json --source-lang en --target-lang es

# Test different batch sizes
python -m app.tools.benchmark --model translation_en_es --batch-sizes "1,2,4,8,16" --test-data tests/data/en_samples.json --source-lang en --target-lang es
```

### What It Does

1. Measures model performance with detailed metrics
2. Supports batch processing optimization
3. Compares multiple models with report generation
4. Tracks memory usage and verification quality
5. Generates comprehensive benchmark results

---

## Additional Tips

### Environment Shortcuts

You can create aliases for common operations:

```bash
# Add these to your .bashrc or .zshrc
alias casa-dev='./start-dev.sh'
alias casa-prod='./start-prod.sh'
alias casa-test='./test.sh'
alias casa-models='python -m app.tools.download_models'
```

### Script Permissions

Make sure to set executable permissions on all scripts:

```bash
chmod +x *.sh
```

### Troubleshooting

If you encounter issues with the scripts, check the following:

1. Ensure Python 3.10+ is installed
2. Verify that the virtual environment is correctly activated
3. Check that all required dependencies are installed
4. Look for error messages in the log files (logs/ directory)
5. Make sure configuration files are properly set up

For specific error messages, consult the CasaLingua documentation or contact Exygy support.