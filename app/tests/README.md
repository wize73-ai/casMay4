# CasaLingua Testing

This directory contains tests for the CasaLingua API and its components.

## Testing Components

### Core Component Tests

- **test_simple.py**: Basic tests for individual components to verify proper operation
  - Route cache
  - Batch optimizer
  - Streaming utilities
  - Error handling

### Unit Tests

- **test_route_cache.py**: Comprehensive tests for route cache
  - Basic caching functionality
  - Cache expiration
  - Cache eviction
  - Bloom compatibility

- **test_batch_optimizer.py**: Comprehensive tests for batch optimizer
  - Single item processing
  - Batch grouping
  - Max batch size
  - Batch timing

### Integration Tests

- **test_optimizations.py**: Tests all optimizations by making API calls
  - Route caching
  - Batch processing
  - Streaming response
  - Error handling

## Running Tests

### Running Simple Component Tests

```bash
# Set PYTHONPATH to ensure imports work correctly
PYTHONPATH=/path/to/project python app/tests/test_simple.py
```

### Running Unit Tests

```bash
# Run route cache tests
PYTHONPATH=/path/to/project pytest app/tests/test_route_cache.py -v

# Run batch optimizer tests
PYTHONPATH=/path/to/project pytest app/tests/test_batch_optimizer.py -v
```

### Running Integration Tests

```bash
# Start the server first
python app/main.py

# Then in another terminal, run the tests
PYTHONPATH=/path/to/project python app/tests/test_optimizations.py
```

## Test Runner

For convenience, a test runner script is provided that runs all component tests:

```bash
python app/tests/run_tests.py
```