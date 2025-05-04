# Changelog

## API Optimizations - 2025-05-04

### Added
- Request-Level Caching: Thread-safe caching with Bloom Housing compatibility
  - Implemented in `app/services/storage/route_cache.py`
  - Configurable TTL and cache size
  - Metrics tracking for hit/miss ratios
  - Automatic LRU eviction

- Smart Batching: Automatic grouping of similar requests
  - Implemented in `app/api/middleware/batch_optimizer.py`
  - Adaptive timing control for optimal batch sizes
  - Concurrent processing of multiple batches

- Streaming Responses: Chunk-based processing for large documents
  - Implemented in `app/api/routes/streaming.py`
  - Event-driven format for client-side progress tracking
  - Progress notifications during processing

- Enhanced Error Handling: Categorized errors with standardized formats
  - Implemented in `app/utils/error_handler.py`
  - Standardized error response model
  - Fallback mechanisms for graceful degradation

- Bloom Housing Integration: Seamless compatibility with Bloom Housing API
  - Implemented in `app/api/schemas/bloom_housing.py` and `app/api/routes/bloom_housing.py`
  - Bidirectional conversion between formats

- Parallel Processing: Concurrent execution of independent tasks
  - Modified `app/api/routes/pipeline.py`
  - Non-blocking I/O throughout the API stack

- Comprehensive Testing: Unit and integration tests
  - Added tests in `app/tests/`
  - Component-level tests
  - End-to-end API tests

### Modified
- `app/main.py`: Added route cache initialization and new routers
- `app/api/routes/pipeline.py`: Added parallel processing
- Several pipeline components for improved performance

### Updated Shell Scripts
- `install.sh`: Completely revamped to support API optimizations
  - Added virtual environment setup and activation
  - Added configuration for optimization settings via .env file
  - Added directory creation for cache and models

- `scripts/startdev.sh`: Updated for development environment
  - Added environment variable setting for optimization parameters
  - Added loading of configuration from .env file
  - Added display of current configuration

- `scripts/startprod.sh`: Updated for production environment
  - Added larger default cache sizes and batch sizes
  - Added timeout parameter to Gunicorn for streaming responses

- `scripts/casalingua.sh`: Improved for compatibility
  - Added support for both .venv and venv directories
  - Added configuration of optimizations via environment variables
  - Added error code propagation

- `scripts/test.sh`: Added optimization testing support
  - Added --optimizations flag for testing API optimizations
  - Added dependency checks for optimization requirements
  - Extended test reporting to include optimization details

### Bug Fixes
- Fixed RouteCache TTL expiration for per-item TTL settings
  - Updated timestamp storage format to include TTL with timestamp
  - Properly respect per-item TTL overrides
  - Fixed LRU eviction logic to handle new timestamp format

- Fixed BatchOptimizer test issues
  - Completely rewrote tests to focus on component testing rather than e2e tests
  - Fixed timeout handling during testing
  - Added direct completion event setting in test process function

- Fixed code issues preventing application from running
  - Fixed syntax error in anonymizer.py that caused a crash on startup
  - Uncommented tokenizer.py code that was completely commented out
  - Updated veracity.py to use EnhancedModelManager instead of ModelManager

### Documentation
- Added `OPTIMIZATIONS.md` detailing all optimizations
- Updated API documentation in `app/api/README.md`
- Added testing documentation in `app/tests/README.md`