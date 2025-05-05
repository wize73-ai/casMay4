# Changelog

## Database Configuration and Persistence - 2025-05-05

### Added
- Multi-database support: Flexible persistence layer supporting both SQLite and PostgreSQL
  - Implemented in `app/services/storage/casalingua_persistence/database.py`
  - Support for SQLite (default) for development and small deployments
  - Support for PostgreSQL for production and larger deployments
  - Automatic query adaptation between database types

- Database utility scripts: Easy database management and configuration
  - Added `scripts/toggle_db_config.py` for switching between database types
  - Added `scripts/init_postgres.py` for initializing PostgreSQL database
  - Added `scripts/check_db_status.py` for viewing database configuration
  - Added `scripts/demo_persistent_memory.py` for testing persistence

- Comprehensive database documentation: Detailed setup and configuration instructions
  - Added `docs/configuration/database.md` with complete documentation
  - Added `docs/configuration/README.md` as a configuration guide index
  - Updated existing documentation to reference database options

### Modified
- `app/services/storage/casalingua_persistence/database.py`: Enhanced for multi-database support
  - Added database type detection and connection handling
  - Added query placeholder conversion (? to $1, $2, etc. for PostgreSQL)
  - Added database-specific optimization methods

- `app/services/storage/casalingua_persistence/manager.py`: Updated for database flexibility
  - Added database configuration handling
  - Updated backup and restore methods for different database types
  - Enhanced error handling for database operations

- `config/default.json`: Added comprehensive database configuration options
  - Added connection pooling parameters
  - Added database-specific connection arguments
  - Added performance tuning options

### Documentation
- Added comprehensive database configuration guide
- Updated architecture documentation to include storage options
- Added examples for both SQLite and PostgreSQL configuration

### Problems and Fixes
1. **Directory Structure Issues**
   - Problem: Installer script did not create all required directories, causing errors when trying to access or write to non-existent paths
   - Fix: Updated install.sh and all database utility scripts to properly create all necessary directory structures including data and backup directories

2. **Query Parameter Style Mismatch**
   - Problem: SQLite uses `?` for parameter placeholders while PostgreSQL uses `$1`, `$2`, etc.
   - Fix: Implemented automatic parameter placeholder conversion in `database.py` that detects database type and converts placeholders appropriately

3. **Thread Safety Issues**
   - Problem: SQLite connections were not thread-safe by default
   - Fix: Added `check_same_thread=False` for SQLite connections and implemented connection pooling for both database types

4. **Database-Specific Optimization**
   - Problem: Different databases require different optimization approaches (VACUUM for SQLite vs ANALYZE for PostgreSQL)
   - Fix: Implemented database-specific optimization methods in the DatabaseManager

5. **Connection Handling**
   - Problem: Direct connection string parsing was error-prone
   - Fix: Implemented robust connection string parsing using urlparse to handle different connection string formats

6. **Backup and Restore Compatibility**
   - Problem: SQLite has a built-in backup API while PostgreSQL requires pg_dump/pg_restore
   - Fix: Created database-specific backup and restore implementations that work transparently for both database types

7. **Table Schema Differences**
   - Problem: Some SQLite-specific types don't exist in PostgreSQL
   - Fix: Standardized table schemas to use types compatible with both database systems

8. **Postgres Connection Errors**
   - Problem: Unable to connect to PostgreSQL on Raspberry Pi
   - Fix: Added detailed connection error handling and comprehensive troubleshooting guide in documentation

9. **Fallback Mechanism**
   - Problem: Deployments might need to work even if PostgreSQL server is unreachable
   - Fix: Implemented intelligent fallback to SQLite if PostgreSQL connection fails, with appropriate logging

10. **JSON Handling Differences**
   - Problem: PostgreSQL has native JSONB type while SQLite stores JSON as text
   - Fix: Implemented consistent JSON serialization/deserialization across database types

11. **Performance Monitoring**
    - Problem: No visibility into database performance
    - Fix: Added performance logging options and metrics collection in the database manager

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