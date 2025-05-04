# GitHub Issues to Create

Once you have access to the GitHub repository, create the following issues and mark them as closed:

## Previous API Optimization Issues

The following issues are from our previous API optimization work and have been completed:

## Issue 1: Implement Request-Level Caching
**Title**: Implement Request-Level Caching
**Labels**: enhancement, performance
**Body**:
Implement a thread-safe, async-compatible caching system optimized for API routes with Bloom Housing compatibility.

Key Features:
- Thread-safe operations with asyncio.Lock
- Configurable TTL and cache size
- Automatic LRU eviction
- Metrics tracking
- Bloom Housing compatibility

**Status**: Closed ✅

## Issue 2: Implement Smart Batching for Similar Requests
**Title**: Implement Smart Batching for Similar Requests
**Labels**: enhancement, performance
**Body**:
Create a batching system that automatically groups similar small requests to improve processing efficiency.

Key Features:
- Automatic grouping based on request properties
- Adaptive timing control for optimal batch sizes
- Concurrent processing of multiple batches
- Result mapping back to original requesters

**Status**: Closed ✅

## Issue 3: Add Streaming Response Support
**Title**: Add Streaming Response Support
**Labels**: enhancement, feature
**Body**:
Implement streaming responses for large document processing to avoid timeouts.

Key Features:
- Chunk-based processing for large documents
- Event-driven format for client-side progress tracking
- Progress notifications during processing
- Resilience against timeouts for large operations

**Status**: Closed ✅

## Issue 4: Enhance Error Handling
**Title**: Enhance Error Handling
**Labels**: enhancement, reliability
**Body**:
Implement standardized error categorization with comprehensive fallbacks.

Key Features:
- Categorized errors for better client handling
- Standardized response format
- Fallback mechanisms for graceful degradation
- Request ID tracking for improved debugging

**Status**: Closed ✅

## Issue 5: Implement Bloom Housing Compatibility
**Title**: Implement Bloom Housing Compatibility
**Labels**: enhancement, integration
**Body**:
Create a compatibility layer for seamless integration with Bloom Housing API.

Key Features:
- Compatible schemas for both systems
- Bidirectional conversion of request/response formats
- Transparent processing with core pipeline

**Status**: Closed ✅

## Issue 6: Implement Parallel Processing
**Title**: Implement Parallel Processing
**Labels**: enhancement, performance
**Body**:
Add concurrent execution of independent tasks to improve throughput.

Key Features:
- Parallel preprocessing using asyncio.gather
- Non-blocking I/O throughout the API stack
- Background tasks for metrics and logging

**Status**: Closed ✅

## Issue 7: Add Comprehensive Tests
**Title**: Add Comprehensive Tests
**Labels**: testing, quality
**Body**:
Create tests for all new optimizations to ensure reliability.

Key Features:
- Unit tests for components
- Integration tests for API
- Component tests for basic functionality
- Documentation for testing

**Status**: Closed ✅

## Issue 8: Update Shell Scripts for API Optimizations
**Title**: Update Shell Scripts for API Optimizations
**Labels**: enhancement, devops
**Body**:
Update all shell scripts to support the new API optimizations and ensure proper virtual environment setup.

Key Features:
- Update install.sh to configure optimization settings
- Update startdev.sh and startprod.sh for appropriate environments
- Update casalingua.sh for compatibility with both venv and .venv
- Update test.sh to support API optimization testing
- Add proper environment variable handling for all optimization parameters

**Status**: Closed ✅

## MBART Translation and Anonymization Issues

The following issues relate to our MBART translation implementation and anonymization enhancements:

## Issue 9: Implement MBART Translation Models
**Title**: Implement MBART Translation Models
**Labels**: enhancement, feature
**Body**:
Replace Helsinki-NLP models with MBART-50 models for improved quality and performance.

Key Features:
- MBART-50 models for 50+ language support
- Special language code handling for MBART models
- Improved translation quality for non-European languages
- Better context-aware translation

**Status**: Closed ✅

## Issue 10: Enhance Anonymization with Deterministic Replacement
**Title**: Enhance Anonymization with Deterministic Replacement
**Labels**: enhancement, security
**Body**:
Implement deterministic entity replacement for consistent anonymization across multiple translations.

Key Features:
- Consistent replacement of entities across jobs
- Preserves formatting and capitalizations
- Configurable replacement strategies
- Support for 20+ entity types

**Status**: Closed ✅

## Health Check Implementation Issues

The following issues relate to our health check implementation:

## Issue 11: Implement Real Database Health Checks
**Title**: Implement Real Database Health Checks
**Labels**: enhancement, monitoring
**Body**:
Replace placeholder database health checks with real connection tests for monitoring and reliability.

Key Features:
- Test actual database connections with simple queries
- Measure response times for database operations
- Component-level status reporting for all database instances
- Consistent status indicators (healthy, degraded, error)

**Status**: Open 🔄

## Issue 12: Enhance Model Health Check Implementation
**Title**: Enhance Model Health Check Implementation
**Labels**: enhancement, monitoring
**Body**:
Implement comprehensive model health checks with functionality verification.

Key Features:
- Verify model loading and functionality
- Model-specific test cases for different model types
- Critical model identification and verification
- Detailed status reporting with response times

**Status**: Open 🔄

## Issue 13: Implement Kubernetes-Compatible Health Probes
**Title**: Implement Kubernetes-Compatible Health Probes
**Labels**: enhancement, devops
**Body**:
Enhance readiness and liveness probe endpoints for Kubernetes integration.

Key Features:
- Comprehensive readiness checks for component availability
- Lightweight liveness checks for crash detection
- Proper HTTP status codes for Kubernetes interpretation
- Component-specific detailed feedback

**Status**: Open 🔄

## Issue 14: Improve Basic and Detailed Health Check Endpoints
**Title**: Improve Basic and Detailed Health Check Endpoints
**Labels**: enhancement, monitoring
**Body**:
Replace placeholder health checks with real component verification while maintaining API compatibility.

Key Features:
- Real component status checks in basic health endpoint
- Comprehensive component verification in detailed endpoint
- Support for monitoring systems integration
- Proper error handling and degraded states

**Status**: Open 🔄