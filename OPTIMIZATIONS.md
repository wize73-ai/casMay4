# API Optimizations for CasaLingua

This document describes the optimizations implemented for the CasaLingua API to improve performance, reliability, and scalability.

## 1. Request-Level Caching

### Description
A thread-safe, async-compatible caching system optimized for API routes with Bloom Housing compatibility.

### Key Features
- Thread-safe operations with asyncio.Lock
- Configurable TTL and cache size
- Automatic LRU eviction when cache size is exceeded
- Metrics tracking for hit/miss ratios
- Compatible with Bloom Housing API format

### Implementation
- **File**: `app/services/storage/route_cache.py`
- **Components**:
  - `RouteCache`: Core caching class
  - `RouteCacheManager`: Central registry for cache instances

### Example Usage
```python
# Get or create a cache instance
cache = await RouteCacheManager.get_cache(
    name="translation", 
    max_size=2000,
    ttl_seconds=7200,
    bloom_compatible=True
)

# Generate a cache key from request data
key = await cache.generate_key(request_data)

# Try to get from cache
result = await cache.get(key)
if result is not None:
    return result

# Cache miss, compute result
result = await process_request(request_data)

# Store in cache
await cache.set(key, result)
```

## 2. Smart Batching

### Description
Automatic grouping of similar small requests to improve processing efficiency.

### Key Features
- Automatic grouping based on request properties
- Adaptive timing control for optimal batch sizes
- Concurrent processing of multiple batches
- Result mapping back to original requesters

### Implementation
- **File**: `app/api/middleware/batch_optimizer.py`
- **Components**:
  - `BatchOptimizer`: Core batching system
  - `BatchGroup`: Group of similar items
  - `BatchItem`: Individual request with completion tracking

### Example Usage
```python
# Create a batch optimizer
optimizer = BatchOptimizer(
    name="translation_batcher",
    max_batch_size=10,
    max_wait_time=0.2,
    process_function=process_batch
)

# Process a request through the batcher
result = await optimizer.process(request_data)
```

## 3. Streaming Responses

### Description
Support for real-time, chunk-based responses for large document processing.

### Key Features
- Chunk-based processing for large documents
- Event-driven format for client-side progress tracking
- Progress notifications during processing
- Resilience against timeouts for large operations

### Implementation
- **File**: `app/api/routes/streaming.py`
- **Components**:
  - `stream_translation`: Streaming translation endpoint
  - `stream_analysis`: Streaming analysis endpoint

### Example Usage
```python
# Server endpoint
@router.post("/streaming/translate")
async def stream_translate_text(...):
    return StreamingResponse(
        stream_translation(translation_request, current_user),
        media_type="text/event-stream"
    )

# Client-side handling
response = requests.post(
    "http://api.example.com/streaming/translate",
    json={"text": "Large document...", ...},
    stream=True
)

for line in response.iter_lines():
    event = json.loads(line)
    if event["event"] == "chunk_translated":
        print(f"Progress: {event['progress'] * 100:.0f}%")
        # Process event["translated_chunk"]
    elif event["event"] == "translation_completed":
        # Process completed translation
```

## 4. Enhanced Error Handling

### Description
Standardized error categorization with comprehensive fallbacks.

### Key Features
- Categorized errors for better client handling
- Standardized response format
- Fallback mechanisms for graceful degradation
- Request ID tracking for improved debugging

### Implementation
- **File**: `app/utils/error_handler.py`
- **Components**:
  - `APIError`: Base exception for API errors
  - `ErrorCategory`: Enum of error types
  - `ErrorResponse`: Standardized error response model
  - `handle_errors`: Decorator for route functions

### Example Usage
```python
# Define route with error handling
@handle_errors(fallbacks={
    TranslationError: lambda e, req: {"text": "Fallback translation", "error": str(e)}
})
async def translate_text(request, translation_request):
    # Implementation
    if not valid_request(translation_request):
        raise ValidationError("Invalid translation parameters")
    
    # Rest of implementation
```

## 5. Parallel Processing

### Description
Concurrent execution of independent tasks for improved throughput.

### Key Features
- Parallel preprocessing using asyncio.gather
- Non-blocking I/O throughout the API stack
- Background tasks for metrics and logging

### Implementation
- **File**: `app/api/routes/pipeline.py`
- **Example**:
```python
# Set up parallel tasks
preprocessing_tasks = {
    "language_detection": processor.detect_language(text),
    "context_retrieval": retriever.get_context(domain)
}

# Execute all tasks concurrently
results = await asyncio.gather(*preprocessing_tasks.values())
```

## 6. Bloom Housing Integration

### Description
Seamless compatibility with Bloom Housing API format.

### Key Features
- Compatible schemas for both systems
- Bidirectional conversion of request/response formats
- Transparent processing with core pipeline

### Implementation
- **Files**: 
  - `app/api/schemas/bloom_housing.py`
  - `app/api/routes/bloom_housing.py`

### Example Conversion
```python
# Convert Bloom Housing request to CasaLingua format
casa_request = translation_request.to_casa_lingua_format()

# Process with internal endpoint
casa_response = await translate_text(...)

# Convert back to Bloom Housing format
bloom_response = BloomTranslationResponse.from_casa_lingua_response(casa_response)
```

## Performance Impact

The optimizations above have significant impact on API performance:

1. **Caching**: 10-100x speedup for repeated requests
2. **Batching**: 2-5x throughput improvement for similar small requests
3. **Streaming**: Support for documents of unlimited size with minimal memory usage
4. **Parallel Processing**: 1.5-3x speedup for operations with independent subtasks

## Testing

Comprehensive tests are available to verify all optimizations:

- `app/tests/test_route_cache.py`: Tests for the route cache
- `app/tests/test_batch_optimizer.py`: Tests for the batch optimizer
- `app/tests/test_simple.py`: Simple component tests
- `app/tests/test_optimizations.py`: End-to-end API tests

Run the tests with:
```bash
python app/tests/run_tests.py
```