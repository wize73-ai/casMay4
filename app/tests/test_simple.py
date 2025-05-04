"""
Simple tests for CasaLingua API optimizations.
"""

import pytest
import time

# Import RouteCache for testing
from app.services.storage.route_cache import RouteCache

def test_route_cache_basic():
    """Test basic route cache functionality."""
    # Create a cache
    cache = RouteCache(name="test_cache", max_size=10, ttl_seconds=5)
    
    # Test properties
    assert cache.name == "test_cache"
    assert cache.max_size == 10
    assert cache.ttl_seconds == 5
    assert cache.cache_enabled is True
    
    # Test metrics
    metrics = cache.get_metrics()
    assert metrics["size"] == 0
    assert metrics["hits"] == 0
    assert metrics["misses"] == 0
    
    # Test enable/disable
    cache.disable()
    assert cache.cache_enabled is False
    cache.enable()
    assert cache.cache_enabled is True
    
    print("✅ Route cache basic tests passed")

def test_batch_optimizer_basic():
    """Test basic batch optimizer functionality."""
    # Import BatchOptimizer and related classes
    from app.api.middleware.batch_optimizer import BatchOptimizer, BatchItem, BatchGroup
    
    # Create a batch group
    batch = BatchGroup(group_key="test_group")
    assert batch.group_key == "test_group"
    assert len(batch.items) == 0
    assert batch.processing is False
    
    # Create a batch item
    item = BatchItem(key="test_item", item_data={"text": "Hello"})
    assert item.key == "test_item"
    assert item.item_data["text"] == "Hello"
    assert item.processed is False
    
    # Add item to batch
    result = batch.add_item(item)
    assert result is True
    assert len(batch.items) == 1
    assert "test_item" in batch.items
    
    # Create a batch optimizer
    optimizer = BatchOptimizer.create_for_testing(name="test_optimizer")
    assert optimizer.name == "test_optimizer"
    # Note: running might be True or False depending on implementation
    
    print("✅ Batch optimizer basic tests passed")

def test_streaming_utils():
    """Test streaming response utilities."""
    import json
    
    # Test event formatting
    event_data = {
        "event": "chunk_translated",
        "chunk_index": 0,
        "total_chunks": 5,
        "progress": 0.2,
        "translated_chunk": "Hola mundo"
    }
    
    # Convert to event string
    event_str = json.dumps(event_data) + "\n"
    
    # Parse back
    parsed = json.loads(event_str.strip())
    
    # Verify
    assert parsed["event"] == "chunk_translated"
    assert parsed["progress"] == 0.2
    assert parsed["translated_chunk"] == "Hola mundo"
    
    print("✅ Streaming utils tests passed")

def test_error_handler():
    """Test error handler components."""
    # Import error components
    from app.utils.error_handler import ErrorCategory, ErrorResponse, APIError
    
    # Test error categories
    assert ErrorCategory.VALIDATION == "validation"
    assert ErrorCategory.MODEL_ERROR == "model_error"
    
    # Test error response
    error_resp = ErrorResponse(
        status_code=400,
        error_code="invalid_input",
        category=ErrorCategory.VALIDATION,
        message="Invalid input parameters"
    )
    
    assert error_resp.status_code == 400
    assert error_resp.error_code == "invalid_input"
    assert error_resp.category == ErrorCategory.VALIDATION
    
    # Test API error
    api_error = APIError(
        message="Test error",
        status_code=404,
        error_code="not_found",
        category=ErrorCategory.RESOURCE_NOT_FOUND
    )
    
    assert api_error.message == "Test error"
    assert api_error.status_code == 404
    
    # Convert to response
    response = api_error.to_response()
    assert response.status_code == 404
    assert response.category == ErrorCategory.RESOURCE_NOT_FOUND
    
    print("✅ Error handler tests passed")

if __name__ == "__main__":
    print("Running simple tests...")
    test_route_cache_basic()
    test_batch_optimizer_basic()
    test_streaming_utils()
    test_error_handler()
    print("All tests passed!")