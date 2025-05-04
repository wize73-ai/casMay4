import pytest
import asyncio
import time
import uuid
import nest_asyncio
from typing import Dict, Any, Optional

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

# Import the RouteCacheManager to test
from app.services.storage.route_cache import RouteCacheManager, RouteCache

class TestRouteCache:
    """Tests for the RouteCache implementation."""
    
    @pytest.fixture
    def cache(self, event_loop):
        """Create a test cache instance."""
        async def _get_cache():
            # Initialize a small test cache with short TTL
            return await RouteCacheManager.get_cache(
                name=f"test_cache_{uuid.uuid4().hex}", 
                max_size=10,
                ttl_seconds=5,
                bloom_compatible=True
            )
        
        # Execute the coroutine to get the actual cache instance
        return event_loop.run_until_complete(_get_cache())
    
    @pytest.mark.asyncio
    async def test_basic_caching(self, cache):
        """Test basic cache set/get functionality."""
        # Set a test value
        key = "test_key"
        test_value = {"data": "test_data", "timestamp": time.time()}
        
        await cache.set(key, test_value)
        result = await cache.get(key)
        
        assert result is not None
        assert result["data"] == test_value["data"]
        
        # Check cache hit metrics
        metrics = cache.get_metrics()
        assert metrics["hits"] == 1
        assert metrics["misses"] == 0
    
    @pytest.mark.asyncio
    async def test_cache_miss(self, cache):
        """Test cache miss behavior."""
        key = "nonexistent_key"
        result = await cache.get(key)
        
        assert result is None
        
        # Check cache miss metrics
        metrics = cache.get_metrics()
        assert metrics["hits"] == 0
        assert metrics["misses"] == 1
    
    @pytest.mark.asyncio
    async def test_cache_expiration(self, cache):
        """Test that cache entries expire after TTL."""
        key = "expiring_key"
        test_value = {"data": "will_expire"}
        
        await cache.set(key, test_value, ttl=1)  # 1 second TTL
        
        # Verify it's in cache
        result = await cache.get(key)
        assert result is not None
        
        # Wait for expiration
        await asyncio.sleep(2)
        
        # Should be gone now
        result = await cache.get(key)
        assert result is None
    
    @pytest.mark.asyncio
    async def test_cache_eviction(self, cache):
        """Test that cache evicts least recently used items when full."""
        # Fill the cache
        for i in range(15):  # Cache max_size is 10
            await cache.set(f"key_{i}", {"value": i})
        
        # The earliest entries should have been evicted
        for i in range(5):
            result = await cache.get(f"key_{i}")
            assert result is None, f"Expected key_{i} to be evicted"
        
        # Later entries should still be present
        for i in range(5, 15):
            result = await cache.get(f"key_{i}")
            assert result is not None, f"Expected key_{i} to be in cache"
            assert result["value"] == i
    
    @pytest.mark.asyncio
    async def test_bloom_compatible_keys(self, cache):
        """Test that Bloom Housing compatible keys are generated."""
        # Test with CasaLingua format
        casa_request = {
            "text": "Hello world",
            "source_language": "en",
            "target_language": "es"
        }
        
        # Test with Bloom Housing format
        bloom_request = {
            "text": "Hello world",
            "sourceLanguage": "en",
            "targetLanguage": "es"
        }
        
        # Generate keys
        casa_key = await cache.generate_key(casa_request)
        bloom_key = await cache.generate_key(bloom_request)
        
        # If bloom compatible, these should generate identical keys
        assert casa_key == bloom_key, "Bloom compatible key generation failed"
        
    @pytest.mark.asyncio
    async def test_thread_safety(self, cache):
        """Test concurrent access to the cache."""
        key = "concurrent_key"
        test_value = {"count": 0}
        
        # Set initial value
        await cache.set(key, test_value)
        
        # Define increment task
        async def increment():
            for _ in range(10):
                value = await cache.get(key)
                if value:
                    value = dict(value)  # Make a copy to simulate real-world mutation
                    value["count"] += 1
                    await cache.set(key, value)
                await asyncio.sleep(0.01)  # Small delay to force task switching
        
        # Run multiple increment tasks concurrently
        tasks = [increment() for _ in range(5)]
        await asyncio.gather(*tasks)
        
        # Check final value
        final_value = await cache.get(key)
        assert final_value is not None
        
        # With proper locking, count should be 50 (5 tasks × 10 increments)
        # Without locking, it would likely be less due to race conditions
        assert final_value["count"] == 50, "Thread safety issue detected"