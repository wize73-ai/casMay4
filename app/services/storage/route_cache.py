"""
Route cache manager for efficient request-level caching.
This module provides a thread-safe, async-compatible caching system
optimized for API routes with Bloom Housing compatibility.
"""

import asyncio
import time
import hashlib
import logging
import json
from typing import Dict, List, Any, Optional, Set, Union
from datetime import datetime
import collections

from app.utils.logging import get_logger

# Initialize logger
logger = get_logger(__name__)

class RouteCache:
    """
    Thread-safe, async-compatible cache for API route results.
    
    This class provides a simple in-memory cache with TTL support, 
    hit/miss tracking, and thread safety for API responses.
    """
    
    def __init__(
        self,
        name: str,
        max_size: int = 1000,
        ttl_seconds: Optional[int] = 3600,
        bloom_compatible: bool = False
    ):
        """
        Initialize a new route cache.
        
        Args:
            name: Unique name for this cache instance
            max_size: Maximum number of items in the cache
            ttl_seconds: Time-to-live in seconds (None for no expiration)
            bloom_compatible: Whether to support Bloom Housing compatibility
        """
        self.name = name
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.bloom_compatible = bloom_compatible
        self.cache_enabled = True
        
        # Cache state
        self.cache: Dict[str, Any] = {}
        self.timestamps: Dict[str, float] = {}
        self.hit_counts: Dict[str, int] = {}
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        
        # Thread safety
        self.cache_lock = asyncio.Lock()
        
        logger.info(f"Route cache '{name}' initialized with max_size={max_size}, "
                   f"ttl_seconds={ttl_seconds}, bloom_compatible={bloom_compatible}")
                   
    @staticmethod
    def bloom_compatible_key(route_path: str, params: Dict[str, Any]) -> str:
        """
        Generate a cache key compatible with Bloom Housing format.
        
        Args:
            route_path: The API route path
            params: Request parameters to include in the key
            
        Returns:
            str: A deterministic cache key
        """
        try:
            # Create a copy of params to avoid modifying the original
            key_params = dict(params)
            
            # Remove fields that shouldn't affect the cache key
            for field in ["cache", "verify", "apiKey", "request_id"]:
                key_params.pop(field, None)
                
            # Map camelCase to snake_case if needed
            camel_to_snake = {
                "sourceLanguage": "source_language",
                "targetLanguage": "target_language",
                "formatPreservation": "preserve_formatting"
            }
            
            for camel, snake in camel_to_snake.items():
                if camel in key_params:
                    key_params[snake] = key_params.pop(camel)
            
            # Combine route and params into a key
            key_data = {
                "route": route_path,
                "params": key_params
            }
            
            # Generate a deterministic string representation
            serialized = json.dumps(key_data, sort_keys=True)
            
            # Hash the serialized data
            key = hashlib.md5(serialized.encode("utf-8")).hexdigest()
            return key
            
        except Exception as e:
            logger.warning(f"Error generating bloom-compatible cache key: {str(e)}")
            import uuid
            return f"error_key_{str(uuid.uuid4())}"
    
    async def generate_key(self, request_data: Any) -> str:
        """
        Generate a cache key from request data.
        
        Args:
            request_data: The request data to generate a key for
            
        Returns:
            str: A cache key
        """
        try:
            # For dict-like requests, generate a deterministic key
            if isinstance(request_data, dict):
                # Create a normalized copy of the request for consistent key generation
                key_data = dict(request_data)
                
                # Bloom Housing compatibility: map camelCase to snake_case for key fields
                if self.bloom_compatible:
                    camel_to_snake = {
                        "sourceLanguage": "source_language",
                        "targetLanguage": "target_language",
                        "formatPreservation": "preserve_formatting"
                    }
                    
                    for camel, snake in camel_to_snake.items():
                        if camel in key_data:
                            key_data[snake] = key_data.pop(camel)
                
                # Remove fields that shouldn't affect the cache key
                for field in ["cache", "verify", "apiKey", "request_id"]:
                    key_data.pop(field, None)
                
                # Sort keys for deterministic ordering and convert to JSON
                serialized = json.dumps(key_data, sort_keys=True)
            else:
                # For non-dict requests, use string representation
                serialized = str(request_data)
            
            # Generate a hash of the serialized data
            key = hashlib.md5(serialized.encode("utf-8")).hexdigest()
            return key
        
        except Exception as e:
            # If anything goes wrong, create a unique key that won't cause collisions
            logger.warning(f"Error generating cache key: {str(e)}")
            import uuid
            return f"error_key_{str(uuid.uuid4())}"
    
    async def get(self, key: str) -> Optional[Any]:
        """
        Get a value from the cache.
        
        Args:
            key: The cache key
            
        Returns:
            Optional[Any]: The cached value, or None if not found or expired
        """
        if not self.cache_enabled:
            self.misses += 1
            return None
        
        async with self.cache_lock:
            # Check if key exists
            if key not in self.cache:
                self.misses += 1
                return None
            
            # Check for expiration
            if key in self.timestamps:
                timestamp_data = self.timestamps.get(key)
                timestamp_time = timestamp_data.get('time', 0) if isinstance(timestamp_data, dict) else timestamp_data
                ttl = timestamp_data.get('ttl', self.ttl_seconds) if isinstance(timestamp_data, dict) else self.ttl_seconds
                
                if ttl is not None and time.time() - timestamp_time > ttl:
                    # Expired, remove from cache
                    del self.cache[key]
                    del self.timestamps[key]
                    if key in self.hit_counts:
                        del self.hit_counts[key]
                    self.misses += 1
                    return None
            
            # Cache hit
            self.hits += 1
            if key in self.hit_counts:
                self.hit_counts[key] += 1
            else:
                self.hit_counts[key] = 1
            
            # Make a copy to prevent inadvertent modification of cached value
            import copy
            return copy.deepcopy(self.cache[key])
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Set a value in the cache.
        
        Args:
            key: The cache key
            value: The value to cache
            ttl: Optional TTL override for this specific item (in seconds)
        """
        if not self.cache_enabled:
            return
        
        # Make a deep copy to prevent modification after caching
        import copy
        value_copy = copy.deepcopy(value)
        
        # Use lock to ensure thread safety
        async with self.cache_lock:
            # Ensure cache doesn't grow too large
            if len(self.cache) >= self.max_size and key not in self.cache:
                await self._evict_entry()
            
            # Add to cache
            self.cache[key] = value_copy
            
            # Store current time for TTL calculation
            # If ttl parameter is provided, it overrides the instance ttl_seconds
            self.timestamps[key] = {
                'time': time.time(),
                'ttl': ttl if ttl is not None else self.ttl_seconds
            }
            
            self.hit_counts[key] = 0
    
    async def invalidate(self, key: str) -> bool:
        """
        Invalidate a specific cache entry.
        
        Args:
            key: The cache key to invalidate
            
        Returns:
            bool: True if the key was found and invalidated, False otherwise
        """
        async with self.cache_lock:
            if key in self.cache:
                del self.cache[key]
                del self.timestamps[key]
                if key in self.hit_counts:
                    del self.hit_counts[key]
                return True
            return False
    
    async def clear(self) -> None:
        """Clear all entries from the cache."""
        async with self.cache_lock:
            self.cache.clear()
            self.timestamps.clear()
            self.hit_counts.clear()
    
    async def _evict_entry(self) -> None:
        """Evict the least recently used entry from the cache."""
        # Find the oldest entry
        oldest_key = min(self.timestamps.items(), key=lambda x: 
                          x[1]['time'] if isinstance(x[1], dict) else x[1])[0]
        
        # Remove it
        del self.cache[oldest_key]
        del self.timestamps[oldest_key]
        if oldest_key in self.hit_counts:
            del self.hit_counts[oldest_key]
        
        self.evictions += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get cache metrics.
        
        Returns:
            Dict[str, Any]: Cache metrics
        """
        total_requests = self.hits + self.misses
        hit_rate = self.hits / max(1, total_requests)
        
        return {
            "name": self.name,
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "evictions": self.evictions,
            "enabled": self.cache_enabled,
            "ttl_seconds": self.ttl_seconds,
            "bloom_compatible": self.bloom_compatible
        }
    
    def enable(self) -> None:
        """Enable the cache."""
        self.cache_enabled = True
    
    def disable(self) -> None:
        """Disable the cache without clearing it."""
        self.cache_enabled = False

class RouteCacheManager:
    """
    Manager for multiple named route cache instances.
    
    This class provides a centralized way to create, access, and manage
    multiple cache instances.
    """
    
    # Static registry of cache instances
    _caches: Dict[str, RouteCache] = {}
    _lock = asyncio.Lock()
    
    @classmethod
    async def get_cache(
        cls,
        name: str,
        max_size: int = 1000,
        ttl_seconds: Optional[int] = 3600,
        bloom_compatible: bool = False
    ) -> RouteCache:
        """
        Get or create a cache instance by name.
        
        Args:
            name: Unique name for the cache instance
            max_size: Maximum number of items in the cache
            ttl_seconds: Time-to-live in seconds (None for no expiration)
            bloom_compatible: Whether to support Bloom Housing compatibility
            
        Returns:
            RouteCache: The cache instance
        """
        async with cls._lock:
            if name not in cls._caches:
                # Create new cache
                cache = RouteCache(
                    name=name,
                    max_size=max_size,
                    ttl_seconds=ttl_seconds,
                    bloom_compatible=bloom_compatible
                )
                cls._caches[name] = cache
                return cache
            
            # Return existing cache
            return cls._caches[name]
    
    @classmethod
    async def remove_cache(cls, name: str) -> bool:
        """
        Remove a cache instance by name.
        
        Args:
            name: The name of the cache instance to remove
            
        Returns:
            bool: True if the cache was found and removed, False otherwise
        """
        async with cls._lock:
            if name in cls._caches:
                # Clear the cache before removing
                await cls._caches[name].clear()
                del cls._caches[name]
                return True
            return False
    
    @classmethod
    async def clear_all(cls) -> None:
        """Clear all cache instances."""
        async with cls._lock:
            for cache in cls._caches.values():
                await cache.clear()
    
    @classmethod
    async def get_all_stats(cls) -> Dict[str, Dict[str, Any]]:
        """
        Get metrics for all cache instances.
        
        Returns:
            Dict[str, Dict[str, Any]]: Metrics for all cache instances
        """
        async with cls._lock:
            return {name: cache.get_metrics() for name, cache in cls._caches.items()}