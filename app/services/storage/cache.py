"""
Storage Cache Module for CasaLingua
Provides caching mechanisms for optimizing data access and storage
"""

import os
import json
import pickle
import logging
import time
import shutil
import hashlib
import threading
import asyncio
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
from datetime import datetime, timedelta
import numpy as np
from collections import OrderedDict

# # Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger(__name__)

# class CacheEntry:
#     """Class representing a cache entry with metadata"""
    
#     def __init__(self, key: str, value: Any, expiry: Optional[datetime] = None, 
#                 metadata: Optional[Dict[str, Any]] = None, now_func: Optional[Callable[[], datetime]] = None):
#         """
#         Initialize a cache entry
        
#         Args:
#             key (str): Cache key
#             value (Any): Cached value
#             expiry (datetime, optional): Expiry timestamp
#             metadata (Dict, optional): Additional metadata
#         """
#         self.key = key
#         self.value = value
#         self.expiry = expiry
#         self.metadata = metadata or {}
#         self._now = now_func if now_func is not None else datetime.now
#         self.created_at = self._now()
#         self.last_accessed = self._now()
#         self.access_count = 0
    
#     def is_expired(self) -> bool:
#         """
#         Check if the cache entry is expired
        
#         Returns:
#             bool: True if expired, False otherwise
#         """
#         if self.expiry is None:
#             return False
#         return self._now() > self.expiry
    
#     def access(self) -> None:
#         """Update last accessed time and access count"""
#         self.last_accessed = self._now()
#         self.access_count += 1
    
#     def get_age(self) -> timedelta:
#         """
#         Get age of the cache entry
        
#         Returns:
#             timedelta: Age of the cache entry
#         """
#         return self._now() - self.created_at
    
#     def get_size(self) -> int:
#         """
#         Estimate size of the cache entry in bytes
        
#         Returns:
#             int: Estimated size in bytes
#         """
#         try:
#             # Try to get size with pickle
#             return len(pickle.dumps(self.value))
#         except:
#             # Fallback for non-picklable objects
#             return 0


# class MemoryCache:
#     """In-memory cache with expiration and LRU eviction"""
    
#     def __init__(self, max_size: int = 1000, ttl: Optional[int] = None, now_func: Optional[Callable[[], datetime]] = None):
#         """
#         Initialize memory cache
        
#         Args:
#             max_size (int): Maximum number of items in cache
#             ttl (int, optional): Default time-to-live in seconds
#         """
#         self.cache = OrderedDict()  # type: OrderedDict[str, CacheEntry]
#         self.max_size = max_size
#         self.ttl = ttl
#         self.lock = threading.RLock()
#         self.stats = {
#             "hits": 0,
#             "misses": 0,
#             "evictions": 0,
#             "expirations": 0
#         }
#         self._now = now_func if now_func is not None else datetime.now
    
#     def get(self, key: str, default: Any = None) -> Any:
#         """
#         Get a value from the cache
        
#         Args:
#             key (str): Cache key
#             default (Any): Default value if key not found
            
#         Returns:
#             Any: Cached value or default
#         """
#         with self.lock:
#             if key not in self.cache:
#                 self.stats["misses"] += 1
#                 return default
            
#             entry = self.cache[key]
            
#             # Check for expiration
#             if entry.is_expired():
#                 self.stats["expirations"] += 1
#                 del self.cache[key]
#                 return default
            
#             # Update access info and move to end (LRU)
#             entry.access()
#             self.cache.move_to_end(key)
            
#             self.stats["hits"] += 1
#             return entry.value
    
#     def set(self, key: str, value: Any, ttl: Optional[int] = None, 
#            metadata: Optional[Dict[str, Any]] = None) -> None:
#         """
#         Set a value in the cache
        
#         Args:
#             key (str): Cache key
#             value (Any): Value to cache
#             ttl (int, optional): Time-to-live in seconds
#             metadata (Dict, optional): Additional metadata
#         """
#         with self.lock:
#             # Calculate expiry
#             expiry = None
#             now = self._now()
#             if ttl is not None:
#                 expiry = now + timedelta(seconds=ttl)
#             elif self.ttl is not None:
#                 expiry = now + timedelta(seconds=self.ttl)
#             # Create entry
#             entry = CacheEntry(key, value, expiry, metadata, now_func=self._now)
#             # Evict if at capacity
#             if len(self.cache) >= self.max_size and key not in self.cache:
#                 self.cache.popitem(last=False)  # Remove oldest item
#                 self.stats["evictions"] += 1
#             # Add to cache
#             self.cache[key] = entry
#             self.cache.move_to_end(key)
    
#     def delete(self, key: str) -> bool:
#         """
#         Delete an entry from the cache
        
#         Args:
#             key (str): Cache key
            
#         Returns:
#             bool: True if deleted, False if not found
#         """
#         with self.lock:
#             if key in self.cache:
#                 del self.cache[key]
#                 return True
#             return False
    
#     def clear(self) -> None:
#         """Clear all cache entries"""
#         with self.lock:
#             self.cache.clear()
    
#     def get_stats(self, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
#         """
#         Get cache statistics
        
#         Returns:
#             Dict[str, Any]: Cache statistics
#         """
#         with self.lock:
#             stats = self.stats.copy()
#             stats["size"] = len(self.cache)
#             stats["max_size"] = self.max_size
#             # Calculate hit rate
#             total_requests = stats["hits"] + stats["misses"]
#             stats["hit_rate"] = stats["hits"] / total_requests if total_requests > 0 else 0
#             # Accept options for future extensibility (currently unused)
#             return stats
    
#     def get_keys(self) -> List[str]:
#         """
#         Get all cache keys
        
#         Returns:
#             List[str]: List of cache keys
#         """
#         with self.lock:
#             return list(self.cache.keys())
    
#     def get_metadata(self, key: str) -> Optional[Dict[str, Any]]:
#         """
#         Get metadata for a cache entry
        
#         Args:
#             key (str): Cache key
            
#         Returns:
#             Dict[str, Any]: Entry metadata or None if not found
#         """
#         with self.lock:
#             if key in self.cache:
#                 entry = self.cache[key]
#                 return {
#                     "created_at": entry.created_at,
#                     "last_accessed": entry.last_accessed,
#                     "access_count": entry.access_count,
#                     "expiry": entry.expiry,
#                     "is_expired": entry.is_expired(),
#                     "age": entry.get_age().total_seconds(),
#                     "size": entry.get_size(),
#                     **entry.metadata
#                 }
#             return None
    
#     def cleanup(self, options: Optional[Dict[str, Any]] = None) -> int:
#         """
#         Remove expired entries from the cache
        
#         Returns:
#             int: Number of entries removed
#         """
#         with self.lock:
#             expired_keys = [k for k, v in self.cache.items() if v.is_expired()]
#             for key in expired_keys:
#                 del self.cache[key]
#             self.stats["expirations"] += len(expired_keys)
#             # Accept options for future extensibility (currently unused)
#             return len(expired_keys)


# class DiskCache:
#     """Persistent disk-based cache with expiration"""
    
#     def __init__(self, cache_dir: str, max_size_mb: int = 1000, ttl: Optional[int] = None,
#                 cleanup_interval: int = 3600, now_func: Optional[Callable[[], datetime]] = None):
#         """
#         Initialize disk cache
        
#         Args:
#             cache_dir (str): Directory to store cache files
#             max_size_mb (int): Maximum size in megabytes
#             ttl (int, optional): Default time-to-live in seconds
#             cleanup_interval (int): Interval between cleanup runs in seconds
#         """
#         self.cache_dir = cache_dir
#         self.max_size_bytes = max_size_mb * 1024 * 1024
#         self.ttl = ttl
#         self.cleanup_interval = cleanup_interval
#         self.lock = threading.RLock()
#         self.stats = {
#             "hits": 0,
#             "misses": 0,
#             "evictions": 0,
#             "expirations": 0
#         }
#         self._now = now_func if now_func is not None else datetime.now
#         # Create cache directory if it doesn't exist
#         os.makedirs(cache_dir, exist_ok=True)
#         # Create metadata directory
#         self.metadata_dir = os.path.join(cache_dir, "_metadata")
#         os.makedirs(self.metadata_dir, exist_ok=True)
#         # Load stats if they exist
#         self._load_stats()
#         # Schedule cleanup
#         self._schedule_cleanup()
    
#     def _hash_key(self, key: str) -> str:
#         """
#         Hash a key to get a filename
        
#         Args:
#             key (str): Cache key
            
#         Returns:
#             str: Hashed key
#         """
#         hash_obj = hashlib.md5(key.encode())
#         return hash_obj.hexdigest()
    
#     def _get_path(self, key: str) -> str:
#         """
#         Get path to cache file for a key
        
#         Args:
#             key (str): Cache key
            
#         Returns:
#             str: Path to cache file
#         """
#         hashed = self._hash_key(key)
#         return os.path.join(self.cache_dir, hashed)
    
#     def _get_metadata_path(self, key: str) -> str:
#         """
#         Get path to metadata file for a key
        
#         Args:
#             key (str): Cache key
            
#         Returns:
#             str: Path to metadata file
#         """
#         hashed = self._hash_key(key)
#         return os.path.join(self.metadata_dir, f"{hashed}.meta")
    
#     def _save_metadata(self, key: str, metadata: Dict[str, Any]) -> None:
#         """
#         Save metadata for a cache entry
        
#         Args:
#             key (str): Cache key
#             metadata (Dict): Metadata to save
#         """
#         try:
#             path = self._get_metadata_path(key)
#             # Convert datetime objects to strings
#             meta_copy = {}
#             for k, v in metadata.items():
#                 if isinstance(v, datetime):
#                     meta_copy[k] = v.isoformat()
#                 else:
#                     meta_copy[k] = v
#             with open(path, 'w') as f:
#                 json.dump(meta_copy, f)
#         except Exception as e:
#             logger.warning(f"Failed to save metadata for {key}: {e}", exc_info=True)
    
#     def _load_metadata(self, key: str) -> Optional[Dict[str, Any]]:
#         """
#         Load metadata for a cache entry
        
#         Args:
#             key (str): Cache key
            
#         Returns:
#             Dict[str, Any]: Entry metadata or None if not found
#         """
#         try:
#             path = self._get_metadata_path(key)
#             if not os.path.exists(path):
#                 return None
#             with open(path, 'r') as f:
#                 metadata = json.load(f)
#             # Convert string dates back to datetime objects
#             for k in ["created_at", "last_accessed", "expiry"]:
#                 if k in metadata and metadata[k]:
#                     try:
#                         metadata[k] = datetime.fromisoformat(metadata[k])
#                     except Exception:
#                         pass
#             return metadata
#         except Exception as e:
#             logger.warning(f"Failed to load metadata for {key}: {e}", exc_info=True)
#             return None
    
#     def _save_stats(self) -> None:
#         """Save cache statistics to disk"""
#         try:
#             stats_path = os.path.join(self.cache_dir, "_stats.json")
#             with open(stats_path, 'w') as f:
#                 json.dump(self.stats, f)
#         except Exception as e:
#             logger.warning(f"Failed to save cache stats: {e}", exc_info=True)
    
#     def _load_stats(self) -> None:
#         """Load cache statistics from disk"""
#         try:
#             stats_path = os.path.join(self.cache_dir, "_stats.json")
#             if os.path.exists(stats_path):
#                 with open(stats_path, 'r') as f:
#                     self.stats.update(json.load(f))
#         except Exception as e:
#             logger.warning(f"Failed to load cache stats: {e}", exc_info=True)
    
#     def _schedule_cleanup(self) -> None:
#         """Schedule periodic cache cleanup"""
#         def cleanup_task():
#             while True:
#                 time.sleep(self.cleanup_interval)
#                 try:
#                     self.cleanup()
#                 except Exception as e:
#                     logger.error(f"Error during cache cleanup: {e}", exc_info=True)
#         cleanup_thread = threading.Thread(target=cleanup_task, daemon=True)
#         cleanup_thread.start()
    
#     def get(self, key: str, default: Any = None) -> Any:
#         """
#         Get a value from the cache
        
#         Args:
#             key (str): Cache key
#             default (Any): Default value if key not found
            
#         Returns:
#             Any: Cached value or default
#         """
#         with self.lock:
#             path = self._get_path(key)
#             if not os.path.exists(path):
#                 self.stats["misses"] += 1
#                 return default
#             # Load metadata
#             metadata = self._load_metadata(key)
#             # Check for expiration
#             if metadata and "expiry" in metadata and metadata["expiry"]:
#                 expiry = metadata["expiry"]
#                 if isinstance(expiry, str):
#                     expiry = datetime.fromisoformat(expiry)
#                 if expiry < self._now():
#                     # Expired, delete files
#                     try:
#                         os.remove(path)
#                         meta_path = self._get_metadata_path(key)
#                         if os.path.exists(meta_path):
#                             os.remove(meta_path)
#                     except Exception:
#                         pass
#                     self.stats["expirations"] += 1
#                     return default
#             # Update access info
#             if metadata:
#                 metadata["last_accessed"] = self._now()
#                 metadata["access_count"] = metadata.get("access_count", 0) + 1
#                 self._save_metadata(key, metadata)
#             # Load value
#             try:
#                 with open(path, 'rb') as f:
#                     value = pickle.load(f)
#                 self.stats["hits"] += 1
#                 self._save_stats()
#                 return value
#             except Exception as e:
#                 logger.warning(f"Failed to load cache value for {key}: {e}", exc_info=True)
#                 self.stats["misses"] += 1
#                 return default
    
#     def set(self, key: str, value: Any, ttl: Optional[int] = None,
#            metadata: Optional[Dict[str, Any]] = None) -> bool:
#         """
#         Set a value in the cache
        
#         Args:
#             key (str): Cache key
#             value (Any): Value to cache
#             ttl (int, optional): Time-to-live in seconds
#             metadata (Dict, optional): Additional metadata
            
#         Returns:
#             bool: Success status
#         """
#         with self.lock:
#             # Check for available space
#             self._ensure_space()
#             path = self._get_path(key)
#             # Calculate expiry
#             expiry = None
#             now = self._now()
#             if ttl is not None:
#                 expiry = now + timedelta(seconds=ttl)
#             elif self.ttl is not None:
#                 expiry = now + timedelta(seconds=self.ttl)
#             # Prepare metadata
#             entry_metadata = metadata or {}
#             entry_metadata.update({
#                 "key": key,
#                 "created_at": now,
#                 "last_accessed": now,
#                 "access_count": 0,
#                 "expiry": expiry
#             })
#             # Save value
#             try:
#                 with open(path, 'wb') as f:
#                     pickle.dump(value, f)
#                 # Save metadata
#                 self._save_metadata(key, entry_metadata)
#                 return True
#             except Exception as e:
#                 logger.error(f"Failed to save cache value for {key}: {e}", exc_info=True)
#                 return False
    
#     def _ensure_space(self) -> None:
#         """Ensure there's enough space for new entries by removing old ones if needed"""
#         try:
#             # Get current size
#             current_size = sum(os.path.getsize(os.path.join(self.cache_dir, f))
#                             for f in os.listdir(self.cache_dir)
#                             if os.path.isfile(os.path.join(self.cache_dir, f)))
            
#             # Check if we need to evict
#             if current_size < self.max_size_bytes:
#                 return
            
#             # Get all cache entries with their metadata
#             entries = []
#             for filename in os.listdir(self.cache_dir):
#                 if filename.startswith("_"):
#                     continue
                
#                 file_path = os.path.join(self.cache_dir, filename)
#                 if not os.path.isfile(file_path):
#                     continue
                
#                 # Try to load metadata
#                 try:
#                     key = filename  # This is the hashed key
#                     metadata = self._load_metadata(key)
                    
#                     if metadata:
#                         entries.append({
#                             "key": key,
#                             "path": file_path,
#                             "size": os.path.getsize(file_path),
#                             "last_accessed": metadata.get("last_accessed", datetime.min),
#                             "access_count": metadata.get("access_count", 0)
#                         })
#                 except:
#                     # If metadata can't be loaded, use file stats
#                     stat = os.stat(file_path)
#                     entries.append({
#                         "key": filename,
#                         "path": file_path,
#                         "size": stat.st_size,
#                         "last_accessed": datetime.fromtimestamp(stat.st_atime),
#                         "access_count": 0
#                     })
            
#             # Sort by last accessed (oldest first)
#             entries.sort(key=lambda x: x["last_accessed"])
            
#             # Remove entries until we have enough space
#             space_to_free = current_size - (self.max_size_bytes * 0.8)  # Free 20% of max size
#             freed = 0
#             evicted = 0
            
#             for entry in entries:
#                 if freed >= space_to_free:
#                     break
                
#                 try:
#                     # Remove cache file
#                     os.remove(entry["path"])
#                     freed += entry["size"]
#                     evicted += 1
                    
#                     # Remove metadata file
#                     meta_path = self._get_metadata_path(entry["key"])
#                     if os.path.exists(meta_path):
#                         os.remove(meta_path)
#                 except Exception as e:
#                     logger.warning(f"Failed to remove cache entry {entry['key']}: {e}", exc_info=True)
            
#             self.stats["evictions"] += evicted
            
#         except Exception as e:
#             logger.error(f"Error ensuring cache space: {e}", exc_info=True)
    
#     def delete(self, key: str) -> bool:
#         """
#         Delete an entry from the cache
        
#         Args:
#             key (str): Cache key
            
#         Returns:
#             bool: True if deleted, False if not found
#         """
#         with self.lock:
#             path = self._get_path(key)
#             meta_path = self._get_metadata_path(key)
            
#             if not os.path.exists(path):
#                 return False
            
#             try:
#                 os.remove(path)
#                 if os.path.exists(meta_path):
#                     os.remove(meta_path)
#                 return True
#             except Exception as e:
#                 logger.warning(f"Failed to delete cache entry {key}: {e}", exc_info=True)
#                 return False
    
#     def clear(self) -> None:
#         """Clear all cache entries"""
#         with self.lock:
#             try:
#                 # Remove all files except stats and system files
#                 for filename in os.listdir(self.cache_dir):
#                     if filename.startswith("_"):
#                         continue
                    
#                     file_path = os.path.join(self.cache_dir, filename)
#                     if os.path.isfile(file_path):
#                         os.remove(file_path)
                
#                 # Clear metadata directory
#                 for filename in os.listdir(self.metadata_dir):
#                     file_path = os.path.join(self.metadata_dir, filename)
#                     if os.path.isfile(file_path):
#                         os.remove(file_path)
                
#                 # Reset stats
#                 self.stats = {
#                     "hits": 0,
#                     "misses": 0,
#                     "evictions": 0,
#                     "expirations": 0
#                 }
#                 self._save_stats()
                
#             except Exception as e:
#                 logger.error(f"Error clearing cache: {e}", exc_info=True)
    
#     def get_stats(self, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
#         """
#         Get cache statistics
        
#         Returns:
#             Dict[str, Any]: Cache statistics
#         """
#         with self.lock:
#             stats = self.stats.copy()
#             try:
#                 # Count files
#                 count = sum(1 for f in os.listdir(self.cache_dir) 
#                           if os.path.isfile(os.path.join(self.cache_dir, f)) and not f.startswith("_"))
#                 # Calculate size
#                 size_bytes = sum(os.path.getsize(os.path.join(self.cache_dir, f))
#                               for f in os.listdir(self.cache_dir)
#                               if os.path.isfile(os.path.join(self.cache_dir, f)))
#                 stats["size"] = count
#                 stats["size_bytes"] = size_bytes
#                 stats["size_mb"] = size_bytes / (1024 * 1024)
#                 stats["max_size_mb"] = self.max_size_bytes / (1024 * 1024)
#                 # Calculate hit rate
#                 total_requests = stats["hits"] + stats["misses"]
#                 stats["hit_rate"] = stats["hits"] / total_requests if total_requests > 0 else 0
#             except Exception as e:
#                 logger.error(f"Error calculating cache stats: {e}", exc_info=True)
#             # Accept options for future extensibility (currently unused)
#             return stats
    
#     def get_keys(self) -> List[str]:
#         """
#         Get all cache keys (this is expensive for disk cache)
        
#         Returns:
#             List[str]: List of cache keys
#         """
#         with self.lock:
#             keys = []
            
#             try:
#                 # Scan metadata directory for keys
#                 for filename in os.listdir(self.metadata_dir):
#                     if not filename.endswith(".meta"):
#                         continue
                    
#                     file_path = os.path.join(self.metadata_dir, filename)
#                     try:
#                         with open(file_path, 'r') as f:
#                             metadata = json.load(f)
#                             if "key" in metadata:
#                                 keys.append(metadata["key"])
#                     except:
#                         pass
                
#             except Exception as e:
#                 logger.error(f"Error getting cache keys: {e}", exc_info=True)
            
#             return keys
    
#     def get_metadata(self, key: str) -> Optional[Dict[str, Any]]:
#         """
#         Get metadata for a cache entry
        
#         Args:
#             key (str): Cache key
            
#         Returns:
#             Dict[str, Any]: Entry metadata or None if not found
#         """
#         with self.lock:
#             metadata = self._load_metadata(key)
            
#             if metadata:
#                 # Add file size
#                 path = self._get_path(key)
#                 if os.path.exists(path):
#                     metadata["size"] = os.path.getsize(path)
                
#                 # Calculate age
#                 if "created_at" in metadata:
#                     created_at = metadata["created_at"]
#                     if isinstance(created_at, str):
#                         created_at = datetime.fromisoformat(created_at)
                    
#                     metadata["age"] = (datetime.now() - created_at).total_seconds()
                
#                 # Check expiry
#                 if "expiry" in metadata and metadata["expiry"]:
#                     expiry = metadata["expiry"]
#                     if isinstance(expiry, str):
#                         expiry = datetime.fromisoformat(expiry)
                    
#                     metadata["is_expired"] = expiry < datetime.now()
            
#             return metadata
    
#     def cleanup(self, options: Optional[Dict[str, Any]] = None) -> int:
#         """
#         Remove expired entries from the cache
        
#         Returns:
#             int: Number of entries removed
#         """
#         with self.lock:
#             removed = 0
#             try:
#                 # Scan metadata for expired entries
#                 for filename in os.listdir(self.metadata_dir):
#                     if not filename.endswith(".meta"):
#                         continue
#                     file_path = os.path.join(self.metadata_dir, filename)
#                     try:
#                         with open(file_path, 'r') as f:
#                             metadata = json.load(f)
#                         # Check expiry
#                         if "expiry" in metadata and metadata["expiry"]:
#                             expiry = metadata["expiry"]
#                             if isinstance(expiry, str):
#                                 expiry = datetime.fromisoformat(expiry)
#                             if expiry < self._now():
#                                 # Expired, delete files
#                                 key = metadata.get("key", filename[:-5])  # Remove .meta
#                                 if self.delete(key):
#                                     removed += 1
#                     except Exception:
#                         pass
#                 self.stats["expirations"] += removed
#                 self._save_stats()
#             except Exception as e:
#                 logger.error(f"Error during cache cleanup: {e}", exc_info=True)
#             # Accept options for future extensibility (currently unused)
#             return removed


# class EmbeddingCache:
#     """Specialized cache for vector embeddings"""
    
#     def __init__(self, base_cache: Union[MemoryCache, DiskCache], 
#                 similarity_threshold: float = 0.9):
#         """
#         Initialize embedding cache
        
#         Args:
#             base_cache: Underlying cache implementation
#             similarity_threshold (float): Threshold for semantic similarity matching
#         """
#         self.cache = base_cache
#         self.similarity_threshold = similarity_threshold
#         self.vector_keys = {}  # Maps vector hash to actual key
    
#     def _hash_vector(self, vector: np.ndarray) -> str:
#         """
#         Create a hash for a vector
        
#         Args:
#             vector (np.ndarray): Vector to hash
            
#         Returns:
#             str: Vector hash
#         """
#         # Quantize vector to reduce sensitivity
#         quantized = (vector * 100).astype(np.int32)
#         return hashlib.md5(quantized.tobytes()).hexdigest()
    
#     def _cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
#         """
#         Calculate cosine similarity between two vectors
        
#         Args:
#             v1 (np.ndarray): First vector
#             v2 (np.ndarray): Second vector
            
#         Returns:
#             float: Cosine similarity (-1 to 1)
#         """
#         dot_product = np.dot(v1, v2)
#         norm_v1 = np.linalg.norm(v1)
#         norm_v2 = np.linalg.norm(v2)
        
#         if norm_v1 == 0 or norm_v2 == 0:
#             return 0.0
            
#         return dot_product / (norm_v1 * norm_v2)
    
#     def get(self, vector: np.ndarray, text: Optional[str] = None, 
#            default: Any = None, exact_match: bool = False) -> Tuple[Any, float]:
#         """
#         Get a value from the cache based on vector similarity
        
#         Args:
#             vector (np.ndarray): Query vector
#             text (str, optional): Text representation for exact matching
#             default (Any): Default value if no match found
#             exact_match (bool): Whether to require exact match
            
#         Returns:
#             Tuple[Any, float]: (value, similarity) tuple
#         """
#         if exact_match and text:
#             # Try exact text match first
#             text_key = f"text:{text}"
#             result = self.cache.get(text_key)
#             if result is not None:
#                 return result, 1.0
        
#         # Try vector match
#         vector_hash = self._hash_vector(vector)
#         vector_key = f"vector:{vector_hash}"
        
#         # Check if we have an exact match
#         result = self.cache.get(vector_key)
#         if result is not None:
#             return result, 1.0
        
#         if exact_match:
#             return default, 0.0
        
#         # Look for similar vectors
#         best_match = None
#         best_similarity = 0.0
        
#         # Get cached vector metadata
#         for key in self.vector_keys.values():
#             metadata = self.cache.get_metadata(key)
#             if metadata and "vector" in metadata:
#                 cached_vector = np.array(metadata["vector"])
#                 similarity = self._cosine_similarity(vector, cached_vector)
                
#                 if similarity > best_similarity:
#                     best_similarity = similarity
#                     if similarity >= self.similarity_threshold:
#                         result = self.cache.get(key)
#                         if result is not None:
#                             best_match = result
        
#         if best_match is not None and best_similarity >= self.similarity_threshold:
#             return best_match, best_similarity
        
#         return default, 0.0
    
#     def set(self, vector: np.ndarray, value: Any, text: Optional[str] = None,
#            ttl: Optional[int] = None) -> None:
#         """
#         Set a value in the cache
        
#         Args:
#             vector (np.ndarray): Vector to cache
#             value (Any): Value to cache
#             text (str, optional): Text representation for exact matching
#             ttl (int, optional): Time-to-live in seconds
#         """
#         # Store with vector hash
#         vector_hash = self._hash_vector(vector)
#         vector_key = f"vector:{vector_hash}"
        
#         # Store vector in metadata for similarity calculations
#         metadata = {
#             "vector": vector.tolist(),
#             "hash": vector_hash,
#             "dim": vector.shape[0]
#         }
        
#         if text:
#             metadata["text"] = text
#             # Also store with text key for exact matching
#             text_key = f"text:{text}"
#             self.cache.set(text_key, value, ttl)
        
#         # Store with vector key
#         self.cache.set(vector_key, value, ttl, metadata)
        
#         # Keep track of vector keys
#         self.vector_keys[vector_hash] = vector_key
    
#     def delete(self, vector: np.ndarray = None, text: str = None) -> bool:
#         """
#         Delete entries from the cache
        
#         Args:
#             vector (np.ndarray, optional): Vector to delete
#             text (str, optional): Text representation to delete
            
#         Returns:
#             bool: True if any entries deleted, False otherwise
#         """
#         deleted = False
        
#         if vector is not None:
#             vector_hash = self._hash_vector(vector)
#             vector_key = f"vector:{vector_hash}"
            
#             if self.cache.delete(vector_key):
#                 deleted = True
#                 if vector_hash in self.vector_keys:
#                     del self.vector_keys[vector_hash]
        
#         if text:
#             text_key = f"text:{text}"
#             if self.cache.delete(text_key):
#                 deleted = True
        
#         return deleted
    
#     def clear(self) -> None:
#         """Clear all cache entries"""
#         self.cache.clear()
#         self.vector_keys = {}
    
#     def get_stats(self) -> Dict[str, Any]:
#         """
#         Get cache statistics
        
#         Returns:
#             Dict[str, Any]: Cache statistics
#         """
#         stats = self.cache.get_stats()
#         stats["vector_entries"] = len(self.vector_keys)
#         return stats


# class TwoLevelCache:
#     """Two-level cache with memory and disk layers"""
    
#     def __init__(self, memory_size: int = 1000, disk_cache_dir: str = "./cache",
#                disk_size_mb: int = 1000, ttl: Optional[int] = None, now_func: Optional[Callable[[], datetime]] = None):
#         """
#         Initialize two-level cache
        
#         Args:
#             memory_size (int): Max items in memory cache
#             disk_cache_dir (str): Directory for disk cache
#             disk_size_mb (int): Max size of disk cache in MB
#             ttl (int, optional): Default time-to-live in seconds
#         """
#         self.memory_cache = MemoryCache(memory_size, ttl, now_func=now_func)
#         self.disk_cache = DiskCache(disk_cache_dir, disk_size_mb, ttl, now_func=now_func)
#         self.ttl = ttl
    
#     def get(self, key: str, default: Any = None) -> Any:
#         """
#         Get a value from the cache
        
#         Args:
#             key (str): Cache key
#             default (Any): Default value if key not found
            
#         Returns:
#             Any: Cached value or default
#         """
#         # Try memory cache first
#         value = self.memory_cache.get(key)
#         if value is not None:
#             return value
        
#         # Try disk cache
#         value = self.disk_cache.get(key)
#         if value is not None:
#             # Promote to memory cache
#             self.memory_cache.set(key, value)
#             return value
        
#         return default
    
#     def set(self, key: str, value: Any, ttl: Optional[int] = None,
#            metadata: Optional[Dict[str, Any]] = None, memory_only: bool = False) -> None:
#         """
#         Set a value in the cache
        
#         Args:
#             key (str): Cache key
#             value (Any): Value to cache
#             ttl (int, optional): Time-to-live in seconds
#             metadata (Dict, optional): Additional metadata
#             memory_only (bool): Whether to store in memory only
#         """
#         # Always set in memory cache
#         self.memory_cache.set(key, value, ttl, metadata)
        
#         # Set in disk cache if not memory only
#         if not memory_only:
#             self.disk_cache.set(key, value, ttl, metadata)
    
#     def delete(self, key: str) -> bool:
#         """
#         Delete an entry from the cache
        
#         Args:
#             key (str): Cache key
            
#         Returns:
#             bool: True if deleted, False if not found
#         """
#         mem_deleted = self.memory_cache.delete(key)
#         disk_deleted = self.disk_cache.delete(key)
#         return mem_deleted or disk_deleted
    
#     def clear(self) -> None:
#         """Clear all cache entries"""
#         self.memory_cache.clear()
#         self.disk_cache.clear()
    
#     def get_stats(self, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
#         """
#         Get cache statistics
        
#         Returns:
#             Dict[str, Any]: Cache statistics
#         """
#         mem_stats = self.memory_cache.get_stats(options)
#         disk_stats = self.disk_cache.get_stats(options)
#         # Combine stats
#         return {
#             "memory": mem_stats,
#             "disk": disk_stats,
#             "combined_hits": mem_stats.get("hits", 0) + disk_stats.get("hits", 0),
#             "combined_misses": mem_stats.get("misses", 0) + disk_stats.get("misses", 0)
#         }
    
#     def cleanup(self, options: Optional[Dict[str, Any]] = None) -> Dict[str, int]:
#         """
#         Remove expired entries from both caches
        
#         Returns:
#             Dict[str, int]: Cleanup results
#         """
#         memory_removed = self.memory_cache.cleanup(options)
#         disk_removed = self.disk_cache.cleanup(options)
#         return {
#             "memory_removed": memory_removed,
#             "disk_removed": disk_removed,
#             "total_removed": memory_removed + disk_removed
#         }


# class CacheManager:
#     """Main cache manager for the application"""
    
#     def __init__(self, config: Optional[Dict[str, Any]] = None, now_func: Optional[Callable[[], datetime]] = None):
#         """
#         Initialize cache manager
        
#         Args:
#             config (Dict, optional): Cache configuration
#         """
#         self.config = {
#             "memory_size": 1000,
#             "disk_cache_dir": "./cache",
#             "disk_size_mb": 1000,
#             "default_ttl": 86400,  # 1 day
#             "enable_embedding_cache": True,
#             "similarity_threshold": 0.9
#         }
        
#         if config:
#             self.config.update(config)
        
#         # Create caches
#         self._now = now_func if now_func is not None else datetime.now
#         self.general_cache = TwoLevelCache(
#             self.config["memory_size"],
#             self.config["disk_cache_dir"],
#             self.config["disk_size_mb"],
#             self.config["default_ttl"],
#             now_func=self._now
#         )
#         # Create embedding cache if enabled
#         if self.config["enable_embedding_cache"]:
#             self.embedding_cache = EmbeddingCache(
#                 self.general_cache.disk_cache,
#                 self.config["similarity_threshold"]
#             )
#         else:
#             self.embedding_cache = None
    
#     def get(self, key: str, default: Any = None) -> Any:
#         """
#         Get a value from the general cache
        
#         Args:
#             key (str): Cache key
#             default (Any): Default value if key not found
            
#         Returns:
#             Any: Cached value or default
#         """
#         return self.general_cache.get(key, default)
    
#     def set(self, key: str, value: Any, ttl: Optional[int] = None,
#            metadata: Optional[Dict[str, Any]] = None, memory_only: bool = False) -> None:
#         """
#         Set a value in the general cache
        
#         Args:
#             key (str): Cache key
#             value (Any): Value to cache
#             ttl (int, optional): Time-to-live in seconds
#             metadata (Dict, optional): Additional metadata
#             memory_only (bool): Whether to store in memory only
#         """
#         self.general_cache.set(key, value, ttl, metadata, memory_only)
    
#     def get_embedding(self, vector: np.ndarray, text: Optional[str] = None,
#                     default: Any = None, exact_match: bool = False) -> Tuple[Any, float]:
#         """
#         Get a value from the embedding cache
        
#         Args:
#             vector (np.ndarray): Query vector
#             text (str, optional): Text representation for exact matching
#             default (Any): Default value if no match found
#             exact_match (bool): Whether to require exact match
            
#         Returns:
#             Tuple[Any, float]: (value, similarity) tuple
#         """
#         if self.embedding_cache:
#             return self.embedding_cache.get(vector, text, default, exact_match)
#         return default, 0.0
    
#     def set_embedding(self, vector: np.ndarray, value: Any, text: Optional[str] = None,
#                     ttl: Optional[int] = None) -> None:
#         """
#         Set a value in the embedding cache
        
#         Args:
#             vector (np.ndarray): Vector to cache
#             value (Any): Value to cache
#             text (str, optional): Text representation for exact matching
#             ttl (int, optional): Time-to-live in seconds
#         """
#         if self.embedding_cache:
#             self.embedding_cache.set(vector, value, text, ttl)
    
#     def delete(self, key: str) -> bool:
#         """
#         Delete an entry from the general cache
        
#         Args:
#             key (str): Cache key
            
#         Returns:
#             bool: True if deleted, False if not found
#         """
#         return self.general_cache.delete(key)
    
#     def delete_embedding(self, vector: np.ndarray = None, text: str = None) -> bool:
#         """
#         Delete entries from the embedding cache
        
#         Args:
#             vector (np.ndarray, optional): Vector to delete
#             text (str, optional): Text representation to delete
            
#         Returns:
#             bool: True if any entries deleted, False otherwise
#         """
#         if self.embedding_cache:
#             return self.embedding_cache.delete(vector, text)
#         return False
    
#     def clear(self) -> None:
#         """Clear all cache entries"""
#         self.general_cache.clear()
#         if self.embedding_cache:
#             self.embedding_cache.clear()
    
#     def get_stats(self, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
#         """
#         Get cache statistics
        
#         Returns:
#             Dict[str, Any]: Cache statistics
#         """
#         stats = {
#             "general": self.general_cache.get_stats(options)
#         }
#         if self.embedding_cache:
#             stats["embedding"] = self.embedding_cache.get_stats()
#         return stats
    
#     def cleanup(self, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
#         """
#         Remove expired entries from all caches
        
#         Returns:
#             Dict[str, Any]: Cleanup results
#         """
#         return self.general_cache.cleanup(options)


# # Example usage
# if __name__ == "__main__":
#     # Create cache manager
#     cache_manager = CacheManager({
#         "memory_size": 500,
#         "disk_cache_dir": "./cache",
#         "disk_size_mb": 500,
#         "default_ttl": 3600  # 1 hour
#     })
    
#     # Set a value in the general cache
#     cache_manager.set("example_key", {"data": "example value"})
    
#     # Get a value from the general cache
#     value = cache_manager.get("example_key")
#     print(f"Retrieved value: {value}")
    
#     # Set an embedding
#     if cache_manager.embedding_cache:
#         vector = np.random.rand(384)  # Example embedding vector
#         cache_manager.set_embedding(vector, {"text": "example embedding"}, "example text")
        
#         # Get an embedding
#         result, similarity = cache_manager.get_embedding(vector)
#         print(f"Retrieved embedding with similarity {similarity}: {result}")
    
#     # Get cache stats
#     stats = cache_manager.get_stats()
#     print(f"Cache stats: {stats}")