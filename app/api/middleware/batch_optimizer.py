"""
Batch optimizer middleware for grouping similar small requests.
This module provides a smart batching system that automatically groups
similar requests together for more efficient processing.
"""

import os
import time
import uuid
import json
import hashlib
import logging
import asyncio
from typing import Dict, List, Any, Optional, Callable, Union, Set, Awaitable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from app.utils.logging import get_logger

# Initialize logger
logger = get_logger(__name__)

@dataclass
class BatchItem:
    """Represents a single item in a batch."""
    key: str
    item_data: Any
    created_at: float = field(default_factory=time.time)
    completion_event: Optional[asyncio.Event] = None
    result: Any = None
    processed: bool = False
    error: Optional[Exception] = None

@dataclass
class BatchGroup:
    """Group of similar items to be processed together."""
    group_key: str
    items: Dict[str, BatchItem] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    processing: bool = False
    max_size: int = 10
    
    def add_item(self, item: BatchItem) -> bool:
        """
        Add an item to this batch group if there's room.
        
        Args:
            item: The batch item to add
            
        Returns:
            bool: True if item was added, False if the batch is full
        """
        if self.processing or len(self.items) >= self.max_size:
            return False
        
        self.items[item.key] = item
        return True
    
    def is_ready(self, max_wait_time: float) -> bool:
        """
        Check if this batch is ready for processing.
        
        Args:
            max_wait_time: Maximum time in seconds to wait before processing
            
        Returns:
            bool: True if the batch should be processed now
        """
        # Process immediately if full
        if len(self.items) >= self.max_size:
            return True
        
        # Process if oldest item has waited long enough
        oldest_item_time = min(item.created_at for item in self.items.values())
        return time.time() - oldest_item_time >= max_wait_time
    
    def mark_as_processing(self) -> None:
        """Mark this batch as currently being processed."""
        self.processing = True
    
    def get_items_list(self) -> List[BatchItem]:
        """Get all items as a list."""
        return list(self.items.values())
    
    def get_items_data(self) -> List[Any]:
        """Extract the data from all items as a list."""
        return [item.item_data for item in self.items.values()]
    
    def complete_with_results(self, results: Dict[str, Any]) -> None:
        """
        Complete all items with their respective results and trigger their completion events.
        
        Args:
            results: Dictionary mapping item keys to their results
        """
        for key, result in results.items():
            if key in self.items:
                item = self.items[key]
                item.result = result
                item.processed = True
                
                # Trigger completion event if available
                if item.completion_event:
                    item.completion_event.set()
    
    def complete_with_error(self, error: Exception) -> None:
        """
        Complete all items with an error and trigger their completion events.
        
        Args:
            error: The exception that occurred during processing
        """
        for key, item in self.items.items():
            item.error = error
            item.processed = True
            
            # Trigger completion event if available
            if item.completion_event:
                item.completion_event.set()

class BatchOptimizer:
    """
    Smart batching system for automatically grouping similar small requests.
    
    This class provides a way to transparently batch similar requests together
    to improve processing efficiency. It handles grouping, timing control,
    and result mapping.
    """
    
    def __init__(
        self,
        name: str,
        max_batch_size: int = 10,
        max_wait_time: float = 0.2,
        process_function: Optional[Callable[[List[BatchItem]], Dict[str, Any]]] = None,
        auto_start: bool = True
    ):
        """
        Initialize batch optimizer.
        
        Args:
            name: Unique name for this batch optimizer instance
            max_batch_size: Maximum number of items in a batch
            max_wait_time: Maximum time in seconds to wait for a batch to fill
            process_function: Function to process a batch of items
            auto_start: Whether to start the background task automatically
        """
        self.name = name
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.process_function = process_function
        
        # State
        self.batch_groups: Dict[str, BatchGroup] = {}
        self.processing_batches: Set[str] = set()
        self.batch_task = None
        self.running = True
        
        # Async controls
        self.lock = asyncio.Lock()
        self.batch_event = asyncio.Event()
        
        # Start background task if auto_start is True
        if auto_start:
            try:
                self.start_background_task()
            except RuntimeError:
                # No running event loop, this is likely a test environment
                logger.warning(f"No running event loop, batch optimizer '{name}' won't start background task")
                self.running = False
        
        logger.info(f"Batch optimizer '{name}' initialized with max_batch_size={max_batch_size}, "
                   f"max_wait_time={max_wait_time}s")
    
    @classmethod
    def create_for_testing(cls, name: str, process_function: Optional[Callable] = None):
        """
        Create a batch optimizer instance for testing without starting the background task.
        
        Args:
            name: Name for the test optimizer
            process_function: Optional function to process batches
            
        Returns:
            BatchOptimizer: An instance ready for testing
        """
        return cls(
            name=name,
            process_function=process_function,
            auto_start=False
        )
    
    def start_background_task(self) -> None:
        """Start the background task that processes batches."""
        self.batch_task = asyncio.create_task(self._batch_processing_loop())
        logger.debug(f"Started batch processing loop for '{self.name}'")
    
    async def cleanup(self) -> None:
        """Clean up resources used by this batch optimizer."""
        logger.info(f"Cleaning up batch optimizer '{self.name}'")
        
        # Mark as not running to stop the background task
        self.running = False
        
        # Cancel the background task if it exists
        if self.batch_task:
            self.batch_task.cancel()
            
            try:
                await self.batch_task
            except asyncio.CancelledError:
                pass
        
        # Complete any remaining items with an error
        async with self.lock:
            for group_key, batch in list(self.batch_groups.items()):
                batch.complete_with_error(Exception("Batch optimizer shutting down"))
            
            # Clear all batches
            self.batch_groups.clear()
            self.processing_batches.clear()
        
        logger.info(f"Batch optimizer '{self.name}' cleaned up")
    
    async def _batch_processing_loop(self) -> None:
        """Background task that processes batches when they're ready."""
        try:
            while self.running:
                # Wait for the batch event to be set
                await self.batch_event.wait()
                
                # Clear the event immediately
                self.batch_event.clear()
                
                # Find ready batches
                ready_batches = []
                
                async with self.lock:
                    # Check all batch groups
                    for group_key, batch in list(self.batch_groups.items()):
                        if group_key not in self.processing_batches and batch.is_ready(self.max_wait_time):
                            # Mark as processing
                            batch.mark_as_processing()
                            self.processing_batches.add(group_key)
                            ready_batches.append(batch)
                
                # Process all ready batches concurrently
                if ready_batches:
                    await asyncio.gather(*[self._process_batch(batch) for batch in ready_batches])
                
                # Small sleep to avoid tight loops
                await asyncio.sleep(0.01)
        
        except asyncio.CancelledError:
            logger.info(f"Batch processing loop for '{self.name}' cancelled")
            raise
        
        except Exception as e:
            logger.error(f"Error in batch processing loop for '{self.name}': {str(e)}", exc_info=True)
    
    async def _process_batch(self, batch: BatchGroup) -> None:
        """
        Process a single batch.
        
        Args:
            batch: The batch group to process
        """
        batch_size = len(batch.items)
        group_key = batch.group_key
        
        try:
            logger.debug(f"Processing batch '{group_key}' with {batch_size} items")
            
            # Get all items as a list
            items_list = batch.get_items_list()
            
            # Process the batch using the process function
            if self.process_function:
                results = await self.process_function(items_list)
                
                # Complete all items with their results
                batch.complete_with_results(results)
            else:
                # No process function, just mark as processed
                for item in items_list:
                    item.processed = True
                    if item.completion_event:
                        item.completion_event.set()
            
            logger.debug(f"Completed batch '{group_key}' with {batch_size} items")
        
        except Exception as e:
            logger.error(f"Error processing batch '{group_key}': {str(e)}", exc_info=True)
            # Complete with error
            batch.complete_with_error(e)
        
        finally:
            # Remove from processing list and batch groups
            async with self.lock:
                if group_key in self.processing_batches:
                    self.processing_batches.remove(group_key)
                
                if group_key in self.batch_groups:
                    del self.batch_groups[group_key]
    
    def _calculate_group_key(self, item_data: Any) -> str:
        """
        Calculate a group key for an item based on its data.
        
        Args:
            item_data: The data to group by
            
        Returns:
            str: A string key representing the group this item belongs to
        """
        try:
            # Handle dictionaries
            if isinstance(item_data, dict):
                # For translation requests, group by language pair
                if all(key in item_data for key in ["source_language", "target_language"]):
                    return f"translation_{item_data['source_language']}_{item_data['target_language']}"
                
                # For analysis requests, group by analysis types
                if "analysis_types" in item_data:
                    analysis_types = sorted(item_data.get("analysis_types", []))
                    return f"analysis_{','.join(analysis_types)}"
            
            # Fall back to a hash of the item type
            item_type = type(item_data).__name__
            return f"type_{item_type}_{str(uuid.uuid4())[:8]}"
        
        except Exception as e:
            # If anything goes wrong, use a random group key
            logger.warning(f"Error calculating group key: {str(e)}")
            return f"random_{str(uuid.uuid4())}"
    
    async def process(self, item_data: Any, group_key: Optional[str] = None, timeout_seconds: float = 5.0) -> Any:
        """
        Process a single item, potentially as part of a batch.
        
        Args:
            item_data: The item data to process
            group_key: Optional group key for manual grouping
            timeout_seconds: Maximum time to wait for processing to complete
            
        Returns:
            Any: The processing result
            
        Raises:
            asyncio.TimeoutError: If processing times out
            Exception: If an error occurs during processing
        """
        # Generate item key and calculate group key if not provided
        item_key = str(uuid.uuid4())
        if group_key is None:
            group_key = self._calculate_group_key(item_data)
        
        # Create completion event and batch item
        completion_event = asyncio.Event()
        item = BatchItem(
            key=item_key,
            item_data=item_data,
            created_at=time.time(),
            completion_event=completion_event
        )
        
        # For testing, if we're not running with a background task
        if not self.running:
            # Create a dummy batch with just this item
            batch = BatchGroup(group_key=group_key)
            batch.add_item(item)
            
            # Process the item directly
            items_list = [item]
            
            try:
                if self.process_function is not None:
                    # This is a direct call to the process function which must return a dictionary
                    # with results keyed by item keys
                    results = await self.process_function(items_list)
                    
                    # Set the result and mark as processed
                    if item_key in results:
                        item.result = results[item_key]
                    else:
                        # Fall back to just returning the first result if the key is not found
                        logger.debug(f"Item key {item_key} not found in results, using first result")
                        for result_key, result_value in results.items():
                            item.result = result_value
                            break
                else:
                    # No process function, just return the original data
                    logger.debug(f"No process function for batch optimizer '{self.name}', returning original data")
                    item.result = item_data
                
                # Mark as processed and set completion event  
                item.processed = True
                if item.completion_event:
                    item.completion_event.set()  # Set the completion event
                
                # Return the result directly
                return item.result
                
            except Exception as e:
                # Propagate the error
                logger.error(f"Error in direct processing: {str(e)}", exc_info=True)
                item.error = e
                item.processed = True
                if item.completion_event:
                    item.completion_event.set()  # Set the completion event even on error
                raise e
                
        # Normal batch processing
        added_to_batch = False
        
        async with self.lock:
            # Check if group already exists and has room
            if group_key in self.batch_groups and not self.batch_groups[group_key].processing:
                added_to_batch = self.batch_groups[group_key].add_item(item)
            
            # Create new batch group if needed
            if not added_to_batch:
                batch_group = BatchGroup(
                    group_key=group_key,
                    max_size=self.max_batch_size
                )
                batch_group.add_item(item)
                self.batch_groups[group_key] = batch_group
            
            # Signal that a new item has been added
            self.batch_event.set()
        
        # Wait for processing to complete with timeout
        try:
            await asyncio.wait_for(completion_event.wait(), timeout=timeout_seconds)
        except asyncio.TimeoutError:
            logger.error(f"Timeout waiting for item {item_key} in group {group_key}")
            raise
        
        # Check for errors
        if item.error:
            logger.error(f"Error processing item {item_key}: {str(item.error)}")
            raise item.error
        
        # Return the result
        return item.result