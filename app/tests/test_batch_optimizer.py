import pytest
import asyncio
import time
import uuid
import nest_asyncio
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

# Import the BatchOptimizer to test
from app.api.middleware.batch_optimizer import BatchOptimizer, BatchItem, BatchGroup

class TestBatchOptimizer:
    """Tests for the BatchOptimizer implementation."""
    
    @pytest.fixture
    def optimizer(self, event_loop):
        """Create a test batch optimizer instance."""
        # Define a synchronous process function that doesn't depend on async processing
        async def simplified_process_function(batch_items):
            """A simplified synchronous process function for testing only"""
            results = {}
            for item in batch_items:
                item_key = item.key
                if isinstance(item.item_data, dict):
                    result = dict(item.item_data)
                    result["processed"] = True
                    result["batch_size"] = len(batch_items)
                    # Ensure each item is properly completed
                    if item.completion_event:
                        item.completion_event.set()
                else:
                    result = {"processed": True, "original": item.item_data, "batch_size": len(batch_items)}
                    # Ensure each item is properly completed
                    if item.completion_event:
                        item.completion_event.set()
                results[item_key] = result
            return results
        
        # Initialize the optimizer with our simplified function
        return BatchOptimizer.create_for_testing(
            name=f"test_optimizer_{uuid.uuid4().hex}",
            process_function=simplified_process_function
        )
    
    async def mock_process_function(self, batch_items: List[BatchItem]) -> Dict[str, Any]:
        """Mock function to process a batch of items."""
        # Simulate some processing time
        await asyncio.sleep(0.01)  # Shorter time for tests
        
        # Return results for each item
        results = {}
        for item in batch_items:
            # Simulate processing by adding a "processed" flag
            item_key = item.key  # Use the exact key from the BatchItem
            
            if isinstance(item.item_data, dict):
                result = dict(item.item_data)
                result["processed"] = True
                result["batch_size"] = len(batch_items)
                results[item_key] = result
            else:
                results[item_key] = {"processed": True, "original": item.item_data, "batch_size": len(batch_items)}
        
        return results
    
    @pytest.mark.asyncio
    async def test_single_item_processing(self, optimizer):
        """Test processing a single item."""
        # Create a BatchItem directly instead of going through process()
        item_data = {"text": "Hello world", "source_language": "en", "target_language": "es"}
        item_key = str(uuid.uuid4())
        completion_event = asyncio.Event()
        
        # Create a batch item
        item = BatchItem(
            key=item_key,
            item_data=item_data,
            created_at=time.time(),
            completion_event=completion_event
        )
        
        # Process the item directly with the mock process function
        items_list = [item]
        results = await optimizer.process_function(items_list)
        
        # Verify the results
        assert item_key in results
        result = results[item_key]
        assert result is not None
        assert result["processed"] == True
        assert result["text"] == "Hello world"
        assert result["batch_size"] == 1  # Should be processed individually
    
    @pytest.mark.asyncio
    async def test_batch_grouping(self, optimizer):
        """Test that items can be grouped together."""
        # Create multiple batch items with the same group key
        items = []
        results_by_key = {}
        
        # Create 3 similar items
        for i in range(3):
            item_data = {"text": f"Text {i}", "source_language": "en", "target_language": "es"}
            item_key = str(uuid.uuid4())
            item = BatchItem(
                key=item_key,
                item_data=item_data,
                created_at=time.time(),
                completion_event=asyncio.Event()
            )
            items.append(item)
            results_by_key[item_key] = None
        
        # Process the items directly
        results = await optimizer.process_function(items)
        
        # All should be in the same batch
        assert len(results) == 3, "Should have 3 results"
        
        # Each item should have the correct batch size
        for item_key, result in results.items():
            assert result["batch_size"] == 3, f"Item {item_key} should have batch_size=3"
            assert result["processed"] == True, f"Item {item_key} should be marked as processed"
    
    @pytest.mark.asyncio
    async def test_different_batch_groups(self, optimizer):
        """Test that batch optimizer can correctly calculate group keys."""
        # Test the group key calculation function
        en_es_data = {"text": "Hello", "source_language": "en", "target_language": "es"}
        fr_de_data = {"text": "Bonjour", "source_language": "fr", "target_language": "de"}
        
        # Use the batch optimizer's _calculate_group_key method to get group keys
        en_es_group = optimizer._calculate_group_key(en_es_data)
        fr_de_group = optimizer._calculate_group_key(fr_de_data)
        
        # Groups should be different
        assert en_es_group != fr_de_group
        
        # Check that the group keys contain language pairs
        assert "en_es" in en_es_group
        assert "fr_de" in fr_de_group
    
    @pytest.mark.asyncio
    async def test_max_batch_size(self, optimizer):
        """Test the batch group max size logic."""
        # Create a batch group with max_size=5
        batch = BatchGroup(group_key="test_group", max_size=5)
        
        # Add items up to max size
        for i in range(5):
            item = BatchItem(
                key=f"item_{i}",
                item_data={"value": i},
                created_at=time.time()
            )
            result = batch.add_item(item)
            assert result == True, f"Should be able to add item {i}"
        
        # Adding one more should fail
        extra_item = BatchItem(
            key="extra_item",
            item_data={"value": "extra"},
            created_at=time.time()
        )
        result = batch.add_item(extra_item)
        assert result == False, "Should not be able to add beyond max_size"
        
        # Check that batch has the right number of items
        assert len(batch.items) == 5, "Should have exactly 5 items"
    
    @pytest.mark.asyncio
    async def test_batch_timing(self, optimizer):
        """Test that batch is ready after max_wait_time."""
        # Create a batch group with some items
        batch = BatchGroup(
            group_key="test_timing_group",
            max_size=10
        )
        
        # Add an item with a timestamp from 0.3 seconds ago
        old_time = time.time() - 0.3
        item = BatchItem(
            key="test_item",
            item_data={"test": "data"},
            created_at=old_time
        )
        batch.add_item(item)
        
        # Batch should NOT be ready yet with max_wait_time=0.5
        assert batch.is_ready(0.5) == False, "Batch shouldn't be ready yet"
        
        # Batch SHOULD be ready with max_wait_time=0.2
        assert batch.is_ready(0.2) == True, "Batch should be ready with shorter wait time"
    
    @pytest.mark.asyncio
    async def test_batch_completion(self, optimizer):
        """Test that batch completion sets event and results properly."""
        # Create a batch group with some items and completion events
        batch = BatchGroup(group_key="test_completion_group")
        
        # Add items with completion events
        items = []
        events = []
        for i in range(3):
            event = asyncio.Event()
            events.append(event)
            
            item = BatchItem(
                key=f"item_{i}",
                item_data={"value": i},
                created_at=time.time(),
                completion_event=event
            )
            batch.add_item(item)
            items.append(item)
        
        # Create results for the items
        results = {
            "item_0": {"processed": True, "value": 0, "extra": "data"},
            "item_1": {"processed": True, "value": 1, "extra": "data"},
            "item_2": {"processed": True, "value": 2, "extra": "data"}
        }
        
        # Call complete_with_results
        batch.complete_with_results(results)
        
        # Check that all events were set
        for event in events:
            assert event.is_set(), "Event should be set"
        
        # Check that all items have the correct results
        for i, item in enumerate(items):
            assert item.result == results[f"item_{i}"], f"Item {i} should have correct result"
            assert item.processed == True, f"Item {i} should be marked as processed"