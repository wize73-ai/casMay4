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
        # Process a single item
        item_data = {"text": "Hello world", "source_language": "en", "target_language": "es"}
        result = await optimizer.process(item_data)
        
        # Verify result
        assert result is not None
        assert result["processed"] == True
        assert result["text"] == "Hello world"
        assert result["batch_size"] == 1  # Should be processed individually
    
    @pytest.mark.asyncio
    async def test_batch_grouping(self, optimizer):
        """Test that similar items are grouped together."""
        # Create tasks for similar items that should be batched
        async def submit_item(text: str):
            item_data = {"text": text, "source_language": "en", "target_language": "es"}
            return await optimizer.process(item_data)
        
        # Submit similar items nearly simultaneously
        tasks = [submit_item(f"Text {i}") for i in range(3)]
        results = await asyncio.gather(*tasks)
        
        # All should be in the same batch
        assert all(result["batch_size"] == 3 for result in results)
        assert all(result["processed"] == True for result in results)
    
    @pytest.mark.asyncio
    async def test_different_batch_groups(self, optimizer):
        """Test that different items go to different batch groups."""
        # Create tasks for different batches
        async def submit_en_es(text: str):
            item_data = {"text": text, "source_language": "en", "target_language": "es"}
            return await optimizer.process(item_data)
        
        async def submit_fr_de(text: str):
            item_data = {"text": text, "source_language": "fr", "target_language": "de"}
            return await optimizer.process(item_data)
        
        # Submit different batches concurrently
        en_es_tasks = [submit_en_es(f"Text EN-ES {i}") for i in range(3)]
        fr_de_tasks = [submit_fr_de(f"Text FR-DE {i}") for i in range(2)]
        
        all_tasks = en_es_tasks + fr_de_tasks
        results = await asyncio.gather(*all_tasks)
        
        # Check en-es batch
        en_es_results = results[:3]
        assert all(result["batch_size"] == 3 for result in en_es_results)
        
        # Check fr-de batch
        fr_de_results = results[3:]
        assert all(result["batch_size"] == 2 for result in fr_de_results)
    
    @pytest.mark.asyncio
    async def test_max_batch_size(self, optimizer):
        """Test that batches respect max_batch_size."""
        # Create more tasks than max_batch_size
        async def submit_item(i: int):
            item_data = {"text": f"Text {i}", "source_language": "en", "target_language": "es"}
            return await optimizer.process(item_data)
        
        # Submit more than max batch size (5) items
        tasks = [submit_item(i) for i in range(8)]
        results = await asyncio.gather(*tasks)
        
        # Count batches of different sizes
        batch_sizes = [result["batch_size"] for result in results]
        
        # Should never exceed max_batch_size
        assert max(batch_sizes) <= 5
        
        # Should have some batches of size 5 (max) and some smaller
        assert 5 in batch_sizes
    
    @pytest.mark.asyncio
    async def test_batch_timing(self, optimizer):
        """Test that batches are processed after max_wait_time."""
        # Submit one item and immediately start timer
        async def submit_delayed_item():
            await asyncio.sleep(0.1)  # Short delay
            item_data = {"text": "Delayed item", "source_language": "en", "target_language": "es"}
            return await optimizer.process(item_data)
        
        # Submit first item
        start_time = time.time()
        item_data = {"text": "First item", "source_language": "en", "target_language": "es"}
        first_result = await optimizer.process(item_data)
        
        # Should wait for max_wait_time (0.5s) before processing
        processing_time = time.time() - start_time
        assert processing_time >= 0.4  # Allow a small margin for timing variations
        
        # Should be processed individually since second item arrives too late
        assert first_result["batch_size"] == 1
    
    @pytest.mark.asyncio
    async def test_concurrent_processing(self, optimizer):
        """Test that multiple batches can be processed concurrently."""
        # Create tasks for different batch groups
        async def submit_item(lang_pair: str, i: int):
            if lang_pair == "en-es":
                item_data = {"text": f"Text {i}", "source_language": "en", "target_language": "es"}
            else:
                item_data = {"text": f"Text {i}", "source_language": "fr", "target_language": "de"}
            return await optimizer.process(item_data)
        
        # Submit many items from different groups concurrently
        tasks = []
        for i in range(10):
            if i % 2 == 0:
                tasks.append(submit_item("en-es", i))
            else:
                tasks.append(submit_item("fr-de", i))
        
        # All should complete without deadlock or errors
        results = await asyncio.gather(*tasks)
        assert len(results) == 10
        assert all(result["processed"] == True for result in results)