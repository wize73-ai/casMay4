#!/usr/bin/env python
"""
Test script for CasaLingua API optimizations.
This script makes basic API calls to test route caching, batch optimization, 
streaming responses, and error handling.
"""

import os
import sys
import asyncio
import logging
import json
import time
import uuid
from typing import Dict, List, Any, Optional
import httpx
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("optimization_tester")

# API endpoint settings
API_BASE_URL = "http://localhost:8000"
API_TIMEOUT = 30.0  # seconds

async def test_route_caching():
    """Test the route cache functionality."""
    logger.info("Testing route cache...")
    
    async with httpx.AsyncClient(timeout=API_TIMEOUT) as client:
        # First request (should be cache miss)
        start_time = time.time()
        response = await client.post(
            f"{API_BASE_URL}/pipeline/translate",
            json={
                "text": "Hello world, this is a test of the route caching system.",
                "source_language": "en",
                "target_language": "es"
            }
        )
        first_request_time = time.time() - start_time
        
        # Check response
        if response.status_code != 200:
            logger.error(f"Translation request failed: {response.text}")
            return False
        
        logger.info(f"First request (cache miss) took {first_request_time:.4f} seconds")
        
        # Second identical request (should be cache hit)
        start_time = time.time()
        response = await client.post(
            f"{API_BASE_URL}/pipeline/translate",
            json={
                "text": "Hello world, this is a test of the route caching system.",
                "source_language": "en",
                "target_language": "es"
            }
        )
        second_request_time = time.time() - start_time
        
        # Check response
        if response.status_code != 200:
            logger.error(f"Second translation request failed: {response.text}")
            return False
        
        logger.info(f"Second request (cache hit) took {second_request_time:.4f} seconds")
        
        # Check if cache is working
        speedup = first_request_time / max(0.001, second_request_time)
        logger.info(f"Cache speedup factor: {speedup:.2f}x")
        
        if speedup >= 2.0:
            logger.info("✅ Route cache is working effectively")
            return True
        else:
            logger.warning("⚠️ Route cache may not be working as expected")
            return False

async def test_batch_optimization():
    """Test the batch optimization system."""
    logger.info("Testing batch optimization...")
    
    async with httpx.AsyncClient(timeout=API_TIMEOUT) as client:
        # Create multiple similar small requests
        texts = [
            "Hello world",
            "How are you?",
            "This is a test",
            "Batch processing is efficient",
            "Natural language processing"
        ]
        
        # Sequential requests (without batching)
        start_time = time.time()
        sequential_results = []
        
        for text in texts:
            response = await client.post(
                f"{API_BASE_URL}/pipeline/translate",
                json={
                    "text": text,
                    "source_language": "en",
                    "target_language": "es",
                    "smart_batching": False  # Disable batching
                }
            )
            
            if response.status_code != 200:
                logger.error(f"Sequential translation request failed: {response.text}")
                return False
            
            sequential_results.append(response.json())
        
        sequential_time = time.time() - start_time
        logger.info(f"Sequential requests took {sequential_time:.4f} seconds")
        
        # Batch requests (with smart batching)
        start_time = time.time()
        batch_tasks = []
        
        for text in texts:
            batch_tasks.append(
                client.post(
                    f"{API_BASE_URL}/pipeline/translate",
                    json={
                        "text": text,
                        "source_language": "en",
                        "target_language": "es",
                        "smart_batching": True  # Enable batching
                    }
                )
            )
        
        batch_responses = await asyncio.gather(*batch_tasks)
        batch_results = [response.json() for response in batch_responses]
        
        batch_time = time.time() - start_time
        logger.info(f"Batch requests took {batch_time:.4f} seconds")
        
        # Check if batching is working
        speedup = sequential_time / max(0.001, batch_time)
        logger.info(f"Batch speedup factor: {speedup:.2f}x")
        
        if speedup >= 1.5:
            logger.info("✅ Batch optimization is working effectively")
            return True
        else:
            logger.warning("⚠️ Batch optimization may not be working as expected")
            return False

async def test_streaming_response():
    """Test the streaming response functionality."""
    logger.info("Testing streaming response...")
    
    async with httpx.AsyncClient(timeout=API_TIMEOUT) as client:
        # Make a streaming request
        start_time = time.time()
        
        try:
            with client.stream(
                "POST",
                f"{API_BASE_URL}/streaming/translate",
                json={
                    "text": "Hello world. This is a test of the streaming translation system. It should return partial results as they become available. This helps with large documents that would otherwise timeout.",
                    "source_language": "en",
                    "target_language": "es"
                },
                timeout=60.0
            ) as response:
                
                if response.status_code != 200:
                    logger.error(f"Streaming request failed: {response.status_code}")
                    return False
                
                # Process the stream
                event_count = 0
                chunk_count = 0
                
                async for chunk in response.aiter_lines():
                    if not chunk.strip():
                        continue
                    
                    try:
                        event = json.loads(chunk)
                        event_count += 1
                        
                        if event.get("event") == "chunk_translated":
                            chunk_count += 1
                            logger.info(f"Received chunk {chunk_count}/{event.get('total_chunks', '?')}: "
                                        f"{event.get('progress', 0)*100:.0f}% complete")
                        
                        if event.get("event") == "translation_completed":
                            logger.info("Translation complete!")
                    except json.JSONDecodeError:
                        logger.warning(f"Could not parse stream chunk: {chunk}")
            
            total_time = time.time() - start_time
            logger.info(f"Streaming translation took {total_time:.4f} seconds")
            logger.info(f"Received {event_count} events, {chunk_count} translated chunks")
            
            if event_count > 0 and chunk_count > 0:
                logger.info("✅ Streaming response is working")
                return True
            else:
                logger.warning("⚠️ No streaming events received")
                return False
            
        except Exception as e:
            logger.error(f"Error testing streaming response: {str(e)}")
            return False

async def test_error_handling():
    """Test the enhanced error handling system."""
    logger.info("Testing error handling...")
    
    async with httpx.AsyncClient(timeout=API_TIMEOUT) as client:
        # Test with an invalid request that should trigger validation error
        response = await client.post(
            f"{API_BASE_URL}/pipeline/translate",
            json={
                "text": "Hello world",
                "source_language": "invalid_language_code",  # Invalid language code
                "target_language": "es"
            }
        )
        
        # Check for proper error response format
        if response.status_code == 400:
            error_data = response.json()
            
            # Check for our standardized error format
            if all(key in error_data for key in ["status_code", "error_code", "category", "message"]):
                logger.info(f"Received properly formatted error response: {error_data.get('error_code')} - "
                           f"{error_data.get('category')} - {error_data.get('message')}")
                logger.info("✅ Error handling is working properly")
                return True
            else:
                logger.warning(f"Error response doesn't match expected format: {error_data}")
                return False
        else:
            logger.warning(f"Expected 400 status code, got {response.status_code}: {response.text}")
            return False

async def main():
    """Run all optimization tests."""
    logger.info("Starting optimization tests...")
    
    # Run tests
    cache_result = await test_route_caching()
    batch_result = await test_batch_optimization()
    streaming_result = await test_streaming_response()
    error_result = await test_error_handling()
    
    # Calculate overall success
    success_count = sum([
        1 if cache_result else 0,
        1 if batch_result else 0,
        1 if streaming_result else 0,
        1 if error_result else 0
    ])
    
    logger.info("\n======= TEST RESULTS =======")
    logger.info(f"Route Caching: {'✅ PASS' if cache_result else '❌ FAIL'}")
    logger.info(f"Batch Optimization: {'✅ PASS' if batch_result else '❌ FAIL'}")
    logger.info(f"Streaming Response: {'✅ PASS' if streaming_result else '❌ FAIL'}")
    logger.info(f"Error Handling: {'✅ PASS' if error_result else '❌ FAIL'}")
    logger.info(f"Overall: {success_count}/4 tests passed")
    
    return success_count == 4

if __name__ == "__main__":
    # Run the tests with asyncio
    success = asyncio.run(main())
    sys.exit(0 if success else 1)