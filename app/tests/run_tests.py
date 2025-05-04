#!/usr/bin/env python
"""
Test runner for CasaLingua API optimizations.
This script runs tests for the route cache and batch optimizer implementations.
"""

import os
import sys
import logging
import asyncio
import pytest
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_runner")

async def run_tests():
    """Run all optimization tests."""
    logger.info("Starting optimization tests")
    start_time = time.time()
    
    # Route cache tests
    logger.info("Running route cache tests...")
    cache_result = pytest.main(["-xvs", "test_route_cache.py"])
    
    # Batch optimizer tests
    logger.info("Running batch optimizer tests...")
    batch_result = pytest.main(["-xvs", "test_batch_optimizer.py"])
    
    # Check results
    if cache_result == 0 and batch_result == 0:
        logger.info("All tests passed!")
    else:
        logger.error("Some tests failed:")
        if cache_result != 0:
            logger.error("- Route cache tests failed with exit code %d", cache_result)
        if batch_result != 0:
            logger.error("- Batch optimizer tests failed with exit code %d", batch_result)
    
    # Report timing
    elapsed = time.time() - start_time
    logger.info(f"Testing completed in {elapsed:.2f} seconds")

if __name__ == "__main__":
    # Change to the tests directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Run the tests with asyncio
    asyncio.run(run_tests())