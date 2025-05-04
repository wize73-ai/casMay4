#!/usr/bin/env python3
"""
Test script for health checks using a mock FastAPI application
"""

import sys
import os
import asyncio
import json
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
import uvicorn
from contextlib import asynccontextmanager
from datetime import datetime
import time
import logging
from unittest.mock import MagicMock, AsyncMock

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import health check endpoints
from app.api.routes.health import (
    health_check, detailed_health_check, model_health_check,
    database_health_check, readiness_probe, liveness_probe
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create mock objects
class MockDatabaseManager:
    def __init__(self):
        self.connection_status = "healthy"
    
    def execute_query(self, query):
        if self.connection_status == "error":
            raise Exception("Database connection error")
        return [{"result": 1}]

class MockUserManager:
    def __init__(self, manager):
        self.db_manager = manager
    
    def execute_query(self, query):
        return self.db_manager.execute_query(query)

class MockContentManager:
    def __init__(self, manager):
        self.db_manager = manager
    
    def execute_query(self, query):
        return self.db_manager.execute_query(query)

class MockProgressManager:
    def __init__(self, manager):
        self.db_manager = manager
    
    def execute_query(self, query):
        return self.db_manager.execute_query(query)

class MockPersistenceManager:
    def __init__(self):
        self.db_manager = MockDatabaseManager()
        self.user_manager = MockUserManager(self.db_manager)
        self.content_manager = MockContentManager(self.db_manager)
        self.progress_manager = MockProgressManager(self.db_manager)

class MockProcessor:
    def __init__(self):
        self.persistence_manager = MockPersistenceManager()
        self.pipelines = ["translation", "language_detection", "simplification"]
        self.version = "1.0.0"
    
    async def translate_text(self, text, source_lang, target_lang, model_name=None):
        return "Translated text"
    
    async def detect_language(self, text):
        return {"language": "en", "confidence": 0.95}
    
    async def simplify_text(self, text, target_level=None):
        return "Simplified text"

class MockModelManager:
    def __init__(self, models_loaded=True):
        self.models_loaded = models_loaded
    
    async def get_model_info(self):
        if self.models_loaded:
            return {
                "loaded_models": ["translation_model", "language_detection_model", "simplifier_model"],
                "device": "cpu",
                "low_memory_mode": False
            }
        else:
            return {
                "loaded_models": [],
                "device": "cpu",
                "low_memory_mode": False
            }

class MockMetrics:
    def get_system_metrics(self):
        return {
            "request_metrics": {
                "total_requests": 100,
                "successful_requests": 95,
                "failed_requests": 5,
                "avg_response_time": 0.1
            },
            "uptime_seconds": 3600
        }

# Create FastAPI app
app = FastAPI()

# Initialize app state directly
db_manager = MockDatabaseManager()
persistence_manager = MockPersistenceManager()
processor = MockProcessor()
model_manager = MockModelManager()
metrics = MockMetrics()

# Set up MockRequest class to simulate FastAPI request
class MockRequest:
    def __init__(self):
        self.app = MagicMock()
        self.app.state = MagicMock()
        self.app.state.start_time = time.time()
        self.app.state.config = {
            "version": "1.0.0",
            "environment": "test",
            "build_date": "2025-05-04T00:00:00Z",
            "build_id": "test-123",
            "git_commit": "abc123"
        }
        self.app.state.processor = processor
        self.app.state.model_manager = model_manager
        self.app.state.metrics = metrics
        self.app.state.hardware_info = {
            "total_memory": 16 * 1024 * 1024 * 1024,
            "available_memory": 8 * 1024 * 1024 * 1024,
            "has_gpu": False
        }
        self.app.state.tokenizer = MagicMock()
        self.app.state.audit_logger = MagicMock()
        
        # Set up model registry
        self.app.state.model_registry = MagicMock()
        self.app.state.model_registry.get_registry_summary = MagicMock(return_value={
            "model_counts": {"total": 3},
            "supported_languages": ["en", "es", "fr"],
            "supported_tasks": ["translation", "language_detection", "simplification"]
        })
        
        # Add headers
        self.headers = {}

# Add our custom endpoints
@app.get("/health")
async def health_endpoint():
    mock_request = MockRequest()
    return await health_check(mock_request)

@app.get("/health/detailed")
async def detailed_health_endpoint():
    mock_request = MockRequest()
    return await detailed_health_check(mock_request)

@app.get("/health/models") 
async def models_health_endpoint():
    mock_request = MockRequest()
    return await model_health_check(mock_request)

@app.get("/health/database")
async def database_health_endpoint():
    mock_request = MockRequest()
    return await database_health_check(mock_request)

@app.get("/readiness")
async def readiness_endpoint(response = None):
    mock_request = MockRequest()
    return await readiness_probe(mock_request, response)

@app.get("/liveness")
async def liveness_endpoint(response = None):
    mock_request = MockRequest()
    return await liveness_probe(mock_request, response)

# Create test client
test_client = TestClient(app)

def run_tests():
    """Run tests for all health check endpoints"""
    # Test basic health check
    logger.info("Testing basic health check...")
    response = test_client.get("/health")
    logger.info(f"Status code: {response.status_code}")
    if response.status_code == 200:
        logger.info(f"Response: {json.dumps(response.json(), indent=2)}")
    else:
        logger.error(f"Error: {response.text}")
    
    # Test detailed health check
    logger.info("\nTesting detailed health check...")
    response = test_client.get("/health/detailed")
    logger.info(f"Status code: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        logger.info(f"Overall status: {result['status']}")
        logger.info(f"Component count: {len(result['components'])}")
        for component in result['components']:
            logger.info(f"  Component: {component['name']}, Status: {component['status']}")
    else:
        logger.error(f"Error: {response.text}")
    
    # Test model health check
    logger.info("\nTesting model health check...")
    response = test_client.get("/health/models")
    logger.info(f"Status code: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        logger.info(f"Status: {result['status']}")
        logger.info(f"Loaded models: {len(result['loaded_models'])}")
    else:
        logger.error(f"Error: {response.text}")
    
    # Test database health check
    logger.info("\nTesting database health check...")
    response = test_client.get("/health/database")
    logger.info(f"Status code: {response.status_code}")
    if response.status_code == 200:
        logger.info(f"Response: {json.dumps(response.json(), indent=2)}")
    else:
        logger.error(f"Error: {response.text}")
    
    # Test readiness probe
    logger.info("\nTesting readiness probe...")
    response = test_client.get("/readiness")
    logger.info(f"Status code: {response.status_code}")
    if response.status_code == 200:
        logger.info(f"Response: {json.dumps(response.json(), indent=2)}")
    else:
        logger.error(f"Error: {response.text}")
    
    # Test liveness probe
    logger.info("\nTesting liveness probe...")
    response = test_client.get("/liveness")
    logger.info(f"Status code: {response.status_code}")
    if response.status_code == 200:
        logger.info(f"Response: {json.dumps(response.json(), indent=2)}")
    else:
        logger.error(f"Error: {response.text}")

if __name__ == "__main__":
    run_tests()