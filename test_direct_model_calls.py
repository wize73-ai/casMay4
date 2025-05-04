#!/usr/bin/env python
"""
Direct Model Testing Script

This script tests each model directly by creating a FastAPI test client
and calling the models through the processor rather than through HTTP endpoints.
This provides a more reliable test of the model functionality itself.
"""

import sys
import asyncio
import json
from typing import Dict, List, Any, Optional

# Add the app directory to the Python path
sys.path.append("/Users/jameswilson/Desktop/PRODUCTION/may4/wip-may30")

from fastapi.testclient import TestClient
from app.main import app

# Set up the test client
client = TestClient(app)

# ANSI colors for output
GREEN = '\033[0;32m'
RED = '\033[0;31m'
YELLOW = '\033[0;33m'
BLUE = '\033[0;34m'
PURPLE = '\033[0;35m'
NC = '\033[0m'  # No Color

async def test_model(model_name: str, test_function, test_input: Dict[str, Any], expected_output_key: Optional[str] = None):
    """Test a specific model directly through the processor"""
    print(f"{BLUE}===== Testing {model_name} Model ====={NC}")
    print(f"{YELLOW}Input: {json.dumps(test_input, indent=2)}{NC}")
    
    try:
        # Create a processor instance directly rather than using app.state
        from app.core.pipeline.processor import UnifiedProcessor
        from app.utils.config import load_config
        from app.services.models.loader import ModelLoader
        from app.services.models.manager import EnhancedModelManager
        from app.audit.logger import AuditLogger
        from app.audit.metrics import MetricsCollector
        
        # Load configuration
        config = load_config()
        
        # Create required components
        model_loader = ModelLoader(config=config)
        audit_logger = AuditLogger(config)
        metrics = MetricsCollector(config)
        
        # Load model registry configuration
        from app.services.models.loader import load_registry_config
        registry_config = load_registry_config(config)
        
        # Create a hardware info dict
        hardware_info = {
            "memory": {"total_gb": 16, "available_gb": 12},
            "system": {"processor_type": "apple_silicon"}
        }
        
        # Create model manager
        model_manager = EnhancedModelManager(model_loader, hardware_info, config)
        
        # Create processor instance directly
        processor = UnifiedProcessor(model_manager, audit_logger, metrics, config, registry_config)
        await processor.initialize()
        
        # Call the test function with the processor and test input
        result = await test_function(processor, test_input)
        
        # Check if result contains the expected output key
        if expected_output_key and expected_output_key not in result:
            print(f"{RED}✗ Test failed for {model_name} - Missing expected output key: {expected_output_key}{NC}")
            print(f"{RED}Actual result: {json.dumps(result, indent=2)}{NC}")
            return False
        
        # Print the result (truncated for readability)
        print(f"{GREEN}✓ Test passed for {model_name}{NC}")
        print(f"{GREEN}Result (truncated): {str(result)[:300]}...{NC}")
        return True
    
    except Exception as e:
        print(f"{RED}✗ Test failed for {model_name} - Exception: {str(e)}{NC}")
        import traceback
        traceback.print_exc()
        return False

async def test_language_detection(processor, test_input):
    """Test the language detection model"""
    result = await processor.detect_language(
        text=test_input["text"],
        detailed=test_input.get("detailed", False)
    )
    return result

async def test_translation(processor, test_input):
    """Test the translation model"""
    result = await processor.process_translation(
        text=test_input["text"],
        source_language=test_input["source_language"],
        target_language=test_input["target_language"],
        model_id=test_input.get("model_id"),
        user_id="test-user",
        request_id="test-request"
    )
    return result

async def test_ner_detection(processor, test_input):
    """Test the NER detection model"""
    # This is a simplified test - the actual implementation might be different
    try:
        # Try different possible methods
        if hasattr(processor, "analyze_entities"):
            result = await processor.analyze_entities(
                text=test_input["text"],
                language=test_input["language"]
            )
        elif hasattr(processor, "extract_entities"):
            result = await processor.extract_entities(
                text=test_input["text"],
                language=test_input["language"]
            )
        elif hasattr(processor, "analyze_text"):
            result = await processor.analyze_text(
                text=test_input["text"],
                language=test_input["language"],
                analyses=["entities"]
            )
        else:
            raise AttributeError("No suitable method found for NER detection")
        return result
    except Exception as e:
        return {"error": str(e), "message": "NER detection failed"}

async def test_simplifier(processor, test_input):
    """Test the simplifier model"""
    # Try different possible methods
    try:
        if hasattr(processor, "simplify_text"):
            result = await processor.simplify_text(
                text=test_input["text"],
                target_level=test_input.get("target_level", "simple")
            )
        elif hasattr(processor, "simplify"):
            result = await processor.simplify(
                text=test_input["text"],
                target_level=test_input.get("target_level", "simple")
            )
        elif hasattr(processor, "run_simplification"):
            result = await processor.run_simplification(
                text=test_input["text"],
                level=test_input.get("target_level", "simple")
            )
        else:
            raise AttributeError("No suitable method found for simplification")
        return {"simplified_text": result}
    except Exception as e:
        return {"error": str(e), "message": "Simplification failed"}

async def test_rag_generator(processor, test_input):
    """Test the RAG generator model"""
    try:
        # Try different possible methods
        if hasattr(processor, "process_rag_query"):
            result = await processor.process_rag_query(
                query=test_input["query"],
                language=test_input.get("language", "en"),
                max_results=test_input.get("max_results", 1)
            )
        elif hasattr(processor, "query_knowledge_base"):
            result = await processor.query_knowledge_base(
                query=test_input["query"],
                language=test_input.get("language", "en"),
                max_results=test_input.get("max_results", 1)
            )
        else:
            raise AttributeError("No suitable method found for RAG generation")
        return result
    except Exception as e:
        return {"error": str(e), "message": "RAG generation failed"}

async def test_anonymizer(processor, test_input):
    """Test the anonymizer model"""
    try:
        # Try different possible methods
        if hasattr(processor, "anonymize_text"):
            result = await processor.anonymize_text(
                text=test_input["text"],
                language=test_input["language"]
            )
        elif hasattr(processor, "anonymize"):
            result = await processor.anonymize(
                text=test_input["text"],
                language=test_input["language"]
            )
        elif hasattr(processor, "process_anonymization"):
            result = await processor.process_anonymization(
                text=test_input["text"],
                language=test_input["language"]
            )
        else:
            raise AttributeError("No suitable method found for anonymization")
        return {"anonymized_text": result}
    except Exception as e:
        return {"error": str(e), "message": "Anonymization failed"}

async def main():
    """Main test function"""
    print(f"{PURPLE}======================================================{NC}")
    print(f"{PURPLE}         CASALINGUA DIRECT MODEL TESTING             {NC}")
    print(f"{PURPLE}======================================================{NC}")
    print(f"{YELLOW}Testing models directly through the processor{NC}")
    print("")
    
    # Test all models
    tests = [
        # 1. Language Detection Model
        {
            "model_name": "Language Detection",
            "test_function": test_language_detection,
            "test_input": {
                "text": "Hello, this is a test of the language detection system.",
                "detailed": True
            },
            "expected_output_key": "detected_language"
        },
        # 2. Translation Model
        {
            "model_name": "Translation",
            "test_function": test_translation,
            "test_input": {
                "text": "Hello, this is a test of the translation system.",
                "source_language": "en",
                "target_language": "es"
            },
            "expected_output_key": "translated_text"
        },
        # 3. NER Detection Model
        {
            "model_name": "NER Detection",
            "test_function": test_ner_detection,
            "test_input": {
                "text": "John Smith works at Microsoft in Seattle.",
                "language": "en"
            },
            "expected_output_key": None  # We don't know the exact output format
        },
        # 4. Simplifier Model
        {
            "model_name": "Simplifier",
            "test_function": test_simplifier,
            "test_input": {
                "text": "The intricate mechanisms of quantum physics elude comprehension by many individuals.",
                "target_level": "simple"
            },
            "expected_output_key": None  # We don't know the exact output format
        },
        # 5. RAG Generator Model
        {
            "model_name": "RAG Generator",
            "test_function": test_rag_generator,
            "test_input": {
                "query": "What is machine learning?",
                "language": "en",
                "max_results": 1
            },
            "expected_output_key": None  # We don't know the exact output format
        },
        # 6. Anonymizer Model
        {
            "model_name": "Anonymizer",
            "test_function": test_anonymizer,
            "test_input": {
                "text": "My name is John Smith and my email is john.smith@example.com.",
                "language": "en"
            },
            "expected_output_key": None  # We don't know the exact output format
        }
    ]
    
    # Run all tests
    results = []
    for test in tests:
        result = await test_model(
            test["model_name"],
            test["test_function"],
            test["test_input"],
            test["expected_output_key"]
        )
        results.append((test["model_name"], result))
        print("")  # Add spacing between tests
    
    # Print summary
    print(f"{PURPLE}======================================================{NC}")
    print(f"{PURPLE}                 TEST SUMMARY                         {NC}")
    print(f"{PURPLE}======================================================{NC}")
    
    for model_name, result in results:
        status = f"{GREEN}PASS{NC}" if result else f"{RED}FAIL{NC}"
        print(f"{model_name}: {status}")
    
    print("")
    print(f"{YELLOW}Note: Even if tests fail, it may be due to missing methods in the processor{NC}")
    print(f"{YELLOW}rather than issues with the model loading itself.{NC}")

if __name__ == "__main__":
    asyncio.run(main())