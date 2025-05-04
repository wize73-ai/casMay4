#!/usr/bin/env python
"""
Test script for MBART translation and anonymization changes.
"""

import asyncio
import sys
import logging
from pathlib import Path

# Configure basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("test_script")

# Add app directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from app.core.pipeline.translator import TranslationPipeline
from app.core.pipeline.anonymizer import AnonymizationPipeline, EntityType
from app.services.models.manager import EnhancedModelManager
from app.utils.config import load_config
from app.api.schemas.translation import TranslationRequest

async def test_mbart_language_codes():
    """Test the MBART language code conversion"""
    translator = TranslationPipeline(model_manager=None)
    
    # Test a few language codes
    test_codes = ["en", "es", "fr", "de", "zh", "ja", "ru"]
    logger.info("Testing MBART language code conversion:")
    
    for code in test_codes:
        mbart_code = translator._get_mbart_language_code(code)
        logger.info(f"  {code} -> {mbart_code}")
    
    # Test an invalid code
    invalid_code = "xx"
    mbart_code = translator._get_mbart_language_code(invalid_code)
    logger.info(f"  Invalid code '{invalid_code}' -> {mbart_code}")
    
    return True

async def test_anonymization():
    """Test the anonymization functionality"""
    # Create an anonymization pipeline without models
    anonymizer = AnonymizationPipeline(model_manager=None)
    
    # Test text with entities
    test_text = """
    John Smith lives at 123 Main Street, Springfield, IL 62701.
    His phone number is (555) 123-4567 and his email is john.smith@example.com.
    He works at Acme Corporation and his social security number is 123-45-6789.
    His credit card number is 4111-1111-1111-1111 and it expires on 01/25.
    """
    
    # Test different anonymization strategies
    strategies = ["mask", "redact", "pseudonymize"]
    
    logger.info("Testing anonymization with different strategies:")
    for strategy in strategies:
        logger.info(f"\nStrategy: {strategy}")
        entity_types = [
            EntityType.PERSON,
            EntityType.LOCATION,
            EntityType.ORGANIZATION,
            EntityType.PHONE,
            EntityType.EMAIL,
            EntityType.SSN,
            EntityType.CREDIT_CARD,
            EntityType.DATE,
            EntityType.ADDRESS
        ]
        
        # Create options dict
        options = {
            "strategy": strategy,
            "entity_types": entity_types
        }
        
        # Process the text
        anonymized_text, entities = await anonymizer.process(
            text=test_text,
            language="en",
            options=options
        )
        
        logger.info(f"Original:   {test_text[:60]}...")
        logger.info(f"Anonymized: {anonymized_text[:60]}...")
        logger.info(f"Entities detected: {len(entities)}")
        
        # Check for consistency by anonymizing the same text again
        anonymized_text2, entities2 = await anonymizer.process(
            text=test_text,
            language="en",
            options=options
        )
        
        is_consistent = anonymized_text == anonymized_text2
        logger.info(f"Anonymization is consistent: {is_consistent}")
    
    return True

async def main():
    """Run all tests"""
    logger.info("Starting tests")
    
    test_results = {}
    
    # Test MBART language codes
    logger.info("\n=== Testing MBART Language Codes ===")
    try:
        test_results["mbart_language_codes"] = await test_mbart_language_codes()
    except Exception as e:
        logger.error(f"MBART language code test failed: {e}")
        test_results["mbart_language_codes"] = False
    
    # Test anonymization
    logger.info("\n=== Testing Anonymization ===")
    try:
        test_results["anonymization"] = await test_anonymization()
    except Exception as e:
        logger.error(f"Anonymization test failed: {e}")
        test_results["anonymization"] = False
    
    # Print summary
    logger.info("\n=== Test Results ===")
    for test, result in test_results.items():
        logger.info(f"{test}: {'PASS' if result else 'FAIL'}")
    
    # Return overall success/failure
    return all(test_results.values())

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)