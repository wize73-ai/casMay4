#!/usr/bin/env python
"""
Test script for MBART translation and anonymization changes.
"""

import asyncio
import sys
import logging
import time
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
    
    # Comprehensive test of language codes
    test_pairs = [
        ("en", "en_XX"),   # English
        ("es", "es_XX"),   # Spanish
        ("fr", "fr_XX"),   # French
        ("de", "de_DE"),   # German
        ("zh", "zh_CN"),   # Chinese
        ("ja", "ja_XX"),   # Japanese
        ("ru", "ru_RU"),   # Russian
        ("ar", "ar_AR"),   # Arabic
        ("nl", "nl_XX"),   # Dutch
        ("pt", "pt_XX"),   # Portuguese
        ("pl", "pl_PL"),   # Polish
        ("tr", "tr_TR"),   # Turkish
        ("it", "it_IT"),   # Italian
        ("ko", "ko_KR"),   # Korean
        ("fi", "fi_FI"),   # Finnish
        ("hi", "hi_IN"),   # Hindi
        
        # Test invalid and edge cases
        ("xx", "en_XX"),   # Invalid code should return default
        ("", "en_XX"),     # Empty string should return default
        (None, "en_XX"),   # None should return default
    ]
    
    logger.info("Testing MBART language code conversion:")
    
    all_passed = True
    for iso_code, expected_mbart_code in test_pairs:
        actual_mbart_code = translator._get_mbart_language_code(iso_code)
        if actual_mbart_code == expected_mbart_code:
            logger.info(f"  ✓ {iso_code} -> {actual_mbart_code}")
        else:
            logger.error(f"  ✗ {iso_code} -> {actual_mbart_code} (expected {expected_mbart_code})")
            all_passed = False
    
    # Verify all MBART language codes are mapped
    total_languages = len(translator.mbart_language_codes)
    logger.info(f"Total MBART language codes supported: {total_languages}")
    if total_languages < 50: # MBART supports 50+ languages
        logger.warning(f"Expected at least 50 language codes, found {total_languages}")
    
    # Return overall success status
    return all_passed

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
    all_consistent = True
    all_strategies_captured_entities = True
    
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
        
        # Create options dict with consistency enabled
        options = {
            "strategy": strategy,
            "entity_types": entity_types,
            "consistency": True
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
        
        # Check that we found some entities
        if len(entities) < 4:  # We should find at least a name, email, phone, and SSN
            logger.error(f"Few entities detected ({len(entities)}). Expected at least 4")
            all_strategies_captured_entities = False
        
        # Check entity types
        entity_types_found = set(entity["type"] for entity in entities)
        logger.info(f"Entity types found: {', '.join(entity_types_found)}")
        
        # Test consistency by running multiple times
        consistency_results = []
        for i in range(3):  # Run 3 more times
            anon_text_repeat, entities_repeat = await anonymizer.process(
                text=test_text,
                language="en",
                options=options
            )
            consistency_results.append(anonymized_text == anon_text_repeat)
        
        is_consistent = all(consistency_results)
        logger.info(f"Anonymization is consistent across multiple runs: {is_consistent}")
        if not is_consistent:
            all_consistent = False
        
        # Test deterministic replacement for specific entity types
        if strategy == "pseudonymize":
            # Create text with repeated entities
            repeated_text = """
            John Smith is a software engineer. John Smith lives in Springfield.
            Contact John Smith at john.smith@example.com or john.smith@example.com.
            His friend Jane Doe also lives in Springfield.
            """
            
            # Process with deterministic replacement
            options["consistency"] = True
            anon_repeated, entities_repeated = await anonymizer.process(
                text=repeated_text,
                language="en",
                options=options
            )
            
            logger.info("Testing deterministic replacement of repeated entities:")
            logger.info(f"Original: {repeated_text[:60]}...")
            logger.info(f"Anonymized: {anon_repeated[:60]}...")
            
            # Check if the same entity (John Smith) was replaced consistently
            replacements = {}
            for entity in entities_repeated:
                if entity["original_text"] not in replacements:
                    replacements[entity["original_text"]] = entity["anonymized_text"]
                else:
                    # Check if the replacement is the same
                    if replacements[entity["original_text"]] != entity["anonymized_text"]:
                        logger.error(f"Inconsistent replacement for '{entity['original_text']}': "
                                    f"{replacements[entity['original_text']]} vs {entity['anonymized_text']}")
                        all_consistent = False
            
            logger.info(f"All repeated entities were replaced consistently: {all_consistent}")
    
    # Test the _deterministic_random method directly
    seed_texts = ["John Smith", "john.smith@example.com", "123-45-6789"]
    results = {}
    
    logger.info("\nTesting _deterministic_random directly:")
    for seed_text in seed_texts:
        # Run multiple times with the same seed text
        values = [anonymizer._deterministic_random(1000, seed_text) for _ in range(5)]
        
        # All values should be the same for the same seed text
        is_consistent = len(set(values)) == 1
        results[seed_text] = is_consistent
        
        logger.info(f"  Seed: '{seed_text}' -> Values: {values[0]} (Consistent: {is_consistent})")
    
    deterministic_random_consistent = all(results.values())
    logger.info(f"_deterministic_random produces consistent results: {deterministic_random_consistent}")
    
    # Test with patterns from different languages
    logger.info("\nTesting anonymization with patterns from different languages:")
    languages = ["en", "es", "fr"]  # Focusing on the main languages we support
    patterns_found = {}
    
    for language in languages:
        patterns = anonymizer._get_patterns_for_language(language)
        patterns_found[language] = len(patterns) > 0
        logger.info(f"  {language}: {len(patterns)} patterns found")
    
    all_languages_have_patterns = all(patterns_found.values())
    logger.info(f"All tested languages have detection patterns: {all_languages_have_patterns}")
    
    # Return overall success status
    overall_success = all_consistent and deterministic_random_consistent and all_languages_have_patterns and all_strategies_captured_entities
    
    if overall_success:
        logger.info("\nAll anonymization tests passed successfully")
    else:
        logger.error("\nSome anonymization tests failed")
        if not all_consistent:
            logger.error("- Consistency tests failed")
        if not deterministic_random_consistent:
            logger.error("- Deterministic random tests failed")
        if not all_languages_have_patterns:
            logger.error("- Language pattern tests failed")
        if not all_strategies_captured_entities:
            logger.error("- Entity detection tests failed")
    
    return overall_success

async def main():
    """Run all tests"""
    logger.info("Starting tests for MBART implementation and enhanced anonymization")
    
    test_results = {}
    start_time = time.time()
    
    # Test MBART language codes
    logger.info("\n=== Testing MBART Language Codes ===")
    try:
        mbart_result = await test_mbart_language_codes()
        test_results["MBART Language Codes"] = mbart_result
        logger.info(f"MBART test completed: {'PASS' if mbart_result else 'FAIL'}")
    except Exception as e:
        logger.error(f"MBART language code test failed with exception: {e}", exc_info=True)
        test_results["MBART Language Codes"] = False
    
    # Test anonymization
    logger.info("\n=== Testing Anonymization ===")
    try:
        anon_result = await test_anonymization()
        test_results["Anonymization"] = anon_result
        logger.info(f"Anonymization test completed: {'PASS' if anon_result else 'FAIL'}")
    except Exception as e:
        logger.error(f"Anonymization test failed with exception: {e}", exc_info=True)
        test_results["Anonymization"] = False
    
    # Calculate time taken
    duration = time.time() - start_time
    
    # Print summary
    logger.info("\n=== Test Results Summary ===")
    passed = 0
    failed = 0
    
    for test, result in test_results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        if result:
            passed += 1
            logger.info(f"{status} - {test}")
        else:
            failed += 1
            logger.error(f"{status} - {test}")
    
    total = passed + failed
    pass_rate = (passed / total) * 100 if total > 0 else 0
    
    logger.info(f"\nTest run completed in {duration:.2f} seconds")
    logger.info(f"Tests: {total}, Passed: {passed}, Failed: {failed}, Pass rate: {pass_rate:.1f}%")
    
    # Overall status
    all_passed = all(test_results.values())
    if all_passed:
        logger.info("\n✅ ALL TESTS PASSED - Implementation is ready for use")
    else:
        logger.error("\n❌ SOME TESTS FAILED - Implementation needs review")
    
    # Return overall success/failure
    return all_passed

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)