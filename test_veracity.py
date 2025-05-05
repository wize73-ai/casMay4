#!/usr/bin/env python3
"""
Test script for veracity auditing system.

This script tests the veracity auditing capabilities for both
translation and simplification operations.
"""

import asyncio
import json
import os
import sys
from typing import Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from app.audit.veracity import VeracityAuditor
from app.utils.config import load_config

# ANSI color codes for prettier output
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BLUE = "\033[94m"
ENDC = "\033[0m"
BOLD = "\033[1m"


async def test_simplification_verification():
    """Test verification of text simplification."""
    print(f"\n{BOLD}{BLUE}Testing Housing Legal Text Simplification Verification{ENDC}")
    print("-" * 80)
    
    # Initialize the veracity auditor
    config = load_config()
    auditor = VeracityAuditor(config=config)
    await auditor.initialize()
    
    # Test case 1: Good simplification
    original1 = (
        "The tenant shall indemnify and hold harmless the landlord from and against any and all claims, "
        "actions, suits, judgments and demands brought or recovered against the landlord by reason of any "
        "negligent or willful act or omission of the tenant, its agents, servants, employees, licensees, "
        "visitors, invitees, and any of the tenant's contractors and subcontractors."
    )
    simplified1 = (
        "The tenant must protect the landlord from all claims, lawsuits, and demands caused by the tenant's "
        "careless or deliberate actions. This includes actions by the tenant's employees, guests, visitors, "
        "and contractors."
    )
    
    # Test case 2: Poor simplification (too similar to original)
    original2 = (
        "In accordance with paragraph 12(b) of the aforesaid Lease Agreement, the Lessee is obligated to "
        "remit payment for all utilities, including but not limited to water, electricity, gas, and "
        "telecommunications services, consumed or utilized on the premises during the term of occupancy."
    )
    simplified2 = (
        "In accordance with paragraph 12(b) of the Lease Agreement, the tenant is obligated to pay for all utilities, "
        "including water, electricity, gas, and telecommunications services, consumed on the premises during the "
        "lease term."
    )
    
    # Test case 3: Bad simplification (changes meaning)
    original3 = (
        "Lessor reserves the right to access the premises for inspection purposes with 24 hours advance "
        "notice provided to the lessee, except in cases of emergency wherein immediate access may be required."
    )
    simplified3 = (
        "The landlord can enter your home without notice for any reason."
    )
    
    # Set up test cases
    test_cases = [
        {"original": original1, "simplified": simplified1, "name": "Good Simplification"},
        {"original": original2, "simplified": simplified2, "name": "Minimal Simplification"},
        {"original": original3, "simplified": simplified3, "name": "Meaning-Altering Simplification"}
    ]
    
    # Run tests
    for i, case in enumerate(test_cases, 1):
        print(f"\n{BOLD}Test Case {i}: {case['name']}{ENDC}")
        print(f"{BOLD}Original:{ENDC} {case['original']}")
        print(f"{BOLD}Simplified:{ENDC} {case['simplified']}")
        
        # Options indicating this is a simplification verification
        options = {
            "operation": "simplification",
            "source_language": "en",
            "domain": "legal-housing"
        }
        
        # Run verification
        result = await auditor.check(case["original"], case["simplified"], options)
        
        # Print results with color coding
        status_color = GREEN if result.get("verified", False) else (YELLOW if result.get("score", 0) > 0.5 else RED)
        print(f"\n{BOLD}Verification Result:{ENDC}")
        print(f"Verified: {status_color}{result.get('verified', False)}{ENDC}")
        print(f"Score: {status_color}{result.get('score', 0):.2f}{ENDC}")
        print(f"Confidence: {result.get('confidence', 0):.2f}")
        
        if "metrics" in result:
            print(f"\n{BOLD}Metrics:{ENDC}")
            for key, value in result.get("metrics", {}).items():
                if isinstance(value, float):
                    print(f"- {key}: {value:.2f}")
                else:
                    print(f"- {key}: {value}")
        
        if "issues" in result and result["issues"]:
            print(f"\n{BOLD}Issues:{ENDC}")
            for issue in result.get("issues", []):
                severity = issue.get("severity", "unknown")
                severity_color = RED if severity == "critical" else (YELLOW if severity == "warning" else BLUE)
                print(f"- [{severity_color}{severity}{ENDC}] {issue.get('type')}: {issue.get('message')}")
        else:
            print(f"\n{GREEN}No issues detected!{ENDC}")

        print("-" * 80)


async def test_translation_verification():
    """Test verification of text translation."""
    print(f"\n{BOLD}{BLUE}Testing Translation Verification{ENDC}")
    print("-" * 80)
    
    # Initialize the veracity auditor
    config = load_config()
    auditor = VeracityAuditor(config=config)
    await auditor.initialize()
    
    # Test case 1: Good translation
    english1 = (
        "The lease agreement requires tenants to maintain the property in good condition "
        "and report any damages immediately."
    )
    spanish1 = (
        "El contrato de arrendamiento requiere que los inquilinos mantengan la propiedad en buenas "
        "condiciones e informen de cualquier daño inmediatamente."
    )
    
    # Test case 2: Bad translation (missing content)
    english2 = (
        "Tenants must pay rent by the 1st of each month. Late fees of $50 will be charged for "
        "payments received after the 5th day."
    )
    spanish2 = (
        "Los inquilinos deben pagar el alquiler antes del día 1 de cada mes."
    )
    
    # Test case 3: Incorrect translation (wrong numbers)
    english3 = (
        "The security deposit of $1,500 will be returned within 30 days after the end of the lease term."
    )
    spanish3 = (
        "El depósito de seguridad de $500 será devuelto dentro de los 60 días posteriores al final del contrato."
    )
    
    # Set up test cases
    test_cases = [
        {"source": english1, "translation": spanish1, "source_lang": "en", "target_lang": "es", "name": "Good Translation"},
        {"source": english2, "translation": spanish2, "source_lang": "en", "target_lang": "es", "name": "Incomplete Translation"},
        {"source": english3, "translation": spanish3, "source_lang": "en", "target_lang": "es", "name": "Translation with Incorrect Numbers"}
    ]
    
    # Run tests
    for i, case in enumerate(test_cases, 1):
        print(f"\n{BOLD}Test Case {i}: {case['name']}{ENDC}")
        print(f"{BOLD}Source ({case['source_lang']}):{ENDC} {case['source']}")
        print(f"{BOLD}Translation ({case['target_lang']}):{ENDC} {case['translation']}")
        
        # Options for translation verification
        options = {
            "operation": "translation",
            "source_language": case["source_lang"],
            "target_language": case["target_lang"]
        }
        
        # Run verification
        result = await auditor.check(case["source"], case["translation"], options)
        
        # Print results with color coding
        status_color = GREEN if result.get("verified", False) else (YELLOW if result.get("score", 0) > 0.5 else RED)
        print(f"\n{BOLD}Verification Result:{ENDC}")
        print(f"Verified: {status_color}{result.get('verified', False)}{ENDC}")
        print(f"Score: {status_color}{result.get('score', 0):.2f}{ENDC}")
        print(f"Confidence: {result.get('confidence', 0):.2f}")
        
        if "metrics" in result:
            print(f"\n{BOLD}Metrics:{ENDC}")
            for key, value in result.get("metrics", {}).items():
                if isinstance(value, float):
                    print(f"- {key}: {value:.2f}")
                else:
                    print(f"- {key}: {value}")
        
        if "issues" in result and result["issues"]:
            print(f"\n{BOLD}Issues:{ENDC}")
            for issue in result.get("issues", []):
                severity = issue.get("severity", "unknown")
                severity_color = RED if severity == "critical" else (YELLOW if severity == "warning" else BLUE)
                print(f"- [{severity_color}{severity}{ENDC}] {issue.get('type')}: {issue.get('message')}")
        else:
            print(f"\n{GREEN}No issues detected!{ENDC}")

        print("-" * 80)


async def run_tests():
    """Run all veracity auditing tests."""
    print(f"{BOLD}{BLUE}=== Veracity Auditing System Tests ==={ENDC}")
    
    try:
        await test_simplification_verification()
        await test_translation_verification()
        
        print(f"\n{BOLD}{GREEN}All tests completed!{ENDC}")
    except Exception as e:
        print(f"\n{BOLD}{RED}Error during testing: {str(e)}{ENDC}")
        import traceback
        traceback.print_exc()
    

if __name__ == "__main__":
    asyncio.run(run_tests())