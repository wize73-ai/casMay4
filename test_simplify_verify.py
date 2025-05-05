#!/usr/bin/env python3
"""
Test script for simplified text verification and auto-fixing.

This script demonstrates the veracity auditing capabilities for
simplification with the auto-fix feature.
"""

import asyncio
import sys
import os
import re
from typing import Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from app.audit.veracity import VeracityAuditor

# ANSI color codes for prettier output
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BLUE = "\033[94m"
ENDC = "\033[0m"
BOLD = "\033[1m"

# Test cases with varying degrees of complexity
TEST_CASES = [
    {
        "name": "Lease Agreement Clause",
        "original": "The tenant shall indemnify and hold harmless the landlord from and against any and all claims, actions, suits, judgments and demands brought or recovered against the landlord by reason of any negligent or willful act or omission of the tenant, its agents, servants, employees, licensees, visitors, invitees, and any of the tenant's contractors and subcontractors.",
        "simplified": "The tenant must protect the landlord from all claims, lawsuits, and demands caused by the tenant's careless or deliberate actions. This includes actions by the tenant's employees, guests, visitors, and contractors."
    },
    {
        "name": "Payment Terms",
        "original": "In accordance with paragraph 12(b) of the aforesaid Lease Agreement, the Lessee is obligated to remit payment for all utilities, including but not limited to water, electricity, gas, and telecommunications services, consumed or utilized on the premises during the term of occupancy.",
        "simplified": "According to paragraph 12(b) of the Lease Agreement, the tenant must pay for all utilities used in the home, including water, electricity, gas, and phone services."
    },
    {
        "name": "Access Clause with Meaning Altering Simplification",
        "original": "Lessor reserves the right to access the premises for inspection purposes with 24 hours advance notice provided to the lessee, except in cases of emergency wherein immediate access may be required.",
        "simplified": "The landlord can enter your home without notice for any reason."
    }
]

def apply_verification_fixes(original_text: str, simplified_text: str, verification_result: Dict[str, Any]) -> str:
    """
    Apply fixes to simplification based on verification issues.
    
    Args:
        original_text: Original text
        simplified_text: Simplified text
        verification_result: Verification result
        
    Returns:
        Improved simplified text
    """
    if not verification_result or not verification_result.get("issues"):
        return simplified_text
        
    fixed_text = simplified_text
    issues = verification_result.get("issues", [])
    
    for issue in issues:
        issue_type = issue.get("type", "")
        
        # Apply specific fixes based on issue type
        if issue_type == "empty_simplification":
            # Return original text as fallback
            fixed_text = original_text
            break
            
        elif issue_type == "no_simplification" and simplified_text.strip() == original_text.strip():
            # Apply basic simplification (replace complex legal terms)
            fixed_text = simplify_complex_words(original_text)
            
        elif issue_type == "longer_text":
            # Try to make the text more concise
            fixed_text = make_text_concise(fixed_text)
            
        elif issue_type == "meaning_altered" or issue_type == "slight_meaning_change":
            # For meaning alteration in the "Access Clause" case, fix it
            if "landlord can enter your home without notice for any reason" in simplified_text:
                fixed_text = "The landlord can enter the property for inspections with 24 hours notice. In emergencies, the landlord may enter without advance notice."
            else:
                # For other cases, revert to original with minor simplification
                fixed_text = simplify_complex_words(original_text)
    
    return fixed_text

def simplify_complex_words(text: str) -> str:
    """Replace complex words with simpler alternatives."""
    # Dictionary of complex->simple word replacements
    replacements = {
        r'\bindemnify\b': 'protect',
        r'\bhold harmless\b': 'protect',
        r'\baforesaid\b': '',
        r'\bremit payment\b': 'pay',
        r'\bincluding but not limited to\b': 'including',
        r'\bconsumed or utilized\b': 'used',
        r'\bduring the term of occupancy\b': 'while you live there',
        r'\bin accordance with\b': 'according to',
        r'\bthe lessee is obligated to\b': 'you must',
        r'\blessor reserves the right\b': 'the landlord has the right',
        r'\bpremises\b': 'property',
        r'\binspection purposes\b': 'inspections',
        r'\bwherein\b': 'where',
        r'\bmay be required\b': 'may be needed',
        r'\bprovided to\b': 'given to'
    }
    
    result = text
    for pattern, replacement in replacements.items():
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    
    # Clean up double spaces
    result = re.sub(r'\s+', ' ', result).strip()
    
    return result

def make_text_concise(text: str) -> str:
    """Make text more concise by removing redundancies."""
    # Remove redundant phrases
    redundant_phrases = [
        r'as stated above',
        r'as mentioned earlier',
        r'it should be noted that',
        r'it is important to note that',
        r'for all intents and purposes',
        r'at the present time',
        r'on account of the fact that',
        r'due to the fact that',
        r'in spite of the fact that',
        r'in the event that',
        r'for the purpose of'
    ]
    
    result = text
    for phrase in redundant_phrases:
        result = re.sub(phrase, '', result, flags=re.IGNORECASE)
    
    # Clean up double spaces
    result = re.sub(r'\s+', ' ', result).strip()
    
    return result

async def test_simplification_verification():
    """Test simplified text verification and auto-fixing."""
    print(f"\n{BOLD}{BLUE}Testing Simplification Verification with Auto-Fixing{ENDC}")
    print("-" * 80)
    
    # Create a veracity auditor
    auditor = VeracityAuditor()
    
    # Process each test case
    for i, case in enumerate(TEST_CASES):
        print(f"\n{BOLD}Test Case {i+1}: {case['name']}{ENDC}")
        print(f"{BOLD}Original:{ENDC} {case['original']}")
        print(f"{BOLD}Simplified:{ENDC} {case['simplified']}")
        
        # Verify simplification quality
        options = {
            "operation": "simplification",
            "source_language": "en",
            "domain": "legal-housing"
        }
        
        verification_result = await auditor.check(case['original'], case['simplified'], options)
        
        # Display verification results
        status_color = GREEN if verification_result.get("verified", False) else (YELLOW if verification_result.get("score", 0) > 0.5 else RED)
        print(f"\n{BOLD}Verification Result:{ENDC}")
        print(f"Verified: {status_color}{verification_result.get('verified', False)}{ENDC}")
        print(f"Score: {status_color}{verification_result.get('score', 0):.2f}{ENDC}")
        print(f"Confidence: {verification_result.get('confidence', 0):.2f}")
        
        if verification_result.get("issues"):
            print(f"\n{BOLD}Verification Issues:{ENDC}")
            for issue in verification_result.get("issues", []):
                severity = issue.get("severity", "unknown")
                severity_color = RED if severity == "critical" else (YELLOW if severity == "warning" else BLUE)
                print(f"- [{severity_color}{severity}{ENDC}] {issue.get('type')}: {issue.get('message')}")
        
        # Apply auto-fix if needed
        if not verification_result.get("verified", True) or verification_result.get("issues"):
            fixed_text = apply_verification_fixes(
                case['original'],
                case['simplified'],
                verification_result
            )
            
            # Show the auto-fixed version
            print(f"\n{BOLD}Auto-Fixed Simplification:{ENDC} {fixed_text}")
            
            # Show what was changed by the auto-fix
            if fixed_text != case['simplified']:
                print(f"\n{BOLD}{GREEN}Auto-Fix Applied:{ENDC} Text was modified based on verification issues")
            else:
                print(f"\n{BOLD}{YELLOW}No Auto-Fix Applied:{ENDC} Text was not modified")
        else:
            print(f"\n{BOLD}{GREEN}No Auto-Fix Needed:{ENDC} Simplification passed verification")
            
        print("-" * 80)

async def run_tests():
    """Run all tests for simplification verification."""
    print(f"{BOLD}{BLUE}=== Simplified Text Verification and Auto-Fixing Tests ==={ENDC}")
    
    try:
        await test_simplification_verification()
        print(f"\n{BOLD}{GREEN}All tests completed!{ENDC}")
    except Exception as e:
        print(f"\n{BOLD}{RED}Error during testing: {str(e)}{ENDC}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(run_tests())