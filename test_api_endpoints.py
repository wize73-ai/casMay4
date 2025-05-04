#!/usr/bin/env python3
"""
API Endpoint Testing Script

This script tests all the available API endpoints in the CasaLingua application,
verifying their functionality and response formats.

Usage:
    python test_api_endpoints.py [--url BASE_URL] [--env development]

Options:
    --url BASE_URL    Base URL of the API (default: http://localhost:8000)
    --env ENV         Environment setting for auth bypass (default: development)
"""

import argparse
import asyncio
import json
import os
import sys
import time
from typing import Dict, List, Any, Optional, Union
import aiohttp
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("endpoint_tester")

# ANSI colors for better output readability
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def log_success(message):
    logger.info(f"{Colors.GREEN}✓ {message}{Colors.ENDC}")

def log_failure(message):
    logger.error(f"{Colors.RED}✗ {message}{Colors.ENDC}")

def log_warning(message):
    logger.warning(f"{Colors.YELLOW}⚠ {message}{Colors.ENDC}")

def log_info(message):
    logger.info(f"{Colors.BLUE}ℹ {message}{Colors.ENDC}")

def log_header(message):
    logger.info(f"\n{Colors.HEADER}{Colors.BOLD}=== {message} ==={Colors.ENDC}")

class APIEndpointTester:
    def __init__(self, base_url: str, env: str = "development"):
        self.base_url = base_url
        self.env = env
        self.session = None
        self.results = {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "tests": []
        }
    
    async def setup(self):
        """Initialize the HTTP session with proper headers"""
        # Set environment variable for auth bypass if needed
        if self.env == "development":
            os.environ["CASALINGUA_ENV"] = "development"
            log_info(f"Setting CASALINGUA_ENV=development for auth bypass")
        
        # Create HTTP session
        self.session = aiohttp.ClientSession(
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json"
            }
        )
    
    async def teardown(self):
        """Close the HTTP session"""
        if self.session:
            await self.session.close()
        
        # Unset environment variable
        if "CASALINGUA_ENV" in os.environ:
            del os.environ["CASALINGUA_ENV"]
    
    async def test_endpoint(self, method: str, endpoint: str, payload: Optional[Dict] = None, 
                            expected_status: int = 200, description: str = ""):
        """Test an API endpoint and record the result"""
        self.results["total"] += 1
        url = f"{self.base_url}{endpoint}"
        
        log_info(f"Testing {method} {endpoint} - {description}")
        
        start_time = time.time()
        try:
            if method.upper() == "GET":
                async with self.session.get(url) as response:
                    status = response.status
                    try:
                        data = await response.json()
                    except:
                        data = await response.text()
            else:  # POST
                async with self.session.post(url, json=payload) as response:
                    status = response.status
                    try:
                        data = await response.json()
                    except:
                        data = await response.text()
            
            duration = time.time() - start_time
            
            # Format the response data for display
            if isinstance(data, dict):
                formatted_data = json.dumps(data, indent=2)
                # Truncate if too long
                if len(formatted_data) > 500:
                    formatted_data = formatted_data[:500] + "..."
            else:
                formatted_data = str(data)
                if len(formatted_data) > 500:
                    formatted_data = formatted_data[:500] + "..."
            
            # Check if status matches expected
            if status == expected_status:
                log_success(f"Status: {status} (Expected: {expected_status}) - {duration:.2f}s")
                self.results["passed"] += 1
                
                # Additional validation for successful responses
                if status == 200 and isinstance(data, dict):
                    if "status" in data and data["status"] == "success":
                        log_success(f"Response indicates success")
                    elif "status" in data:
                        log_warning(f"Response status is {data['status']}, expected 'success'")
            else:
                log_failure(f"Status: {status} (Expected: {expected_status}) - {duration:.2f}s")
                self.results["failed"] += 1
            
            log_info(f"Response: {formatted_data}")
            
            # Record test result
            self.results["tests"].append({
                "endpoint": endpoint,
                "method": method,
                "description": description,
                "status": status,
                "expected_status": expected_status,
                "passed": status == expected_status,
                "duration": duration,
                "response": data if isinstance(data, dict) else str(data)
            })
            
            return status, data
            
        except Exception as e:
            duration = time.time() - start_time
            log_failure(f"Request failed: {str(e)} - {duration:.2f}s")
            self.results["failed"] += 1
            
            # Record test failure
            self.results["tests"].append({
                "endpoint": endpoint,
                "method": method,
                "description": description,
                "status": None,
                "expected_status": expected_status,
                "passed": False,
                "duration": duration,
                "error": str(e)
            })
            
            return None, str(e)
    
    async def test_health_endpoints(self):
        """Test health and status endpoints"""
        log_header("Testing Health & Status Endpoints")
        
        await self.test_endpoint("GET", "/health", description="Basic health check")
        await self.test_endpoint("GET", "/health/detailed", description="Detailed health check")
        await self.test_endpoint("GET", "/health/models", description="Model health check")
        await self.test_endpoint("GET", "/readiness", description="Readiness probe")
        await self.test_endpoint("GET", "/liveness", description="Liveness probe")
    
    async def test_translation_endpoints(self):
        """Test translation endpoints"""
        log_header("Testing Translation Endpoints")
        
        # Test basic translation
        await self.test_endpoint(
            "POST", 
            "/pipeline/translate",
            payload={
                "text": "Hello, how are you?",
                "source_language": "en",
                "target_language": "es"
            },
            description="Basic text translation (EN to ES)"
        )
        
        # Test translation with auto-detection
        await self.test_endpoint(
            "POST", 
            "/pipeline/translate",
            payload={
                "text": "Bonjour, comment ça va?",
                "source_language": "auto",
                "target_language": "en"
            },
            description="Translation with auto-detection (FR to EN)"
        )
        
        # Test batch translation
        await self.test_endpoint(
            "POST", 
            "/pipeline/translate/batch",
            payload={
                "texts": ["Hello, how are you?", "The weather is nice today"],
                "source_language": "en",
                "target_language": "es"
            },
            description="Batch translation (EN to ES)"
        )
    
    async def test_language_detection_endpoints(self):
        """Test language detection endpoints"""
        log_header("Testing Language Detection Endpoints")
        
        # Test basic language detection
        await self.test_endpoint(
            "POST", 
            "/pipeline/detect",
            payload={
                "text": "Hello, how are you?"
            },
            description="Language detection (English)"
        )
        
        # Test language detection with detailed analysis
        await self.test_endpoint(
            "POST", 
            "/pipeline/detect",
            payload={
                "text": "Hola, ¿cómo estás?",
                "detailed": True
            },
            description="Detailed language detection (Spanish)"
        )
        
        # Test with alias endpoint
        await self.test_endpoint(
            "POST", 
            "/pipeline/detect-language",
            payload={
                "text": "Guten Tag, wie geht es Ihnen?"
            },
            description="Language detection alias endpoint (German)"
        )
    
    async def test_text_simplification_endpoint(self):
        """Test text simplification endpoint"""
        log_header("Testing Text Simplification Endpoint")
        
        await self.test_endpoint(
            "POST", 
            "/pipeline/simplify",
            payload={
                "text": "The mitochondrion is a double membrane-bound organelle found in most eukaryotic organisms. Mitochondria use aerobic respiration to generate most of the cell's supply of adenosine triphosphate (ATP), which is used as a source of chemical energy.",
                "language": "en",
                "target_level": "simple"
            },
            description="Text simplification (English)"
        )
    
    async def test_text_anonymization_endpoint(self):
        """Test text anonymization endpoint"""
        log_header("Testing Text Anonymization Endpoint")
        
        await self.test_endpoint(
            "POST", 
            "/pipeline/anonymize",
            payload={
                "text": "John Smith lives at 123 Main St, New York and his email is john.smith@example.com. His phone number is (555) 123-4567.",
                "language": "en",
                "strategy": "mask"
            },
            description="Text anonymization (English)"
        )
    
    async def test_text_analysis_endpoints(self):
        """Test text analysis endpoints"""
        log_header("Testing Text Analysis Endpoints")
        
        # Test sentiment analysis
        await self.test_endpoint(
            "POST", 
            "/pipeline/analyze",
            payload={
                "text": "I love this product! It's amazing and works really well.",
                "language": "en",
                "analyses": ["sentiment"]
            },
            description="Sentiment analysis (Positive)"
        )
        
        # Test entity recognition
        await self.test_endpoint(
            "POST", 
            "/pipeline/analyze",
            payload={
                "text": "Apple Inc. is headquartered in Cupertino, California and was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne.",
                "language": "en",
                "analyses": ["entities"]
            },
            description="Entity recognition"
        )
        
        # Test multiple analyses
        await self.test_endpoint(
            "POST", 
            "/pipeline/analyze",
            payload={
                "text": "Google announced a new partnership with Microsoft yesterday. The tech giants will collaborate on AI research.",
                "language": "en",
                "analyses": ["sentiment", "entities", "topics"]
            },
            description="Multiple analyses"
        )
    
    async def test_summarization_endpoint(self):
        """Test text summarization endpoint"""
        log_header("Testing Text Summarization Endpoint")
        
        await self.test_endpoint(
            "POST", 
            "/pipeline/summarize",
            payload={
                "text": """
                Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to natural intelligence displayed by animals including humans. 
                AI research has been defined as the field of study of intelligent agents, which refers to any system that perceives its environment and takes actions that maximize its chance of achieving its goals.
                The term "artificial intelligence" had previously been used to describe machines that mimic and display "human" cognitive skills that are associated with the human mind, such as "learning" and "problem-solving." 
                This definition has since been rejected by major AI researchers who now describe AI in terms of rationality and acting rationally, which does not limit how intelligence can be articulated.
                AI applications include advanced web search engines (Google Search), recommendation systems (YouTube, Amazon, and Netflix), understanding human speech (Siri and Alexa), self-driving cars (Tesla), generative or creative tools (ChatGPT and AI art), automated decision-making, and competing at the highest level in strategic game systems (chess and Go).
                """,
                "language": "en"
            },
            description="Text summarization (English)"
        )
    
    async def print_summary(self):
        """Print a summary of all test results"""
        log_header("Test Summary")
        
        # Calculate overall pass rate
        pass_rate = (self.results["passed"] / self.results["total"]) * 100 if self.results["total"] > 0 else 0
        
        log_info(f"Total Tests: {self.results['total']}")
        log_success(f"Passed: {self.results['passed']}")
        log_failure(f"Failed: {self.results['failed']}")
        
        if self.results["skipped"] > 0:
            log_warning(f"Skipped: {self.results['skipped']}")
        
        if pass_rate >= 90:
            log_success(f"Pass Rate: {pass_rate:.1f}%")
        elif pass_rate >= 75:
            log_warning(f"Pass Rate: {pass_rate:.1f}%")
        else:
            log_failure(f"Pass Rate: {pass_rate:.1f}%")
        
        # Generate a detailed report of failed tests
        if self.results["failed"] > 0:
            log_header("Failed Tests")
            failed_tests = [test for test in self.results["tests"] if not test["passed"]]
            for i, test in enumerate(failed_tests, 1):
                log_failure(f"{i}. {test['method']} {test['endpoint']} - {test['description']}")
                if "error" in test:
                    log_failure(f"   Error: {test['error']}")
                else:
                    log_failure(f"   Status: {test['status']} (Expected: {test['expected_status']})")
        
        # Save the full results to a JSON file
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"api_test_results_{timestamp}.json"
        with open(filename, "w") as f:
            json.dump(self.results, f, indent=2)
        
        log_info(f"Full test results saved to: {filename}")
        
        # Return code based on pass rate
        return 0 if pass_rate == 100 else 1

    async def run_all_tests(self):
        """Run all endpoint tests"""
        await self.setup()
        
        try:
            # Test health endpoints
            await self.test_health_endpoints()
            
            # Test all pipeline endpoints
            await self.test_translation_endpoints()
            await self.test_language_detection_endpoints()
            await self.test_text_simplification_endpoint()
            await self.test_text_anonymization_endpoint()
            await self.test_text_analysis_endpoints()
            await self.test_summarization_endpoint()
            
            # Print summary
            return await self.print_summary()
            
        finally:
            await self.teardown()

async def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test API endpoints")
    parser.add_argument("--url", type=str, default="http://localhost:8000", 
                        help="Base URL of the API (default: http://localhost:8000)")
    parser.add_argument("--env", type=str, default="development", 
                        help="Environment setting for auth bypass (default: development)")
    args = parser.parse_args()
    
    # Create and run the tester
    tester = APIEndpointTester(args.url, args.env)
    return await tester.run_all_tests()

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        log_warning("Test interrupted by user")
        sys.exit(130)
    except Exception as e:
        log_failure(f"Unhandled exception: {str(e)}")
        sys.exit(1)