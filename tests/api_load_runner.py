#!/usr/bin/env python3
"""
API Load Runner for CasaLingua

This script provides a Python-based API load testing utility that can be used
programmatically without requiring Locust. It includes features for concurrency,
detailed metrics, and visualization.

Example usage:
    python api_load_runner.py --endpoint translate --concurrency 10 --duration 60
"""

import argparse
import asyncio
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple

import aiohttp
import numpy as np
import pandas as pd
import psutil
from tqdm import tqdm

try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Common test data - same as in the Locust script
TEST_TEXTS = {
    "en": [
        "Hello, how are you doing today?",
        "The quick brown fox jumps over the lazy dog.",
        "I hope this message finds you well.",
        "Please let me know if you have any questions.",
        "We need to discuss this matter as soon as possible.",
    ],
    "es": [
        "Hola, ¿cómo estás hoy?",
        "El rápido zorro marrón salta sobre el perro perezoso.",
        "Espero que este mensaje te encuentre bien.",
        "Por favor, avísame si tienes alguna pregunta.",
        "Necesitamos discutir este asunto lo antes posible.",
    ],
    "fr": [
        "Bonjour, comment vas-tu aujourd'hui?",
        "Le rapide renard brun saute par-dessus le chien paresseux.",
        "J'espère que ce message vous trouve bien.",
        "S'il vous plaît, faites-moi savoir si vous avez des questions.",
        "Nous devons discuter de cette affaire dès que possible.",
    ],
    "de": [
        "Hallo, wie geht es dir heute?",
        "Der schnelle braune Fuchs springt über den faulen Hund.",
        "Ich hoffe, diese Nachricht findet dich gut.",
        "Bitte lass mich wissen, wenn du Fragen hast.",
        "Wir müssen diese Angelegenheit so schnell wie möglich besprechen.",
    ],
}

LANGUAGES = ["en", "es", "fr", "de", "it", "pt", "ru", "ja", "zh", "ar"]
TARGET_LEVELS = ["simple", "intermediate", "advanced"]
FORMALITY_LEVELS = ["informal", "neutral", "formal"]

# Configuration
API_KEY = os.environ.get("CASALINGUA_API_KEY", "cslg_8f4b2d1e7a3c5b9e2f1d8a7c4b2e5f9d")
BASE_URL = os.environ.get("CASALINGUA_API_URL", "http://localhost:8000")


class APILoadRunner:
    """API Load Runner for CasaLingua"""

    def __init__(
        self,
        base_url: str = BASE_URL,
        api_key: str = API_KEY,
        concurrency: int = 10,
        duration: int = 60,
        ramp_up: int = 5,
        cooldown: int = 5,
        log_dir: Optional[str] = None,
    ):
        """Initialize the load runner"""
        self.base_url = base_url
        self.api_key = api_key
        self.concurrency = concurrency
        self.duration = duration
        self.ramp_up = ramp_up
        self.cooldown = cooldown
        
        # Logging and metrics
        self.start_time = None
        self.end_time = None
        self.results = []
        self.log_dir = log_dir or os.path.join(parent_dir, "logs", "load_tests")
        os.makedirs(self.log_dir, exist_ok=True)
        
        # System metrics
        self.system_metrics = {
            "timestamp": [],
            "cpu_percent": [],
            "memory_percent": [],
            "active_connections": [],
        }
        
        # Limit for visualization
        self.max_points = 1000

    def get_headers(self) -> Dict[str, str]:
        """Get headers for API requests"""
        return {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-API-Key": self.api_key,
        }

    def get_random_text(self, language: str = "en") -> str:
        """Get a random test text in the specified language"""
        # If we don't have texts for this language, fall back to English
        if language not in TEST_TEXTS:
            language = "en"
        return random.choice(TEST_TEXTS[language])

    def get_language_pair(self) -> Tuple[str, str]:
        """Get a random language pair for translation"""
        source = random.choice(LANGUAGES)
        target = random.choice([lang for lang in LANGUAGES if lang != source])
        return source, target

    async def health_check(self, session: aiohttp.ClientSession) -> Dict[str, Any]:
        """Test the health check endpoint"""
        start_time = time.time()
        url = f"{self.base_url}/health"
        
        try:
            async with session.get(url) as response:
                response_time = time.time() - start_time
                status = response.status
                try:
                    data = await response.json()
                except:
                    data = {"error": "Failed to parse JSON"}
                
                success = status == 200
                
                return {
                    "endpoint": "health",
                    "status": status,
                    "response_time": response_time,
                    "success": success,
                    "timestamp": datetime.now().isoformat(),
                }
        
        except Exception as e:
            response_time = time.time() - start_time
            return {
                "endpoint": "health",
                "status": 0,
                "response_time": response_time,
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    async def translate_text(self, session: aiohttp.ClientSession) -> Dict[str, Any]:
        """Test the translation endpoint"""
        start_time = time.time()
        url = f"{self.base_url}/pipeline/translate"
        
        source_lang, target_lang = self.get_language_pair()
        payload = {
            "text": self.get_random_text(source_lang),
            "source_language": source_lang,
            "target_language": target_lang,
            "preserve_formatting": True,
        }
        
        try:
            async with session.post(url, json=payload) as response:
                response_time = time.time() - start_time
                status = response.status
                try:
                    data = await response.json()
                except:
                    data = {"error": "Failed to parse JSON"}
                
                success = status == 200
                
                return {
                    "endpoint": "translate",
                    "status": status,
                    "response_time": response_time,
                    "success": success,
                    "source_lang": source_lang,
                    "target_lang": target_lang,
                    "text_length": len(payload["text"]),
                    "timestamp": datetime.now().isoformat(),
                }
        
        except Exception as e:
            response_time = time.time() - start_time
            return {
                "endpoint": "translate",
                "status": 0,
                "response_time": response_time,
                "success": False,
                "error": str(e),
                "source_lang": source_lang,
                "target_lang": target_lang,
                "text_length": len(payload["text"]),
                "timestamp": datetime.now().isoformat(),
            }

    async def detect_language(self, session: aiohttp.ClientSession) -> Dict[str, Any]:
        """Test the language detection endpoint"""
        start_time = time.time()
        url = f"{self.base_url}/pipeline/detect"
        
        lang = random.choice(list(TEST_TEXTS.keys()))
        text = self.get_random_text(lang)
        
        payload = {
            "text": text,
            "detailed": random.choice([True, False]),
        }
        
        try:
            async with session.post(url, json=payload) as response:
                response_time = time.time() - start_time
                status = response.status
                try:
                    data = await response.json()
                except:
                    data = {"error": "Failed to parse JSON"}
                
                success = status == 200
                
                return {
                    "endpoint": "detect",
                    "status": status,
                    "response_time": response_time,
                    "success": success,
                    "actual_lang": lang,
                    "text_length": len(text),
                    "timestamp": datetime.now().isoformat(),
                }
        
        except Exception as e:
            response_time = time.time() - start_time
            return {
                "endpoint": "detect",
                "status": 0,
                "response_time": response_time,
                "success": False,
                "error": str(e),
                "actual_lang": lang,
                "text_length": len(text),
                "timestamp": datetime.now().isoformat(),
            }

    async def analyze_text(self, session: aiohttp.ClientSession) -> Dict[str, Any]:
        """Test the text analysis endpoint"""
        start_time = time.time()
        url = f"{self.base_url}/pipeline/analyze"
        
        lang = random.choice(list(TEST_TEXTS.keys()))
        text = self.get_random_text(lang)
        
        payload = {
            "text": text,
            "language": lang,
            "include_sentiment": True,
            "include_entities": True,
        }
        
        try:
            async with session.post(url, json=payload) as response:
                response_time = time.time() - start_time
                status = response.status
                try:
                    data = await response.json()
                except:
                    data = {"error": "Failed to parse JSON"}
                
                success = status == 200
                
                return {
                    "endpoint": "analyze",
                    "status": status,
                    "response_time": response_time,
                    "success": success,
                    "language": lang,
                    "text_length": len(text),
                    "timestamp": datetime.now().isoformat(),
                }
        
        except Exception as e:
            response_time = time.time() - start_time
            return {
                "endpoint": "analyze",
                "status": 0,
                "response_time": response_time,
                "success": False,
                "error": str(e),
                "language": lang,
                "text_length": len(text),
                "timestamp": datetime.now().isoformat(),
            }

    async def simplify_text(self, session: aiohttp.ClientSession) -> Dict[str, Any]:
        """Test the text simplification endpoint"""
        start_time = time.time()
        url = f"{self.base_url}/pipeline/simplify"
        
        text = self.get_random_text("en")  # Simplification works best in English
        
        payload = {
            "text": text,
            "language": "en",
            "target_level": random.choice(TARGET_LEVELS),
        }
        
        try:
            async with session.post(url, json=payload) as response:
                response_time = time.time() - start_time
                status = response.status
                try:
                    data = await response.json()
                except:
                    data = {"error": "Failed to parse JSON"}
                
                success = status == 200
                
                return {
                    "endpoint": "simplify",
                    "status": status,
                    "response_time": response_time,
                    "success": success,
                    "target_level": payload["target_level"],
                    "text_length": len(text),
                    "timestamp": datetime.now().isoformat(),
                }
        
        except Exception as e:
            response_time = time.time() - start_time
            return {
                "endpoint": "simplify",
                "status": 0,
                "response_time": response_time,
                "success": False,
                "error": str(e),
                "target_level": payload["target_level"],
                "text_length": len(text),
                "timestamp": datetime.now().isoformat(),
            }

    async def anonymize_text(self, session: aiohttp.ClientSession) -> Dict[str, Any]:
        """Test the text anonymization endpoint"""
        start_time = time.time()
        url = f"{self.base_url}/pipeline/anonymize"
        
        # Create text with PII
        pii_texts = [
            "My name is John Smith and my email is john.smith@example.com.",
            "Please contact Sarah Johnson at (555) 123-4567 or sarah@example.org.",
            "David Wilson lives at 123 Main Street, Boston, MA 02108.",
            "Social Security Number: 123-45-6789, Credit Card: 4111-1111-1111-1111.",
            "Patient ID: MRN-12345678, DOB: 01/15/1980."
        ]
        
        payload = {
            "text": random.choice(pii_texts),
            "language": "en",
            "strategy": "mask",
        }
        
        try:
            async with session.post(url, json=payload) as response:
                response_time = time.time() - start_time
                status = response.status
                try:
                    data = await response.json()
                except:
                    data = {"error": "Failed to parse JSON"}
                
                success = status == 200
                
                return {
                    "endpoint": "anonymize",
                    "status": status,
                    "response_time": response_time,
                    "success": success,
                    "strategy": payload["strategy"],
                    "text_length": len(payload["text"]),
                    "timestamp": datetime.now().isoformat(),
                }
        
        except Exception as e:
            response_time = time.time() - start_time
            return {
                "endpoint": "anonymize",
                "status": 0,
                "response_time": response_time,
                "success": False,
                "error": str(e),
                "strategy": payload["strategy"],
                "text_length": len(payload["text"]),
                "timestamp": datetime.now().isoformat(),
            }

    async def summarize_text(self, session: aiohttp.ClientSession) -> Dict[str, Any]:
        """Test the text summarization endpoint"""
        start_time = time.time()
        url = f"{self.base_url}/pipeline/summarize"
        
        # Create longer texts for summarization
        long_texts = [
            """Climate change is the long-term alteration of temperature and typical weather patterns in a place. 
            Climate change could refer to a particular location or the planet as a whole. Climate change may 
            cause weather patterns to be less predictable. These unexpected weather patterns can make it 
            difficult to maintain and grow crops in regions that rely on farming. Climate change has also been 
            connected with other damaging weather events such as more frequent and more intense hurricanes, 
            floods, downpours, and winter storms.""",
            
            """Artificial intelligence (AI) is intelligence—perceiving, synthesizing, and inferring information—demonstrated 
            by machines, as opposed to intelligence displayed by humans or by other animals. Example tasks in which 
            AI is applied include speech recognition, computer vision, translation between (natural) languages, 
            decision-making, and producing creative content such as images, text, music, and inventions.""",
            
            """Machine learning is a field of inquiry devoted to understanding and building methods that 'learn', 
            that is, methods that leverage data to improve performance on some set of tasks. It is considered 
            a part of artificial intelligence. Machine learning algorithms build a model based on sample data, 
            known as training data, in order to make predictions or decisions without being explicitly programmed to do so."""
        ]
        
        payload = {
            "text": random.choice(long_texts),
            "language": "en",
        }
        
        try:
            async with session.post(url, json=payload) as response:
                response_time = time.time() - start_time
                status = response.status
                try:
                    data = await response.json()
                except:
                    data = {"error": "Failed to parse JSON"}
                
                success = status == 200
                
                return {
                    "endpoint": "summarize",
                    "status": status,
                    "response_time": response_time,
                    "success": success,
                    "text_length": len(payload["text"]),
                    "timestamp": datetime.now().isoformat(),
                }
        
        except Exception as e:
            response_time = time.time() - start_time
            return {
                "endpoint": "summarize",
                "status": 0,
                "response_time": response_time,
                "success": False,
                "error": str(e),
                "text_length": len(payload["text"]),
                "timestamp": datetime.now().isoformat(),
            }

    async def mixed_operations(self, session: aiohttp.ClientSession) -> Dict[str, Any]:
        """Test a sequence of operations to simulate a real user workflow"""
        start_time = time.time()
        
        # First, detect language
        lang = random.choice(list(TEST_TEXTS.keys()))
        text = self.get_random_text(lang)
        
        detect_url = f"{self.base_url}/pipeline/detect"
        detect_payload = {
            "text": text,
            "detailed": True,
        }
        
        try:
            # Step 1: Detect language
            detect_start = time.time()
            async with session.post(detect_url, json=detect_payload) as detect_response:
                detect_time = time.time() - detect_start
                
                if detect_response.status != 200:
                    return {
                        "endpoint": "mixed",
                        "status": detect_response.status,
                        "response_time": time.time() - start_time,
                        "success": False,
                        "step_failed": "detect",
                        "timestamp": datetime.now().isoformat(),
                    }
                
                try:
                    detect_data = await detect_response.json()
                    detected = detect_data.get("language", "en")
                except:
                    detected = "en"
            
            # Step 2: Translate
            translate_url = f"{self.base_url}/pipeline/translate"
            target_lang = random.choice([l for l in LANGUAGES if l != detected])
            translate_payload = {
                "text": text,
                "source_language": detected,
                "target_language": target_lang,
            }
            
            translate_start = time.time()
            async with session.post(translate_url, json=translate_payload) as translate_response:
                translate_time = time.time() - translate_start
                
                if translate_response.status != 200:
                    return {
                        "endpoint": "mixed",
                        "status": translate_response.status,
                        "response_time": time.time() - start_time,
                        "success": False,
                        "step_failed": "translate",
                        "timestamp": datetime.now().isoformat(),
                    }
                
                try:
                    translate_data = await translate_response.json()
                    translated_text = translate_data.get("translated_text", text)
                except:
                    translated_text = text
            
            # Step 3: Analyze
            analyze_url = f"{self.base_url}/pipeline/analyze"
            analyze_payload = {
                "text": translated_text,
                "language": target_lang,
                "include_sentiment": True,
            }
            
            analyze_start = time.time()
            async with session.post(analyze_url, json=analyze_payload) as analyze_response:
                analyze_time = time.time() - analyze_start
                
                success = analyze_response.status == 200
                
                return {
                    "endpoint": "mixed",
                    "status": analyze_response.status,
                    "response_time": time.time() - start_time,
                    "success": success,
                    "detect_time": detect_time,
                    "translate_time": translate_time,
                    "analyze_time": analyze_time,
                    "source_lang": detected,
                    "target_lang": target_lang,
                    "text_length": len(text),
                    "timestamp": datetime.now().isoformat(),
                }
        
        except Exception as e:
            return {
                "endpoint": "mixed",
                "status": 0,
                "response_time": time.time() - start_time,
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    async def collect_system_metrics(self, active_connections: int = 0):
        """Collect system metrics during the test"""
        self.system_metrics["timestamp"].append(time.time())
        self.system_metrics["cpu_percent"].append(psutil.cpu_percent())
        self.system_metrics["memory_percent"].append(psutil.virtual_memory().percent)
        self.system_metrics["active_connections"].append(active_connections)

    async def worker(self, endpoint: str, semaphore: asyncio.Semaphore):
        """Worker function to make API calls"""
        # Get the appropriate endpoint function
        endpoint_functions = {
            "health": self.health_check,
            "translate": self.translate_text,
            "detect": self.detect_language,
            "analyze": self.analyze_text,
            "simplify": self.simplify_text,
            "anonymize": self.anonymize_text,
            "summarize": self.summarize_text,
            "mixed": self.mixed_operations,
        }
        
        func = endpoint_functions.get(endpoint, self.health_check)
        
        # Create a session for this worker
        timeout = aiohttp.ClientTimeout(total=60)  # 60 second timeout
        async with aiohttp.ClientSession(headers=self.get_headers(), timeout=timeout) as session:
            # Acquire semaphore to limit concurrency
            async with semaphore:
                # Make the API call
                result = await func(session)
                self.results.append(result)
                
                # Sleep a bit to avoid hammering the server
                await asyncio.sleep(random.uniform(0.1, 0.5))

    async def run_test(self, endpoint: str = "mixed", tick_interval: float = 0.5):
        """Run the load test"""
        # Register start time
        self.start_time = time.time()
        print(f"Starting load test on {endpoint} endpoint with {self.concurrency} concurrent users")
        print(f"Test will run for {self.duration} seconds (plus {self.ramp_up}s ramp-up, {self.cooldown}s cooldown)")
        
        # Create semaphore for limiting concurrency
        semaphore = asyncio.Semaphore(self.concurrency)
        
        # Task to collect system metrics
        async def metrics_collector():
            while time.time() - self.start_time < self.duration + self.ramp_up + self.cooldown:
                active = self.concurrency  # In a real system, you'd count active connections
                await self.collect_system_metrics(active)
                await asyncio.sleep(1)  # Collect every second
        
        # Start metrics collector
        metrics_task = asyncio.create_task(metrics_collector())
        
        # Create tasks progressively during ramp-up
        tasks = []
        with tqdm(total=self.duration, desc="Running test") as pbar:
            # Ramp-up phase
            ramp_up_end = self.start_time + self.ramp_up
            while time.time() < ramp_up_end:
                # Calculate how many tasks to add based on ramp-up progress
                progress = (time.time() - self.start_time) / self.ramp_up
                target_tasks = int(progress * self.concurrency)
                
                # Add tasks until we reach the target
                while len(tasks) < target_tasks:
                    task = asyncio.create_task(self.worker(endpoint, semaphore))
                    tasks.append(task)
                
                # Update progress bar
                pbar.update(tick_interval)
                await asyncio.sleep(tick_interval)
            
            # Steady state phase
            steady_end = ramp_up_end + self.duration
            while time.time() < steady_end:
                # Maintain constant number of tasks
                completed = [t for t in tasks if t.done()]
                for t in completed:
                    tasks.remove(t)
                
                # Add new tasks to replace completed ones
                while len(tasks) < self.concurrency:
                    task = asyncio.create_task(self.worker(endpoint, semaphore))
                    tasks.append(task)
                
                # Update progress bar
                pbar.update(tick_interval)
                await asyncio.sleep(tick_interval)
            
            # Cooldown phase
            cooldown_end = steady_end + self.cooldown
            while time.time() < cooldown_end:
                # Gradually reduce the number of tasks
                progress = (time.time() - steady_end) / self.cooldown
                target_tasks = int((1 - progress) * self.concurrency)
                
                # Remove tasks if we have too many
                while len(tasks) > target_tasks:
                    if tasks:
                        task = tasks.pop()
                        task.cancel()
                
                # Update progress bar
                pbar.update(tick_interval)
                await asyncio.sleep(tick_interval)
        
        # Wait for remaining tasks to complete
        if tasks:
            print(f"Waiting for {len(tasks)} remaining tasks to complete...")
            await asyncio.gather(*tasks, return_exceptions=True)
        
        # Wait for metrics collector to finish
        metrics_task.cancel()
        try:
            await metrics_task
        except asyncio.CancelledError:
            pass
        
        # Register end time
        self.end_time = time.time()
        print(f"Test completed in {self.end_time - self.start_time:.2f} seconds")

    def analyze_results(self) -> Dict[str, Any]:
        """Analyze the results"""
        if not self.results:
            return {"error": "No results to analyze"}
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(self.results)
        
        # Basic statistics
        stats = {
            "total_requests": len(df),
            "successful_requests": df["success"].sum(),
            "failed_requests": len(df) - df["success"].sum(),
            "success_rate": (df["success"].sum() / len(df)) * 100,
            "total_duration": self.end_time - self.start_time,
            "requests_per_second": len(df) / (self.end_time - self.start_time),
        }
        
        # Response time statistics
        stats.update({
            "response_time": {
                "min": df["response_time"].min(),
                "max": df["response_time"].max(),
                "mean": df["response_time"].mean(),
                "median": df["response_time"].median(),
                "p95": df["response_time"].quantile(0.95),
                "p99": df["response_time"].quantile(0.99),
            }
        })
        
        # Endpoint-specific statistics
        endpoint_stats = {}
        for endpoint in df["endpoint"].unique():
            endpoint_df = df[df["endpoint"] == endpoint]
            endpoint_stats[endpoint] = {
                "total_requests": len(endpoint_df),
                "successful_requests": endpoint_df["success"].sum(),
                "failed_requests": len(endpoint_df) - endpoint_df["success"].sum(),
                "success_rate": (endpoint_df["success"].sum() / len(endpoint_df)) * 100,
                "response_time": {
                    "min": endpoint_df["response_time"].min(),
                    "max": endpoint_df["response_time"].max(),
                    "mean": endpoint_df["response_time"].mean(),
                    "median": endpoint_df["response_time"].median(),
                    "p95": endpoint_df["response_time"].quantile(0.95),
                    "p99": endpoint_df["response_time"].quantile(0.99),
                }
            }
        
        stats["endpoints"] = endpoint_stats
        
        # Error analysis
        if "error" in df.columns:
            errors = df[~df["success"]]["error"].value_counts().to_dict()
            stats["errors"] = errors
        
        # System metrics
        if self.system_metrics["timestamp"]:
            stats["system_metrics"] = {
                "cpu_percent": {
                    "min": min(self.system_metrics["cpu_percent"]),
                    "max": max(self.system_metrics["cpu_percent"]),
                    "mean": np.mean(self.system_metrics["cpu_percent"]),
                },
                "memory_percent": {
                    "min": min(self.system_metrics["memory_percent"]),
                    "max": max(self.system_metrics["memory_percent"]),
                    "mean": np.mean(self.system_metrics["memory_percent"]),
                },
            }
        
        return stats

    def plot_results(self) -> Optional[List[Figure]]:
        """Generate plots of the results"""
        if not HAS_MATPLOTLIB:
            print("Matplotlib is not installed. Cannot generate plots.")
            return None
        
        if not self.results:
            return None
        
        figures = []
        
        # Convert to DataFrame for easier plotting
        df = pd.DataFrame(self.results)
        
        # Add relative timestamp for plotting
        if "timestamp" in df.columns:
            df["relative_time"] = pd.to_datetime(df["timestamp"]).astype(int) / 10**9
            df["relative_time"] = df["relative_time"] - df["relative_time"].min()
        
        # 1. Response time over time
        fig, ax = plt.subplots(figsize=(10, 6))
        for endpoint in df["endpoint"].unique():
            endpoint_df = df[df["endpoint"] == endpoint]
            if "relative_time" in endpoint_df.columns:
                ax.scatter(
                    endpoint_df["relative_time"],
                    endpoint_df["response_time"],
                    alpha=0.5,
                    label=endpoint,
                )
        
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Response Time (seconds)")
        ax.set_title("Response Time Over Time by Endpoint")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.7)
        figures.append(fig)
        
        # 2. Response time distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        for endpoint in df["endpoint"].unique():
            endpoint_df = df[df["endpoint"] == endpoint]
            ax.hist(
                endpoint_df["response_time"],
                bins=20,
                alpha=0.5,
                label=endpoint,
            )
        
        ax.set_xlabel("Response Time (seconds)")
        ax.set_ylabel("Frequency")
        ax.set_title("Response Time Distribution by Endpoint")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.7)
        figures.append(fig)
        
        # 3. Success rate by endpoint
        fig, ax = plt.subplots(figsize=(10, 6))
        success_rates = []
        endpoints = []
        
        for endpoint in df["endpoint"].unique():
            endpoint_df = df[df["endpoint"] == endpoint]
            success_rate = (endpoint_df["success"].sum() / len(endpoint_df)) * 100
            success_rates.append(success_rate)
            endpoints.append(endpoint)
        
        ax.bar(endpoints, success_rates)
        ax.set_xlabel("Endpoint")
        ax.set_ylabel("Success Rate (%)")
        ax.set_title("Success Rate by Endpoint")
        ax.set_ylim(0, 100)
        ax.grid(True, linestyle="--", alpha=0.7)
        for i, v in enumerate(success_rates):
            ax.text(i, v + 1, f"{v:.1f}%", ha="center")
        figures.append(fig)
        
        # 4. System metrics over time
        if self.system_metrics["timestamp"]:
            # Normalize timestamps to relative time
            relative_timestamps = [t - self.system_metrics["timestamp"][0] for t in self.system_metrics["timestamp"]]
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
            
            # CPU usage
            ax1.plot(relative_timestamps, self.system_metrics["cpu_percent"], "b-", label="CPU")
            ax1.set_ylabel("CPU Usage (%)")
            ax1.set_title("System Metrics During Test")
            ax1.grid(True, linestyle="--", alpha=0.7)
            ax1.legend(loc="upper right")
            
            # Memory usage
            ax2.plot(relative_timestamps, self.system_metrics["memory_percent"], "r-", label="Memory")
            ax2.set_xlabel("Time (seconds)")
            ax2.set_ylabel("Memory Usage (%)")
            ax2.grid(True, linestyle="--", alpha=0.7)
            ax2.legend(loc="upper right")
            
            fig.tight_layout()
            figures.append(fig)
        
        return figures

    def save_results(self, stats: Dict[str, Any], plots: Optional[List[Figure]] = None) -> str:
        """Save results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save raw results
        results_file = os.path.join(self.log_dir, f"load_test_raw_{timestamp}.csv")
        pd.DataFrame(self.results).to_csv(results_file, index=False)
        
        # Save statistics
        stats_file = os.path.join(self.log_dir, f"load_test_stats_{timestamp}.json")
        with open(stats_file, "w") as f:
            json.dump(stats, f, indent=2)
        
        # Save system metrics
        if self.system_metrics["timestamp"]:
            metrics_file = os.path.join(self.log_dir, f"load_test_metrics_{timestamp}.csv")
            metrics_df = pd.DataFrame(self.system_metrics)
            metrics_df["relative_time"] = metrics_df["timestamp"] - metrics_df["timestamp"].iloc[0]
            metrics_df.to_csv(metrics_file, index=False)
        
        # Save plots
        if plots and HAS_MATPLOTLIB:
            for i, fig in enumerate(plots):
                plot_file = os.path.join(self.log_dir, f"load_test_plot_{i+1}_{timestamp}.png")
                fig.savefig(plot_file, dpi=300, bbox_inches="tight")
        
        # Generate HTML report
        html_file = os.path.join(self.log_dir, f"load_test_report_{timestamp}.html")
        self._generate_html_report(html_file, stats, timestamp)
        
        return html_file

    def _generate_html_report(self, file_path: str, stats: Dict[str, Any], timestamp: str) -> None:
        """Generate HTML report"""
        # Simple HTML template
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>API Load Test Report - {timestamp}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                h1, h2, h3 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .card {{ background-color: #fff; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); padding: 20px; margin-bottom: 20px; }}
                .success {{ color: green; }}
                .failure {{ color: red; }}
                .plot {{ max-width: 100%; height: auto; margin: 20px 0; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>API Load Test Report</h1>
                <p>Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                
                <div class="card">
                    <h2>Test Configuration</h2>
                    <table>
                        <tr><th>Parameter</th><th>Value</th></tr>
                        <tr><td>Base URL</td><td>{self.base_url}</td></tr>
                        <tr><td>Concurrency</td><td>{self.concurrency}</td></tr>
                        <tr><td>Duration</td><td>{self.duration} seconds</td></tr>
                        <tr><td>Ramp-up</td><td>{self.ramp_up} seconds</td></tr>
                        <tr><td>Cooldown</td><td>{self.cooldown} seconds</td></tr>
                        <tr><td>Total Test Time</td><td>{stats.get("total_duration", "N/A"):.2f} seconds</td></tr>
                    </table>
                </div>
                
                <div class="card">
                    <h2>Summary</h2>
                    <table>
                        <tr><th>Metric</th><th>Value</th></tr>
                        <tr><td>Total Requests</td><td>{stats.get("total_requests", "N/A")}</td></tr>
                        <tr><td>Successful Requests</td><td>{stats.get("successful_requests", "N/A")}</td></tr>
                        <tr><td>Failed Requests</td><td>{stats.get("failed_requests", "N/A")}</td></tr>
                        <tr><td>Success Rate</td><td>{stats.get("success_rate", "N/A"):.2f}%</td></tr>
                        <tr><td>Requests Per Second</td><td>{stats.get("requests_per_second", "N/A"):.2f}</td></tr>
                    </table>
                </div>
                
                <div class="card">
                    <h2>Response Time Statistics (seconds)</h2>
                    <table>
                        <tr><th>Metric</th><th>Value</th></tr>
        """
        
        if "response_time" in stats:
            response_times = stats["response_time"]
            html += f"""
                        <tr><td>Minimum</td><td>{response_times.get("min", "N/A"):.4f}</td></tr>
                        <tr><td>Maximum</td><td>{response_times.get("max", "N/A"):.4f}</td></tr>
                        <tr><td>Mean</td><td>{response_times.get("mean", "N/A"):.4f}</td></tr>
                        <tr><td>Median</td><td>{response_times.get("median", "N/A"):.4f}</td></tr>
                        <tr><td>95th Percentile</td><td>{response_times.get("p95", "N/A"):.4f}</td></tr>
                        <tr><td>99th Percentile</td><td>{response_times.get("p99", "N/A"):.4f}</td></tr>
            """
        
        html += """
                    </table>
                </div>
                
                <div class="card">
                    <h2>Endpoint Performance</h2>
        """
        
        if "endpoints" in stats:
            for endpoint, endpoint_stats in stats["endpoints"].items():
                html += f"""
                    <h3>{endpoint}</h3>
                    <table>
                        <tr><th>Metric</th><th>Value</th></tr>
                        <tr><td>Total Requests</td><td>{endpoint_stats.get("total_requests", "N/A")}</td></tr>
                        <tr><td>Successful Requests</td><td>{endpoint_stats.get("successful_requests", "N/A")}</td></tr>
                        <tr><td>Failed Requests</td><td>{endpoint_stats.get("failed_requests", "N/A")}</td></tr>
                        <tr><td>Success Rate</td><td>{endpoint_stats.get("success_rate", "N/A"):.2f}%</td></tr>
                    </table>
                    
                    <h4>Response Time (seconds)</h4>
                    <table>
                        <tr><th>Metric</th><th>Value</th></tr>
                """
                
                if "response_time" in endpoint_stats:
                    response_times = endpoint_stats["response_time"]
                    html += f"""
                        <tr><td>Minimum</td><td>{response_times.get("min", "N/A"):.4f}</td></tr>
                        <tr><td>Maximum</td><td>{response_times.get("max", "N/A"):.4f}</td></tr>
                        <tr><td>Mean</td><td>{response_times.get("mean", "N/A"):.4f}</td></tr>
                        <tr><td>Median</td><td>{response_times.get("median", "N/A"):.4f}</td></tr>
                        <tr><td>95th Percentile</td><td>{response_times.get("p95", "N/A"):.4f}</td></tr>
                        <tr><td>99th Percentile</td><td>{response_times.get("p99", "N/A"):.4f}</td></tr>
                    """
                
                html += """
                    </table>
                """
        
        # System metrics
        if "system_metrics" in stats:
            html += """
                <div class="card">
                    <h2>System Metrics</h2>
                    <table>
                        <tr><th>Metric</th><th>Min</th><th>Max</th><th>Mean</th></tr>
            """
            
            cpu_metrics = stats["system_metrics"].get("cpu_percent", {})
            memory_metrics = stats["system_metrics"].get("memory_percent", {})
            
            html += f"""
                        <tr>
                            <td>CPU Usage (%)</td>
                            <td>{cpu_metrics.get("min", "N/A"):.2f}</td>
                            <td>{cpu_metrics.get("max", "N/A"):.2f}</td>
                            <td>{cpu_metrics.get("mean", "N/A"):.2f}</td>
                        </tr>
                        <tr>
                            <td>Memory Usage (%)</td>
                            <td>{memory_metrics.get("min", "N/A"):.2f}</td>
                            <td>{memory_metrics.get("max", "N/A"):.2f}</td>
                            <td>{memory_metrics.get("mean", "N/A"):.2f}</td>
                        </tr>
            """
            
            html += """
                    </table>
                </div>
            """
        
        # Error analysis
        if "errors" in stats and stats["errors"]:
            html += """
                <div class="card">
                    <h2>Error Analysis</h2>
                    <table>
                        <tr><th>Error</th><th>Count</th></tr>
            """
            
            for error, count in stats["errors"].items():
                html += f"""
                        <tr><td>{error}</td><td>{count}</td></tr>
                """
            
            html += """
                    </table>
                </div>
            """
        
        # Plots
        if HAS_MATPLOTLIB:
            html += """
                <div class="card">
                    <h2>Plots</h2>
            """
            
            # Get list of plot files
            plot_dir = os.path.dirname(file_path)
            base_name = os.path.basename(file_path).replace("report", "plot").rsplit("_", 1)[0]
            plot_files = [f for f in os.listdir(plot_dir) if f.startswith(base_name)]
            
            for plot_file in sorted(plot_files):
                plot_path = os.path.join(os.path.basename(plot_dir), plot_file)
                html += f"""
                    <img src="{plot_path}" alt="Performance Plot" class="plot">
                """
            
            html += """
                </div>
            """
        
        html += """
            </div>
        </body>
        </html>
        """
        
        with open(file_path, "w") as f:
            f.write(html)

    async def run(self, endpoint: str = "mixed"):
        """Run the load test and analyze results"""
        await self.run_test(endpoint)
        stats = self.analyze_results()
        plots = self.plot_results()
        report_file = self.save_results(stats, plots)
        
        print(f"\nLoad test completed!")
        print(f"Report saved to: {report_file}")
        
        # Print summary
        print("\nSummary:")
        print(f"Total Requests: {stats.get('total_requests', 'N/A')}")
        print(f"Success Rate: {stats.get('success_rate', 'N/A'):.2f}%")
        print(f"Requests/second: {stats.get('requests_per_second', 'N/A'):.2f}")
        
        if "response_time" in stats:
            print("\nResponse Time (seconds):")
            print(f"  Min: {stats['response_time'].get('min', 'N/A'):.4f}")
            print(f"  Max: {stats['response_time'].get('max', 'N/A'):.4f}")
            print(f"  Mean: {stats['response_time'].get('mean', 'N/A'):.4f}")
            print(f"  Median: {stats['response_time'].get('median', 'N/A'):.4f}")
            print(f"  95th Percentile: {stats['response_time'].get('p95', 'N/A'):.4f}")
        
        return stats


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="API Load Testing Tool")
    parser.add_argument("--endpoint", type=str, default="mixed",
                        choices=["health", "translate", "detect", "analyze", 
                                 "simplify", "anonymize", "summarize", "mixed"],
                        help="Endpoint to test")
    parser.add_argument("-c", "--concurrency", type=int, default=10,
                        help="Number of concurrent users")
    parser.add_argument("-d", "--duration", type=int, default=60,
                        help="Test duration in seconds")
    parser.add_argument("-r", "--ramp-up", type=int, default=5,
                        help="Ramp-up time in seconds")
    parser.add_argument("--cooldown", type=int, default=5,
                        help="Cooldown time in seconds")
    parser.add_argument("--url", type=str, default=BASE_URL,
                        help="Base URL for API")
    parser.add_argument("--api-key", type=str, default=API_KEY,
                        help="API Key for authentication")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output directory for reports")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Create load runner
    runner = APILoadRunner(
        base_url=args.url,
        api_key=args.api_key,
        concurrency=args.concurrency,
        duration=args.duration,
        ramp_up=args.ramp_up,
        cooldown=args.cooldown,
        log_dir=args.output,
    )
    
    # Run the test
    asyncio.run(runner.run(args.endpoint))