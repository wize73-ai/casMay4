#!/usr/bin/env python3
"""
API Load Testing for CasaLingua

This script performs load testing on CasaLingua API endpoints using Locust.
It simulates various user behaviors like translation, language detection,
and other NLP tasks with different concurrency patterns.

Run with:
    locust -f test_api_load.py

Or headless mode with:
    locust -f test_api_load.py --headless -u 100 -r 10 -t 5m
"""

import json
import os
import random
import time
from typing import Dict, List, Optional, Union

import locust
from locust import HttpUser, TaskSet, between, tag, task

# Common test data
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

# Test configuration
API_KEY = os.environ.get("CASALINGUA_API_KEY", "cslg_8f4b2d1e7a3c5b9e2f1d8a7c4b2e5f9d")
BASE_URL = os.environ.get("CASALINGUA_API_URL", "http://localhost:8000")


class CasaLinguaAPI(TaskSet):
    """Task set for CasaLingua API endpoints"""

    def on_start(self):
        """Set up the test session"""
        self.client.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-API-Key": API_KEY,
        }

    def get_random_text(self, language: str = "en") -> str:
        """Get a random test text in the specified language"""
        # If we don't have texts for this language, fall back to English
        if language not in TEST_TEXTS:
            language = "en"
        return random.choice(TEST_TEXTS[language])

    def get_language_pair(self) -> tuple:
        """Get a random language pair for translation"""
        source = random.choice(LANGUAGES)
        target = random.choice([lang for lang in LANGUAGES if lang != source])
        return source, target

    @tag("health")
    @task(1)
    def health_check(self):
        """Test the basic health check endpoint"""
        with self.client.get("/health", catch_response=True, name="/health") as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Health check failed: {response.status_code}")

    @tag("health")
    @task(1)
    def health_detailed(self):
        """Test the detailed health check endpoint"""
        with self.client.get("/health/detailed", catch_response=True, name="/health/detailed") as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Detailed health check failed: {response.status_code}")

    @tag("translation")
    @task(10)
    def translate_text(self):
        """Test the translation endpoint"""
        source_lang, target_lang = self.get_language_pair()
        payload = {
            "text": self.get_random_text(source_lang),
            "source_language": source_lang,
            "target_language": target_lang,
            "preserve_formatting": True,
        }
        
        with self.client.post(
            "/pipeline/translate",
            json=payload,
            catch_response=True,
            name="/pipeline/translate",
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Translation failed: {response.status_code} - {response.text}")

    @tag("language_detection")
    @task(5)
    def detect_language(self):
        """Test the language detection endpoint"""
        # Randomly select a language and text
        lang = random.choice(list(TEST_TEXTS.keys()))
        text = self.get_random_text(lang)
        
        payload = {
            "text": text,
            "detailed": random.choice([True, False]),
        }
        
        with self.client.post(
            "/pipeline/detect",
            json=payload,
            catch_response=True,
            name="/pipeline/detect",
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Language detection failed: {response.status_code} - {response.text}")

    @tag("analysis")
    @task(3)
    def analyze_text(self):
        """Test the text analysis endpoint"""
        lang = random.choice(list(TEST_TEXTS.keys()))
        text = self.get_random_text(lang)
        
        payload = {
            "text": text,
            "language": lang,
            "include_sentiment": True,
            "include_entities": True,
        }
        
        with self.client.post(
            "/pipeline/analyze",
            json=payload,
            catch_response=True,
            name="/pipeline/analyze",
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Text analysis failed: {response.status_code} - {response.text}")

    @tag("simplification")
    @task(3)
    def simplify_text(self):
        """Test the text simplification endpoint"""
        text = self.get_random_text("en")  # Simplification works best in English
        
        payload = {
            "text": text,
            "language": "en",
            "target_level": random.choice(TARGET_LEVELS),
        }
        
        with self.client.post(
            "/pipeline/simplify",
            json=payload,
            catch_response=True,
            name="/pipeline/simplify",
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Text simplification failed: {response.status_code} - {response.text}")

    @tag("anonymization")
    @task(3)
    def anonymize_text(self):
        """Test the text anonymization endpoint"""
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
        
        with self.client.post(
            "/pipeline/anonymize",
            json=payload,
            catch_response=True,
            name="/pipeline/anonymize",
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Text anonymization failed: {response.status_code} - {response.text}")

    @tag("summarization")
    @task(2)
    def summarize_text(self):
        """Test the text summarization endpoint"""
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
        
        with self.client.post(
            "/pipeline/summarize",
            json=payload,
            catch_response=True,
            name="/pipeline/summarize",
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Text summarization failed: {response.status_code} - {response.text}")

    @tag("mixed_workload")
    @task(5)
    def mixed_operations(self):
        """Simulate a user making multiple API calls in sequence"""
        # First detect language
        lang = random.choice(list(TEST_TEXTS.keys()))
        text = self.get_random_text(lang)
        
        detect_payload = {
            "text": text,
            "detailed": True,
        }
        
        with self.client.post(
            "/pipeline/detect",
            json=detect_payload,
            catch_response=True,
            name="/pipeline/detect (mixed)",
        ) as response:
            if response.status_code != 200:
                response.failure(f"Language detection failed: {response.status_code}")
                return
            
            try:
                detected = response.json().get("language", "en")
            except json.JSONDecodeError:
                detected = "en"
        
        # Then translate it
        target_lang = random.choice([l for l in LANGUAGES if l != detected])
        translate_payload = {
            "text": text,
            "source_language": detected,
            "target_language": target_lang,
        }
        
        with self.client.post(
            "/pipeline/translate",
            json=translate_payload,
            catch_response=True,
            name="/pipeline/translate (mixed)",
        ) as response:
            if response.status_code != 200:
                response.failure(f"Translation failed: {response.status_code}")
                return
            
            try:
                translated_text = response.json().get("translated_text", text)
            except json.JSONDecodeError:
                translated_text = text
        
        # Finally, analyze the translated text
        analyze_payload = {
            "text": translated_text,
            "language": target_lang,
            "include_sentiment": True,
        }
        
        with self.client.post(
            "/pipeline/analyze",
            json=analyze_payload,
            catch_response=True,
            name="/pipeline/analyze (mixed)",
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Text analysis failed: {response.status_code}")


class CasaLinguaUser(HttpUser):
    tasks = [CasaLinguaAPI]
    host = BASE_URL
    # Adjust wait time between requests (1-5 seconds)
    wait_time = between(1, 5)


# Standalone execution for local testing without Locust UI
if __name__ == "__main__":
    import argparse
    from locust.env import Environment
    from locust.stats import stats_printer
    from locust.log import setup_logging
    import gevent
    
    setup_logging("INFO", None)
    
    parser = argparse.ArgumentParser(description="Run load tests without Locust UI")
    parser.add_argument("--users", "-u", type=int, default=10, help="Number of concurrent users")
    parser.add_argument("--spawn-rate", "-r", type=float, default=2, help="User spawn rate")
    parser.add_argument("--time", "-t", type=int, default=60, help="Test duration in seconds")
    parser.add_argument("--host", type=str, default=BASE_URL, help="Host to test")
    parser.add_argument("--tags", type=str, help="Comma-separated list of tags to include")
    args = parser.parse_args()
    
    # Manually create an Environment with the specified User class
    env = Environment(user_classes=[CasaLinguaUser])
    env.create_local_runner()
    
    # Configure the environment
    env.host = args.host
    if args.tags:
        env.tags = set(args.tags.split(","))
    
    # Start the test by spawning users
    env.runner.start(args.users, spawn_rate=args.spawn_rate)
    
    # Print stats during the test
    gevent.spawn(stats_printer(env.stats))
    
    # Stop the test after the specified time
    gevent.spawn_later(args.time, lambda: env.runner.quit())
    
    # Wait for the gevent tasks to complete
    env.runner.greenlet.join()
    
    # Print final stats
    print("\nFinal Statistics:")
    env.stats.log_stats()