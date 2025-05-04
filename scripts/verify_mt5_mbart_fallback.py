#!/usr/bin/env python3
"""
Test script to verify the MT5-to-MBART fallback mechanism.

This script tests the translation pipeline by:
1. Sending translation requests with various language pairs
2. Verifying that MT5 fallback to MBART works correctly when needed
3. Comparing the quality of translations with and without fallback

Author: CasaLingua Team
"""

import sys
import os
import argparse
import json
import time
import requests
from typing import Dict, List, Optional, Tuple, Any
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("mt5_mbart_fallback_test")

# Test configuration
DEFAULT_API_URL = "http://localhost:8001"  # Default API URL
MT5_MODEL_NAME = "mt5_translation"        # MT5 model name
MBART_MODEL_NAME = "mbart_translation"    # MBART model name

# Test sentences for different languages
TEST_SENTENCES = {
    "en": [
        "The quick brown fox jumps over the lazy dog.",
        "I would like to book a flight to New York for next week.",
        "Can you recommend a good restaurant in the downtown area?"
    ],
    "es": [
        "El zorro marrón rápido salta sobre el perro perezoso.",
        "Me gustaría reservar un vuelo a Nueva York para la próxima semana.",
        "¿Puedes recomendar un buen restaurante en el centro de la ciudad?"
    ],
    "fr": [
        "Le rapide renard brun saute par-dessus le chien paresseux.",
        "Je voudrais réserver un vol pour New York pour la semaine prochaine.",
        "Pouvez-vous recommander un bon restaurant dans le centre-ville?"
    ],
    "de": [
        "Der schnelle braune Fuchs springt über den faulen Hund.",
        "Ich möchte einen Flug nach New York für nächste Woche buchen.",
        "Können Sie ein gutes Restaurant in der Innenstadt empfehlen?"
    ],
    "it": [
        "La rapida volpe marrone salta sopra il cane pigro.",
        "Vorrei prenotare un volo per New York per la prossima settimana.",
        "Puoi consigliare un buon ristorante nel centro della città?"
    ],
    "pt": [
        "A rápida raposa marrom pula sobre o cachorro preguiçoso.",
        "Eu gostaria de reservar um voo para Nova York para a próxima semana.",
        "Você pode recomendar um bom restaurante no centro da cidade?"
    ],
    "ru": [
        "Быстрая коричневая лиса прыгает через ленивую собаку.",
        "Я хотел бы забронировать рейс в Нью-Йорк на следующую неделю.",
        "Можете порекомендовать хороший ресторан в центре города?"
    ],
    "zh": [
        "快速的棕色狐狸跳过懒狗。",
        "我想预订下周去纽约的航班。",
        "您能推荐市中心的一家好餐厅吗？"
    ],
    "ja": [
        "素早い茶色のキツネは怠け者の犬を飛び越えます。",
        "来週ニューヨークへの便を予約したいです。",
        "ダウンタウンの良いレストランを推薦してもらえますか？"
    ]
}

# Language pairs to test (source -> target)
LANGUAGE_PAIRS = [
    ("en", "es"),  # English to Spanish
    ("en", "fr"),  # English to French
    ("en", "de"),  # English to German
    ("en", "it"),  # English to Italian
    ("en", "pt"),  # English to Portuguese
    ("en", "ru"),  # English to Russian
    ("en", "zh"),  # English to Chinese
    ("en", "ja"),  # English to Japanese
    ("es", "en"),  # Spanish to English
    ("fr", "en"),  # French to English
    ("de", "en"),  # German to English
    ("it", "en"),  # Italian to English
    ("pt", "en"),  # Portuguese to English
    ("ru", "en"),  # Russian to English
    ("zh", "en"),  # Chinese to English
    ("ja", "en"),  # Japanese to English
    ("es", "fr"),  # Spanish to French
    ("fr", "es"),  # French to Spanish
    ("de", "fr"),  # German to French
    ("it", "es"),  # Italian to Spanish
]

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test MT5-to-MBART fallback mechanism")
    parser.add_argument("--api-url", type=str, default=DEFAULT_API_URL,
                        help=f"API base URL (default: {DEFAULT_API_URL})")
    parser.add_argument("--api-key", type=str, default=None,
                        help="API key for authentication")
    parser.add_argument("--disable-fallback", action="store_true",
                        help="Disable fallback mechanism to compare quality")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Set logging level")
    parser.add_argument("--output", type=str, default="fallback_test_results.json",
                        help="Output file for test results")
    parser.add_argument("--retry", type=int, default=3,
                        help="Number of retries for HTTP requests")
    parser.add_argument("--timeout", type=int, default=30,
                        help="Timeout for HTTP requests in seconds")
    parser.add_argument("--pairs", type=str, nargs="+",
                        help="Specific language pairs to test (e.g., 'en-es fr-en')")
    
    return parser.parse_args()

def setup_session(api_key: Optional[str] = None) -> requests.Session:
    """Set up a session with authorization headers if needed."""
    session = requests.Session()
    if api_key:
        session.headers.update({"Authorization": f"Bearer {api_key}"})
    return session

def translate_text(
    session: requests.Session,
    api_url: str,
    text: str,
    source_lang: str,
    target_lang: str,
    model_name: Optional[str] = None,
    use_fallback: bool = True,
    timeout: int = 30
) -> Dict[str, Any]:
    """
    Call the translation API to translate text.
    
    Args:
        session: Requests session
        api_url: Base API URL
        text: Text to translate
        source_lang: Source language code
        target_lang: Target language code
        model_name: Specific model to use
        use_fallback: Whether to allow fallback to MBART
        timeout: Request timeout in seconds
        
    Returns:
        Dictionary with translation result or error
    """
    endpoint = f"{api_url}/pipeline/translate"
    
    payload = {
        "text": text,
        "source_language": source_lang,
        "target_language": target_lang,
        "preserve_formatting": True,
    }
    
    if model_name:
        payload["model_name"] = model_name
        
    # Add flag to control fallback if needed
    if not use_fallback:
        payload["parameters"] = {"disable_fallback": True}
        
    try:
        response = session.post(endpoint, json=payload, timeout=timeout)
        response.raise_for_status()
        
        # Parse response
        result = response.json()
        if "data" in result:
            return result["data"]
        else:
            return {
                "error": "Invalid response format",
                "response": result
            }
            
    except requests.RequestException as e:
        logger.error(f"API request failed: {str(e)}")
        return {
            "error": str(e),
            "status": getattr(e.response, "status_code", None) if hasattr(e, "response") else None
        }
        
def evaluate_translation(
    original: str,
    translation: str,
    metrics: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Perform basic evaluation of translation quality.
    
    Args:
        original: Original text
        translation: Translated text
        metrics: Metrics from the translation API
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Check for typical MT5 issues
    has_special_tokens = "<extra_id" in translation or "</s>" in translation or "<pad>" in translation
    
    # Calculate length ratio (very rough quality indicator)
    src_length = len(original)
    tgt_length = len(translation)
    length_ratio = tgt_length / max(1, src_length)
    
    # Check for repeated patterns (indicating hallucination)
    words = translation.split()
    repeated_patterns = 0
    if len(words) >= 4:
        for i in range(len(words) - 3):
            pattern = " ".join(words[i:i+2])
            if pattern in " ".join(words[i+2:]):
                repeated_patterns += 1
    
    return {
        "has_special_tokens": has_special_tokens,
        "repeated_patterns": repeated_patterns,
        "length_ratio": length_ratio,
        "confidence": metrics.get("confidence", 0.0),
        "quality_issues": has_special_tokens or repeated_patterns > 2 or length_ratio < 0.2 or length_ratio > 5.0,
        "used_fallback": metrics.get("used_fallback", False),
        "fallback_model": metrics.get("fallback_model")
    }

def format_results_report(results: List[Dict[str, Any]]) -> str:
    """Format test results into a readable report."""
    report = "MT5 to MBART Fallback Test Results\n"
    report += "================================\n\n"
    
    # Count successful and failed tests
    total_tests = len(results)
    success_count = sum(1 for r in results if not r.get("error"))
    fallback_count = sum(1 for r in results if r.get("evaluation", {}).get("used_fallback", False))
    
    # Summary
    report += f"Total tests: {total_tests}\n"
    report += f"Successful translations: {success_count}\n"
    report += f"Fallback to MBART used: {fallback_count}\n\n"
    
    # Group by language pair
    by_lang_pair = {}
    for result in results:
        pair = f"{result['source_lang']}-{result['target_lang']}"
        if pair not in by_lang_pair:
            by_lang_pair[pair] = []
        by_lang_pair[pair].append(result)
    
    # Report by language pair
    for pair, pair_results in by_lang_pair.items():
        report += f"Language Pair: {pair}\n"
        report += "-" * 50 + "\n"
        
        pair_success = sum(1 for r in pair_results if not r.get("error"))
        pair_fallback = sum(1 for r in pair_results if r.get("evaluation", {}).get("used_fallback", False))
        
        report += f"  Tests: {len(pair_results)}, Successful: {pair_success}, Used Fallback: {pair_fallback}\n\n"
        
        for i, result in enumerate(pair_results):
            report += f"  Test {i+1}: "
            
            if result.get("error"):
                report += f"ERROR: {result['error']}\n"
                continue
                
            report += "SUCCESS\n"
            report += f"    Original: {result['original'][:60]}{'...' if len(result['original']) > 60 else ''}\n"
            report += f"    Translation: {result['translation'][:60]}{'...' if len(result['translation']) > 60 else ''}\n"
            
            eval_data = result.get("evaluation", {})
            if eval_data.get("used_fallback"):
                report += f"    Used fallback: Yes (model: {eval_data.get('fallback_model', 'unknown')})\n"
            else:
                report += f"    Used fallback: No\n"
                
            if eval_data.get("quality_issues"):
                report += f"    Quality issues detected: Yes\n"
                if eval_data.get("has_special_tokens"):
                    report += f"      - Contains special tokens\n"
                if eval_data.get("repeated_patterns", 0) > 2:
                    report += f"      - Contains repeated patterns ({eval_data.get('repeated_patterns')})\n"
                if eval_data.get("length_ratio", 1.0) < 0.2 or eval_data.get("length_ratio", 1.0) > 5.0:
                    report += f"      - Unusual length ratio: {eval_data.get('length_ratio', 1.0):.2f}\n"
            
            report += "\n"
        
        report += "\n"
    
    return report

def run_tests(args):
    """Run all fallback mechanism tests."""
    logger.info("Starting MT5-to-MBART fallback mechanism tests")
    
    # Set up HTTP session
    session = setup_session(args.api_key)
    
    # Parse language pairs
    if args.pairs:
        pairs = []
        for pair_str in args.pairs:
            if "-" in pair_str:
                src, tgt = pair_str.split("-")
                pairs.append((src, tgt))
            else:
                logger.warning(f"Invalid language pair format: {pair_str}, skipping")
        if pairs:
            language_pairs = pairs
        else:
            language_pairs = LANGUAGE_PAIRS
    else:
        language_pairs = LANGUAGE_PAIRS
    
    logger.info(f"Testing {len(language_pairs)} language pairs with fallback {'disabled' if args.disable_fallback else 'enabled'}")
    
    # Store all test results
    all_results = []
    
    # Run tests for each language pair
    for src_lang, tgt_lang in language_pairs:
        # Get test sentences for source language
        if src_lang not in TEST_SENTENCES:
            logger.warning(f"No test sentences for {src_lang}, skipping {src_lang}-{tgt_lang}")
            continue
            
        sentences = TEST_SENTENCES[src_lang]
        logger.info(f"Testing {src_lang} to {tgt_lang} translation with {len(sentences)} sentences")
        
        for i, sentence in enumerate(sentences):
            logger.info(f"  Translating sentence {i+1}/{len(sentences)}")
            
            # Try multiple times in case of failures
            for attempt in range(args.retry):
                try:
                    # Perform translation with MT5 model
                    result = translate_text(
                        session=session,
                        api_url=args.api_url,
                        text=sentence,
                        source_lang=src_lang,
                        target_lang=tgt_lang,
                        model_name=MT5_MODEL_NAME,
                        use_fallback=not args.disable_fallback,
                        timeout=args.timeout
                    )
                    
                    # Check for errors
                    if "error" in result:
                        if attempt < args.retry - 1:
                            logger.warning(f"  Attempt {attempt+1} failed: {result['error']}, retrying...")
                            time.sleep(1)  # Small delay before retry
                            continue
                        else:
                            logger.error(f"  All {args.retry} attempts failed for {src_lang}-{tgt_lang}")
                            all_results.append({
                                "original": sentence,
                                "source_lang": src_lang,
                                "target_lang": tgt_lang,
                                "error": result["error"]
                            })
                            break
                    
                    # Get the translated text
                    translation = result.get("translated_text", "")
                    
                    # Evaluate the translation
                    evaluation = evaluate_translation(sentence, translation, result)
                    
                    # Log results
                    if evaluation.get("used_fallback"):
                        logger.info(f"  Used fallback for {src_lang}-{tgt_lang}: Yes")
                    else:
                        logger.info(f"  Used fallback for {src_lang}-{tgt_lang}: No")
                    
                    # Add to results
                    all_results.append({
                        "original": sentence,
                        "translation": translation,
                        "source_lang": src_lang,
                        "target_lang": tgt_lang,
                        "model": result.get("model_used", MT5_MODEL_NAME),
                        "evaluation": evaluation
                    })
                    
                    # Success, break retry loop
                    break
                    
                except Exception as e:
                    logger.error(f"  Unexpected error: {str(e)}")
                    if attempt < args.retry - 1:
                        logger.warning(f"  Retrying...")
                        time.sleep(1)
                    else:
                        all_results.append({
                            "original": sentence,
                            "source_lang": src_lang,
                            "target_lang": tgt_lang,
                            "error": str(e)
                        })
    
    # Save results to file
    if args.output:
        try:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)
            logger.info(f"Results saved to {args.output}")
        except Exception as e:
            logger.error(f"Failed to save results: {str(e)}")
    
    # Generate and print report
    report = format_results_report(all_results)
    print("\n" + report)
    
    # Save report
    report_path = args.output.replace(".json", ".txt")
    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        logger.info(f"Report saved to {report_path}")
    except Exception as e:
        logger.error(f"Failed to save report: {str(e)}")
    
    return all_results

if __name__ == "__main__":
    args = parse_arguments()
    
    # Set log level
    logger.setLevel(getattr(logging, args.log_level))
    
    try:
        run_tests(args)
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unhandled exception: {str(e)}", exc_info=True)
        sys.exit(1)