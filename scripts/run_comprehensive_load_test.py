#!/usr/bin/env python3
"""
Comprehensive Load Testing for CasaLingua API

This script runs a series of load tests across different endpoints with
varying concurrency levels to analyze system performance under different
conditions. It generates a combined report with recommendations.
"""

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Import the APILoadRunner from the api_load_runner module
from tests.api_load_runner import APILoadRunner

# Try to import visualization libraries
try:
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    HAS_VISUALIZATION = True
except ImportError:
    HAS_VISUALIZATION = False

# Configuration
BASE_URL = os.environ.get("CASALINGUA_API_URL", "http://localhost:8000")
API_KEY = os.environ.get("CASALINGUA_API_KEY", "cslg_8f4b2d1e7a3c5b9e2f1d8a7c4b2e5f9d")
LOG_DIR = os.path.join(parent_dir, "logs", "load_tests")

# Test scenarios
DEFAULT_SCENARIOS = [
    {
        "name": "baseline",
        "endpoint": "health", 
        "concurrency": 5, 
        "duration": 30,
        "description": "Baseline test with health endpoint"
    },
    {
        "name": "light_translation",
        "endpoint": "translate", 
        "concurrency": 5, 
        "duration": 60,
        "description": "Light load on translation endpoint"
    },
    {
        "name": "medium_translation",
        "endpoint": "translate", 
        "concurrency": 20, 
        "duration": 60,
        "description": "Medium load on translation endpoint"
    },
    {
        "name": "heavy_translation",
        "endpoint": "translate", 
        "concurrency": 50, 
        "duration": 120,
        "description": "Heavy load on translation endpoint"
    },
    {
        "name": "light_mixed",
        "endpoint": "mixed", 
        "concurrency": 10, 
        "duration": 60,
        "description": "Light mixed workload simulating real usage"
    },
    {
        "name": "medium_mixed",
        "endpoint": "mixed", 
        "concurrency": 30, 
        "duration": 120,
        "description": "Medium mixed workload simulating real usage"
    },
    {
        "name": "language_detection",
        "endpoint": "detect", 
        "concurrency": 20, 
        "duration": 60,
        "description": "Test focused on language detection"
    },
    {
        "name": "text_analysis",
        "endpoint": "analyze", 
        "concurrency": 15, 
        "duration": 60,
        "description": "Test focused on text analysis"
    },
    {
        "name": "text_simplification",
        "endpoint": "simplify", 
        "concurrency": 10, 
        "duration": 60,
        "description": "Test focused on text simplification"
    }
]


async def run_scenario(scenario: Dict[str, Any], output_dir: str) -> Dict[str, Any]:
    """Run a single test scenario and return the results"""
    print(f"\n{'=' * 80}")
    print(f"Running scenario: {scenario['name']}")
    print(f"Endpoint: {scenario['endpoint']}, Concurrency: {scenario['concurrency']}, Duration: {scenario['duration']}s")
    print(f"{'=' * 80}")
    
    # Create a subdirectory for this scenario
    scenario_dir = os.path.join(output_dir, scenario["name"])
    os.makedirs(scenario_dir, exist_ok=True)
    
    # Create and run the load runner
    runner = APILoadRunner(
        base_url=BASE_URL,
        api_key=API_KEY,
        concurrency=scenario["concurrency"],
        duration=scenario["duration"],
        ramp_up=5,
        cooldown=5,
        log_dir=scenario_dir,
    )
    
    # Run the test
    await runner.run_test(scenario["endpoint"])
    
    # Analyze results
    stats = runner.analyze_results()
    
    # Save results
    if HAS_VISUALIZATION:
        plots = runner.plot_results()
        report_file = runner.save_results(stats, plots)
    else:
        report_file = runner.save_results(stats)
    
    # Add scenario info to stats
    stats["scenario"] = scenario
    stats["report_file"] = report_file
    
    return stats


async def run_comprehensive_test(scenarios: List[Dict[str, Any]], output_dir: str) -> Dict[str, Any]:
    """Run all test scenarios and generate a comprehensive report"""
    # Create the output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Run all scenarios
    all_results = []
    for scenario in scenarios:
        result = await run_scenario(scenario, output_dir)
        all_results.append(result)
        
        # Short pause between scenarios to let system recover
        print(f"Waiting 10 seconds before next scenario...")
        await asyncio.sleep(10)
    
    # Analyze combined results
    combined_analysis = analyze_combined_results(all_results)
    
    # Generate combined report
    report_file = generate_combined_report(all_results, combined_analysis, output_dir)
    
    return {
        "results": all_results,
        "analysis": combined_analysis,
        "report_file": report_file
    }


def analyze_combined_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze combined results from all scenarios"""
    analysis = {
        "bottlenecks": [],
        "performance_summary": {},
        "recommendations": []
    }
    
    # Performance summary
    total_requests = sum(result.get("total_requests", 0) for result in results)
    total_failed = sum(result.get("failed_requests", 0) for result in results)
    
    if total_requests > 0:
        overall_success_rate = ((total_requests - total_failed) / total_requests) * 100
    else:
        overall_success_rate = 0
    
    analysis["performance_summary"] = {
        "total_scenarios": len(results),
        "total_requests": total_requests,
        "total_failed_requests": total_failed,
        "overall_success_rate": overall_success_rate,
    }
    
    # Find slowest endpoints
    endpoint_metrics = {}
    for result in results:
        if "endpoints" in result:
            for endpoint, stats in result["endpoints"].items():
                if endpoint not in endpoint_metrics:
                    endpoint_metrics[endpoint] = []
                
                if "response_time" in stats and "mean" in stats["response_time"]:
                    endpoint_metrics[endpoint].append(stats["response_time"]["mean"])
    
    # Calculate average response time for each endpoint
    for endpoint, times in endpoint_metrics.items():
        if times:
            avg_time = sum(times) / len(times)
            if avg_time > 1.0:  # If average response time is over 1 second
                analysis["bottlenecks"].append({
                    "endpoint": endpoint,
                    "avg_response_time": avg_time,
                    "severity": "high" if avg_time > 2.0 else "medium"
                })
    
    # Check for error patterns
    error_patterns = {}
    for result in results:
        if "errors" in result:
            for error, count in result["errors"].items():
                if error not in error_patterns:
                    error_patterns[error] = 0
                error_patterns[error] += count
    
    # Add frequent errors to bottlenecks
    for error, count in error_patterns.items():
        if count > 10:  # If an error occurs more than 10 times
            analysis["bottlenecks"].append({
                "type": "error",
                "error": error,
                "count": count,
                "severity": "high" if count > 50 else "medium"
            })
    
    # Generate recommendations
    if analysis["bottlenecks"]:
        for bottleneck in analysis["bottlenecks"]:
            if "endpoint" in bottleneck:
                endpoint = bottleneck["endpoint"]
                time = bottleneck["avg_response_time"]
                
                if endpoint == "translate":
                    analysis["recommendations"].append({
                        "component": endpoint,
                        "issue": f"Slow response time ({time:.2f}s)",
                        "suggestion": "Consider optimizing the translation model or implementing caching for common translations"
                    })
                elif endpoint == "analyze":
                    analysis["recommendations"].append({
                        "component": endpoint,
                        "issue": f"Slow response time ({time:.2f}s)",
                        "suggestion": "Review the analysis pipeline for optimizations, consider parallel processing"
                    })
                else:
                    analysis["recommendations"].append({
                        "component": endpoint,
                        "issue": f"Slow response time ({time:.2f}s)",
                        "suggestion": "Profile the endpoint to identify bottlenecks"
                    })
            
            if "type" in bottleneck and bottleneck["type"] == "error":
                error = bottleneck["error"]
                analysis["recommendations"].append({
                    "component": "error_handling",
                    "issue": f"Frequent error: {error}",
                    "suggestion": "Implement better error handling and retry logic"
                })
    
    # Check for scaling issues
    concurrency_performance = {}
    for result in results:
        scenario = result.get("scenario", {})
        if scenario.get("endpoint") == "translate":
            concurrency = scenario.get("concurrency", 0)
            if "response_time" in result and "mean" in result["response_time"]:
                if "translate" not in concurrency_performance:
                    concurrency_performance["translate"] = []
                
                concurrency_performance["translate"].append({
                    "concurrency": concurrency,
                    "response_time": result["response_time"]["mean"],
                    "success_rate": result.get("success_rate", 0)
                })
    
    # Analyze scaling efficiency
    for endpoint, data in concurrency_performance.items():
        if len(data) > 1:
            # Sort by concurrency
            data.sort(key=lambda x: x["concurrency"])
            
            # Check if response time increases disproportionately with concurrency
            base_concurrency = data[0]["concurrency"]
            base_response_time = data[0]["response_time"]
            
            for entry in data[1:]:
                concurrency_ratio = entry["concurrency"] / base_concurrency
                response_time_ratio = entry["response_time"] / base_response_time
                
                # If response time grows faster than 1.5x the concurrency ratio, there's a scaling issue
                if response_time_ratio > concurrency_ratio * 1.5:
                    analysis["bottlenecks"].append({
                        "type": "scaling",
                        "endpoint": endpoint,
                        "base_concurrency": base_concurrency,
                        "high_concurrency": entry["concurrency"],
                        "response_time_ratio": response_time_ratio,
                        "concurrency_ratio": concurrency_ratio,
                        "severity": "high"
                    })
                    
                    analysis["recommendations"].append({
                        "component": f"{endpoint}_scaling",
                        "issue": f"Poor scaling efficiency when concurrency increases from {base_concurrency} to {entry['concurrency']}",
                        "suggestion": "Implement connection pooling, optimize model loading, or add caching"
                    })
    
    # General recommendations
    if total_failed > 0 and total_requests > 0:
        failure_rate = (total_failed / total_requests) * 100
        if failure_rate > 5:
            analysis["recommendations"].append({
                "component": "reliability",
                "issue": f"High failure rate: {failure_rate:.2f}%",
                "suggestion": "Implement retry logic and improve error handling"
            })
    
    return analysis


def generate_combined_report(
    results: List[Dict[str, Any]], 
    analysis: Dict[str, Any], 
    output_dir: str
) -> str:
    """Generate a combined HTML report for all scenarios"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = os.path.join(output_dir, f"comprehensive_load_test_report_{timestamp}.html")
    
    # Simple HTML template
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Comprehensive API Load Test Report - {timestamp}</title>
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
            .warning {{ color: orange; }}
            .high {{ color: red; font-weight: bold; }}
            .medium {{ color: orange; }}
            .low {{ color: green; }}
            .plot {{ max-width: 100%; height: auto; margin: 20px 0; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Comprehensive API Load Test Report</h1>
            <p>Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            
            <div class="card">
                <h2>Executive Summary</h2>
                <p>This report presents the results of a comprehensive load testing of the CasaLingua API across different endpoints and load scenarios.</p>
                
                <h3>Performance Summary</h3>
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Total Scenarios Tested</td><td>{analysis["performance_summary"].get("total_scenarios", "N/A")}</td></tr>
                    <tr><td>Total Requests</td><td>{analysis["performance_summary"].get("total_requests", "N/A")}</td></tr>
                    <tr><td>Total Failed Requests</td><td>{analysis["performance_summary"].get("total_failed_requests", "N/A")}</td></tr>
                    <tr><td>Overall Success Rate</td><td>{analysis["performance_summary"].get("overall_success_rate", "N/A"):.2f}%</td></tr>
                </table>
            </div>
    """
    
    # Performance bottlenecks
    if analysis["bottlenecks"]:
        html += """
            <div class="card">
                <h2>Performance Bottlenecks</h2>
                <table>
                    <tr><th>Type</th><th>Component</th><th>Details</th><th>Severity</th></tr>
        """
        
        for bottleneck in analysis["bottlenecks"]:
            severity_class = bottleneck.get("severity", "medium")
            
            if "endpoint" in bottleneck:
                html += f"""
                    <tr>
                        <td>Slow endpoint</td>
                        <td>{bottleneck.get("endpoint", "N/A")}</td>
                        <td>Average response time: {bottleneck.get("avg_response_time", "N/A"):.4f}s</td>
                        <td class="{severity_class}">{bottleneck.get("severity", "medium").upper()}</td>
                    </tr>
                """
            elif "type" in bottleneck and bottleneck["type"] == "error":
                html += f"""
                    <tr>
                        <td>Error pattern</td>
                        <td>Error handling</td>
                        <td>"{bottleneck.get("error", "Unknown error")}" occurred {bottleneck.get("count", 0)} times</td>
                        <td class="{severity_class}">{bottleneck.get("severity", "medium").upper()}</td>
                    </tr>
                """
            elif "type" in bottleneck and bottleneck["type"] == "scaling":
                html += f"""
                    <tr>
                        <td>Scaling issue</td>
                        <td>{bottleneck.get("endpoint", "N/A")}</td>
                        <td>Response time grew {bottleneck.get("response_time_ratio", "N/A"):.2f}x when concurrency increased {bottleneck.get("concurrency_ratio", "N/A"):.2f}x</td>
                        <td class="{severity_class}">{bottleneck.get("severity", "medium").upper()}</td>
                    </tr>
                """
        
        html += """
                </table>
            </div>
        """
    
    # Recommendations
    if analysis["recommendations"]:
        html += """
            <div class="card">
                <h2>Recommendations</h2>
                <table>
                    <tr><th>Component</th><th>Issue</th><th>Suggestion</th></tr>
        """
        
        for recommendation in analysis["recommendations"]:
            html += f"""
                <tr>
                    <td>{recommendation.get("component", "N/A")}</td>
                    <td>{recommendation.get("issue", "N/A")}</td>
                    <td>{recommendation.get("suggestion", "N/A")}</td>
                </tr>
            """
        
        html += """
                </table>
            </div>
        """
    
    # Scenario results
    html += """
        <div class="card">
            <h2>Scenario Results</h2>
    """
    
    for result in results:
        scenario = result.get("scenario", {})
        html += f"""
            <h3>{scenario.get("name", "Unknown Scenario")}</h3>
            <p><strong>Description:</strong> {scenario.get("description", "N/A")}</p>
            <p><strong>Endpoint:</strong> {scenario.get("endpoint", "N/A")}, <strong>Concurrency:</strong> {scenario.get("concurrency", "N/A")}, <strong>Duration:</strong> {scenario.get("duration", "N/A")}s</p>
            
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Total Requests</td><td>{result.get("total_requests", "N/A")}</td></tr>
                <tr><td>Success Rate</td><td>{result.get("success_rate", "N/A"):.2f}%</td></tr>
                <tr><td>Requests Per Second</td><td>{result.get("requests_per_second", "N/A"):.2f}</td></tr>
        """
        
        if "response_time" in result:
            response_times = result["response_time"]
            html += f"""
                <tr><td>Min Response Time</td><td>{response_times.get("min", "N/A"):.4f}s</td></tr>
                <tr><td>Max Response Time</td><td>{response_times.get("max", "N/A"):.4f}s</td></tr>
                <tr><td>Mean Response Time</td><td>{response_times.get("mean", "N/A"):.4f}s</td></tr>
                <tr><td>95th Percentile</td><td>{response_times.get("p95", "N/A"):.4f}s</td></tr>
            """
        
        html += f"""
            </table>
            
            <p><a href="{os.path.relpath(result.get('report_file', '#'), output_dir)}" target="_blank">View detailed report</a></p>
        """
    
    html += """
        </div>
    """
    
    # If matplotlib is available, generate some combined visualizations
    if HAS_VISUALIZATION:
        html += """
            <div class="card">
                <h2>Comparative Analysis</h2>
        """
        
        # Response time by endpoint and concurrency
        plt.figure(figsize=(12, 8))
        
        # Prepare data for plotting
        endpoints = set()
        scenario_data = []
        
        for result in results:
            if "endpoints" in result:
                scenario = result.get("scenario", {})
                concurrency = scenario.get("concurrency", 0)
                
                for endpoint, stats in result["endpoints"].items():
                    endpoints.add(endpoint)
                    
                    if "response_time" in stats and "mean" in stats["response_time"]:
                        scenario_data.append({
                            "endpoint": endpoint,
                            "concurrency": concurrency,
                            "response_time": stats["response_time"]["mean"]
                        })
        
        # Only create visualization if we have data
        if scenario_data:
            # Convert to DataFrame
            df = pd.DataFrame(scenario_data)
            
            # Group by endpoint and concurrency
            grouped = df.groupby(["endpoint", "concurrency"]).mean().reset_index()
            
            # Plot
            plt.figure(figsize=(12, 8))
            for endpoint in endpoints:
                endpoint_data = grouped[grouped["endpoint"] == endpoint]
                if not endpoint_data.empty:
                    plt.plot(
                        endpoint_data["concurrency"], 
                        endpoint_data["response_time"], 
                        "o-", 
                        label=endpoint
                    )
            
            plt.xlabel("Concurrency")
            plt.ylabel("Average Response Time (s)")
            plt.title("Response Time by Endpoint and Concurrency")
            plt.grid(True, linestyle="--", alpha=0.7)
            plt.legend()
            
            # Save plot
            plot_file = os.path.join(output_dir, f"response_time_by_concurrency_{timestamp}.png")
            plt.savefig(plot_file, dpi=300)
            plt.close()
            
            # Add to report
            html += f"""
                <h3>Response Time by Endpoint and Concurrency</h3>
                <img src="{os.path.basename(plot_file)}" alt="Response Time Plot" class="plot">
            """
        
        # Success rate by scenario
        plt.figure(figsize=(12, 8))
        
        # Prepare data
        scenario_names = []
        success_rates = []
        
        for result in results:
            scenario = result.get("scenario", {})
            if "name" in scenario and "success_rate" in result:
                scenario_names.append(scenario["name"])
                success_rates.append(result["success_rate"])
        
        if scenario_names:
            # Plot
            plt.figure(figsize=(12, 8))
            plt.bar(scenario_names, success_rates)
            plt.xlabel("Scenario")
            plt.ylabel("Success Rate (%)")
            plt.title("Success Rate by Scenario")
            plt.ylim(0, 100)
            plt.grid(True, linestyle="--", alpha=0.7)
            plt.xticks(rotation=45)
            
            # Save plot
            plot_file = os.path.join(output_dir, f"success_rate_by_scenario_{timestamp}.png")
            plt.savefig(plot_file, dpi=300, bbox_inches="tight")
            plt.close()
            
            # Add to report
            html += f"""
                <h3>Success Rate by Scenario</h3>
                <img src="{os.path.basename(plot_file)}" alt="Success Rate Plot" class="plot">
            """
        
        html += """
            </div>
        """
    
    html += """
        </div>
    </body>
    </html>
    """
    
    with open(report_file, "w") as f:
        f.write(html)
    
    return report_file


async def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Comprehensive API Load Testing Tool")
    parser.add_argument("--output", type=str, default=LOG_DIR,
                        help="Output directory for reports")
    parser.add_argument("--url", type=str, default=BASE_URL,
                        help="Base URL for API")
    parser.add_argument("--api-key", type=str, default=API_KEY,
                        help="API Key for authentication")
    parser.add_argument("--scenarios", type=str, default=None,
                        help="JSON file with custom scenarios")
    parser.add_argument("--skip", type=str, default="",
                        help="Comma-separated list of scenario names to skip")
    
    args = parser.parse_args()
    
    # Set global variables
    global BASE_URL, API_KEY
    BASE_URL = args.url
    API_KEY = args.api_key
    
    # Create timestamp-based output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output, f"comprehensive_test_{timestamp}")
    
    # Load scenarios
    if args.scenarios and os.path.exists(args.scenarios):
        with open(args.scenarios, "r") as f:
            scenarios = json.load(f)
    else:
        scenarios = DEFAULT_SCENARIOS
    
    # Skip scenarios if requested
    if args.skip:
        skip_names = args.skip.split(",")
        scenarios = [s for s in scenarios if s["name"] not in skip_names]
    
    print(f"Starting comprehensive load test with {len(scenarios)} scenarios")
    print(f"Results will be saved to: {output_dir}")
    
    # Run tests
    result = await run_comprehensive_test(scenarios, output_dir)
    
    print(f"\nComprehensive test completed!")
    print(f"Report saved to: {result['report_file']}")


if __name__ == "__main__":
    asyncio.run(main())