#!/usr/bin/env python3
"""
Test script for health check endpoints

This script tests all health check endpoints in the CasaLingua system:
- Basic health check (/health)
- Detailed health check (/health/detailed)
- Model health check (/health/models)
- Database health check (/health/database)
- Readiness probe (/readiness)
- Liveness probe (/liveness)

Usage:
    python test_health_checks.py [--host HOST] [--port PORT]

"""

import argparse
import json
import requests
import sys
import time
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax

# Initialize rich console
console = Console()

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Test CasaLingua health check endpoints")
    parser.add_argument("--host", default="localhost", help="API host (default: localhost)")
    parser.add_argument("--port", default=8000, type=int, help="API port (default: 8000)")
    return parser.parse_args()

def make_request(url, endpoint, timeout=10):
    """Make a request to the API endpoint and return the response"""
    full_url = f"{url}{endpoint}"
    try:
        console.print(f"[bold cyan]Requesting:[/] {full_url}")
        start_time = time.time()
        response = requests.get(full_url, timeout=timeout)
        elapsed = time.time() - start_time
        
        console.print(f"[bold green]Status:[/] {response.status_code} [bold blue]Time:[/] {elapsed:.2f}s")
        
        return {
            "status_code": response.status_code,
            "elapsed": elapsed,
            "success": response.status_code < 400,
            "data": response.json() if response.status_code < 400 else None,
            "error": response.text if response.status_code >= 400 else None
        }
    except requests.exceptions.RequestException as e:
        console.print(f"[bold red]Error:[/] {str(e)}")
        return {
            "status_code": None,
            "elapsed": time.time() - start_time,
            "success": False,
            "data": None,
            "error": str(e)
        }

def print_response(response):
    """Pretty print a JSON response"""
    if not response["success"]:
        console.print(Panel(f"[bold red]ERROR:[/] {response['error']}", title="Error Response"))
        return
    
    # Format JSON response nicely
    json_str = json.dumps(response["data"], indent=2)
    syntax = Syntax(json_str, "json", theme="monokai", line_numbers=True)
    console.print(syntax)

def run_health_tests(base_url):
    """Run tests for all health check endpoints"""
    results = {}
    
    # Test basic health check
    console.rule("[bold green]Testing Basic Health Check[/]")
    results["basic"] = make_request(base_url, "/health")
    print_response(results["basic"])
    
    # Test detailed health check
    console.rule("[bold green]Testing Detailed Health Check[/]")
    results["detailed"] = make_request(base_url, "/health/detailed")
    print_response(results["detailed"])
    
    # Test model health check
    console.rule("[bold green]Testing Model Health Check[/]")
    results["models"] = make_request(base_url, "/health/models")
    print_response(results["models"])
    
    # Test database health check
    console.rule("[bold green]Testing Database Health Check[/]")
    results["database"] = make_request(base_url, "/health/database")
    print_response(results["database"])
    
    # Test readiness probe
    console.rule("[bold green]Testing Readiness Probe[/]")
    results["readiness"] = make_request(base_url, "/readiness")
    print_response(results["readiness"])
    
    # Test liveness probe
    console.rule("[bold green]Testing Liveness Probe[/]")
    results["liveness"] = make_request(base_url, "/liveness")
    print_response(results["liveness"])
    
    return results

def print_summary(results):
    """Print a summary of all test results"""
    console.rule("[bold blue]Health Check Summary[/]")
    
    table = Table(title="Health Check Results")
    table.add_column("Endpoint", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Response Time", style="yellow")
    table.add_column("Health Status", style="magenta")
    
    for endpoint, result in results.items():
        status = "✅ Success" if result["success"] else "❌ Failed"
        response_time = f"{result['elapsed']:.2f}s"
        
        if result["success"] and result["data"]:
            health_status = result["data"].get("status", "N/A")
            # Apply color based on health status
            if health_status == "healthy":
                health_status = f"[green]{health_status}[/]"
            elif health_status == "degraded":
                health_status = f"[yellow]{health_status}[/]"
            elif health_status in ["error", "unhealthy"]:
                health_status = f"[red]{health_status}[/]"
            elif health_status in ["alive", "ready"]:
                health_status = f"[green]{health_status}[/]"
        else:
            health_status = "N/A"
        
        table.add_row(endpoint, status, response_time, health_status)
    
    console.print(table)

def main():
    """Main function"""
    args = parse_args()
    base_url = f"http://{args.host}:{args.port}"
    
    console.print(Panel.fit(
        f"[bold]CasaLingua Health Check Test[/]\n"
        f"Testing against: [cyan]{base_url}[/]\n"
        f"Time: [yellow]{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/]",
        title="Health Check Tester",
        border_style="blue"
    ))
    
    try:
        results = run_health_tests(base_url)
        print_summary(results)
        
        # Determine overall success
        all_success = all(result["success"] for result in results.values())
        if all_success:
            console.print(Panel("[bold green]ALL HEALTH CHECKS PASSED[/]", border_style="green"))
            return 0
        else:
            failed = [endpoint for endpoint, result in results.items() if not result["success"]]
            console.print(Panel(f"[bold red]SOME HEALTH CHECKS FAILED:[/] {', '.join(failed)}", border_style="red"))
            return 1
    except Exception as e:
        console.print_exception()
        console.print(Panel(f"[bold red]ERROR:[/] {str(e)}", border_style="red"))
        return 1

if __name__ == "__main__":
    sys.exit(main())