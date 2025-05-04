#!/usr/bin/env python3
"""
Simple CasaLingua Health Check Test Script

This script tests the health check endpoints of a running CasaLingua instance.
It's a simpler version of the comprehensive test script that focuses on basic validation.
"""

import requests
import time
import sys
import json
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress

# Initialize rich console
console = Console()

# Base URL for the CasaLingua API (default to localhost:8000)
BASE_URL = "http://localhost:8000"

def test_endpoint(endpoint, description):
    """Test a health check endpoint and return the result"""
    url = f"{BASE_URL}{endpoint}"
    console.print(f"[cyan]Testing endpoint:[/cyan] [bold]{url}[/bold]")
    
    try:
        start_time = time.time()
        # Use a shorter timeout for /liveness endpoint
        timeout = 3 if endpoint == "/liveness" else 10
        response = requests.get(url, timeout=timeout)
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            console.print(f"[green]✓[/green] Status code: {response.status_code} ({elapsed:.2f}s)")
            try:
                data = response.json()
                return {
                    "status": "success",
                    "status_code": response.status_code,
                    "response_time": elapsed,
                    "data": data
                }
            except json.JSONDecodeError:
                console.print("[yellow]⚠[/yellow] Response is not valid JSON")
                return {
                    "status": "warning",
                    "status_code": response.status_code,
                    "response_time": elapsed,
                    "data": response.text[:100] + "..." if len(response.text) > 100 else response.text
                }
        else:
            console.print(f"[red]✗[/red] Status code: {response.status_code} ({elapsed:.2f}s)")
            try:
                data = response.json()
            except:
                data = response.text[:100] + "..." if len(response.text) > 100 else response.text
            
            return {
                "status": "error",
                "status_code": response.status_code,
                "response_time": elapsed,
                "data": data
            }
            
    except requests.exceptions.ConnectionError:
        console.print("[red]✗[/red] Connection error - Is the server running?")
        return {
            "status": "error",
            "error": "connection_error",
            "message": "Failed to connect to server"
        }
    except requests.exceptions.Timeout:
        console.print("[red]✗[/red] Request timed out")
        return {
            "status": "error",
            "error": "timeout",
            "message": "Request timed out"
        }
    except Exception as e:
        console.print(f"[red]✗[/red] Error: {str(e)}")
        return {
            "status": "error",
            "error": "unexpected_error",
            "message": str(e)
        }

def run_tests():
    """Run all health check endpoint tests"""
    console.print(Panel(
        "[bold cyan]CasaLingua Health Check Tester[/bold cyan]\n"
        f"Testing against: [yellow]{BASE_URL}[/yellow]",
        border_style="blue"
    ))
    
    # List of endpoints to test with descriptions
    endpoints = [
        ("/liveness", "Liveness probe"),
        ("/readiness", "Readiness probe"),
        ("/health", "Basic health check"),
        ("/health/models", "Model health check"),
        ("/health/database", "Database health check")
    ]
    
    # Create a table for results
    table = Table(title="Health Check Results")
    table.add_column("Endpoint", style="cyan")
    table.add_column("Status", style="bold")
    table.add_column("Time", style="magenta")
    table.add_column("Details", style="dim")
    
    # Test each endpoint
    results = {}
    with Progress() as progress:
        task = progress.add_task("[cyan]Testing endpoints...", total=len(endpoints))
        
        for endpoint, description in endpoints:
            result = test_endpoint(endpoint, description)
            results[endpoint] = result
            
            # Update the table
            status_style = "green" if result["status"] == "success" else "red"
            details = ""
            if "data" in result and isinstance(result["data"], dict) and "status" in result["data"]:
                details = f"Service status: {result['data']['status']}"
            elif "message" in result:
                details = result["message"]
                
            table.add_row(
                endpoint,
                f"[{status_style}]{result['status'].upper()}[/{status_style}]",
                f"{result.get('response_time', 0):.2f}s" if "response_time" in result else "N/A",
                details
            )
            
            # Pause between requests to avoid overwhelming the server
            time.sleep(0.5)
            progress.update(task, advance=1)
    
    # Print the results table
    console.print(table)
    
    # Print detailed results for each endpoint
    console.print("\n[bold]Detailed Results:[/bold]")
    for endpoint, result in results.items():
        if result["status"] == "success":
            console.print(Panel(
                f"[bold green]✓ {endpoint}[/bold green]\n"
                f"Response time: {result.get('response_time', 0):.2f}s\n\n"
                f"{json.dumps(result.get('data', {}), indent=2, sort_keys=True)}",
                border_style="green"
            ))
        else:
            console.print(Panel(
                f"[bold red]✗ {endpoint}[/bold red]\n"
                f"Status code: {result.get('status_code', 'N/A')}\n"
                f"Error: {result.get('error', 'unknown')}\n\n"
                f"{json.dumps(result.get('data', {}), indent=2, sort_keys=True) if isinstance(result.get('data', {}), dict) else result.get('data', 'No data')}",
                border_style="red"
            ))

if __name__ == "__main__":
    # Check if a different base URL was provided
    if len(sys.argv) > 1:
        BASE_URL = sys.argv[1]
        if not BASE_URL.startswith("http"):
            BASE_URL = f"http://{BASE_URL}"
    
    run_tests()