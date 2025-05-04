#!/usr/bin/env python3
"""
Simple test script to verify the liveness endpoint
"""

import requests
import json
from rich.console import Console

console = Console()

def test_liveness():
    """Test the liveness endpoint"""
    url = "http://localhost:8000/liveness"
    
    console.print(f"[bold cyan]Testing liveness endpoint:[/] {url}")
    
    try:
        response = requests.get(url, timeout=5)
        console.print(f"[bold green]Status code:[/] {response.status_code}")
        
        if response.status_code < 400:
            result = response.json()
            console.print(f"[bold green]Response:[/]")
            console.print(json.dumps(result, indent=2))
            return True
        else:
            console.print(f"[bold red]Error:[/] {response.text}")
            return False
    except Exception as e:
        console.print(f"[bold red]Exception:[/] {str(e)}")
        return False

if __name__ == "__main__":
    test_liveness()