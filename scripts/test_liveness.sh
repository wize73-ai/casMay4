#!/bin/bash

# Ultra-minimal script to test just the liveness endpoint

echo "Testing CasaLingua liveness endpoint..."
echo "Request: curl -s http://localhost:8000/liveness"
echo "-----------------------------------------------"
curl -s http://localhost:8000/liveness
echo
echo "-----------------------------------------------"
echo "Test complete."