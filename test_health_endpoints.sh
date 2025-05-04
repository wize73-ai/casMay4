#!/bin/bash
# Test all health endpoints

# Set base URL - default to localhost:8000 if not provided
BASE_URL=${1:-http://localhost:8000}
echo "Testing health endpoints at $BASE_URL"

# Function to make a GET request and display results
test_endpoint() {
  local endpoint=$1
  local description=$2
  
  echo -e "\n=== Testing $description ($endpoint) ==="
  curl -s "$BASE_URL$endpoint" | jq '.' 2>/dev/null || echo "Failed to parse response as JSON"
}

# Test each health endpoint
test_endpoint "/health" "Basic health check"
test_endpoint "/health/detailed" "Detailed health check"
test_endpoint "/health/models" "Model health check"
test_endpoint "/health/database" "Database health check"
test_endpoint "/readiness" "Kubernetes readiness probe"
test_endpoint "/liveness" "Kubernetes liveness probe"

echo -e "\n=== Health checks completed ==="