#!/bin/bash
# Script to run API load tests with various configurations

# Set default values
USERS=10
SPAWN_RATE=2
TEST_DURATION=60
HOST="http://localhost:8000"
TAGS=""
OUTPUT_DIR="../logs/load_tests"
WEB_UI=false
TEST_SCRIPT="../tests/test_api_load.py"
TEST_PROFILE="default"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -u|--users)
      USERS="$2"
      shift 2
      ;;
    -r|--spawn-rate)
      SPAWN_RATE="$2"
      shift 2
      ;;
    -t|--time)
      TEST_DURATION="$2"
      shift 2
      ;;
    -h|--host)
      HOST="$2"
      shift 2
      ;;
    --tags)
      TAGS="$2"
      shift 2
      ;;
    -o|--output)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    -w|--web-ui)
      WEB_UI=true
      shift
      ;;
    -p|--profile)
      TEST_PROFILE="$2"
      shift 2
      ;;
    -s|--script)
      TEST_SCRIPT="$2"
      shift 2
      ;;
    --help)
      echo "Usage: $0 [options]"
      echo ""
      echo "Options:"
      echo "  -u, --users N          Number of concurrent users (default: 10)"
      echo "  -r, --spawn-rate N     User spawn rate (default: 2)"
      echo "  -t, --time N           Test duration in seconds (default: 60)"
      echo "  -h, --host URL         Host to test (default: http://localhost:8000)"
      echo "  --tags TAGS            Comma-separated list of tags to include"
      echo "  -o, --output DIR       Output directory for reports (default: ../logs/load_tests)"
      echo "  -w, --web-ui           Start with web UI"
      echo "  -p, --profile NAME     Use a predefined test profile"
      echo "  -s, --script PATH      Path to test script (default: ../tests/test_api_load.py)"
      echo "  --help                 Show this help message"
      echo ""
      echo "Available profiles:"
      echo "  light    - Light load (10 users, 1 spawn rate, 60s)"
      echo "  medium   - Medium load (50 users, 5 spawn rate, 120s)"
      echo "  heavy    - Heavy load (100 users, 10 spawn rate, 300s)"
      echo "  stress   - Stress test (200 users, 20 spawn rate, 600s)"
      echo "  endurance - Endurance test (30 users, 2 spawn rate, 1800s)"
      echo "  mixed    - Only test mixed workload tag"
      echo "  translate - Only test translation endpoints"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Set timestamp for unique filenames
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Load predefined profiles
case $TEST_PROFILE in
  light)
    USERS=10
    SPAWN_RATE=1
    TEST_DURATION=60
    ;;
  medium)
    USERS=50
    SPAWN_RATE=5
    TEST_DURATION=120
    ;;
  heavy)
    USERS=100
    SPAWN_RATE=10
    TEST_DURATION=300
    ;;
  stress)
    USERS=200
    SPAWN_RATE=20
    TEST_DURATION=600
    ;;
  endurance)
    USERS=30
    SPAWN_RATE=2
    TEST_DURATION=1800
    ;;
  mixed)
    TAGS="mixed_workload"
    ;;
  translate)
    TAGS="translation"
    ;;
esac

# Prepare the CSV report filename
CSV_REPORT="$OUTPUT_DIR/load_test_${TEST_PROFILE}_${TIMESTAMP}.csv"
HTML_REPORT="$OUTPUT_DIR/load_test_${TEST_PROFILE}_${TIMESTAMP}.html"

echo "Starting load test with the following configuration:"
echo "Users: $USERS"
echo "Spawn Rate: $SPAWN_RATE"
echo "Duration: ${TEST_DURATION}s"
echo "Host: $HOST"
[ -n "$TAGS" ] && echo "Tags: $TAGS"
echo "Output: $CSV_REPORT"
echo "Profile: $TEST_PROFILE"
echo ""

# Build the command based on whether we want the web UI
if [ "$WEB_UI" = true ]; then
  # Run with web UI
  CMD="locust -f $TEST_SCRIPT --host=$HOST"
  [ -n "$TAGS" ] && CMD="$CMD --tags $TAGS"
  
  echo "Running with web UI. Open browser to http://localhost:8089/"
  $CMD
else
  # Run headless
  CMD="locust -f $TEST_SCRIPT --headless -u $USERS -r $SPAWN_RATE -t ${TEST_DURATION}s --host=$HOST --csv=$CSV_REPORT --html=$HTML_REPORT"
  [ -n "$TAGS" ] && CMD="$CMD --tags $TAGS"
  
  echo "Running headless load test..."
  echo "$CMD"
  $CMD
  
  # Display summary when finished
  echo ""
  echo "Load test completed!"
  echo "Results saved to: $CSV_REPORT"
  echo "HTML report saved to: $HTML_REPORT"
  
  # Optional: Generate a quick summary from the CSV
  if [ -f "${CSV_REPORT}_stats.csv" ]; then
    echo ""
    echo "Summary of results:"
    echo "-----------------------------------------------------"
    echo "Name, # Requests, # Failures, Median Response Time, 95% Response Time, Requests/s"
    head -n 10 "${CSV_REPORT}_stats.csv" | tail -n +2 | awk -F, '{print $1", "$2", "$3", "$4", "$5", "$6}'
    echo "-----------------------------------------------------"
  fi
fi