#!/bin/bash
# Initialize PostgreSQL database for CasaLingua

# Ensure we're in the project root directory
cd "$(dirname "$0")"

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not found."
    exit 1
fi

# Verify psycopg2 is installed
python3 -c "import psycopg2" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing psycopg2-binary..."
    pip install psycopg2-binary
fi

# Run the initialization script
echo "Initializing PostgreSQL database..."
python3 scripts/init_postgres.py

# Check the result
if [ $? -eq 0 ]; then
    echo "✅ PostgreSQL database initialization complete!"
    echo "CasaLingua is now configured to use PostgreSQL at 192.168.1.105"
else
    echo "❌ PostgreSQL database initialization failed. Check the logs for details."
    exit 1
fi

# Make the script executable
chmod +x init_postgres.sh

echo "You can run ./init_postgres.sh to initialize the database anytime."