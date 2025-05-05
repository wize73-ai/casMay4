#!/usr/bin/env python3
"""
Test Database Connection and Functionality
Verifies that the database is properly configured and accessible
"""

import os
import sys
import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_db")

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

def load_config():
    """Load configuration from default.json"""
    config_path = os.path.join(project_root, "config/default.json")
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return {}

def test_sqlite_connection(db_path):
    """Test connection to SQLite database"""
    import sqlite3
    
    logger.info(f"Testing SQLite connection to: {db_path}")
    
    try:
        # Connect to database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create a test table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS test_connection (
            id INTEGER PRIMARY KEY,
            name TEXT,
            timestamp TEXT
        )
        """)
        
        # Insert a test record
        import datetime
        timestamp = datetime.datetime.now().isoformat()
        cursor.execute(
            "INSERT INTO test_connection (name, timestamp) VALUES (?, ?)",
            ("test_user", timestamp)
        )
        conn.commit()
        
        # Read the record back
        cursor.execute("SELECT * FROM test_connection ORDER BY id DESC LIMIT 1")
        result = cursor.fetchone()
        
        # Clean up
        cursor.execute("DROP TABLE test_connection")
        conn.commit()
        conn.close()
        
        if result:
            logger.info(f"✅ Successfully connected to SQLite database")
            logger.info(f"✅ Read test record: {result}")
            return True
        else:
            logger.error("❌ Could not read test record")
            return False
            
    except Exception as e:
        logger.error(f"❌ SQLite connection error: {e}")
        return False

def main():
    """Main function to test database connection"""
    config = load_config()
    db_config = config.get("database", {})
    
    # Check database configuration
    if not db_config:
        logger.error("❌ No database configuration found in config/default.json")
        return 1
    
    # Get database URL
    db_url = db_config.get("url", "")
    if not db_url:
        logger.error("❌ No database URL specified in configuration")
        return 1
    
    logger.info(f"Database URL: {db_url}")
    
    # Determine database type
    if db_url.startswith("sqlite:///"):
        # For SQLite, extract the database path
        db_path = db_url.replace("sqlite:///", "")
        
        # If it's a relative path, make it absolute
        if not os.path.isabs(db_path):
            db_path = os.path.join(project_root, db_path)
            
        # Check if the directory exists
        db_dir = os.path.dirname(db_path)
        if not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
            logger.info(f"Created directory: {db_dir}")
        
        # Test SQLite connection
        if test_sqlite_connection(db_path):
            return 0
        else:
            return 1
            
    elif db_url.startswith(("postgresql://", "postgres://")):
        logger.info("PostgreSQL configuration detected, but we're using SQLite for local testing")
        return 1
    else:
        logger.error(f"❌ Unsupported database type: {db_url}")
        return 1

if __name__ == "__main__":
    sys.exit(main())