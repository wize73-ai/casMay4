#!/usr/bin/env python3
"""
Database Status Check for CasaLingua

Displays detailed information about the database configuration and status
"""

import os
import sys
import json
import logging
from pathlib import Path
from urllib.parse import urlparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("db_status")

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

def get_db_info(db_url):
    """Extract database information from URL"""
    parsed_url = urlparse(db_url)
    
    if parsed_url.scheme == "sqlite":
        # SQLite database
        if parsed_url.netloc:
            # Format is sqlite://hostname/path
            db_path = os.path.join(parsed_url.netloc, parsed_url.path.lstrip('/'))
        else:
            # Format is sqlite:///path
            db_path = parsed_url.path
            if db_path.startswith("/"):
                db_path = db_path[1:]
                
        # If path is relative, make it absolute
        if not os.path.isabs(db_path):
            db_path = os.path.join(project_root, db_path)
            
        # Get database file size and last modified time
        db_size = 0
        db_modified = None
        
        if os.path.exists(db_path):
            db_size = os.path.getsize(db_path)
            db_modified = os.path.getmtime(db_path)
            
        return {
            "type": "SQLite",
            "path": db_path,
            "exists": os.path.exists(db_path),
            "size_bytes": db_size,
            "size_kb": round(db_size / 1024, 2),
            "last_modified": db_modified
        }
    elif parsed_url.scheme in ["postgresql", "postgres"]:
        # PostgreSQL database
        return {
            "type": "PostgreSQL",
            "host": parsed_url.hostname,
            "port": parsed_url.port or 5432,
            "database": parsed_url.path.lstrip('/'),
            "user": parsed_url.username
        }
    else:
        return {
            "type": "Unknown",
            "url": db_url
        }

def check_table_count(db_type, db_path=None, db_url=None):
    """Check the number of tables in the database"""
    try:
        if db_type == "SQLite" and db_path and os.path.exists(db_path):
            import sqlite3
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            cursor.close()
            conn.close()
            return [table[0] for table in tables]
        elif db_type == "PostgreSQL" and db_url:
            try:
                import psycopg2
                conn = psycopg2.connect(db_url)
                cursor = conn.cursor()
                cursor.execute("SELECT tablename FROM pg_tables WHERE schemaname='public'")
                tables = cursor.fetchall()
                cursor.close()
                conn.close()
                return [table[0] for table in tables]
            except Exception as e:
                logger.error(f"Error connecting to PostgreSQL: {e}")
                return []
        return []
    except Exception as e:
        logger.error(f"Error checking table count: {e}")
        return []

def main():
    """Main function to display database status"""
    logger.info("Checking CasaLingua database status...")
    
    # Load configuration
    config = load_config()
    db_config = config.get("database", {})
    
    # Check if configuration exists
    if not db_config:
        logger.error("No database configuration found!")
        return 1
    
    # Get database URL
    db_url = db_config.get("url")
    if not db_url:
        logger.error("No database URL specified!")
        return 1
    
    # Get database information
    db_info = get_db_info(db_url)
    
    # Display database information
    print("\n=== DATABASE CONFIGURATION ===")
    print(f"Type: {db_info['type']}")
    
    if db_info['type'] == "SQLite":
        print(f"Path: {db_info['path']}")
        print(f"Exists: {db_info['exists']}")
        if db_info['exists']:
            print(f"Size: {db_info['size_kb']} KB")
            
            # Check data directory structure
            data_dir = os.path.dirname(db_info['path'])
            print(f"\nData Directory: {data_dir}")
            if os.path.exists(data_dir):
                files = os.listdir(data_dir)
                print(f"Files: {', '.join(files)}")
                
                # Check for backup directory
                backup_dir = os.path.join(data_dir, "backups")
                if os.path.exists(backup_dir):
                    backups = os.listdir(backup_dir)
                    print(f"Backups: {len(backups)} files")
                else:
                    print("Backup directory does not exist! Creating it now...")
                    os.makedirs(backup_dir, exist_ok=True)
                    print(f"Created backup directory: {backup_dir}")
            else:
                print("Data directory does not exist! Creating it now...")
                os.makedirs(data_dir, exist_ok=True)
                print(f"Created data directory: {data_dir}")
                
            # Check tables
            tables = check_table_count("SQLite", db_path=db_info['path'])
            print(f"\nDatabase Tables: {len(tables)}")
            if tables:
                for table in tables:
                    print(f"  - {table}")
            
    elif db_info['type'] == "PostgreSQL":
        print(f"Host: {db_info['host']}")
        print(f"Port: {db_info['port']}")
        print(f"Database: {db_info['database']}")
        print(f"User: {db_info['user']}")
        
        # Check connection
        print("\nConnection Status: ", end="")
        try:
            import psycopg2
            conn = psycopg2.connect(
                host=db_info['host'],
                port=db_info['port'],
                dbname=db_info['database'],
                user=db_info['user'],
                password=urlparse(db_url).password,
                connect_timeout=5
            )
            conn.close()
            print("✅ Connected successfully")
            
            # Check tables
            tables = check_table_count("PostgreSQL", db_url=db_url)
            print(f"\nDatabase Tables: {len(tables)}")
            if tables:
                for table in tables:
                    print(f"  - {table}")
        except Exception as e:
            print(f"❌ Connection failed: {e}")
    
    print("\n=== CONFIGURATION DETAILS ===")
    print(f"Pool Size: {db_config.get('pool_size', 'Not specified')}")
    print(f"Max Overflow: {db_config.get('max_overflow', 'Not specified')}")
    print(f"Echo: {db_config.get('echo', 'Not specified')}")
    print(f"Connect Args: {json.dumps(db_config.get('connect_args', {}), indent=2)}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())