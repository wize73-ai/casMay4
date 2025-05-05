#!/usr/bin/env python3
"""
Toggle Database Configuration for CasaLingua

Switches between SQLite and PostgreSQL configurations
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("toggle_db")

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
        return config, config_path
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return {}, config_path

def save_config(config, config_path):
    """Save configuration to default.json"""
    try:
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Error saving configuration: {e}")
        return False

def get_sqlite_config():
    """Get SQLite database configuration"""
    return {
        "url": "sqlite:///data/casalingua.db",
        "pool_size": 5,
        "max_overflow": 10,
        "echo": False,
        "connect_args": {
            "check_same_thread": False
        }
    }

def get_postgres_config(host="192.168.1.105", port=5432, username="postgres", password="postgres", database="casalingua"):
    """Get PostgreSQL database configuration"""
    return {
        "url": f"postgresql://{username}:{password}@{host}:{port}/{database}",
        "pool_size": 10,
        "max_overflow": 20,
        "echo": False,
        "connect_args": {
            "connect_timeout": 10
        }
    }

def main():
    """Main function to toggle database configuration"""
    parser = argparse.ArgumentParser(description="Toggle database configuration between SQLite and PostgreSQL")
    parser.add_argument("--type", choices=["sqlite", "postgres"], help="Database type to use")
    parser.add_argument("--host", default="192.168.1.105", help="PostgreSQL host (default: 192.168.1.105)")
    parser.add_argument("--port", type=int, default=5432, help="PostgreSQL port (default: 5432)")
    parser.add_argument("--username", default="postgres", help="PostgreSQL username (default: postgres)")
    parser.add_argument("--password", default="postgres", help="PostgreSQL password (default: postgres)")
    parser.add_argument("--database", default="casalingua", help="PostgreSQL database name (default: casalingua)")
    args = parser.parse_args()
    
    # Load current configuration
    config, config_path = load_config()
    
    # Get current database configuration
    current_db_config = config.get("database", {})
    current_db_url = current_db_config.get("url", "")
    
    # Determine current type
    current_type = "unknown"
    if current_db_url.startswith("sqlite:"):
        current_type = "sqlite"
    elif current_db_url.startswith(("postgresql:", "postgres:")):
        current_type = "postgres"
    
    # Determine target type
    target_type = args.type
    if not target_type:
        # If no type specified, toggle
        target_type = "postgres" if current_type == "sqlite" else "sqlite"
    
    logger.info(f"Current database type: {current_type}")
    logger.info(f"Switching to: {target_type}")
    
    # Create backup of current config
    backup_path = os.path.join(project_root, "config/database_backup.json")
    with open(backup_path, "w") as f:
        json.dump(current_db_config, f, indent=2)
    logger.info(f"Backed up current configuration to {backup_path}")
    
    # Set new configuration
    if target_type == "sqlite":
        config["database"] = get_sqlite_config()
    else:
        config["database"] = get_postgres_config(
            host=args.host,
            port=args.port,
            username=args.username,
            password=args.password,
            database=args.database
        )
    
    # Save new configuration
    if save_config(config, config_path):
        logger.info(f"Successfully switched to {target_type} configuration")
        
        # Ensure data directory structure exists (for both database types)
        data_dir = os.path.join(project_root, "data")
        backup_dir = os.path.join(data_dir, "backups")
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(backup_dir, exist_ok=True)
        logger.info(f"Ensured data directory structure exists: {data_dir}")
        
        # If switching to SQLite, ensure specific SQLite directories
        if target_type == "sqlite":
            logger.info("Creating SQLite database structure")
        
        # Display the new URL
        new_url = config["database"]["url"]
        if target_type == "postgres":
            # Hide password for display
            masked_url = new_url.replace(args.password, "*****")
            logger.info(f"New database URL: {masked_url}")
        else:
            logger.info(f"New database URL: {new_url}")
            
        return 0
    else:
        logger.error("Failed to save configuration")
        return 1

if __name__ == "__main__":
    sys.exit(main())