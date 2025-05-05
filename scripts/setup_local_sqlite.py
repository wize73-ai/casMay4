#!/usr/bin/env python3
"""
Setup Local SQLite Configuration for CasaLingua

This script creates a local SQLite database configuration
for testing when PostgreSQL is not available.
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
logger = logging.getLogger("setup_sqlite")

def main():
    """Main function to set up SQLite configuration"""
    # Determine the project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Path to config file
    config_path = os.path.join(project_root, "config/default.json")
    
    # Load current configuration
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return 1
    
    # Create data directory if it doesn't exist
    data_dir = os.path.join(project_root, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    # Save the current database configuration as backup
    if "database" in config:
        orig_config_path = os.path.join(project_root, "config/database_postgres_backup.json")
        try:
            with open(orig_config_path, "w") as f:
                json.dump(config["database"], f, indent=2)
            logger.info(f"Original database configuration saved to {orig_config_path}")
        except Exception as e:
            logger.error(f"Error saving backup configuration: {e}")
    
    # Update with SQLite configuration
    config["database"] = {
        "url": "sqlite:///data/casalingua.db",
        "pool_size": 5,
        "max_overflow": 10,
        "echo": False,
        "connect_args": {
            "check_same_thread": False
        }
    }
    
    # Save the updated configuration
    try:
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        logger.info(f"Configuration updated with SQLite database")
    except Exception as e:
        logger.error(f"Error saving configuration: {e}")
        return 1
    
    # Create empty database files for SQLite
    for db_name in ["users", "content", "progress"]:
        db_path = os.path.join(data_dir, f"{db_name}.db")
        try:
            # Create an empty file if it doesn't exist
            if not os.path.exists(db_path):
                Path(db_path).touch()
                logger.info(f"Created empty database file: {db_path}")
            else:
                logger.info(f"Database file already exists: {db_path}")
        except Exception as e:
            logger.error(f"Error creating database file {db_path}: {e}")
    
    logger.info("SQLite configuration complete! CasaLingua is now set to use local SQLite databases.")
    logger.info(f"Database files are located in: {data_dir}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())