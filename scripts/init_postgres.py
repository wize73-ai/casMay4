#!/usr/bin/env python3
"""
PostgreSQL Initialization Script for CasaLingua
Creates necessary database tables for PostgreSQL persistence

This script should be run after configuring the PostgreSQL connection
in the config/default.json file.
"""

import os
import sys
import json
import logging
import argparse
import psycopg2
from urllib.parse import urlparse
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("init_postgres")

# Add the project root to the path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

# User table schema
USER_TABLE_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    username TEXT UNIQUE NOT NULL,
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP WITH TIME ZONE,
    active BOOLEAN DEFAULT TRUE
);

CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
"""

# Content table schema
CONTENT_TABLE_SCHEMA = """
CREATE TABLE IF NOT EXISTS content (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    type TEXT NOT NULL,
    language TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    content TEXT NOT NULL,
    metadata JSONB
);

CREATE INDEX IF NOT EXISTS idx_content_type ON content(type);
CREATE INDEX IF NOT EXISTS idx_content_language ON content(language);
"""

# Progress table schema
PROGRESS_TABLE_SCHEMA = """
CREATE TABLE IF NOT EXISTS progress (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    content_id TEXT NOT NULL,
    status TEXT NOT NULL,
    progress FLOAT NOT NULL,
    last_accessed TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

CREATE INDEX IF NOT EXISTS idx_progress_user ON progress(user_id);
CREATE INDEX IF NOT EXISTS idx_progress_content ON progress(content_id);
"""

def load_config():
    """Load database configuration from default.json"""
    config_path = os.path.join(project_root, "config/default.json")
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
        return config.get("database", {})
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return {}

def create_database(connection_url, db_name):
    """Create the PostgreSQL database if it doesn't exist"""
    # Parse the URL to get connection parameters
    url = urlparse(connection_url)
    
    # Extract username, password, host, port from URL
    username = url.username
    password = url.password
    host = url.hostname
    port = url.port or 5432
    
    # Connect to PostgreSQL server (default database)
    conn = None
    try:
        # Connect to default postgres database
        conn = psycopg2.connect(
            host=host,
            port=port,
            user=username,
            password=password,
            dbname="postgres"
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        
        # Check if database exists
        with conn.cursor() as cursor:
            cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s;", (db_name,))
            exists = cursor.fetchone()
            
            if not exists:
                logger.info(f"Creating database '{db_name}'...")
                cursor.execute(f"CREATE DATABASE {db_name};")
                logger.info(f"Database '{db_name}' created successfully")
            else:
                logger.info(f"Database '{db_name}' already exists")
                
        return True
    except Exception as e:
        logger.error(f"Error creating database: {e}")
        return False
    finally:
        if conn:
            conn.close()

def setup_tables(connection_url):
    """Set up the necessary tables in the PostgreSQL database"""
    try:
        # Connect to the specific database
        conn = psycopg2.connect(connection_url)
        
        # Create tables
        with conn.cursor() as cursor:
            logger.info("Creating users table...")
            cursor.execute(USER_TABLE_SCHEMA)
            
            logger.info("Creating content table...")
            cursor.execute(CONTENT_TABLE_SCHEMA)
            
            logger.info("Creating progress table...")
            cursor.execute(PROGRESS_TABLE_SCHEMA)
            
            conn.commit()
            
        logger.info("All tables created successfully")
        return True
    except Exception as e:
        logger.error(f"Error setting up tables: {e}")
        return False
    finally:
        if conn:
            conn.close()

def insert_default_admin(connection_url):
    """Insert default admin user if not exists"""
    try:
        # Connect to the database
        conn = psycopg2.connect(connection_url)
        
        # Check if admin user exists
        with conn.cursor() as cursor:
            cursor.execute("SELECT 1 FROM users WHERE username = %s;", ("admin",))
            exists = cursor.fetchone()
            
            if not exists:
                logger.info("Creating default admin user...")
                cursor.execute(
                    "INSERT INTO users (id, username, email, password_hash, role) "
                    "VALUES (%s, %s, %s, %s, %s);",
                    ("admin", "admin", "admin@casalingua.example", "salt$hash", "admin")
                )
                conn.commit()
                logger.info("Default admin user created successfully")
            else:
                logger.info("Admin user already exists")
                
        return True
    except Exception as e:
        logger.error(f"Error creating admin user: {e}")
        return False
    finally:
        if conn:
            conn.close()

def main():
    """Main function to initialize PostgreSQL database"""
    parser = argparse.ArgumentParser(description="Initialize PostgreSQL database for CasaLingua")
    parser.add_argument("--force", action="store_true", help="Force initialization even if database exists")
    args = parser.parse_args()
    
    # Ensure data directory exists
    data_dir = os.path.join(project_root, "data")
    if not os.path.exists(data_dir):
        logger.info(f"Creating data directory: {data_dir}")
        os.makedirs(data_dir, exist_ok=True)
    
    # Load configuration
    db_config = load_config()
    connection_url = db_config.get("url")
    
    if not connection_url or not connection_url.startswith(("postgresql://", "postgres://")):
        logger.error("PostgreSQL connection URL not found or invalid in config/default.json")
        return 1
    
    # Parse the URL to get database name
    url = urlparse(connection_url)
    db_name = url.path.lstrip('/') or "casalingua"
    
    # Create the database
    if not create_database(connection_url, db_name):
        logger.error("Failed to create or verify database")
        return 1
    
    # Set up tables
    if not setup_tables(connection_url):
        logger.error("Failed to set up tables")
        return 1
    
    # Insert default admin user
    if not insert_default_admin(connection_url):
        logger.error("Failed to create default admin user")
        return 1
    
    logger.info("PostgreSQL initialization complete!")
    return 0

if __name__ == "__main__":
    sys.exit(main())