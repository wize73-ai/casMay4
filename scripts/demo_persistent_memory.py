#!/usr/bin/env python3
"""
Persistent Memory Demo for CasaLingua

Demonstrates how to use the persistence layer for storing and retrieving data
"""

import os
import sys
import json
import logging
import argparse
import uuid
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("demo")

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

def create_simple_demo(persistence_manager):
    """Create a simple demo with a user and some content"""
    # Create a test user
    user_id = f"demo_{uuid.uuid4().hex[:8]}"
    test_user = {
        "user_id": user_id,
        "username": f"demo_user_{uuid.uuid4().hex[:6]}",
        "email": f"demo_{uuid.uuid4().hex[:6]}@example.com",
        "password_hash": "demo_hash$salt",
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "settings": {"theme": "dark", "notifications": True},
        "profile": {"bio": "Demo user for testing", "interests": ["language learning", "translation"]}
    }
    
    # Create a test content item
    content_id = f"demo_{uuid.uuid4().hex[:8]}"
    test_content = {
        "content_id": content_id,
        "title": f"Sample Content {uuid.uuid4().hex[:6]}",
        "content_type": "text",
        "language_code": "en",
        "difficulty_level": "intermediate",
        "content": "This is a sample content item stored in persistent memory.",
        "metadata": {"source": "demo", "tags": ["sample", "test"]},
        "tags": "sample,test",
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat()
    }
    
    # Save the user
    try:
        # Check if users table exists, create it if not
        if persistence_manager.is_postgres:
            persistence_manager.user_manager._create_tables()
        
        # Insert user
        user_query = """
        INSERT INTO users 
        (user_id, username, email, password_hash, created_at, updated_at, settings, profile)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        persistence_manager.user_manager.execute_query(
            user_query, 
            (
                test_user["user_id"],
                test_user["username"],
                test_user["email"],
                test_user["password_hash"],
                test_user["created_at"],
                test_user["updated_at"],
                json.dumps(test_user["settings"]),
                json.dumps(test_user["profile"])
            )
        )
        logger.info(f"✅ Created demo user: {test_user['username']} ({test_user['user_id']})")
        
        # Save the content
        if persistence_manager.is_postgres:
            persistence_manager.content_manager._create_tables()
            
        # Insert content
        content_query = """
        INSERT INTO content_items 
        (content_id, title, content_type, language_code, difficulty_level, 
         content, metadata, tags, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        persistence_manager.content_manager.execute_query(
            content_query, 
            (
                test_content["content_id"],
                test_content["title"],
                test_content["content_type"],
                test_content["language_code"],
                test_content["difficulty_level"],
                test_content["content"],
                json.dumps(test_content["metadata"]),
                test_content["tags"],
                test_content["created_at"],
                test_content["updated_at"]
            )
        )
        logger.info(f"✅ Created demo content: {test_content['title']} ({test_content['content_id']})")
        
        return user_id, content_id
    except Exception as e:
        logger.error(f"❌ Error creating demo data: {e}")
        return None, None

def retrieve_demo_data(persistence_manager, user_id, content_id):
    """Retrieve demo data from persistent storage"""
    try:
        # Retrieve user
        user_query = "SELECT * FROM users WHERE user_id = ?"
        user_results = persistence_manager.user_manager.execute_query(user_query, (user_id,))
        
        if user_results and len(user_results) > 0:
            user = dict(user_results[0])
            
            # Parse JSON fields
            try:
                user['settings'] = json.loads(user['settings']) if user['settings'] else {}
                user['profile'] = json.loads(user['profile']) if user['profile'] else {}
            except:
                pass
                
            logger.info(f"✅ Retrieved user: {user['username']} ({user['user_id']})")
            logger.info(f"  Settings: {user['settings']}")
            logger.info(f"  Profile: {user['profile']}")
        else:
            logger.error(f"❌ User not found: {user_id}")
        
        # Retrieve content
        content_query = "SELECT * FROM content_items WHERE content_id = ?"
        content_results = persistence_manager.content_manager.execute_query(content_query, (content_id,))
        
        if content_results and len(content_results) > 0:
            content = dict(content_results[0])
            
            # Parse JSON metadata
            try:
                content['metadata'] = json.loads(content['metadata']) if content['metadata'] else {}
            except:
                content['metadata'] = {}
                
            logger.info(f"✅ Retrieved content: {content['title']} ({content['content_id']})")
            logger.info(f"  Content: {content['content']}")
            logger.info(f"  Metadata: {content['metadata']}")
        else:
            logger.error(f"❌ Content not found: {content_id}")
            
        return True
    except Exception as e:
        logger.error(f"❌ Error retrieving demo data: {e}")
        return False

def cleanup_demo_data(persistence_manager, user_id, content_id):
    """Clean up demo data"""
    try:
        # Delete user
        user_query = "DELETE FROM users WHERE user_id = ?"
        persistence_manager.user_manager.execute_query(user_query, (user_id,))
        logger.info(f"✅ Deleted user: {user_id}")
        
        # Delete content
        content_query = "DELETE FROM content_items WHERE content_id = ?"
        persistence_manager.content_manager.execute_query(content_query, (content_id,))
        logger.info(f"✅ Deleted content: {content_id}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Error cleaning up demo data: {e}")
        return False

def main():
    """Main function to demonstrate persistence layer"""
    parser = argparse.ArgumentParser(description="Demonstrate persistence layer")
    parser.add_argument("--cleanup", action="store_true", help="Clean up after demo")
    args = parser.parse_args()
    
    logger.info("Starting CasaLingua persistence demo...")
    
    # Load configuration
    config = load_config()
    db_config = config.get("database", {})
    
    if not db_config:
        logger.error("No database configuration found!")
        return 1
    
    # Import the persistence manager
    try:
        from app.services.storage.casalingua_persistence.manager import PersistenceManager
        
        # Initialize persistence manager
        data_dir = os.path.join(project_root, "data")
        backup_dir = os.path.join(data_dir, "backups")
        
        # Ensure all directories exist
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(backup_dir, exist_ok=True)
        logger.info(f"Ensured data directories exist: {data_dir}, {backup_dir}")
        
        persistence_manager = PersistenceManager(data_dir, db_config)
        
        # Get database type
        db_type = "SQLite"
        if persistence_manager.is_postgres:
            db_type = "PostgreSQL"
        
        logger.info(f"Using {db_type} database")
        
        # Create demo data
        user_id, content_id = create_simple_demo(persistence_manager)
        
        if user_id and content_id:
            # Retrieve demo data
            logger.info("\nRetrieving data from persistent storage...")
            retrieve_demo_data(persistence_manager, user_id, content_id)
            
            # Clean up if requested
            if args.cleanup:
                logger.info("\nCleaning up demo data...")
                cleanup_demo_data(persistence_manager, user_id, content_id)
            else:
                logger.info("\nLeaving demo data in database for future reference.")
                logger.info(f"User ID: {user_id}")
                logger.info(f"Content ID: {content_id}")
                logger.info("To clean up, run again with --cleanup option.")
            
            logger.info("\nDemo completed successfully!")
        else:
            logger.error("Failed to create demo data.")
            return 1
            
        return 0
    except Exception as e:
        logger.error(f"❌ Error in persistence demo: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())