#!/usr/bin/env python3
"""
Test Database Persistence Manager
Verifies that the persistence layer is working correctly with CasaLingua's classes
"""

import os
import sys
import json
import logging
import uuid
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_persistence")

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

def test_user_persistence(persistence_manager):
    """Test user database operations"""
    logger.info("Testing user persistence...")
    
    # Create a test user
    test_user_id = f"test_{uuid.uuid4().hex[:8]}"
    test_user = {
        "id": test_user_id,
        "username": f"test_user_{uuid.uuid4().hex[:8]}",
        "email": f"test_{uuid.uuid4().hex[:8]}@example.com",
        "password_hash": "test_hash$salt",
        "role": "tester",
        "created_at": datetime.now().isoformat(),
        "last_login": None,
        "active": True
    }
    
    # Save user to database
    try:
        # We don't need to create the table, UserManager does that in _create_tables()
        
        # Then insert the user
        query = """
        INSERT INTO users (user_id, username, email, password_hash, created_at, updated_at, settings, profile)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        persistence_manager.user_manager.execute_query(
            query, 
            (
                test_user["id"],
                test_user["username"],
                test_user["email"],
                test_user["password_hash"],
                test_user["created_at"],
                test_user["created_at"],  # Use same value for updated_at
                json.dumps({}),  # Empty settings
                json.dumps({})   # Empty profile
            )
        )
        logger.info(f"✅ User saved successfully: {test_user['username']}")
        
        # Read user back from database
        query = "SELECT * FROM users WHERE user_id = ?"
        results = persistence_manager.user_manager.execute_query(query, (test_user_id,))
        
        if results and len(results) > 0:
            retrieved_user = results[0]
            logger.info(f"✅ User retrieved successfully: {retrieved_user['username']}")
            
            # Update user
            update_query = "UPDATE users SET updated_at = ? WHERE user_id = ?"
            update_time = datetime.now().isoformat()
            persistence_manager.user_manager.execute_query(update_query, (update_time, test_user_id))
            logger.info(f"✅ User updated successfully")
            
            # Read updated user
            results = persistence_manager.user_manager.execute_query(query, (test_user_id,))
            if results and len(results) > 0:
                updated_user = results[0]
                logger.info(f"✅ Updated user retrieved successfully: {updated_user['username']} (updated_at: {updated_user['updated_at']})")
                
                # Delete user
                delete_query = "DELETE FROM users WHERE user_id = ?"
                persistence_manager.user_manager.execute_query(delete_query, (test_user_id,))
                logger.info(f"✅ User deleted successfully")
                
                # Verify deletion
                results = persistence_manager.user_manager.execute_query(query, (test_user_id,))
                if not results or len(results) == 0:
                    logger.info(f"✅ User deletion verified")
                    return True
                else:
                    logger.error(f"❌ User still exists after deletion")
                    return False
            else:
                logger.error(f"❌ Could not retrieve updated user")
                return False
        else:
            logger.error(f"❌ Could not retrieve user after saving")
            return False
        
    except Exception as e:
        logger.error(f"❌ Error in user persistence test: {e}")
        return False

def test_content_persistence(persistence_manager):
    """Test content database operations"""
    logger.info("Testing content persistence...")
    
    # Create a test content item
    test_content_id = f"test_{uuid.uuid4().hex[:8]}"
    test_content = {
        "id": test_content_id,
        "title": f"Test Content {uuid.uuid4().hex[:8]}",
        "type": "test_document",
        "language": "en",
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "content": "This is test content for database persistence testing",
        "metadata": json.dumps({"test_key": "test_value", "tags": ["test", "persistence"]})
    }
    
    try:
        # First create the content table if it doesn't exist
        persistence_manager.content_manager.execute_query("""
        CREATE TABLE IF NOT EXISTS content (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            type TEXT NOT NULL,
            language TEXT NOT NULL,
            created_at TEXT,
            updated_at TEXT,
            content TEXT NOT NULL,
            metadata TEXT
        )
        """)
        
        # Then insert the content
        query = """
        INSERT INTO content (id, title, type, language, created_at, updated_at, content, metadata)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        persistence_manager.content_manager.execute_query(
            query, 
            (
                test_content["id"],
                test_content["title"],
                test_content["type"],
                test_content["language"],
                test_content["created_at"],
                test_content["updated_at"],
                test_content["content"],
                test_content["metadata"]
            )
        )
        logger.info(f"✅ Content saved successfully: {test_content['title']}")
        
        # Read content back from database
        query = "SELECT * FROM content WHERE id = ?"
        results = persistence_manager.content_manager.execute_query(query, (test_content_id,))
        
        if results and len(results) > 0:
            retrieved_content = results[0]
            logger.info(f"✅ Content retrieved successfully: {retrieved_content['title']}")
            
            # Verify JSON metadata can be parsed
            try:
                metadata = json.loads(retrieved_content['metadata'])
                logger.info(f"✅ Metadata parsed successfully: {metadata}")
            except Exception as e:
                logger.error(f"❌ Error parsing metadata: {e}")
                return False
            
            # Update content
            update_query = "UPDATE content SET updated_at = ?, content = ? WHERE id = ?"
            update_time = datetime.now().isoformat()
            updated_content = "This content has been updated for testing"
            persistence_manager.content_manager.execute_query(
                update_query, 
                (update_time, updated_content, test_content_id)
            )
            logger.info(f"✅ Content updated successfully")
            
            # Delete content
            delete_query = "DELETE FROM content WHERE id = ?"
            persistence_manager.content_manager.execute_query(delete_query, (test_content_id,))
            logger.info(f"✅ Content deleted successfully")
            
            return True
        else:
            logger.error(f"❌ Could not retrieve content after saving")
            return False
        
    except Exception as e:
        logger.error(f"❌ Error in content persistence test: {e}")
        return False

def main():
    """Main function to test persistence manager"""
    config = load_config()
    db_config = config.get("database", {})
    
    # Check database configuration
    if not db_config:
        logger.error("❌ No database configuration found in config/default.json")
        return 1
    
    logger.info("Initializing persistence manager...")
    
    try:
        # Import the persistence manager
        from app.services.storage.casalingua_persistence.manager import PersistenceManager
        
        # Initialize persistence manager
        data_dir = os.path.join(project_root, "data")
        persistence_manager = PersistenceManager(data_dir, db_config)
        logger.info(f"✅ Persistence manager initialized successfully")
        
        # Run persistence tests
        user_test = test_user_persistence(persistence_manager)
        content_test = test_content_persistence(persistence_manager)
        
        # Optimize databases
        logger.info("Optimizing databases...")
        persistence_manager.optimize_all()
        
        # Test backup functionality
        logger.info("Testing database backup...")
        backup_dir = os.path.join(project_root, "data/backups")
        os.makedirs(backup_dir, exist_ok=True)
        
        backup_results = persistence_manager.backup_all(backup_dir)
        if all(backup_results.values()):
            logger.info(f"✅ Database backup successful")
        else:
            logger.error(f"❌ Database backup failed: {backup_results}")
        
        # Overall test results
        if user_test and content_test:
            logger.info("✅ All persistence tests passed successfully!")
            return 0
        else:
            logger.error("❌ Some persistence tests failed")
            return 1
            
    except Exception as e:
        logger.error(f"❌ Error initializing persistence manager: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())