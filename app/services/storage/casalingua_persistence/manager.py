"""
Persistence Manager for CasaLingua
Main class for managing all data persistence components
"""

import os
import logging
import json
from datetime import datetime
from typing import Dict, Any, Optional, List
from .user import UserManager
from .content import ContentManager
from .progress import ProgressManager

logger = logging.getLogger(__name__)

class PersistenceManager:
    """Main class for managing all data persistence"""
    
    def __init__(self, data_dir: str = "./data", db_config: Optional[Dict[str, Any]] = None):
        """
        Initialize persistence manager
        
        Args:
            data_dir (str): Data directory
            db_config (Dict[str, Any], optional): Database configuration
        """
        self.data_dir = data_dir
        self.db_config = db_config or {}
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
        # Determine if we're using PostgreSQL or SQLite
        self.is_postgres = False
        connection_url = ""
        
        if self.db_config and "url" in self.db_config:
            connection_url = self.db_config["url"]
            self.is_postgres = connection_url.startswith(("postgresql://", "postgres://"))
        
        if self.is_postgres:
            # For PostgreSQL, we'll use a single database with different tables for each component
            # We need to construct database URLs with different schema names
            base_url = connection_url
            
            # Parse base URL to check if it has a path component
            from urllib.parse import urlparse, urlunparse
            parsed_url = urlparse(base_url)
            
            # Get the base database name
            db_name = parsed_url.path.lstrip('/') or "casalingua"
            
            # Create connection URLs for each component
            users_url = base_url
            content_url = base_url 
            progress_url = base_url
            
            logger.info(f"Using PostgreSQL database at {base_url}")
            
            # Initialize component managers with PostgreSQL connection strings
            self.user_manager = UserManager(users_url)
            self.content_manager = ContentManager(content_url)
            self.progress_manager = ProgressManager(progress_url)
        else:
            # For SQLite, use separate database files
            logger.info(f"Using SQLite databases in {data_dir}")
            
            # Initialize component managers with SQLite file paths
            self.user_manager = UserManager(os.path.join(data_dir, "users.db"))
            self.content_manager = ContentManager(os.path.join(data_dir, "content.db"))
            self.progress_manager = ProgressManager(os.path.join(data_dir, "progress.db"))
    
    def backup_all(self, backup_dir: str) -> Dict[str, bool]:
        """
        Backup all databases
        
        Args:
            backup_dir (str): Directory to save backups
            
        Returns:
            Dict[str, bool]: Backup results
            
        TypeScript equivalent:
            async backupAll(backupDir: string): Promise<Record<string, boolean>>
        """
        # Create backup directory
        os.makedirs(backup_dir, exist_ok=True)
        
        # Timestamp for backup filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Backup each database
        results = {}
        
        if self.is_postgres:
            # For PostgreSQL, use the extension .dump for clarity
            users_backup = os.path.join(backup_dir, f"users_{timestamp}.dump")
            content_backup = os.path.join(backup_dir, f"content_{timestamp}.dump")
            progress_backup = os.path.join(backup_dir, f"progress_{timestamp}.dump")
        else:
            # For SQLite, use the .db extension
            users_backup = os.path.join(backup_dir, f"users_{timestamp}.db")
            content_backup = os.path.join(backup_dir, f"content_{timestamp}.db")
            progress_backup = os.path.join(backup_dir, f"progress_{timestamp}.db")
        
        # Perform the backups
        results["users"] = self.user_manager.backup(users_backup)
        results["content"] = self.content_manager.backup(content_backup)
        results["progress"] = self.progress_manager.backup(progress_backup)
        
        # Store metadata about the backup
        metadata = {
            "timestamp": timestamp,
            "db_type": "postgresql" if self.is_postgres else "sqlite",
            "components": ["users", "content", "progress"],
            "results": results
        }
        
        # Write metadata to a JSON file
        metadata_path = os.path.join(backup_dir, f"backup_metadata_{timestamp}.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Backup completed with timestamp {timestamp}")
        return results
    
    def restore_all(self, backup_dir: str, timestamp: str) -> Dict[str, bool]:
        """
        Restore all databases from backup
        
        Args:
            backup_dir (str): Directory with backups
            timestamp (str): Backup timestamp to restore
            
        Returns:
            Dict[str, bool]: Restore results
            
        TypeScript equivalent:
            async restoreAll(backupDir: string, timestamp: string): Promise<Record<string, boolean>>
        """
        results = {}
        
        # Try to load metadata to determine if it's a PostgreSQL or SQLite backup
        metadata_path = os.path.join(backup_dir, f"backup_metadata_{timestamp}.json")
        is_postgres_backup = self.is_postgres  # Default to current state
        
        try:
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                is_postgres_backup = metadata.get("db_type") == "postgresql"
                logger.info(f"Detected backup type: {'PostgreSQL' if is_postgres_backup else 'SQLite'}")
            else:
                logger.warning(f"No backup metadata found for timestamp {timestamp}, assuming {'PostgreSQL' if self.is_postgres else 'SQLite'} format")
        except Exception as e:
            logger.error(f"Error reading backup metadata: {e}")
        
        # Determine file extensions based on backup type
        if is_postgres_backup:
            extension = ".dump"
        else:
            extension = ".db"
        
        # Construct backup paths
        users_backup = os.path.join(backup_dir, f"users_{timestamp}{extension}")
        content_backup = os.path.join(backup_dir, f"content_{timestamp}{extension}")
        progress_backup = os.path.join(backup_dir, f"progress_{timestamp}{extension}")
        
        # Verify backup files exist
        for path, component in [(users_backup, "users"), (content_backup, "content"), (progress_backup, "progress")]:
            if not os.path.exists(path):
                logger.error(f"Backup file not found: {path}")
                results[component] = False
        
        # If we're restoring PostgreSQL backups to SQLite or vice versa, warn about potential issues
        if is_postgres_backup != self.is_postgres:
            logger.warning(f"Attempting to restore a {'PostgreSQL' if is_postgres_backup else 'SQLite'} backup to a {'PostgreSQL' if self.is_postgres else 'SQLite'} database. This may fail due to format differences.")
        
        # Perform the restores
        if all(os.path.exists(path) for path in [users_backup, content_backup, progress_backup]):
            results["users"] = self.user_manager.restore(users_backup)
            results["content"] = self.content_manager.restore(content_backup)
            results["progress"] = self.progress_manager.restore(progress_backup)
            
            logger.info(f"Restore completed for timestamp {timestamp}")
        else:
            logger.error(f"Restore failed - one or more backup files missing for timestamp {timestamp}")
        
        return results
    
    def optimize_all(self) -> Dict[str, bool]:
        """
        Optimize all databases
        
        Returns:
            Dict[str, bool]: Optimization results
            
        TypeScript equivalent:
            async optimizeAll(): Promise<Record<string, boolean>>
        """
        results = {}
        
        db_type = "PostgreSQL" if self.is_postgres else "SQLite"
        logger.info(f"Optimizing {db_type} databases...")
        
        try:
            self.user_manager.optimize()
            results["users"] = True
            logger.info(f"Successfully optimized users database ({db_type})")
        except Exception as e:
            logger.error(f"Error optimizing users database ({db_type}): {e}", exc_info=True)
            results["users"] = False

        try:
            self.content_manager.optimize()
            results["content"] = True
            logger.info(f"Successfully optimized content database ({db_type})")
        except Exception as e:
            logger.error(f"Error optimizing content database ({db_type}): {e}", exc_info=True)
            results["content"] = False

        try:
            self.progress_manager.optimize()
            results["progress"] = True
            logger.info(f"Successfully optimized progress database ({db_type})")
        except Exception as e:
            logger.error(f"Error optimizing progress database ({db_type}): {e}", exc_info=True)
            results["progress"] = False

        # Log a summary of the results
        success_count = sum(1 for success in results.values() if success)
        logger.info(f"Database optimization complete: {success_count}/{len(results)} successful")
        
        return results