"""
Persistence Manager for CasaLingua
Main class for managing all data persistence components
"""

import os
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from .user import UserManager
from .content import ContentManager
from .progress import ProgressManager

logger = logging.getLogger(__name__)

class PersistenceManager:
    """Main class for managing all data persistence"""
    
    def __init__(self, data_dir: str = "./data"):
        """
        Initialize persistence manager
        
        Args:
            data_dir (str): Data directory
        """
        self.data_dir = data_dir
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
        # Initialize component managers
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
        
        results["users"] = self.user_manager.backup(
            os.path.join(backup_dir, f"users_{timestamp}.db")
        )
        
        results["content"] = self.content_manager.backup(
            os.path.join(backup_dir, f"content_{timestamp}.db")
        )
        
        results["progress"] = self.progress_manager.backup(
            os.path.join(backup_dir, f"progress_{timestamp}.db")
        )
        
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
        
        users_backup = os.path.join(backup_dir, f"users_{timestamp}.db")
        content_backup = os.path.join(backup_dir, f"content_{timestamp}.db")
        progress_backup = os.path.join(backup_dir, f"progress_{timestamp}.db")
        
        results["users"] = self.user_manager.restore(users_backup)
        results["content"] = self.content_manager.restore(content_backup)
        results["progress"] = self.progress_manager.restore(progress_backup)
        
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
        
        try:
            self.user_manager.optimize()
            results["users"] = True
        except Exception as e:
            logger.error(f"Error optimizing users database: {e}", exc_info=True)
            results["users"] = False

        try:
            self.content_manager.optimize()
            results["content"] = True
        except Exception as e:
            logger.error(f"Error optimizing content database: {e}", exc_info=True)
            results["content"] = False

        try:
            self.progress_manager.optimize()
            results["progress"] = True
        except Exception as e:
            logger.error(f"Error optimizing progress database: {e}", exc_info=True)
            results["progress"] = False

        return results