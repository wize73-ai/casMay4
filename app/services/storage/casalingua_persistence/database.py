"""
Database Manager for CasaLingua
Provides base database functionality and connection management
"""

import os
import json
import logging
import time
import sqlite3
import threading
from typing import Dict, Any, Optional, List, Tuple, Union, Callable, Set
from datetime import datetime
import numpy as np
from contextlib import contextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DatabaseManager:
    """Base database manager class"""
    
    def __init__(self, db_path: str):
        """
        Initialize database manager
        
        Args:
            db_path (str): Path to database file
        """
        self.db_path = db_path
        self.local = threading.local()
        
    @contextmanager
    def get_connection(self) -> sqlite3.Connection:
        """
        Get a database connection (thread-safe)
        
        Yields:
            sqlite3.Connection: Database connection
            
        TypeScript equivalent:
            async getConnection(): Promise<Connection>
        """
        if not hasattr(self.local, 'connection'):
            self.local.connection = sqlite3.connect(self.db_path)
            self.local.connection.row_factory = sqlite3.Row
        
        try:
            yield self.local.connection
        except Exception as e:
            self.local.connection.rollback()
            logger.error(f"Database error: {e}", exc_info=True)
            raise
    
    @contextmanager
    def get_cursor(self) -> sqlite3.Cursor:
        """
        Get a database cursor (thread-safe)
        
        Yields:
            sqlite3.Cursor: Database cursor
            
        TypeScript equivalent:
            async getCursor(): Promise<Cursor>
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                yield cursor
                conn.commit()
            except Exception as e:
                conn.rollback()
                logger.error(f"Database error: {e}", exc_info=True)
                raise
            finally:
                cursor.close()
    
    def execute_query(self, query: str, params: tuple = ()) -> List[Dict[str, Any]]:
        """
        Execute a database query
        
        Args:
            query (str): SQL query
            params (tuple): Query parameters
            
        Returns:
            List[Dict[str, Any]]: Query results
            
        TypeScript equivalent:
            async executeQuery(query: string, params: any[]): Promise<Record<string, any>[]>
            
        Note for PostgreSQL migration:
        - SQLite uses ? for parameter placeholders
        - PostgreSQL uses $1, $2, etc.
        """
        with self.get_cursor() as cursor:
            cursor.execute(query, params)
            
            if query.strip().upper().startswith("SELECT"):
                columns = [column[0] for column in cursor.description]
                return [dict(zip(columns, row)) for row in cursor.fetchall()]
            
            return []
    
    def execute_many(self, query: str, params_list: List[tuple]) -> int:
        """
        Execute a query with multiple parameter sets
        
        Args:
            query (str): SQL query
            params_list (List[tuple]): List of parameter tuples
            
        Returns:
            int: Number of affected rows
            
        TypeScript equivalent:
            async executeMany(query: string, paramsList: any[][]): Promise<number>
        """
        with self.get_cursor() as cursor:
            cursor.executemany(query, params_list)
            return cursor.rowcount
    
    def execute_script(self, script: str) -> None:
        """
        Execute a SQL script
        
        Args:
            script (str): SQL script
            
        TypeScript equivalent:
            async executeScript(script: string): Promise<void>
        """
        with self.get_connection() as conn:
            conn.executescript(script)
    
    def optimize(self) -> None:
        """
        Optimize database performance
        
        TypeScript equivalent:
            async optimize(): Promise<void>
            
        Note for PostgreSQL migration:
        - PostgreSQL uses VACUUM instead of PRAGMA optimize
        """
        with self.get_cursor() as cursor:
            cursor.execute("PRAGMA optimize")
            cursor.execute("VACUUM")
    
    def backup(self, backup_path: str) -> bool:
        """
        Backup the database to a file
        
        Args:
            backup_path (str): Path to save backup
            
        Returns:
            bool: Success status
            
        TypeScript equivalent:
            async backup(backupPath: string): Promise<boolean>
            
        Note for PostgreSQL migration:
        - Use pg_dump for PostgreSQL backups
        """
        try:
            with self.get_connection() as conn:
                backup_conn = sqlite3.connect(backup_path)
                conn.backup(backup_conn)
                backup_conn.close()
            return True
        except Exception as e:
            logger.error(f"Backup error: {e}", exc_info=True)
            return False
    
    def restore(self, backup_path: str) -> bool:
        """
        Restore database from a backup
        
        Args:
            backup_path (str): Path to backup file
            
        Returns:
            bool: Success status
            
        TypeScript equivalent:
            async restore(backupPath: string): Promise<boolean>
            
        Note for PostgreSQL migration:
        - Use pg_restore for PostgreSQL restores
        """
        try:
            if not os.path.exists(backup_path):
                logger.error(f"Backup file not found: {backup_path}", exc_info=True)
                return False
            # Close current connection
            if hasattr(self.local, 'connection'):
                self.local.connection.close()
                delattr(self.local, 'connection')
            # Restore from backup
            backup_conn = sqlite3.connect(backup_path)
            conn = sqlite3.connect(self.db_path)
            backup_conn.backup(conn)
            backup_conn.close()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"Restore error: {e}", exc_info=True)
            return False