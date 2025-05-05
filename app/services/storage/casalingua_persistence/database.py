"""
Database Manager for CasaLingua
Provides base database functionality and connection management
"""

import os
import json
import logging
import time
import threading
import importlib
from typing import Dict, Any, Optional, List, Tuple, Union, Callable, Set
from datetime import datetime
import numpy as np
from contextlib import contextmanager
from urllib.parse import urlparse

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
            db_path (str): Path to database file or connection URL
        """
        self.db_path = db_path
        self.local = threading.local()
        
        # Determine database type based on the connection string
        self.db_type = self._determine_db_type(db_path)
        self.is_postgres = self.db_type == "postgresql"
        
        # Import appropriate driver modules
        if self.is_postgres:
            import psycopg2
            import psycopg2.extras
            self.db_module = psycopg2
            self.db_extras = psycopg2.extras
        else:
            import sqlite3
            self.db_module = sqlite3
            self.db_extras = None
        
    def _determine_db_type(self, db_path: str) -> str:
        """
        Determine database type from connection string
        
        Args:
            db_path (str): Database path or connection URL
            
        Returns:
            str: Database type ('postgresql' or 'sqlite')
        """
        if db_path.startswith(("postgresql://", "postgres://")):
            return "postgresql"
        elif db_path.startswith("sqlite://"):
            # Extract file path from SQLite connection URL
            url_parts = urlparse(db_path)
            if url_parts.netloc:
                # Format is sqlite://hostname/path
                filepath = os.path.join(url_parts.netloc, url_parts.path.lstrip('/'))
            else:
                # Format is sqlite:///path
                filepath = url_parts.path
                if filepath.startswith("/"):
                    filepath = filepath[1:]
            self.db_path = filepath
            return "sqlite"
        else:
            # Assume it's a direct path to SQLite DB
            return "sqlite"
        
    @contextmanager
    def get_connection(self):
        """
        Get a database connection (thread-safe)
        
        Yields:
            Connection: Database connection object (psycopg2 or sqlite3)
            
        TypeScript equivalent:
            async getConnection(): Promise<Connection>
        """
        if not hasattr(self.local, 'connection'):
            if self.is_postgres:
                # PostgreSQL connection
                self.local.connection = self.db_module.connect(
                    self.db_path,
                    cursor_factory=self.db_extras.RealDictCursor
                )
            else:
                # SQLite connection
                self.local.connection = self.db_module.connect(self.db_path)
                self.local.connection.row_factory = self.db_module.Row
        
        try:
            yield self.local.connection
        except Exception as e:
            self.local.connection.rollback()
            logger.error(f"Database error: {e}", exc_info=True)
            raise
            
    @contextmanager
    def get_cursor(self):
        """
        Get a database cursor (thread-safe)
        
        Yields:
            Cursor: Database cursor object (psycopg2 or sqlite3)
            
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
            
        Note:
        - SQLite uses ? for parameter placeholders
        - PostgreSQL uses $1, $2, etc.
        """
        if self.is_postgres:
            # Convert ? to $1, $2, etc. for PostgreSQL
            query = self._convert_query_placeholders(query)
            
        with self.get_cursor() as cursor:
            cursor.execute(query, params)
            
            if query.strip().upper().startswith("SELECT"):
                if self.is_postgres:
                    # For PostgreSQL with RealDictCursor, fetchall() already returns dicts
                    return cursor.fetchall()
                else:
                    # For SQLite, convert rows to dicts
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
        if self.is_postgres:
            # Convert ? to $1, $2, etc. for PostgreSQL
            query = self._convert_query_placeholders(query)
            
        with self.get_cursor() as cursor:
            cursor.executemany(query, params_list)
            return cursor.rowcount
            
    def _convert_query_placeholders(self, query: str) -> str:
        """
        Convert SQLite placeholders (?) to PostgreSQL placeholders ($1, $2, etc.)
        
        Args:
            query (str): Query with SQLite-style placeholders
            
        Returns:
            str: Query with PostgreSQL-style placeholders
        """
        if not self.is_postgres:
            return query
            
        parts = query.split('?')
        if len(parts) <= 1:  # No placeholders
            return query
            
        result = ""
        for i in range(len(parts) - 1):
            result += parts[i] + f"${i + 1}"
        result += parts[-1]
        
        return result
    
    def execute_script(self, script: str) -> None:
        """
        Execute a SQL script
        
        Args:
            script (str): SQL script
            
        TypeScript equivalent:
            async executeScript(script: string): Promise<void>
        """
        with self.get_connection() as conn:
            if self.is_postgres:
                # PostgreSQL doesn't have executescript, so we execute directly
                with conn.cursor() as cursor:
                    cursor.execute(script)
                    conn.commit()
            else:
                # SQLite has executescript method
                conn.executescript(script)
    
    def optimize(self) -> None:
        """
        Optimize database performance
        
        TypeScript equivalent:
            async optimize(): Promise<void>
            
        Note:
        - SQLite uses PRAGMA optimize and VACUUM
        - PostgreSQL uses VACUUM ANALYZE
        """
        with self.get_cursor() as cursor:
            if self.is_postgres:
                cursor.execute("VACUUM ANALYZE")
            else:
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
            
        Note:
        - SQLite uses connection.backup method
        - PostgreSQL uses subprocess to call pg_dump
        """
        try:
            if self.is_postgres:
                # For PostgreSQL, use pg_dump through subprocess
                import subprocess
                import shlex
                
                # Parse connection string to get connection parameters
                url = urlparse(self.db_path)
                dbname = url.path.lstrip('/')
                user = url.username
                password = url.password
                host = url.hostname
                port = url.port or 5432
                
                # Build pg_dump command
                cmd = [
                    "pg_dump",
                    f"-h{host}",
                    f"-p{port}",
                    f"-U{user}",
                    "-Fc",  # Custom format for pg_restore
                    f"-f{backup_path}",
                    dbname
                ]
                
                # Set PGPASSWORD environment variable
                env = os.environ.copy()
                if password:
                    env["PGPASSWORD"] = password
                
                # Execute pg_dump
                result = subprocess.run(
                    cmd,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False
                )
                
                if result.returncode != 0:
                    logger.error(f"pg_dump error: {result.stderr.decode()}")
                    return False
                
                return True
            else:
                # For SQLite, use the backup API
                import sqlite3
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
            
        Note:
        - SQLite uses connection.backup method
        - PostgreSQL uses subprocess to call pg_restore
        """
        try:
            if not os.path.exists(backup_path):
                logger.error(f"Backup file not found: {backup_path}", exc_info=True)
                return False
                
            # Close current connection
            if hasattr(self.local, 'connection'):
                self.local.connection.close()
                delattr(self.local, 'connection')
                
            if self.is_postgres:
                # For PostgreSQL, use pg_restore through subprocess
                import subprocess
                import shlex
                
                # Parse connection string to get connection parameters
                url = urlparse(self.db_path)
                dbname = url.path.lstrip('/')
                user = url.username
                password = url.password
                host = url.hostname
                port = url.port or 5432
                
                # Build pg_restore command
                cmd = [
                    "pg_restore",
                    f"-h{host}",
                    f"-p{port}",
                    f"-U{user}",
                    "-d", dbname,
                    "--clean",  # Clean (drop) database objects before recreating
                    backup_path
                ]
                
                # Set PGPASSWORD environment variable
                env = os.environ.copy()
                if password:
                    env["PGPASSWORD"] = password
                
                # Execute pg_restore
                result = subprocess.run(
                    cmd,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False
                )
                
                if result.returncode != 0:
                    logger.error(f"pg_restore error: {result.stderr.decode()}")
                    return False
                
                return True
            else:
                # For SQLite, use the backup API in reverse
                import sqlite3
                backup_conn = sqlite3.connect(backup_path)
                conn = sqlite3.connect(self.db_path)
                backup_conn.backup(conn)
                backup_conn.close()
                conn.close()
                return True
        except Exception as e:
            logger.error(f"Restore error: {e}", exc_info=True)
            return False