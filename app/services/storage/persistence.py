# """
# Persistence Module for CasaLingua
# Provides data persistence and database operations
# """

# import os
# import json
# import logging
# import time
# import sqlite3
# import threading
# from typing import Dict, Any, Optional, List, Tuple, Union, Callable, Set
# from datetime import datetime
# import numpy as np
# import pandas as pd
# from contextlib import contextmanager

# # Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger(__name__)

# class DatabaseManager:
#     """Base database manager class"""
    
#     def __init__(self, db_path: str):
#         """
#         Initialize database manager
        
#         Args:
#             db_path (str): Path to database file
#         """
#         self.db_path = db_path
#         self.local = threading.local()
        
#     @contextmanager
#     def get_connection(self) -> sqlite3.Connection:
#         """
#         Get a database connection (thread-safe)
        
#         Yields:
#             sqlite3.Connection: Database connection
#         """
#         if not hasattr(self.local, 'connection'):
#             self.local.connection = sqlite3.connect(self.db_path)
#             self.local.connection.row_factory = sqlite3.Row
        
#         try:
#             yield self.local.connection
#         except Exception as e:
#             self.local.connection.rollback()
#             logger.error(f"Database error: {e}", exc_info=True)
#             raise
    
#     @contextmanager
#     def get_cursor(self) -> sqlite3.Cursor:
#         """
#         Get a database cursor (thread-safe)
        
#         Yields:
#             sqlite3.Cursor: Database cursor
#         """
#         with self.get_connection() as conn:
#             cursor = conn.cursor()
#             try:
#                 yield cursor
#                 conn.commit()
#             except Exception as e:
#                 conn.rollback()
#                 logger.error(f"Database error: {e}", exc_info=True)
#                 raise
#             finally:
#                 cursor.close()
    
#     def execute_query(self, query: str, params: tuple = ()) -> List[Dict[str, Any]]:
#         """
#         Execute a database query
        
#         Args:
#             query (str): SQL query
#             params (tuple): Query parameters
            
#         Returns:
#             List[Dict[str, Any]]: Query results
#         """
#         with self.get_cursor() as cursor:
#             cursor.execute(query, params)
            
#             if query.strip().upper().startswith("SELECT"):
#                 columns = [column[0] for column in cursor.description]
#                 return [dict(zip(columns, row)) for row in cursor.fetchall()]
            
#             return []
    
#     def execute_many(self, query: str, params_list: List[tuple]) -> int:
#         """
#         Execute a query with multiple parameter sets
        
#         Args:
#             query (str): SQL query
#             params_list (List[tuple]): List of parameter tuples
            
#         Returns:
#             int: Number of affected rows
#         """
#         with self.get_cursor() as cursor:
#             cursor.executemany(query, params_list)
#             return cursor.rowcount
    
#     def execute_script(self, script: str) -> None:
#         """
#         Execute a SQL script
        
#         Args:
#             script (str): SQL script
#         """
#         with self.get_connection() as conn:
#             conn.executescript(script)
    
#     def optimize(self) -> None:
#         """Optimize database performance"""
#         with self.get_cursor() as cursor:
#             cursor.execute("PRAGMA optimize")
#             cursor.execute("VACUUM")
    
#     def backup(self, backup_path: str) -> bool:
#         """
#         Backup the database to a file
        
#         Args:
#             backup_path (str): Path to save backup
            
#         Returns:
#             bool: Success status
#         """
#         try:
#             with self.get_connection() as conn:
#                 backup_conn = sqlite3.connect(backup_path)
#                 conn.backup(backup_conn)
#                 backup_conn.close()
#             return True
#         except Exception as e:
#             logger.error(f"Backup error: {e}", exc_info=True)
#             return False
    
#     def restore(self, backup_path: str) -> bool:
#         """
#         Restore database from a backup
        
#         Args:
#             backup_path (str): Path to backup file
            
#         Returns:
#             bool: Success status
#         """
#         try:
#             if not os.path.exists(backup_path):
#                 logger.error(f"Backup file not found: {backup_path}", exc_info=True)
#                 return False
#             # Close current connection
#             if hasattr(self.local, 'connection'):
#                 self.local.connection.close()
#                 delattr(self.local, 'connection')
#             # Restore from backup
#             backup_conn = sqlite3.connect(backup_path)
#             conn = sqlite3.connect(self.db_path)
#             backup_conn.backup(conn)
#             backup_conn.close()
#             conn.close()
#             return True
#         except Exception as e:
#             logger.error(f"Restore error: {e}", exc_info=True)
#             return False


# class UserManager(DatabaseManager):
#     """Manages user data persistence"""
    
#     def __init__(self, db_path: str):
#         """
#         Initialize user manager
        
#         Args:
#             db_path (str): Path to database file
#         """
#         super().__init__(db_path)
#         self._create_tables()
    
#     def _create_tables(self) -> None:
#         """Create necessary database tables"""
#         script = """
#         CREATE TABLE IF NOT EXISTS users (
#             user_id TEXT PRIMARY KEY,
#             username TEXT NOT NULL,
#             email TEXT UNIQUE,
#             password_hash TEXT,
#             created_at TEXT NOT NULL,
#             updated_at TEXT NOT NULL,
#             settings TEXT,
#             profile TEXT
#         );

#         CREATE TABLE IF NOT EXISTS user_sessions (
#             session_id TEXT PRIMARY KEY,
#             user_id TEXT NOT NULL,
#             started_at TEXT NOT NULL,
#             expires_at TEXT NOT NULL,
#             ip_address TEXT,
#             user_agent TEXT,
#             is_active BOOLEAN DEFAULT 1,
#             FOREIGN KEY (user_id) REFERENCES users (user_id)
#         );

#         CREATE TABLE IF NOT EXISTS user_languages (
#             id INTEGER PRIMARY KEY AUTOINCREMENT,
#             user_id TEXT NOT NULL,
#             language_code TEXT NOT NULL,
#             proficiency_level TEXT,
#             is_native BOOLEAN DEFAULT 0,
#             is_learning BOOLEAN DEFAULT 1,
#             started_at TEXT,
#             last_activity TEXT,
#             FOREIGN KEY (user_id) REFERENCES users (user_id),
#             UNIQUE (user_id, language_code)
#         );
#         """
#         self.execute_script(script)
    
#     def create_user(self, user_data: Dict[str, Any]) -> str:
#         """
#         Create a new user
        
#         Args:
#             user_data (Dict[str, Any]): User data
            
#         Returns:
#             str: User ID
#         """
#         now = datetime.now().isoformat()
        
#         # Generate user ID if not provided
#         if 'user_id' not in user_data:
#             user_data['user_id'] = f"user_{int(time.time())}_{os.urandom(4).hex()}"
        
#         # Format user data
#         user = {
#             'user_id': user_data['user_id'],
#             'username': user_data['username'],
#             'email': user_data.get('email'),
#             'password_hash': user_data.get('password_hash'),
#             'created_at': now,
#             'updated_at': now,
#             'settings': json.dumps(user_data.get('settings', {})),
#             'profile': json.dumps(user_data.get('profile', {}))
#         }
        
#         # Insert user
#         query = """
#         INSERT INTO users 
#         (user_id, username, email, password_hash, created_at, updated_at, settings, profile)
#         VALUES (?, ?, ?, ?, ?, ?, ?, ?)
#         """
#         params = (
#             user['user_id'], user['username'], user['email'], user['password_hash'],
#             user['created_at'], user['updated_at'], user['settings'], user['profile']
#         )
        
#         try:
#             self.execute_query(query, params)
#             logger.info(f"Created user: {user['username']} ({user['user_id']})")
#             # Add user languages if provided
#             if 'languages' in user_data:
#                 for lang in user_data['languages']:
#                     self.add_user_language(user['user_id'], lang)
#             return user['user_id']
#         except Exception as e:
#             logger.error(f"Error creating user: {e}", exc_info=True)
#             raise
    
#     def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
#         """
#         Get user by ID
        
#         Args:
#             user_id (str): User ID
            
#         Returns:
#             Dict[str, Any]: User data or None if not found
#         """
#         query = "SELECT * FROM users WHERE user_id = ?"
#         results = self.execute_query(query, (user_id,))
        
#         if not results:
#             return None
        
#         user = dict(results[0])
        
#         # Parse JSON fields
#         try:
#             user['settings'] = json.loads(user['settings']) if user['settings'] else {}
#         except:
#             user['settings'] = {}
            
#         try:
#             user['profile'] = json.loads(user['profile']) if user['profile'] else {}
#         except:
#             user['profile'] = {}
        
#         # Get user languages
#         languages_query = "SELECT * FROM user_languages WHERE user_id = ?"
#         languages = self.execute_query(languages_query, (user_id,))
#         user['languages'] = languages
        
#         return user
    
#     def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
#         """
#         Get user by email
        
#         Args:
#             email (str): User email
            
#         Returns:
#             Dict[str, Any]: User data or None if not found
#         """
#         query = "SELECT user_id FROM users WHERE email = ?"
#         results = self.execute_query(query, (email,))
        
#         if not results:
#             return None
            
#         return self.get_user(results[0]['user_id'])
    
#     def update_user(self, user_id: str, user_data: Dict[str, Any]) -> bool:
#         """
#         Update user data
        
#         Args:
#             user_id (str): User ID
#             user_data (Dict[str, Any]): User data to update
            
#         Returns:
#             bool: Success status
#         """
#         # Get current user data
#         current_user = self.get_user(user_id)
#         if not current_user:
#             logger.error(f"User not found: {user_id}")
#             return False
        
#         # Update user fields
#         updates = []
#         params = []
        
#         if 'username' in user_data:
#             updates.append("username = ?")
#             params.append(user_data['username'])
            
#         if 'email' in user_data:
#             updates.append("email = ?")
#             params.append(user_data['email'])
            
#         if 'password_hash' in user_data:
#             updates.append("password_hash = ?")
#             params.append(user_data['password_hash'])
            
#         if 'settings' in user_data:
#             updates.append("settings = ?")
#             params.append(json.dumps(user_data['settings']))
            
#         if 'profile' in user_data:
#             updates.append("profile = ?")
#             params.append(json.dumps(user_data['profile']))
        
#         # Always update the updated_at timestamp
#         updates.append("updated_at = ?")
#         params.append(datetime.now().isoformat())
        
#         # Add user_id to params
#         params.append(user_id)
        
#         if not updates:
#             return True  # Nothing to update
        
#         # Execute update query
#         query = f"UPDATE users SET {', '.join(updates)} WHERE user_id = ?"
#         self.execute_query(query, tuple(params))
        
#         # Update languages if provided
#         if 'languages' in user_data:
#             # Clear existing languages
#             self.execute_query("DELETE FROM user_languages WHERE user_id = ?", (user_id,))
            
#             # Add new languages
#             for lang in user_data['languages']:
#                 self.add_user_language(user_id, lang)
        
#         return True
    
#     def delete_user(self, user_id: str) -> bool:
#         """
#         Delete a user
        
#         Args:
#             user_id (str): User ID
            
#         Returns:
#             bool: Success status
#         """
#         # Delete related records first
#         self.execute_query("DELETE FROM user_languages WHERE user_id = ?", (user_id,))
#         self.execute_query("DELETE FROM user_sessions WHERE user_id = ?", (user_id,))
        
#         # Delete user
#         self.execute_query("DELETE FROM users WHERE user_id = ?", (user_id,))
        
#         logger.info(f"Deleted user: {user_id}")
#         return True
    
#     def add_user_language(self, user_id: str, language_data: Dict[str, Any]) -> int:
#         """
#         Add a language to user profile
        
#         Args:
#             user_id (str): User ID
#             language_data (Dict[str, Any]): Language data
            
#         Returns:
#             int: Language record ID
#         """
#         now = datetime.now().isoformat()
        
#         query = """
#         INSERT OR REPLACE INTO user_languages
#         (user_id, language_code, proficiency_level, is_native, is_learning, started_at, last_activity)
#         VALUES (?, ?, ?, ?, ?, ?, ?)
#         """
        
#         params = (
#             user_id,
#             language_data['language_code'],
#             language_data.get('proficiency_level', 'beginner'),
#             language_data.get('is_native', False),
#             language_data.get('is_learning', True),
#             language_data.get('started_at', now),
#             language_data.get('last_activity', now)
#         )
        
#         self.execute_query(query, params)
        
#         # Get the ID of the inserted record
#         result = self.execute_query(
#             "SELECT id FROM user_languages WHERE user_id = ? AND language_code = ?",
#             (user_id, language_data['language_code'])
#         )
        
#         return result[0]['id'] if result else -1
    
#     def update_language_activity(self, user_id: str, language_code: str) -> bool:
#         """
#         Update last activity timestamp for a language
        
#         Args:
#             user_id (str): User ID
#             language_code (str): Language code
            
#         Returns:
#             bool: Success status
#         """
#         now = datetime.now().isoformat()
        
#         query = """
#         UPDATE user_languages 
#         SET last_activity = ? 
#         WHERE user_id = ? AND language_code = ?
#         """
        
#         self.execute_query(query, (now, user_id, language_code))
#         return True
    
#     def create_session(self, user_id: str, expires_minutes: int = 60*24) -> Dict[str, Any]:
#         """
#         Create a new user session
        
#         Args:
#             user_id (str): User ID
#             expires_minutes (int): Session expiry time in minutes
            
#         Returns:
#             Dict[str, Any]: Session data
#         """
#         session_id = f"session_{int(time.time())}_{os.urandom(8).hex()}"
#         now = datetime.now()
        
#         session = {
#             'session_id': session_id,
#             'user_id': user_id,
#             'started_at': now.isoformat(),
#             'expires_at': (now + pd.Timedelta(minutes=expires_minutes)).isoformat(),
#             'is_active': True
#         }
        
#         query = """
#         INSERT INTO user_sessions
#         (session_id, user_id, started_at, expires_at, is_active)
#         VALUES (?, ?, ?, ?, ?)
#         """
        
#         params = (
#             session['session_id'], 
#             session['user_id'],
#             session['started_at'],
#             session['expires_at'],
#             session['is_active']
#         )
        
#         self.execute_query(query, params)
        
#         return session
    
#     def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
#         """
#         Get session by ID
        
#         Args:
#             session_id (str): Session ID
            
#         Returns:
#             Dict[str, Any]: Session data or None if not found or expired
#         """
#         query = "SELECT * FROM user_sessions WHERE session_id = ?"
#         results = self.execute_query(query, (session_id,))
        
#         if not results:
#             return None
            
#         session = dict(results[0])
        
#         # Check if session is expired
#         now = datetime.now().isoformat()
#         if session['expires_at'] < now or not session['is_active']:
#             self.invalidate_session(session_id)
#             return None
            
#         return session
    
#     def invalidate_session(self, session_id: str) -> bool:
#         """
#         Invalidate a user session
        
#         Args:
#             session_id (str): Session ID
            
#         Returns:
#             bool: Success status
#         """
#         query = "UPDATE user_sessions SET is_active = 0 WHERE session_id = ?"
#         self.execute_query(query, (session_id,))
        
#         return True
    
#     def get_active_users(self, days: int = 30) -> List[Dict[str, Any]]:
#         """
#         Get users who have been active recently
        
#         Args:
#             days (int): Number of days to look back
            
#         Returns:
#             List[Dict[str, Any]]: List of active users
#         """
#         threshold = (datetime.now() - pd.Timedelta(days=days)).isoformat()
        
#         query = """
#         SELECT DISTINCT u.* 
#         FROM users u
#         JOIN user_sessions s ON u.user_id = s.user_id
#         WHERE s.started_at > ?
#         ORDER BY u.username
#         """
        
#         return self.execute_query(query, (threshold,))


# class ContentManager(DatabaseManager):
#     """Manages language learning content persistence"""
    
#     def __init__(self, db_path: str):
#         """
#         Initialize content manager
        
#         Args:
#             db_path (str): Path to database file
#         """
#         super().__init__(db_path)
#         self._create_tables()
    
#     def _create_tables(self) -> None:
#         """Create necessary database tables"""
#         script = """
#         CREATE TABLE IF NOT EXISTS content_items (
#             content_id TEXT PRIMARY KEY,
#             title TEXT NOT NULL,
#             content_type TEXT NOT NULL,
#             language_code TEXT NOT NULL,
#             difficulty_level TEXT,
#             content TEXT NOT NULL,
#             metadata TEXT,
#             tags TEXT,
#             created_at TEXT NOT NULL,
#             updated_at TEXT NOT NULL
#         );

#         CREATE TABLE IF NOT EXISTS content_categories (
#             category_id TEXT PRIMARY KEY,
#             name TEXT NOT NULL,
#             description TEXT,
#             parent_id TEXT,
#             language_code TEXT,
#             created_at TEXT NOT NULL,
#             FOREIGN KEY (parent_id) REFERENCES content_categories (category_id)
#         );

#         CREATE TABLE IF NOT EXISTS content_category_items (
#             id INTEGER PRIMARY KEY AUTOINCREMENT,
#             category_id TEXT NOT NULL,
#             content_id TEXT NOT NULL,
#             position INTEGER,
#             FOREIGN KEY (category_id) REFERENCES content_categories (category_id),
#             FOREIGN KEY (content_id) REFERENCES content_items (content_id),
#             UNIQUE (category_id, content_id)
#         );

#         CREATE TABLE IF NOT EXISTS content_embeddings (
#             id INTEGER PRIMARY KEY AUTOINCREMENT,
#             content_id TEXT NOT NULL,
#             chunk_id TEXT NOT NULL,
#             chunk_text TEXT NOT NULL,
#             embedding BLOB,
#             model_name TEXT,
#             created_at TEXT NOT NULL,
#             FOREIGN KEY (content_id) REFERENCES content_items (content_id),
#             UNIQUE (content_id, chunk_id)
#         );

#         CREATE INDEX IF NOT EXISTS idx_content_language ON content_items (language_code);
#         CREATE INDEX IF NOT EXISTS idx_content_type ON content_items (content_type);
#         CREATE INDEX IF NOT EXISTS idx_content_difficulty ON content_items (difficulty_level);
#         CREATE INDEX IF NOT EXISTS idx_category_language ON content_categories (language_code);
#         CREATE INDEX IF NOT EXISTS idx_category_parent ON content_categories (parent_id);
#         """
#         self.execute_script(script)
    
#     def create_content(self, content_data: Dict[str, Any]) -> str:
#         """
#         Create a new content item
        
#         Args:
#             content_data (Dict[str, Any]): Content data
            
#         Returns:
#             str: Content ID
#         """
#         now = datetime.now().isoformat()
        
#         # Generate content ID if not provided
#         if 'content_id' not in content_data:
#             content_data['content_id'] = f"content_{int(time.time())}_{os.urandom(4).hex()}"
        
#         # Process metadata and tags
#         metadata = json.dumps(content_data.get('metadata', {}))
#         tags = ','.join(content_data.get('tags', []))
        
#         # Format content data
#         content = {
#             'content_id': content_data['content_id'],
#             'title': content_data['title'],
#             'content_type': content_data['content_type'],
#             'language_code': content_data['language_code'],
#             'difficulty_level': content_data.get('difficulty_level', 'intermediate'),
#             'content': content_data['content'],
#             'metadata': metadata,
#             'tags': tags,
#             'created_at': now,
#             'updated_at': now
#         }
        
#         # Insert content
#         query = """
#         INSERT INTO content_items 
#         (content_id, title, content_type, language_code, difficulty_level, 
#          content, metadata, tags, created_at, updated_at)
#         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
#         """
#         params = (
#             content['content_id'], content['title'], content['content_type'],
#             content['language_code'], content['difficulty_level'], content['content'],
#             content['metadata'], content['tags'], content['created_at'], content['updated_at']
#         )
        
#         try:
#             self.execute_query(query, params)
#             logger.info(f"Created content: {content['title']} ({content['content_id']})")
#             # Add to categories if provided
#             if 'categories' in content_data:
#                 for category_id in content_data['categories']:
#                     self.add_content_to_category(content['content_id'], category_id)
#             return content['content_id']
#         except Exception as e:
#             logger.error(f"Error creating content: {e}", exc_info=True)
#             raise
    
#     def get_content(self, content_id: str) -> Optional[Dict[str, Any]]:
#         """
#         Get content by ID
        
#         Args:
#             content_id (str): Content ID
            
#         Returns:
#             Dict[str, Any]: Content data or None if not found
#         """
#         query = "SELECT * FROM content_items WHERE content_id = ?"
#         results = self.execute_query(query, (content_id,))
        
#         if not results:
#             return None
        
#         content = dict(results[0])
        
#         # Parse JSON fields
#         try:
#             content['metadata'] = json.loads(content['metadata']) if content['metadata'] else {}
#         except:
#             content['metadata'] = {}
            
#         # Parse tags
#         content['tags'] = content['tags'].split(',') if content['tags'] else []
        
#         # Get categories
#         categories_query = """
#         SELECT c.* FROM content_categories c
#         JOIN content_category_items ci ON c.category_id = ci.category_id
#         WHERE ci.content_id = ?
#         """
#         categories = self.execute_query(categories_query, (content_id,))
#         content['categories'] = categories
        
#         return content
    
#     def update_content(self, content_id: str, content_data: Dict[str, Any]) -> bool:
#         """
#         Update content data
        
#         Args:
#             content_id (str): Content ID
#             content_data (Dict[str, Any]): Content data to update
            
#         Returns:
#             bool: Success status
#         """
#         # Get current content data
#         current_content = self.get_content(content_id)
#         if not current_content:
#             logger.error(f"Content not found: {content_id}")
#             return False
        
#         # Update content fields
#         updates = []
#         params = []
        
#         if 'title' in content_data:
#             updates.append("title = ?")
#             params.append(content_data['title'])
            
#         if 'content_type' in content_data:
#             updates.append("content_type = ?")
#             params.append(content_data['content_type'])
            
#         if 'language_code' in content_data:
#             updates.append("language_code = ?")
#             params.append(content_data['language_code'])
            
#         if 'difficulty_level' in content_data:
#             updates.append("difficulty_level = ?")
#             params.append(content_data['difficulty_level'])
            
#         if 'content' in content_data:
#             updates.append("content = ?")
#             params.append(content_data['content'])
            
#         if 'metadata' in content_data:
#             updates.append("metadata = ?")
#             params.append(json.dumps(content_data['metadata']))
            
#         if 'tags' in content_data:
#             updates.append("tags = ?")
#             params.append(','.join(content_data['tags']))
        
#         # Always update the updated_at timestamp
#         updates.append("updated_at = ?")
#         params.append(datetime.now().isoformat())
        
#         # Add content_id to params
#         params.append(content_id)
        
#         if not updates:
#             return True  # Nothing to update
        
#         # Execute update query
#         query = f"UPDATE content_items SET {', '.join(updates)} WHERE content_id = ?"
#         self.execute_query(query, tuple(params))
        
#         # Update categories if provided
#         if 'categories' in content_data:
#             # Clear existing categories
#             self.execute_query("DELETE FROM content_category_items WHERE content_id = ?", (content_id,))
            
#             # Add new categories
#             for category_id in content_data['categories']:
#                 self.add_content_to_category(content_id, category_id)
        
#         return True
    
#     def delete_content(self, content_id: str) -> bool:
#         """
#         Delete a content item
        
#         Args:
#             content_id (str): Content ID
            
#         Returns:
#             bool: Success status
#         """
#         # Delete related records first
#         self.execute_query("DELETE FROM content_category_items WHERE content_id = ?", (content_id,))
#         self.execute_query("DELETE FROM content_embeddings WHERE content_id = ?", (content_id,))
        
#         # Delete content
#         self.execute_query("DELETE FROM content_items WHERE content_id = ?", (content_id,))
        
#         logger.info(f"Deleted content: {content_id}")
#         return True
    
#     def create_category(self, category_data: Dict[str, Any]) -> str:
#         """
#         Create a new content category
        
#         Args:
#             category_data (Dict[str, Any]): Category data
            
#         Returns:
#             str: Category ID
#         """
#         now = datetime.now().isoformat()
        
#         # Generate category ID if not provided
#         if 'category_id' not in category_data:
#             category_data['category_id'] = f"category_{int(time.time())}_{os.urandom(4).hex()}"
        
#         # Format category data
#         category = {
#             'category_id': category_data['category_id'],
#             'name': category_data['name'],
#             'description': category_data.get('description', ''),
#             'parent_id': category_data.get('parent_id'),
#             'language_code': category_data.get('language_code'),
#             'created_at': now
#         }
        
#         # Insert category
#         query = """
#         INSERT INTO content_categories 
#         (category_id, name, description, parent_id, language_code, created_at)
#         VALUES (?, ?, ?, ?, ?, ?)
#         """
#         params = (
#             category['category_id'], category['name'], category['description'],
#             category['parent_id'], category['language_code'], category['created_at']
#         )
        
#         try:
#             self.execute_query(query, params)
#             logger.info(f"Created category: {category['name']} ({category['category_id']})")
#             return category['category_id']
#         except Exception as e:
#             logger.error(f"Error creating category: {e}", exc_info=True)
#             raise
    
#     def get_category(self, category_id: str) -> Optional[Dict[str, Any]]:
#         """
#         Get category by ID
        
#         Args:
#             category_id (str): Category ID
            
#         Returns:
#             Dict[str, Any]: Category data or None if not found
#         """
#         query = "SELECT * FROM content_categories WHERE category_id = ?"
#         results = self.execute_query(query, (category_id,))
        
#         if not results:
#             return None
        
#         category = dict(results[0])
        
#         # Get subcategories
#         subcategories_query = "SELECT * FROM content_categories WHERE parent_id = ?"
#         subcategories = self.execute_query(subcategories_query, (category_id,))
#         category['subcategories'] = subcategories
        
#         # Get contents
#         contents_query = """
#         SELECT ci.* FROM content_items ci
#         JOIN content_category_items cci ON ci.content_id = cci.content_id
#         WHERE cci.category_id = ?
#         """
#         contents = self.execute_query(contents_query, (category_id,))
#         category['contents'] = contents
        
#         return category
    
#     def get_categories_by_language(self, language_code: str, parent_id: Optional[str] = None) -> List[Dict[str, Any]]:
#         """
#         Get categories by language
        
#         Args:
#             language_code (str): Language code
#             parent_id (str, optional): Parent category ID
            
#         Returns:
#             List[Dict[str, Any]]: List of categories
#         """
#         if parent_id:
#             query = """
#             SELECT * FROM content_categories 
#             WHERE language_code = ? AND parent_id = ?
#             ORDER BY name
#             """
#             return self.execute_query(query, (language_code, parent_id))
#         else:
#             query = """
#             SELECT * FROM content_categories 
#             WHERE language_code = ? AND parent_id IS NULL
#             ORDER BY name
#             """
#             return self.execute_query(query, (language_code,))
    
#     def add_content_to_category(self, content_id: str, category_id: str, position: Optional[int] = None) -> bool:
#         """
#         Add content to a category
        
#         Args:
#             content_id (str): Content ID
#             category_id (str): Category ID
#             position (int, optional): Position within category
            
#         Returns:
#             bool: Success status
#         """
#         query = """
#         INSERT OR REPLACE INTO content_category_items
#         (content_id, category_id, position)
#         VALUES (?, ?, ?)
#         """
        
#         self.execute_query(query, (content_id, category_id, position))
#         return True
    
#     def get_content_by_type(self, content_type: str, language_code: str, 
#                          difficulty_level: Optional[str] = None, 
#                          limit: int = 100) -> List[Dict[str, Any]]:
#         """
#         Get content by type and language
        
#         Args:
#             content_type (str): Content type
#             language_code (str): Language code
#             difficulty_level (str, optional): Difficulty level
#             limit (int): Maximum number of results
            
#         Returns:
#             List[Dict[str, Any]]: List of content items
#         """
#         if difficulty_level:
#             query = """
#             SELECT * FROM content_items 
#             WHERE content_type = ? AND language_code = ? AND difficulty_level = ?
#             ORDER BY created_at DESC
#             LIMIT ?
#             """
#             return self.execute_query(query, (content_type, language_code, difficulty_level, limit))
#         else:
#             query = """
#             SELECT * FROM content_items 
#             WHERE content_type = ? AND language_code = ?
#             ORDER BY created_at DESC
#             LIMIT ?
#             """
#             return self.execute_query(query, (content_type, language_code, limit))
    
#     def search_content(self, search_term: str, language_code: Optional[str] = None, 
#                     content_type: Optional[str] = None, 
#                     limit: int = 100) -> List[Dict[str, Any]]:
#         """
#         Search content by term
        
#         Args:
#             search_term (str): Search term
#             language_code (str, optional): Language code
#             content_type (str, optional): Content type
#             limit (int): Maximum number of results
            
#         Returns:
#             List[Dict[str, Any]]: List of matching content items
#         """
#         search_param = f"%{search_term}%"
        
#         # Build query based on filters
#         query = "SELECT * FROM content_items WHERE (title LIKE ? OR content LIKE ? OR tags LIKE ?)"
#         params = [search_param, search_param, search_param]
        
#         if language_code:
#             query += " AND language_code = ?"
#             params.append(language_code)
        
#         if content_type:
#             query += " AND content_type = ?"
#             params.append(content_type)
        
#         query += " ORDER BY created_at DESC LIMIT ?"
#         params.append(limit)
        
#         return self.execute_query(query, tuple(params))
    
#     def store_embedding(self, content_id: str, chunk_id: str, chunk_text: str, 
#                       embedding: np.ndarray, model_name: str) -> int:
#         """
#         Store a content embedding
        
#         Args:
#             content_id (str): Content ID
#             chunk_id (str): Chunk ID
#             chunk_text (str): Text chunk
#             embedding (np.ndarray): Embedding vector
#             model_name (str): Embedding model name
            
#         Returns:
#             int: Embedding record ID
#         """
#         now = datetime.now().isoformat()
        
#         # Convert embedding to binary
#         embedding_binary = embedding.tobytes()
        
#         query = """
#         INSERT OR REPLACE INTO content_embeddings
#         (content_id, chunk_id, chunk_text, embedding, model_name, created_at)
#         VALUES (?, ?, ?, ?, ?, ?)
#         """
        
#         params = (content_id, chunk_id, chunk_text, embedding_binary, model_name, now)
        
#         self.execute_query(query, params)
        
#         # Get the ID of the inserted record
#         result = self.execute_query(
#             "SELECT id FROM content_embeddings WHERE content_id = ? AND chunk_id = ?",
#             (content_id, chunk_id)
#         )
        
#         return result[0]['id'] if result else -1
    
#     def get_embedding(self, content_id: str, chunk_id: str) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
#         """
#         Get an embedding by content ID and chunk ID
        
#         Args:
#             content_id (str): Content ID
#             chunk_id (str): Chunk ID
            
#         Returns:
#             Tuple[np.ndarray, Dict[str, Any]]: (embedding, metadata) or None if not found
#         """
#         query = "SELECT * FROM content_embeddings WHERE content_id = ? AND chunk_id = ?"
#         results = self.execute_query(query, (content_id, chunk_id))
        
#         if not results:
#             return None
        
#         record = dict(results[0])
        
#         # Convert binary to numpy array
#         embedding_binary = record['embedding']
#         embedding = np.frombuffer(embedding_binary, dtype=np.float32)
        
#         metadata = {
#             'id': record['id'],
#             'content_id': record['content_id'],
#             'chunk_id': record['chunk_id'],
#             'chunk_text': record['chunk_text'],
#             'model_name': record['model_name'],
#             'created_at': record['created_at']
#         }
        
#         return embedding, metadata
    
#     def get_all_embeddings(self, model_name: Optional[str] = None) -> List[Tuple[np.ndarray, Dict[str, Any]]]:
#         """
#         Get all embeddings
        
#         Args:
#             model_name (str, optional): Filter by model name
            
#         Returns:
#             List[Tuple[np.ndarray, Dict[str, Any]]]: List of (embedding, metadata) tuples
#         """
#         if model_name:
#             query = "SELECT * FROM content_embeddings WHERE model_name = ?"
#             results = self.execute_query(query, (model_name,))
#         else:
#             query = "SELECT * FROM content_embeddings"
#             results = self.execute_query(query)
        
#         embeddings = []
        
#         for record in results:
#             record = dict(record)
            
#             # Convert binary to numpy array
#             embedding_binary = record['embedding']
#             embedding = np.frombuffer(embedding_binary, dtype=np.float32)
            
#             metadata = {
#                 'id': record['id'],
#                 'content_id': record['content_id'],
#                 'chunk_id': record['chunk_id'],
#                 'chunk_text': record['chunk_text'],
#                 'model_name': record['model_name'],
#                 'created_at': record['created_at']
#             }
            
#             embeddings.append((embedding, metadata))
        
#         return embeddings


# class ProgressManager(DatabaseManager):
#     """Manages user learning progress persistence"""
    
#     def __init__(self, db_path: str):
#         """
#         Initialize progress manager
        
#         Args:
#             db_path (str): Path to database file
#         """
#         super().__init__(db_path)
#         self._create_tables()
    
#     def _create_tables(self) -> None:
#         """Create necessary database tables"""
#         script = """
#         CREATE TABLE IF NOT EXISTS vocabulary_items (
#             id INTEGER PRIMARY KEY AUTOINCREMENT,
#             user_id TEXT NOT NULL,
#             language_code TEXT NOT NULL,
#             word TEXT NOT NULL,
#             translation TEXT NOT NULL,
#             context TEXT,
#             category TEXT,
#             level INTEGER DEFAULT 0,
#             added_date TEXT NOT NULL,
#             last_review TEXT,
#             next_review TEXT,
#             UNIQUE (user_id, language_code, word)
#         );
        
#         CREATE TABLE IF NOT EXISTS vocabulary_reviews (
#             id INTEGER PRIMARY KEY AUTOINCREMENT,
#             vocabulary_id INTEGER NOT NULL,
#             review_date TEXT NOT NULL,
#             success BOOLEAN NOT NULL,
#             old_level INTEGER NOT NULL,
#             new_level INTEGER NOT NULL,
#             FOREIGN KEY (vocabulary_id) REFERENCES vocabulary_items (id)
#         );
        
#         CREATE TABLE IF NOT EXISTS grammar_topics (
#             id INTEGER PRIMARY KEY AUTOINCREMENT,
#             user_id TEXT NOT NULL,
#             language_code TEXT NOT NULL,
#             topic_id TEXT NOT NULL,
#             name TEXT NOT NULL,
#             description TEXT,
#             category TEXT,
#             difficulty TEXT,
#             mastery_level REAL DEFAULT 0.0,
#             practice_count INTEGER DEFAULT 0,
#             last_practice TEXT,
#             UNIQUE (user_id, language_code, topic_id)
#         );
        
#         CREATE TABLE IF NOT EXISTS grammar_practices (
#             id INTEGER PRIMARY KEY AUTOINCREMENT,
#             grammar_id INTEGER NOT NULL,
#             practice_date TEXT NOT NULL,
#             score REAL NOT NULL,
#             old_mastery REAL NOT NULL,
#             new_mastery REAL NOT NULL,
#             details TEXT,
#             FOREIGN KEY (grammar_id) REFERENCES grammar_topics (id)
#         );
        
#         CREATE TABLE IF NOT EXISTS learning_activities (
#             id INTEGER PRIMARY KEY AUTOINCREMENT,
#             user_id TEXT NOT NULL,
#             language_code TEXT NOT NULL,
#             activity_type TEXT NOT NULL,
#             duration INTEGER NOT NULL,
#             timestamp TEXT NOT NULL,
#             details TEXT
#         );
        
#         CREATE TABLE IF NOT EXISTS learning_goals (
#             id INTEGER PRIMARY KEY AUTOINCREMENT,
#             user_id TEXT NOT NULL,
#             language_code TEXT NOT NULL,
#             description TEXT NOT NULL,
#             target_date TEXT,
#             created_date TEXT NOT NULL,
#             completed BOOLEAN DEFAULT 0,
#             progress REAL DEFAULT 0.0,
#             metrics TEXT
#         );
        
#         CREATE TABLE IF NOT EXISTS daily_goals (
#             id INTEGER PRIMARY KEY AUTOINCREMENT,
#             user_id TEXT NOT NULL,
#             language_code TEXT NOT NULL,
#             goal_type TEXT NOT NULL,
#             value INTEGER NOT NULL,
#             created_date TEXT NOT NULL,
#             updated_date TEXT NOT NULL,
#             UNIQUE (user_id, language_code, goal_type)
#         );
        
#         CREATE INDEX IF NOT EXISTS idx_vocabulary_user_lang ON vocabulary_items (user_id, language_code);
#         CREATE INDEX IF NOT EXISTS idx_grammar_user_lang ON grammar_topics (user_id, language_code);
#         CREATE INDEX IF NOT EXISTS idx_activities_user_lang ON learning_activities (user_id, language_code);
#         CREATE INDEX IF NOT EXISTS idx_goals_user_lang ON learning_goals (user_id, language_code);
#         """
#         self.execute_script(script)
    
#     def add_vocabulary(self, vocabulary_data: Dict[str, Any]) -> int:
#         """
#         Add a vocabulary word
        
#         Args:
#             vocabulary_data (Dict[str, Any]): Vocabulary data
            
#         Returns:
#             int: Vocabulary record ID
#         """
#         now = datetime.now().isoformat()
        
#         query = """
#         INSERT OR REPLACE INTO vocabulary_items
#         (user_id, language_code, word, translation, context, category, level, 
#          added_date, last_review, next_review)
#         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
#         """
        
#         params = (
#             vocabulary_data['user_id'],
#             vocabulary_data['language_code'],
#             vocabulary_data['word'],
#             vocabulary_data['translation'],
#             vocabulary_data.get('context'),
#             vocabulary_data.get('category'),
#             vocabulary_data.get('level', 0),
#             vocabulary_data.get('added_date', now),
#             vocabulary_data.get('last_review'),
#             vocabulary_data.get('next_review', now)
#         )
        
#         self.execute_query(query, params)
        
#         # Get the ID of the inserted record
#         result = self.execute_query(
#             "SELECT id FROM vocabulary_items WHERE user_id = ? AND language_code = ? AND word = ?",
#             (vocabulary_data['user_id'], vocabulary_data['language_code'], vocabulary_data['word'])
#         )
        
#         return result[0]['id'] if result else -1
    
#     def get_vocabulary(self, user_id: str, language_code: str, word: Optional[str] = None) -> List[Dict[str, Any]]:
#         """
#         Get vocabulary words
        
#         Args:
#             user_id (str): User ID
#             language_code (str): Language code
#             word (str, optional): Specific word to get
            
#         Returns:
#             List[Dict[str, Any]]: List of vocabulary items
#         """
#         if word:
#             query = """
#             SELECT * FROM vocabulary_items 
#             WHERE user_id = ? AND language_code = ? AND word = ?
#             """
#             results = self.execute_query(query, (user_id, language_code, word))
#         else:
#             query = """
#             SELECT * FROM vocabulary_items 
#             WHERE user_id = ? AND language_code = ?
#             ORDER BY word
#             """
#             results = self.execute_query(query, (user_id, language_code))
        
#         # Get review history for each word
#         for item in results:
#             item = dict(item)
#             vocab_id = item['id']
            
#             history_query = """
#             SELECT * FROM vocabulary_reviews 
#             WHERE vocabulary_id = ?
#             ORDER BY review_date DESC
#             """
#             history = self.execute_query(history_query, (vocab_id,))
#             item['review_history'] = history
        
#         return results
    
#     def update_vocabulary_review(self, vocabulary_id: int, success: bool) -> Dict[str, Any]:
#         """
#         Update review progress for a vocabulary word
        
#         Args:
#             vocabulary_id (int): Vocabulary ID
#             success (bool): Whether recall was successful
            
#         Returns:
#             Dict[str, Any]: Updated vocabulary data
#         """
#         now = datetime.now().isoformat()
        
#         # Get current vocabulary data
#         query = "SELECT * FROM vocabulary_items WHERE id = ?"
#         results = self.execute_query(query, (vocabulary_id,))
        
#         if not results:
#             logger.error(f"Vocabulary not found: {vocabulary_id}")
#             return {}
            
#         vocab = dict(results[0])
#         current_level = vocab['level']
        
#         # Update level based on success
#         if success:
#             # Increment level (max 6)
#             new_level = min(current_level + 1, 6)
#         else:
#             # Decrement level (min 0)
#             new_level = max(current_level - 1, 0)
        
#         # Calculate next review date based on spaced repetition
#         days_until_next_review = self._calculate_review_interval(new_level, success)
#         next_review = (datetime.now() + pd.Timedelta(days=days_until_next_review)).isoformat()
        
#         # Update vocabulary data
#         update_query = """
#         UPDATE vocabulary_items 
#         SET level = ?, last_review = ?, next_review = ?
#         WHERE id = ?
#         """
#         self.execute_query(update_query, (new_level, now, next_review, vocabulary_id))
        
#         # Add to review history
#         history_query = """
#         INSERT INTO vocabulary_reviews
#         (vocabulary_id, review_date, success, old_level, new_level)
#         VALUES (?, ?, ?, ?, ?)
#         """
#         self.execute_query(history_query, (vocabulary_id, now, success, current_level, new_level))
        
#         # Get updated vocabulary data
#         updated_query = "SELECT * FROM vocabulary_items WHERE id = ?"
#         updated_results = self.execute_query(updated_query, (vocabulary_id,))
        
#         if updated_results:
#             return dict(updated_results[0])
        
#         return {}
    
#     def _calculate_review_interval(self, level: int, success: bool) -> int:
#         """
#         Calculate days until next review using spaced repetition
        
#         Args:
#             level (int): Current knowledge level
#             success (bool): Whether recall was successful
            
#         Returns:
#             int: Days until next review
#         """
#         if not success:
#             return 1  # Review again tomorrow if failed
        
#         # Spaced repetition intervals based on level
#         intervals = {
#             0: 1,   # New word - review next day
#             1: 2,   # Review after 2 days
#             2: 4,   # Review after 4 days
#             3: 7,   # Review after 1 week
#             4: 14,  # Review after 2 weeks
#             5: 30,  # Review after 1 month
#             6: 90   # Review after 3 months (mastered)
#         }
        
#         return intervals.get(level, 1)
    
#     def get_words_to_review(self, user_id: str, language_code: str, limit: int = 10) -> List[Dict[str, Any]]:
#         """
#         Get words due for review
        
#         Args:
#             user_id (str): User ID
#             language_code (str): Language code
#             limit (int): Maximum number of words to return
            
#         Returns:
#             List[Dict[str, Any]]: Words due for review
#         """
#         now = datetime.now().isoformat()
        
#         query = """
#         SELECT * FROM vocabulary_items 
#         WHERE user_id = ? AND language_code = ? AND next_review <= ?
#         ORDER BY level, next_review
#         LIMIT ?
#         """
        
#         return self.execute_query(query, (user_id, language_code, now, limit))
    
#     def add_grammar_topic(self, grammar_data: Dict[str, Any]) -> int:
#         """
#         Add a grammar topic
        
#         Args:
#             grammar_data (Dict[str, Any]): Grammar topic data
            
#         Returns:
#             int: Grammar record ID
#         """
#         query = """
#         INSERT OR REPLACE INTO grammar_topics
#         (user_id, language_code, topic_id, name, description, category, 
#          difficulty, mastery_level, practice_count, last_practice)
#         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
#         """
        
#         params = (
#             grammar_data['user_id'],
#             grammar_data['language_code'],
#             grammar_data['topic_id'],
#             grammar_data['name'],
#             grammar_data.get('description', ''),
#             grammar_data.get('category', ''),
#             grammar_data.get('difficulty', 'intermediate'),
#             grammar_data.get('mastery_level', 0.0),
#             grammar_data.get('practice_count', 0),
#             grammar_data.get('last_practice')
#         )
        
#         self.execute_query(query, params)
        
#         # Get the ID of the inserted record
#         result = self.execute_query(
#             "SELECT id FROM grammar_topics WHERE user_id = ? AND language_code = ? AND topic_id = ?",
#             (grammar_data['user_id'], grammar_data['language_code'], grammar_data['topic_id'])
#         )
        
#         return result[0]['id'] if result else -1
    
#     def get_grammar_topics(self, user_id: str, language_code: str, 
#                         topic_id: Optional[str] = None) -> List[Dict[str, Any]]:
#         """
#         Get grammar topics
        
#         Args:
#             user_id (str): User ID
#             language_code (str): Language code
#             topic_id (str, optional): Specific topic ID to get
            
#         Returns:
#             List[Dict[str, Any]]: List of grammar topics
#         """
#         if topic_id:
#             query = """
#             SELECT * FROM grammar_topics 
#             WHERE user_id = ? AND language_code = ? AND topic_id = ?
#             """
#             results = self.execute_query(query, (user_id, language_code, topic_id))
#         else:
#             query = """
#             SELECT * FROM grammar_topics 
#             WHERE user_id = ? AND language_code = ?
#             ORDER BY name
#             """
#             results = self.execute_query(query, (user_id, language_code))
        
#         # Get practice history for each topic
#         for item in results:
#             item = dict(item)
#             grammar_id = item['id']
            
#             history_query = """
#             SELECT * FROM grammar_practices 
#             WHERE grammar_id = ?
#             ORDER BY practice_date DESC
#             """
#             history = self.execute_query(history_query, (grammar_id,))
#             item['practice_history'] = history
        
#         return results
    
#     def update_grammar_mastery(self, grammar_id: int, score: float, 
#                              details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
#         """
#         Update mastery level for a grammar topic
        
#         Args:
#             grammar_id (int): Grammar topic ID
#             score (float): Practice score (0.0-1.0)
#             details (Dict, optional): Additional practice details
            
#         Returns:
#             Dict[str, Any]: Updated grammar topic data
#         """
#         now = datetime.now().isoformat()
        
#         # Get current grammar data
#         query = "SELECT * FROM grammar_topics WHERE id = ?"
#         results = self.execute_query(query, (grammar_id,))
        
#         if not results:
#             logger.error(f"Grammar topic not found: {grammar_id}")
#             return {}
            
#         topic = dict(results[0])
#         current_mastery = topic['mastery_level']
#         practice_count = topic['practice_count']
        
#         # Calculate new mastery level with exponential moving average
#         # Give more weight to recent practice sessions
#         alpha = 0.3  # Weight for new score
#         new_mastery = (alpha * score) + ((1 - alpha) * current_mastery)
        
#         # Update grammar data
#         update_query = """
#         UPDATE grammar_topics 
#         SET mastery_level = ?, practice_count = ?, last_practice = ?
#         WHERE id = ?
#         """
#         self.execute_query(update_query, (new_mastery, practice_count + 1, now, grammar_id))
        
#         # Add to practice history
#         history_query = """
#         INSERT INTO grammar_practices
#         (grammar_id, practice_date, score, old_mastery, new_mastery, details)
#         VALUES (?, ?, ?, ?, ?, ?)
#         """
#         details_json = json.dumps(details) if details else None
#         self.execute_query(history_query, (grammar_id, now, score, current_mastery, new_mastery, details_json))
        
#         # Get updated grammar data
#         updated_query = "SELECT * FROM grammar_topics WHERE id = ?"
#         updated_results = self.execute_query(updated_query, (grammar_id,))
        
#         if updated_results:
#             return dict(updated_results[0])
        
#         return {}
    
#     def log_activity(self, user_id: str, language_code: str, activity_type: str, 
#                    duration: int, details: Optional[Dict[str, Any]] = None) -> int:
#         """
#         Log a learning activity
        
#         Args:
#             user_id (str): User ID
#             language_code (str): Language code
#             activity_type (str): Type of activity
#             duration (int): Duration in minutes
#             details (Dict, optional): Additional activity details
            
#         Returns:
#             int: Activity record ID
#         """
#         now = datetime.now().isoformat()
        
#         query = """
#         INSERT INTO learning_activities
#         (user_id, language_code, activity_type, duration, timestamp, details)
#         VALUES (?, ?, ?, ?, ?, ?)
#         """
        
#         details_json = json.dumps(details) if details else None
        
#         self.execute_query(query, (user_id, language_code, activity_type, duration, now, details_json))
        
#         # Get the ID of the inserted record
#         result = self.execute_query(
#             "SELECT id FROM learning_activities WHERE user_id = ? AND timestamp = ?",
#             (user_id, now)
#         )
        
#         return result[0]['id'] if result else -1
    
#     def get_activities(self, user_id: str, language_code: Optional[str] = None, 
#                      days: int = 30) -> List[Dict[str, Any]]:
#         """
#         Get learning activities
        
#         Args:
#             user_id (str): User ID
#             language_code (str, optional): Language code
#             days (int): Number of days to look back
            
#         Returns:
#             List[Dict[str, Any]]: List of activities
#         """
#         threshold = (datetime.now() - pd.Timedelta(days=days)).isoformat()
        
#         if language_code:
#             query = """
#             SELECT * FROM learning_activities 
#             WHERE user_id = ? AND language_code = ? AND timestamp > ?
#             ORDER BY timestamp DESC
#             """
#             results = self.execute_query(query, (user_id, language_code, threshold))
#         else:
#             query = """
#             SELECT * FROM learning_activities 
#             WHERE user_id = ? AND timestamp > ?
#             ORDER BY timestamp DESC
#             """
#             results = self.execute_query(query, (user_id, threshold))
        
#         # Parse details JSON
#         for item in results:
#             item = dict(item)
#             if item['details']:
#                 try:
#                     item['details'] = json.loads(item['details'])
#                 except:
#                     item['details'] = {}
        
#         return results
    
#     def set_daily_goal(self, user_id: str, language_code: str, 
#                      goal_type: str, value: int) -> int:
#         """
#         Set a daily learning goal
        
#         Args:
#             user_id (str): User ID
#             language_code (str): Language code
#             goal_type (str): Type of goal (vocabulary, grammar, time_spent)
#             value (int): Goal value
            
#         Returns:
#             int: Goal record ID
#         """
#         now = datetime.now().isoformat()
        
#         query = """
#         INSERT OR REPLACE INTO daily_goals
#         (user_id, language_code, goal_type, value, created_date, updated_date)
#         VALUES (?, ?, ?, ?, ?, ?)
#         """
        
#         # Check if goal already exists
#         existing = self.execute_query(
#             "SELECT created_date FROM daily_goals WHERE user_id = ? AND language_code = ? AND goal_type = ?",
#             (user_id, language_code, goal_type)
#         )
        
#         created_date = existing[0]['created_date'] if existing else now
        
#         self.execute_query(query, (user_id, language_code, goal_type, value, created_date, now))
        
#         # Get the ID of the inserted record
#         result = self.execute_query(
#             "SELECT id FROM daily_goals WHERE user_id = ? AND language_code = ? AND goal_type = ?",
#             (user_id, language_code, goal_type)
#         )
        
#         return result[0]['id'] if result else -1
    
#     def get_daily_goals(self, user_id: str, language_code: str) -> Dict[str, int]:
#         """
#         Get daily goals
        
#         Args:
#             user_id (str): User ID
#             language_code (str): Language code
            
#         Returns:
#             Dict[str, int]: Goals by type
#         """
#         query = """
#         SELECT goal_type, value FROM daily_goals 
#         WHERE user_id = ? AND language_code = ?
#         """
        
#         results = self.execute_query(query, (user_id, language_code))
        
#         goals = {}
#         for item in results:
#             goals[item['goal_type']] = item['value']
        
#         return goals
    
#     def add_long_term_goal(self, goal_data: Dict[str, Any]) -> int:
#         """
#         Add a long-term learning goal
        
#         Args:
#             goal_data (Dict[str, Any]): Goal data
            
#         Returns:
#             int: Goal record ID
#         """
#         now = datetime.now().isoformat()
        
#         query = """
#         INSERT INTO learning_goals
#         (user_id, language_code, description, target_date, created_date, 
#          completed, progress, metrics)
#         VALUES (?, ?, ?, ?, ?, ?, ?, ?)
#         """
        
#         metrics = json.dumps(goal_data.get('metrics', {}))
        
#         params = (
#             goal_data['user_id'],
#             goal_data['language_code'],
#             goal_data['description'],
#             goal_data.get('target_date'),
#             now,
#             goal_data.get('completed', False),
#             goal_data.get('progress', 0.0),
#             metrics
#         )
        
#         self.execute_query(query, params)
        
#         # Get the ID of the inserted record
#         result = self.execute_query(
#             "SELECT id FROM learning_goals WHERE user_id = ? AND created_date = ?",
#             (goal_data['user_id'], now)
#         )
        
#         return result[0]['id'] if result else -1
    
#     def update_goal_progress(self, goal_id: int, progress: float, 
#                            completed: bool = False) -> bool:
#         """
#         Update progress for a long-term goal
        
#         Args:
#             goal_id (int): Goal ID
#             progress (float): Progress value (0.0-1.0)
#             completed (bool): Whether the goal is completed
            
#         Returns:
#             bool: Success status
#         """
#         query = """
#         UPDATE learning_goals 
#         SET progress = ?, completed = ?
#         WHERE id = ?
#         """
        
#         self.execute_query(query, (progress, completed, goal_id))
#         return True
    
#     def get_learning_goals(self, user_id: str, language_code: str, 
#                          include_completed: bool = False) -> List[Dict[str, Any]]:
#         """
#         Get learning goals
        
#         Args:
#             user_id (str): User ID
#             language_code (str): Language code
#             include_completed (bool): Whether to include completed goals
            
#         Returns:
#             List[Dict[str, Any]]: List of goals
#         """
#         if include_completed:
#             query = """
#             SELECT * FROM learning_goals 
#             WHERE user_id = ? AND language_code = ?
#             ORDER BY target_date, created_date
#             """
#             results = self.execute_query(query, (user_id, language_code))
#         else:
#             query = """
#             SELECT * FROM learning_goals 
#             WHERE user_id = ? AND language_code = ? AND completed = 0
#             ORDER BY target_date, created_date
#             """
#             results = self.execute_query(query, (user_id, language_code))
        
#         # Parse metrics JSON
#         for item in results:
#             item = dict(item)
#             if item['metrics']:
#                 try:
#                     item['metrics'] = json.loads(item['metrics'])
#                 except:
#                     item['metrics'] = {}
        
#         return results
    
#     def get_vocabulary_stats(self, user_id: str, language_code: str) -> Dict[str, Any]:
#         """
#         Get vocabulary statistics
        
#         Args:
#             user_id (str): User ID
#             language_code (str): Language code
            
#         Returns:
#             Dict[str, Any]: Vocabulary statistics
#         """
#         # Count total words
#         total_query = """
#         SELECT COUNT(*) as total FROM vocabulary_items 
#         WHERE user_id = ? AND language_code = ?
#         """
#         total_result = self.execute_query(total_query, (user_id, language_code))
#         total_words = total_result[0]['total'] if total_result else 0
        
#         # Count words by level
#         level_query = """
#         SELECT level, COUNT(*) as count FROM vocabulary_items 
#         WHERE user_id = ? AND language_code = ?
#         GROUP BY level
#         ORDER BY level
#         """
#         level_results = self.execute_query(level_query, (user_id, language_code))
        
#         level_counts = {}
#         for item in level_results:
#             level_counts[item['level']] = item['count']
        
#         # Count mastered words (level >= 5)
#         mastered_query = """
#         SELECT COUNT(*) as count FROM vocabulary_items 
#         WHERE user_id = ? AND language_code = ? AND level >= 5
#         """
#         mastered_result = self.execute_query(mastered_query, (user_id, language_code))
#         mastered_words = mastered_result[0]['count'] if mastered_result else 0
        
#         # Get category distribution
#         category_query = """
#         SELECT category, COUNT(*) as count FROM vocabulary_items 
#         WHERE user_id = ? AND language_code = ?
#         GROUP BY category
#         """
#         category_results = self.execute_query(category_query, (user_id, language_code))
        
#         category_counts = {}
#         for item in category_results:
#             if item['category']:
#                 category_counts[item['category']] = item['count']
        
#         # Calculate daily learning rate (last 30 days)
#         thirty_days_ago = (datetime.now() - pd.Timedelta(days=30)).isoformat()
        
#         daily_query = """
#         SELECT COUNT(*) as count FROM vocabulary_items 
#         WHERE user_id = ? AND language_code = ? AND added_date > ?
#         """
#         daily_result = self.execute_query(daily_query, (user_id, language_code, thirty_days_ago))
#         recent_words = daily_result[0]['count'] if daily_result else 0
        
#         # Create stats object
#         stats = {
#             "total_words": total_words,
#             "mastered_words": mastered_words,
#             "learning_words": total_words - mastered_words,
#             "mastery_percentage": (mastered_words / total_words * 100) if total_words > 0 else 0,
#             "level_distribution": level_counts,
#             "category_distribution": category_counts,
#             "daily_learning_rate": recent_words / 30
#         }
        
#         return stats
    
#     def get_grammar_stats(self, user_id: str, language_code: str) -> Dict[str, Any]:
#         """
#         Get grammar statistics
        
#         Args:
#             user_id (str): User ID
#             language_code (str): Language code
            
#         Returns:
#             Dict[str, Any]: Grammar statistics
#         """
#         # Count total topics
#         total_query = """
#         SELECT COUNT(*) as total FROM grammar_topics 
#         WHERE user_id = ? AND language_code = ?
#         """
#         total_result = self.execute_query(total_query, (user_id, language_code))
#         total_topics = total_result[0]['total'] if total_result else 0
        
#         # Count mastered topics (mastery_level >= 0.8)
#         mastered_query = """
#         SELECT COUNT(*) as count FROM grammar_topics 
#         WHERE user_id = ? AND language_code = ? AND mastery_level >= 0.8
#         """
#         mastered_result = self.execute_query(mastered_query, (user_id, language_code))
#         mastered_topics = mastered_result[0]['count'] if mastered_result else 0
        
#         # Get average mastery level
#         avg_query = """
#         SELECT AVG(mastery_level) as avg_mastery FROM grammar_topics 
#         WHERE user_id = ? AND language_code = ?
#         """
#         avg_result = self.execute_query(avg_query, (user_id, language_code))
#         avg_mastery = avg_result[0]['avg_mastery'] if avg_result and avg_result[0]['avg_mastery'] else 0
        
#         # Get category distribution
#         category_query = """
#         SELECT category, COUNT(*) as count, AVG(mastery_level) as avg_mastery
#         FROM grammar_topics 
#         WHERE user_id = ? AND language_code = ?
#         GROUP BY category
#         """
#         category_results = self.execute_query(category_query, (user_id, language_code))
        
#         category_stats = {}
#         for item in category_results:
#             if item['category']:
#                 category_stats[item['category']] = {
#                     "count": item['count'],
#                     "avg_mastery": item['avg_mastery']
#                 }
        
#         # Get difficulty distribution
#         difficulty_query = """
#         SELECT difficulty, COUNT(*) as count
#         FROM grammar_topics 
#         WHERE user_id = ? AND language_code = ?
#         GROUP BY difficulty
#         """
#         difficulty_results = self.execute_query(difficulty_query, (user_id, language_code))
        
#         difficulty_counts = {}
#         for item in difficulty_results:
#             if item['difficulty']:
#                 difficulty_counts[item['difficulty']] = item['count']
        
#         # Create stats object
#         stats = {
#             "total_topics": total_topics,
#             "mastered_topics": mastered_topics,
#             "learning_topics": total_topics - mastered_topics,
#             "mastery_percentage": (mastered_topics / total_topics * 100) if total_topics > 0 else 0,
#             "avg_mastery": avg_mastery,
#             "category_stats": category_stats,
#             "difficulty_distribution": difficulty_counts
#         }
        
#         return stats
    
#     def get_activity_summary(self, user_id: str, language_code: str, days: int = 30) -> Dict[str, Any]:
#         """
#         Get summary of learning activities
        
#         Args:
#             user_id (str): User ID
#             language_code (str): Language code
#             days (int): Number of days to look back
            
#         Returns:
#             Dict[str, Any]: Activity summary
#         """
#         threshold = (datetime.now() - pd.Timedelta(days=days)).isoformat()
        
#         # Get activities in date range
#         query = """
#         SELECT * FROM learning_activities 
#         WHERE user_id = ? AND language_code = ? AND timestamp > ?
#         ORDER BY timestamp DESC
#         """
#         activities = self.execute_query(query, (user_id, language_code, threshold))
        
#         # Count activities by type
#         type_counts = {}
#         type_durations = {}
#         daily_activity = {}
        
#         for activity in activities:
#             activity_type = activity['activity_type']
            
#             # Count by type
#             if activity_type in type_counts:
#                 type_counts[activity_type] += 1
#                 type_durations[activity_type] += activity['duration']
#             else:
#                 type_counts[activity_type] = 1
#                 type_durations[activity_type] = activity['duration']
            
#             # Aggregate by day
#             activity_date = datetime.fromisoformat(activity['timestamp']).date().isoformat()
            
#             if activity_date in daily_activity:
#                 daily_activity[activity_date] += activity['duration']
#             else:
#                 daily_activity[activity_date] = activity['duration']
        
#         # Calculate streaks
#         activity_dates = sorted(daily_activity.keys())
#         current_streak = 0
#         max_streak = 0
        
#         if activity_dates:
#             # Check if user was active today or yesterday
#             today = datetime.now().date().isoformat()
#             yesterday = (datetime.now().date() - pd.Timedelta(days=1)).isoformat()
            
#             if today in activity_dates or yesterday in activity_dates:
#                 current_streak = 1
                
#                 # Count consecutive days backwards
#                 dates_set = set(activity_dates)
#                 check_date = (datetime.now().date() - pd.Timedelta(days=1))
                
#                 while check_date.isoformat() in dates_set:
#                     current_streak += 1
#                     check_date -= pd.Timedelta(days=1)
            
#             # Find longest streak
#             max_streak = current_streak
#             for i in range(len(activity_dates) - 1):
#                 date1 = datetime.fromisoformat(activity_dates[i]).date()
#                 date2 = datetime.fromisoformat(activity_dates[i+1]).date()
                
#                 if (date1 - date2).days == 1:  # Consecutive days
#                     temp_streak = 1
                    
#                     for j in range(i+1, len(activity_dates) - 1):
#                         date_j = datetime.fromisoformat(activity_dates[j]).date()
#                         date_next = datetime.fromisoformat(activity_dates[j+1]).date()
                        
#                         if (date_j - date_next).days == 1:
#                             temp_streak += 1
#                         else:
#                             break
                    
#                     max_streak = max(max_streak, temp_streak)
        
#         # Create summary object
#         summary = {
#             "period_days": days,
#             "total_activities": len(activities),
#             "total_duration": sum(activity['duration'] for activity in activities),
#             "activity_types": type_counts,
#             "type_durations": type_durations,
#             "daily_activity": daily_activity,
#             "current_streak": current_streak,
#             "longest_streak": max_streak
#         }
        
#         # Calculate averages
#         if activities:
#             summary["avg_duration_per_activity"] = summary["total_duration"] / summary["total_activities"]
#             summary["avg_duration_per_day"] = summary["total_duration"] / days
#         else:
#             summary["avg_duration_per_activity"] = 0
#             summary["avg_duration_per_day"] = 0
        
#         return summary


# class PersistenceManager:
#     """Main class for managing all data persistence"""
    
#     def __init__(self, data_dir: str = "./data"):
#         """
#         Initialize persistence manager
        
#         Args:
#             data_dir (str): Data directory
#         """
#         self.data_dir = data_dir
        
#         # Create data directory if it doesn't exist
#         os.makedirs(data_dir, exist_ok=True)
        
#         # Initialize component managers
#         self.user_manager = UserManager(os.path.join(data_dir, "users.db"))
#         self.content_manager = ContentManager(os.path.join(data_dir, "content.db"))
#         self.progress_manager = ProgressManager(os.path.join(data_dir, "progress.db"))
    
#     def backup_all(self, backup_dir: str) -> Dict[str, bool]:
#         """
#         Backup all databases
        
#         Args:
#             backup_dir (str): Directory to save backups
            
#         Returns:
#             Dict[str, bool]: Backup results
#         """
#         # Create backup directory
#         os.makedirs(backup_dir, exist_ok=True)
        
#         # Timestamp for backup filenames
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
#         # Backup each database
#         results = {}
        
#         results["users"] = self.user_manager.backup(
#             os.path.join(backup_dir, f"users_{timestamp}.db")
#         )
        
#         results["content"] = self.content_manager.backup(
#             os.path.join(backup_dir, f"content_{timestamp}.db")
#         )
        
#         results["progress"] = self.progress_manager.backup(
#             os.path.join(backup_dir, f"progress_{timestamp}.db")
#         )
        
#         return results
    
#     def restore_all(self, backup_dir: str, timestamp: str) -> Dict[str, bool]:
#         """
#         Restore all databases from backup
        
#         Args:
#             backup_dir (str): Directory with backups
#             timestamp (str): Backup timestamp to restore
            
#         Returns:
#             Dict[str, bool]: Restore results
#         """
#         results = {}
        
#         users_backup = os.path.join(backup_dir, f"users_{timestamp}.db")
#         content_backup = os.path.join(backup_dir, f"content_{timestamp}.db")
#         progress_backup = os.path.join(backup_dir, f"progress_{timestamp}.db")
        
#         results["users"] = self.user_manager.restore(users_backup)
#         results["content"] = self.content_manager.restore(content_backup)
#         results["progress"] = self.progress_manager.restore(progress_backup)
        
#         return results
    
#     def optimize_all(self) -> Dict[str, bool]:
#         """
#         Optimize all databases
        
#         Returns:
#             Dict[str, bool]: Optimization results
#         """
#         results = {}
        
#         try:
#             self.user_manager.optimize()
#             results["users"] = True
#         except Exception as e:
#             logger.error(f"Error optimizing users database: {e}", exc_info=True)
#             results["users"] = False

#         try:
#             self.content_manager.optimize()
#             results["content"] = True
#         except Exception as e:
#             logger.error(f"Error optimizing content database: {e}", exc_info=True)
#             results["content"] = False

#         try:
#             self.progress_manager.optimize()
#             results["progress"] = True
#         except Exception as e:
#             logger.error(f"Error optimizing progress database: {e}", exc_info=True)
#             results["progress"] = False

#         return results


# # Example usage
# if __name__ == "__main__":
#     # Initialize persistence manager
#     manager = PersistenceManager("./casalingua_data")
    
#     # Create a user
#     user_id = manager.user_manager.create_user({
#         "username": "language_learner",
#         "email": "learner@example.com",
#         "password_hash": "hashed_password_here",
#         "languages": [
#             {
#                 "language_code": "es",
#                 "proficiency_level": "beginner",
#                 "is_learning": True
#             }
#         ]
#     })
    
#     # Add vocabulary words
#     manager.progress_manager.add_vocabulary({
#         "user_id": user_id,
#         "language_code": "es",
#         "word": "hola",
#         "translation": "hello",
#         "context": "¡Hola, buenos días!",
#         "category": "greetings"
#     })
    
#     # Add a grammar topic
#     manager.progress_manager.add_grammar_topic({
#         "user_id": user_id,
#         "language_code": "es",
#         "topic_id": "present_tense",
#         "name": "Present Tense",
#         "description": "Regular verb conjugation in the present tense",
#         "category": "verb conjugation",
#         "difficulty": "beginner"
#     })
    
#     # Log a learning activity
#     manager.progress_manager.log_activity(
#         user_id, 
#         "es", 
#         "vocabulary_practice", 
#         15,  # 15 minutes
#         {"words_practiced": 10, "success_rate": 0.8}
#     )
    
#     # Add a learning goal
#     manager.progress_manager.add_long_term_goal({
#         "user_id": user_id,
#         "language_code": "es",
#         "description": "Learn 500 common Spanish words",
#         "target_date": (datetime.now() + pd.Timedelta(days=90)).isoformat(),
#         "metrics": {
#             "target_words": 500,
#             "current_words": 1,
#             "measure": "vocabulary_count"
#         }
#     })
    
#     # Create content
#     content_id = manager.content_manager.create_content({
#         "title": "Basic Greetings in Spanish",
#         "content_type": "lesson",
#         "language_code": "es",
#         "difficulty_level": "beginner",
#         "content": "In this lesson, we'll learn basic greetings in Spanish...",
#         "tags": ["greetings", "beginner", "conversation"]
#     })
    
#     print(f"Created user: {user_id}")
#     print(f"Created content: {content_id}")
    
#     # Get user data
#     user = manager.user_manager.get_user(user_id)
#     print(f"User: {user['username']}")
    
#     # Get vocabulary statistics
#     vocab_stats = manager.progress_manager.get_vocabulary_stats(user_id, "es")
#     print(f"Vocabulary stats: {vocab_stats}")