"""
User Manager for CasaLingua
Manages user data persistence and authentication
"""

import os
import json
import logging
import time
import sqlite3
import threading
from typing import Dict, Any, Optional, List, Tuple, Union, Callable, Set
from datetime import datetime
import pandas as pd
from .database import DatabaseManager

logger = logging.getLogger(__name__)

class UserManager(DatabaseManager):
    """Manages user data persistence"""
    
    def __init__(self, db_path: str):
        """
        Initialize user manager
        
        Args:
            db_path (str): Path to database file
        """
        super().__init__(db_path)
        self._create_tables()
    
    def _create_tables(self) -> None:
        """Create necessary database tables"""
        script = """
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            username TEXT NOT NULL,
            email TEXT UNIQUE,
            password_hash TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            settings TEXT,
            profile TEXT
        );

        CREATE TABLE IF NOT EXISTS user_sessions (
            session_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            started_at TEXT NOT NULL,
            expires_at TEXT NOT NULL,
            ip_address TEXT,
            user_agent TEXT,
            is_active BOOLEAN DEFAULT 1,
            FOREIGN KEY (user_id) REFERENCES users (user_id)
        );

        CREATE TABLE IF NOT EXISTS user_languages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            language_code TEXT NOT NULL,
            proficiency_level TEXT,
            is_native BOOLEAN DEFAULT 0,
            is_learning BOOLEAN DEFAULT 1,
            started_at TEXT,
            last_activity TEXT,
            FOREIGN KEY (user_id) REFERENCES users (user_id),
            UNIQUE (user_id, language_code)
        );
        """
        self.execute_script(script)
    
    def create_user(self, user_data: Dict[str, Any]) -> str:
        """
        Create a new user
        
        Args:
            user_data (Dict[str, Any]): User data
            
        Returns:
            str: User ID
            
        TypeScript equivalent:
            async createUser(userData: UserData): Promise<string>
        """
        now = datetime.now().isoformat()
        
        # Generate user ID if not provided
        if 'user_id' not in user_data:
            user_data['user_id'] = f"user_{int(time.time())}_{os.urandom(4).hex()}"
        
        # Format user data
        user = {
            'user_id': user_data['user_id'],
            'username': user_data['username'],
            'email': user_data.get('email'),
            'password_hash': user_data.get('password_hash'),
            'created_at': now,
            'updated_at': now,
            'settings': json.dumps(user_data.get('settings', {})),
            'profile': json.dumps(user_data.get('profile', {}))
        }
        
        # Insert user
        query = """
        INSERT INTO users 
        (user_id, username, email, password_hash, created_at, updated_at, settings, profile)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            user['user_id'], user['username'], user['email'], user['password_hash'],
            user['created_at'], user['updated_at'], user['settings'], user['profile']
        )
        
        try:
            self.execute_query(query, params)
            logger.info(f"Created user: {user['username']} ({user['user_id']})")
            # Add user languages if provided
            if 'languages' in user_data:
                for lang in user_data['languages']:
                    self.add_user_language(user['user_id'], lang)
            return user['user_id']
        except Exception as e:
            logger.error(f"Error creating user: {e}", exc_info=True)
            raise
    
    def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get user by ID
        
        Args:
            user_id (str): User ID
            
        Returns:
            Dict[str, Any]: User data or None if not found
            
        TypeScript equivalent:
            async getUser(userId: string): Promise<UserData | null>
        """
        query = "SELECT * FROM users WHERE user_id = ?"
        results = self.execute_query(query, (user_id,))
        
        if not results:
            return None
        
        user = dict(results[0])
        
        # Parse JSON fields
        try:
            user['settings'] = json.loads(user['settings']) if user['settings'] else {}
        except:
            user['settings'] = {}
            
        try:
            user['profile'] = json.loads(user['profile']) if user['profile'] else {}
        except:
            user['profile'] = {}
        
        # Get user languages
        languages_query = "SELECT * FROM user_languages WHERE user_id = ?"
        languages = self.execute_query(languages_query, (user_id,))
        user['languages'] = languages
        
        return user
    
    def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """
        Get user by email
        
        Args:
            email (str): User email
            
        Returns:
            Dict[str, Any]: User data or None if not found
            
        TypeScript equivalent:
            async getUserByEmail(email: string): Promise<UserData | null>
        """
        query = "SELECT user_id FROM users WHERE email = ?"
        results = self.execute_query(query, (email,))
        
        if not results:
            return None
            
        return self.get_user(results[0]['user_id'])
    
    def update_user(self, user_id: str, user_data: Dict[str, Any]) -> bool:
        """
        Update user data
        
        Args:
            user_id (str): User ID
            user_data (Dict[str, Any]): User data to update
            
        Returns:
            bool: Success status
            
        TypeScript equivalent:
            async updateUser(userId: string, userData: Partial<UserData>): Promise<boolean>
        """
        # Get current user data
        current_user = self.get_user(user_id)
        if not current_user:
            logger.error(f"User not found: {user_id}")
            return False
        
        # Update user fields
        updates = []
        params = []
        
        if 'username' in user_data:
            updates.append("username = ?")
            params.append(user_data['username'])
            
        if 'email' in user_data:
            updates.append("email = ?")
            params.append(user_data['email'])
            
        if 'password_hash' in user_data:
            updates.append("password_hash = ?")
            params.append(user_data['password_hash'])
            
        if 'settings' in user_data:
            updates.append("settings = ?")
            params.append(json.dumps(user_data['settings']))
            
        if 'profile' in user_data:
            updates.append("profile = ?")
            params.append(json.dumps(user_data['profile']))
        
        # Always update the updated_at timestamp
        updates.append("updated_at = ?")
        params.append(datetime.now().isoformat())
        
        # Add user_id to params
        params.append(user_id)
        
        if not updates:
            return True  # Nothing to update
        
        # Execute update query
        query = f"UPDATE users SET {', '.join(updates)} WHERE user_id = ?"
        self.execute_query(query, tuple(params))
        
        # Update languages if provided
        if 'languages' in user_data:
            # Clear existing languages
            self.execute_query("DELETE FROM user_languages WHERE user_id = ?", (user_id,))
            
            # Add new languages
            for lang in user_data['languages']:
                self.add_user_language(user_id, lang)
        
        return True
    
    def delete_user(self, user_id: str) -> bool:
        """
        Delete a user
        
        Args:
            user_id (str): User ID
            
        Returns:
            bool: Success status
            
        TypeScript equivalent:
            async deleteUser(userId: string): Promise<boolean>
        """
        # Delete related records first
        self.execute_query("DELETE FROM user_languages WHERE user_id = ?", (user_id,))
        self.execute_query("DELETE FROM user_sessions WHERE user_id = ?", (user_id,))
        
        # Delete user
        self.execute_query("DELETE FROM users WHERE user_id = ?", (user_id,))
        
        logger.info(f"Deleted user: {user_id}")
        return True
    
    def add_user_language(self, user_id: str, language_data: Dict[str, Any]) -> int:
        """
        Add a language to user profile
        
        Args:
            user_id (str): User ID
            language_data (Dict[str, Any]): Language data
            
        Returns:
            int: Language record ID
            
        TypeScript equivalent:
            async addUserLanguage(userId: string, languageData: LanguageData): Promise<number>
        """
        now = datetime.now().isoformat()
        
        query = """
        INSERT OR REPLACE INTO user_languages
        (user_id, language_code, proficiency_level, is_native, is_learning, started_at, last_activity)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        
        params = (
            user_id,
            language_data['language_code'],
            language_data.get('proficiency_level', 'beginner'),
            language_data.get('is_native', False),
            language_data.get('is_learning', True),
            language_data.get('started_at', now),
            language_data.get('last_activity', now)
        )
        
        self.execute_query(query, params)
        
        # Get the ID of the inserted record
        result = self.execute_query(
            "SELECT id FROM user_languages WHERE user_id = ? AND language_code = ?",
            (user_id, language_data['language_code'])
        )
        
        return result[0]['id'] if result else -1
    
    def update_language_activity(self, user_id: str, language_code: str) -> bool:
        """
        Update last activity timestamp for a language
        
        Args:
            user_id (str): User ID
            language_code (str): Language code
            
        Returns:
            bool: Success status
            
        TypeScript equivalent:
            async updateLanguageActivity(userId: string, languageCode: string): Promise<boolean>
        """
        now = datetime.now().isoformat()
        
        query = """
        UPDATE user_languages 
        SET last_activity = ? 
        WHERE user_id = ? AND language_code = ?
        """
        
        self.execute_query(query, (now, user_id, language_code))
        return True
    
    def create_session(self, user_id: str, expires_minutes: int = 60*24) -> Dict[str, Any]:
        """
        Create a new user session
        
        Args:
            user_id (str): User ID
            expires_minutes (int): Session expiry time in minutes
            
        Returns:
            Dict[str, Any]: Session data
            
        TypeScript equivalent:
            async createSession(userId: string, expiresMinutes?: number): Promise<SessionData>
        """
        session_id = f"session_{int(time.time())}_{os.urandom(8).hex()}"
        now = datetime.now()
        
        session = {
            'session_id': session_id,
            'user_id': user_id,
            'started_at': now.isoformat(),
            'expires_at': (now + pd.Timedelta(minutes=expires_minutes)).isoformat(),
            'is_active': True
        }
        
        query = """
        INSERT INTO user_sessions
        (session_id, user_id, started_at, expires_at, is_active)
        VALUES (?, ?, ?, ?, ?)
        """
        
        params = (
            session['session_id'], 
            session['user_id'],
            session['started_at'],
            session['expires_at'],
            session['is_active']
        )
        
        self.execute_query(query, params)
        
        return session
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get session by ID
        
        Args:
            session_id (str): Session ID
            
        Returns:
            Dict[str, Any]: Session data or None if not found or expired
            
        TypeScript equivalent:
            async getSession(sessionId: string): Promise<SessionData | null>
        """
        query = "SELECT * FROM user_sessions WHERE session_id = ?"
        results = self.execute_query(query, (session_id,))
        
        if not results:
            return None
            
        session = dict(results[0])
        
        # Check if session is expired
        now = datetime.now().isoformat()
        if session['expires_at'] < now or not session['is_active']:
            self.invalidate_session(session_id)
            return None
            
        return session
    
    def invalidate_session(self, session_id: str) -> bool:
        """
        Invalidate a user session
        
        Args:
            session_id (str): Session ID
            
        Returns:
            bool: Success status
            
        TypeScript equivalent:
            async invalidateSession(sessionId: string): Promise<boolean>
        """
        query = "UPDATE user_sessions SET is_active = 0 WHERE session_id = ?"
        self.execute_query(query, (session_id,))
        
        return True
    
    def get_active_users(self, days: int = 30) -> List[Dict[str, Any]]:
        """
        Get users who have been active recently
        
        Args:
            days (int): Number of days to look back
            
        Returns:
            List[Dict[str, Any]]: List of active users
            
        TypeScript equivalent:
            async getActiveUsers(days?: number): Promise<UserData[]>
        """
        threshold = (datetime.now() - pd.Timedelta(days=days)).isoformat()
        
        query = """
        SELECT DISTINCT u.* 
        FROM users u
        JOIN user_sessions s ON u.user_id = s.user_id
        WHERE s.started_at > ?
        ORDER BY u.username
        """
        
        return self.execute_query(query, (threshold,))