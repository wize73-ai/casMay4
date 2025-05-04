"""
Progress Manager for CasaLingua
Manages user learning progress persistence
"""

import os
import json
import logging
import time
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple, Union, Callable, Set
from datetime import datetime
from .database import DatabaseManager

logger = logging.getLogger(__name__)

class ProgressManager(DatabaseManager):
    """Manages user learning progress persistence"""
    
    def __init__(self, db_path: str):
        """
        Initialize progress manager
        
        Args:
            db_path (str): Path to database file
        """
        super().__init__(db_path)
        self._create_tables()
    
    def _create_tables(self) -> None:
        """Create necessary database tables"""
        script = """
        CREATE TABLE IF NOT EXISTS vocabulary_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            language_code TEXT NOT NULL,
            word TEXT NOT NULL,
            translation TEXT NOT NULL,
            context TEXT,
            category TEXT,
            level INTEGER DEFAULT 0,
            added_date TEXT NOT NULL,
            last_review TEXT,
            next_review TEXT,
            UNIQUE (user_id, language_code, word)
        );
        
        CREATE TABLE IF NOT EXISTS vocabulary_reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            vocabulary_id INTEGER NOT NULL,
            review_date TEXT NOT NULL,
            success BOOLEAN NOT NULL,
            old_level INTEGER NOT NULL,
            new_level INTEGER NOT NULL,
            FOREIGN KEY (vocabulary_id) REFERENCES vocabulary_items (id)
        );
        
        CREATE TABLE IF NOT EXISTS grammar_topics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            language_code TEXT NOT NULL,
            topic_id TEXT NOT NULL,
            name TEXT NOT NULL,
            description TEXT,
            category TEXT,
            difficulty TEXT,
            mastery_level REAL DEFAULT 0.0,
            practice_count INTEGER DEFAULT 0,
            last_practice TEXT,
            UNIQUE (user_id, language_code, topic_id)
        );
        
        CREATE TABLE IF NOT EXISTS grammar_practices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            grammar_id INTEGER NOT NULL,
            practice_date TEXT NOT NULL,
            score REAL NOT NULL,
            old_mastery REAL NOT NULL,
            new_mastery REAL NOT NULL,
            details TEXT,
            FOREIGN KEY (grammar_id) REFERENCES grammar_topics (id)
        );
        
        CREATE TABLE IF NOT EXISTS learning_activities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            language_code TEXT NOT NULL,
            activity_type TEXT NOT NULL,
            duration INTEGER NOT NULL,
            timestamp TEXT NOT NULL,
            details TEXT
        );
        
        CREATE TABLE IF NOT EXISTS learning_goals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            language_code TEXT NOT NULL,
            description TEXT NOT NULL,
            target_date TEXT,
            created_date TEXT NOT NULL,
            completed BOOLEAN DEFAULT 0,
            progress REAL DEFAULT 0.0,
            metrics TEXT
        );
        
        CREATE TABLE IF NOT EXISTS daily_goals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            language_code TEXT NOT NULL,
            goal_type TEXT NOT NULL,
            value INTEGER NOT NULL,
            created_date TEXT NOT NULL,
            updated_date TEXT NOT NULL,
            UNIQUE (user_id, language_code, goal_type)
        );
        
        CREATE INDEX IF NOT EXISTS idx_vocabulary_user_lang ON vocabulary_items (user_id, language_code);
        CREATE INDEX IF NOT EXISTS idx_grammar_user_lang ON grammar_topics (user_id, language_code);
        CREATE INDEX IF NOT EXISTS idx_activities_user_lang ON learning_activities (user_id, language_code);
        CREATE INDEX IF NOT EXISTS idx_goals_user_lang ON learning_goals (user_id, language_code);
        """
        self.execute_script(script)
    
    def add_vocabulary(self, vocabulary_data: Dict[str, Any]) -> int:
        """
        Add a vocabulary word
        
        Args:
            vocabulary_data (Dict[str, Any]): Vocabulary data
            
        Returns:
            int: Vocabulary record ID
            
        TypeScript equivalent:
            async addVocabulary(vocabularyData: VocabularyData): Promise<number>
        """
        now = datetime.now().isoformat()
        
        query = """
        INSERT OR REPLACE INTO vocabulary_items
        (user_id, language_code, word, translation, context, category, level, 
         added_date, last_review, next_review)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        params = (
            vocabulary_data['user_id'],
            vocabulary_data['language_code'],
            vocabulary_data['word'],
            vocabulary_data['translation'],
            vocabulary_data.get('context'),
            vocabulary_data.get('category'),
            vocabulary_data.get('level', 0),
            vocabulary_data.get('added_date', now),
            vocabulary_data.get('last_review'),
            vocabulary_data.get('next_review', now)
        )
        
        self.execute_query(query, params)
        
        # Get the ID of the inserted record
        result = self.execute_query(
            "SELECT id FROM vocabulary_items WHERE user_id = ? AND language_code = ? AND word = ?",
            (vocabulary_data['user_id'], vocabulary_data['language_code'], vocabulary_data['word'])
        )
        
        return result[0]['id'] if result else -1
    
    def get_vocabulary(self, user_id: str, language_code: str, word: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get vocabulary words
        
        Args:
            user_id (str): User ID
            language_code (str): Language code
            word (str, optional): Specific word to get
            
        Returns:
            List[Dict[str, Any]]: List of vocabulary items
            
        TypeScript equivalent:
            async getVocabulary(userId: string, languageCode: string, word?: string): Promise<VocabularyData[]>
        """
        if word:
            query = """
            SELECT * FROM vocabulary_items 
            WHERE user_id = ? AND language_code = ? AND word = ?
            """
            results = self.execute_query(query, (user_id, language_code, word))
        else:
            query = """
            SELECT * FROM vocabulary_items 
            WHERE user_id = ? AND language_code = ?
            ORDER BY word
            """
            results = self.execute_query(query, (user_id, language_code))
        
        # Get review history for each word
        for item in results:
            item = dict(item)
            vocab_id = item['id']
            
            history_query = """
            SELECT * FROM vocabulary_reviews 
            WHERE vocabulary_id = ?
            ORDER BY review_date DESC
            """
            history = self.execute_query(history_query, (vocab_id,))
            item['review_history'] = history
        
        return results
    
    def update_vocabulary_review(self, vocabulary_id: int, success: bool) -> Dict[str, Any]:
        """
        Update review progress for a vocabulary word
        
        Args:
            vocabulary_id (int): Vocabulary ID
            success (bool): Whether recall was successful
            
        Returns:
            Dict[str, Any]: Updated vocabulary data
            
        TypeScript equivalent:
            async updateVocabularyReview(vocabularyId: number, success: boolean): Promise<VocabularyData>
        """
        now = datetime.now().isoformat()
        
        # Get current vocabulary data
        query = "SELECT * FROM vocabulary_items WHERE id = ?"
        results = self.execute_query(query, (vocabulary_id,))
        
        if not results:
            logger.error(f"Vocabulary not found: {vocabulary_id}")
            return {}
            
        vocab = dict(results[0])
        current_level = vocab['level']
        
        # Update level based on success
        if success:
            # Increment level (max 6)
            new_level = min(current_level + 1, 6)
        else:
            # Decrement level (min 0)
            new_level = max(current_level - 1, 0)
        
        # Calculate next review date based on spaced repetition
        days_until_next_review = self._calculate_review_interval(new_level, success)
        next_review = (datetime.now() + pd.Timedelta(days=days_until_next_review)).isoformat()
        
        # Update vocabulary data
        update_query = """
        UPDATE vocabulary_items 
        SET level = ?, last_review = ?, next_review = ?
        WHERE id = ?
        """
        self.execute_query(update_query, (new_level, now, next_review, vocabulary_id))
        
        # Add to review history
        history_query = """
        INSERT INTO vocabulary_reviews
        (vocabulary_id, review_date, success, old_level, new_level)
        VALUES (?, ?, ?, ?, ?)
        """
        self.execute_query(history_query, (vocabulary_id, now, success, current_level, new_level))
        
        # Get updated vocabulary data
        updated_query = "SELECT * FROM vocabulary_items WHERE id = ?"
        updated_results = self.execute_query(updated_query, (vocabulary_id,))
        
        if updated_results:
            return dict(updated_results[0])
        
        return {}
    
    def _calculate_review_interval(self, level: int, success: bool) -> int:
        """
        Calculate days until next review using spaced repetition
        
        Args:
            level (int): Current knowledge level
            success (bool): Whether recall was successful
            
        Returns:
            int: Days until next review
            
        TypeScript equivalent:
            private calculateReviewInterval(level: number, success: boolean): number
        """
        if not success:
            return 1  # Review again tomorrow if failed
        
        # Spaced repetition intervals based on level
        intervals = {
            0: 1,   # New word - review next day
            1: 2,   # Review after 2 days
            2: 4,   # Review after 4 days
            3: 7,   # Review after 1 week
            4: 14,  # Review after 2 weeks
            5: 30,  # Review after 1 month
            6: 90   # Review after 3 months (mastered)
        }
        
        return intervals.get(level, 1)
    
    def get_words_to_review(self, user_id: str, language_code: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get words due for review
        
        Args:
            user_id (str): User ID
            language_code (str): Language code
            limit (int): Maximum number of words to return
            
        Returns:
            List[Dict[str, Any]]: Words due for review
            
        TypeScript equivalent:
            async getWordsToReview(userId: string, languageCode: string, limit?: number): Promise<VocabularyData[]>
        """
        now = datetime.now().isoformat()
        
        query = """
        SELECT * FROM vocabulary_items 
        WHERE user_id = ? AND language_code = ? AND next_review <= ?
        ORDER BY level, next_review
        LIMIT ?
        """
        
        return self.execute_query(query, (user_id, language_code, now, limit))
    
    def add_grammar_topic(self, grammar_data: Dict[str, Any]) -> int:
        """
        Add a grammar topic
        
        Args:
            grammar_data (Dict[str, Any]): Grammar topic data
            
        Returns:
            int: Grammar record ID
            
        TypeScript equivalent:
            async addGrammarTopic(grammarData: GrammarData): Promise<number>
        """
        query = """
        INSERT OR REPLACE INTO grammar_topics
        (user_id, language_code, topic_id, name, description, category, 
         difficulty, mastery_level, practice_count, last_practice)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        params = (
            grammar_data['user_id'],
            grammar_data['language_code'],
            grammar_data['topic_id'],
            grammar_data['name'],
            grammar_data.get('description', ''),
            grammar_data.get('category', ''),
            grammar_data.get('difficulty', 'intermediate'),
            grammar_data.get('mastery_level', 0.0),
            grammar_data.get('practice_count', 0),
            grammar_data.get('last_practice')
        )
        
        self.execute_query(query, params)
        
        # Get the ID of the inserted record
        result = self.execute_query(
            "SELECT id FROM grammar_topics WHERE user_id = ? AND language_code = ? AND topic_id = ?",
            (grammar_data['user_id'], grammar_data['language_code'], grammar_data['topic_id'])
        )
        
        return result[0]['id'] if result else -1
    
    def get_grammar_topics(self, user_id: str, language_code: str, 
                        topic_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get grammar topics
        
        Args:
            user_id (str): User ID
            language_code (str): Language code
            topic_id (str, optional): Specific topic ID to get
            
        Returns:
            List[Dict[str, Any]]: List of grammar topics
            
        TypeScript equivalent:
            async getGrammarTopics(userId: string, languageCode: string, topicId?: string): Promise<GrammarData[]>
        """
        if topic_id:
            query = """
            SELECT * FROM grammar_topics 
            WHERE user_id = ? AND language_code = ? AND topic_id = ?
            """
            results = self.execute_query(query, (user_id, language_code, topic_id))
        else:
            query = """
            SELECT * FROM grammar_topics 
            WHERE user_id = ? AND language_code = ?
            ORDER BY name
            """
            results = self.execute_query(query, (user_id, language_code))
        
        # Get practice history for each topic
        for item in results:
            item = dict(item)
            grammar_id = item['id']
            
            history_query = """
            SELECT * FROM grammar_practices 
            WHERE grammar_id = ?
            ORDER BY practice_date DESC
            """
            history = self.execute_query(history_query, (grammar_id,))
            item['practice_history'] = history
        
        return results
    
    def update_grammar_mastery(self, grammar_id: int, score: float, 
                             details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Update mastery level for a grammar topic
        
        Args:
            grammar_id (int): Grammar topic ID
            score (float): Practice score (0.0-1.0)
            details (Dict, optional): Additional practice details
            
        Returns:
            Dict[str, Any]: Updated grammar topic data
            
        TypeScript equivalent:
            async updateGrammarMastery(grammarId: number, score: number, 
                                    details?: Record<string, any>): Promise<GrammarData>
        """
        now = datetime.now().isoformat()
        
        # Get current grammar data
        query = "SELECT * FROM grammar_topics WHERE id = ?"
        results = self.execute_query(query, (grammar_id,))
        
        if not results:
            logger.error(f"Grammar topic not found: {grammar_id}")
            return {}
            
        topic = dict(results[0])
        current_mastery = topic['mastery_level']
        practice_count = topic['practice_count']
        
        # Calculate new mastery level with exponential moving average
        # Give more weight to recent practice sessions
        alpha = 0.3  # Weight for new score
        new_mastery = (alpha * score) + ((1 - alpha) * current_mastery)
        
        # Update grammar data
        update_query = """
        UPDATE grammar_topics 
        SET mastery_level = ?, practice_count = ?, last_practice = ?
        WHERE id = ?
        """
        self.execute_query(update_query, (new_mastery, practice_count + 1, now, grammar_id))
        
        # Add to practice history
        history_query = """
        INSERT INTO grammar_practices
        (grammar_id, practice_date, score, old_mastery, new_mastery, details)
        VALUES (?, ?, ?, ?, ?, ?)
        """
        details_json = json.dumps(details) if details else None
        self.execute_query(history_query, (grammar_id, now, score, current_mastery, new_mastery, details_json))
        
        # Get updated grammar data
        updated_query = "SELECT * FROM grammar_topics WHERE id = ?"
        updated_results = self.execute_query(updated_query, (grammar_id,))
        
        if updated_results:
            return dict(updated_results[0])
        
        return {}
    
    def log_activity(self, user_id: str, language_code: str, activity_type: str, 
                   duration: int, details: Optional[Dict[str, Any]] = None) -> int:
        """
        Log a learning activity
        
        Args:
            user_id (str): User ID
            language_code (str): Language code
            activity_type (str): Type of activity
            duration (int): Duration in minutes
            details (Dict, optional): Additional activity details
            
        Returns:
            int: Activity record ID
            
        TypeScript equivalent:
            async logActivity(userId: string, languageCode: string, activityType: string, 
                            duration: number, details?: Record<string, any>): Promise<number>
        """
        now = datetime.now().isoformat()
        
        query = """
        INSERT INTO learning_activities
        (user_id, language_code, activity_type, duration, timestamp, details)
        VALUES (?, ?, ?, ?, ?, ?)
        """
        
        details_json = json.dumps(details) if details else None
        
        self.execute_query(query, (user_id, language_code, activity_type, duration, now, details_json))
        
        # Get the ID of the inserted record
        result = self.execute_query(
            "SELECT id FROM learning_activities WHERE user_id = ? AND timestamp = ?",
            (user_id, now)
        )
        
        return result[0]['id'] if result else -1
    
    def get_activities(self, user_id: str, language_code: Optional[str] = None, 
                     days: int = 30) -> List[Dict[str, Any]]:
        """
        Get learning activities
        
        Args:
            user_id (str): User ID
            language_code (str, optional): Language code
            days (int): Number of days to look back
            
        Returns:
            List[Dict[str, Any]]: List of activities
            
        TypeScript equivalent:
            async getActivities(userId: string, languageCode?: string, days?: number): Promise<ActivityData[]>
        """
        threshold = (datetime.now() - pd.Timedelta(days=days)).isoformat()
        
        if language_code:
            query = """
            SELECT * FROM learning_activities 
            WHERE user_id = ? AND language_code = ? AND timestamp > ?
            ORDER BY timestamp DESC
            """
            results = self.execute_query(query, (user_id, language_code, threshold))
        else:
            query = """
            SELECT * FROM learning_activities 
            WHERE user_id = ? AND timestamp > ?
            ORDER BY timestamp DESC
            """
            results = self.execute_query(query, (user_id, threshold))
        
        # Parse details JSON
        for item in results:
            item = dict(item)
            if item['details']:
                try:
                    item['details'] = json.loads(item['details'])
                except:
                    item['details'] = {}
        
        return results
    
    def set_daily_goal(self, user_id: str, language_code: str, 
                     goal_type: str, value: int) -> int:
        """
        Set a daily learning goal
        
        Args:
            user_id (str): User ID
            language_code (str): Language code
            goal_type (str): Type of goal (vocabulary, grammar, time_spent)
            value (int): Goal value
            
        Returns:
            int: Goal record ID
            
        TypeScript equivalent:
            async setDailyGoal(userId: string, languageCode: string, 
                             goalType: string, value: number): Promise<number>
        """
        now = datetime.now().isoformat()
        
        query = """
        INSERT OR REPLACE INTO daily_goals
        (user_id, language_code, goal_type, value, created_date, updated_date)
        VALUES (?, ?, ?, ?, ?, ?)
        """
        
        # Check if goal already exists
        existing = self.execute_query(
            "SELECT created_date FROM daily_goals WHERE user_id = ? AND language_code = ? AND goal_type = ?",
            (user_id, language_code, goal_type)
        )
        
        created_date = existing[0]['created_date'] if existing else now
        
        self.execute_query(query, (user_id, language_code, goal_type, value, created_date, now))
        
        # Get the ID of the inserted record
        result = self.execute_query(
            "SELECT id FROM daily_goals WHERE user_id = ? AND language_code = ? AND goal_type = ?",
            (user_id, language_code, goal_type)
        )
        
        return result[0]['id'] if result else -1
    
    def get_daily_goals(self, user_id: str, language_code: str) -> Dict[str, int]:
        """
        Get daily goals
        
        Args:
            user_id (str): User ID
            language_code (str): Language code
            
        Returns:
            Dict[str, int]: Goals by type
            
        TypeScript equivalent:
            async getDailyGoals(userId: string, languageCode: string): Promise<Record<string, number>>
        """
        query = """
        SELECT goal_type, value FROM daily_goals 
        WHERE user_id = ? AND language_code = ?
        """
        
        results = self.execute_query(query, (user_id, language_code))
        
        goals = {}
        for item in results:
            goals[item['goal_type']] = item['value']
        
        return goals