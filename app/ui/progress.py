"""
CasaLingua Progress Tracker

Tracks vocabulary, grammar, and learning goals. Supports spaced repetition,
category breakdowns, and visualizations using matplotlib.
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")

import os
import json
import logging
import datetime
from datetime import timedelta
from typing import List, Dict, Any, Optional, Tuple, Union, Set
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import pandas as pd
from collections import defaultdict, Counter

# Centralized constants
MASTERY_LEVELS = list(range(7))  # 0 to 6
DIFFICULTY_LEVELS = {"beginner", "intermediate", "advanced"}
DEFAULT_DATA_DIR = Path("./data/progress")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class VocabularyTracker:
    """Tracks vocabulary learning progress"""
    
    def __init__(self, user_id: str, language: str, data_dir: Union[str, Path] = DEFAULT_DATA_DIR):
        """
        Initialize vocabulary tracker
        
        Args:
            user_id (str): User identifier
            language (str): Language code
            data_dir (str): Directory to store progress data
        """
        self.user_id = user_id
        self.language = language
        self.data_dir = Path(data_dir)
        # Create data directory if it doesn't exist
        (self.data_dir / user_id).mkdir(parents=True, exist_ok=True)
        # Load vocabulary data
        self.vocab_file = self.data_dir / user_id / f"{language}_vocabulary.json"
        self.vocabulary = self._load_vocabulary()
        
        # Learning stats
        self.stats = {
            "total_words": 0,
            "learning_words": 0,
            "mastered_words": 0,
            "last_update": None
        }
        
        self._update_stats()
    
    def _load_vocabulary(self) -> Dict[str, Dict[str, Any]]:
        """
        Load vocabulary data from file
        
        Returns:
            Dict[str, Dict[str, Any]]: Vocabulary data
        """
        if self.vocab_file.exists():
            try:
                with open(self.vocab_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading vocabulary data: {e}")
                return {}
        return {}
    
    def _save_vocabulary(self) -> bool:
        """
        Save vocabulary data to file
        
        Returns:
            bool: Success status
        """
        try:
            with open(self.vocab_file, 'w', encoding='utf-8') as f:
                json.dump(self.vocabulary, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error saving vocabulary data: {e}")
            return False
    
    def _update_stats(self) -> None:
        """Update vocabulary statistics"""
        self.stats["total_words"] = len(self.vocabulary)
        self.stats["mastered_words"] = sum(1 for word in self.vocabulary.values() if word.get("level", 0) >= 5)
        self.stats["learning_words"] = self.stats["total_words"] - self.stats["mastered_words"]
        self.stats["last_update"] = datetime.datetime.now().isoformat()
    
    def add_word(self, 
               word: str, 
               translation: str, 
               context: Optional[str] = None,
               category: Optional[str] = None) -> bool:
        """
        Add a new vocabulary word
        
        Args:
            word (str): Word in the target language
            translation (str): Translation to user's native language
            context (str, optional): Example sentence
            category (str, optional): Word category
            
        Returns:
            bool: Success status
        """
        word_key = word.lower().strip()
        
        # Check if word already exists
        if word_key in self.vocabulary:
            # Update existing word
            self.vocabulary[word_key]["translation"] = translation
            if context:
                self.vocabulary[word_key]["context"] = context
            if category:
                self.vocabulary[word_key]["category"] = category
            
            logger.info(f"Updated existing word: {word}")
        else:
            # Add new word
            self.vocabulary[word_key] = {
                "word": word,
                "translation": translation,
                "context": context,
                "category": category,
                "level": 0,
                "added_date": datetime.datetime.now().isoformat(),
                "last_review": None,
                "next_review": datetime.datetime.now().isoformat(),
                "review_history": []
            }
            
            logger.info(f"Added new word: {word}")
        
        self._update_stats()
        return self._save_vocabulary()
    
    def add_words_batch(self, words: List[Dict[str, Any]]) -> int:
        """
        Add multiple vocabulary words at once
        
        Args:
            words (List[Dict]): List of word dictionaries
            
        Returns:
            int: Number of words added
        """
        count = 0
        for word_data in words:
            if "word" in word_data and "translation" in word_data:
                if self.add_word(
                    word_data["word"],
                    word_data["translation"],
                    word_data.get("context"),
                    word_data.get("category")
                ):
                    count += 1
        
        return count
    
    def remove_word(self, word: str) -> bool:
        """
        Remove a vocabulary word
        
        Args:
            word (str): Word to remove
            
        Returns:
            bool: Success status
        """
        word_key = word.lower().strip()
        
        if word_key in self.vocabulary:
            del self.vocabulary[word_key]
            logger.info(f"Removed word: {word}")
            self._update_stats()
            return self._save_vocabulary()
        
        return False
    
    def update_review(self, word: str, success: bool) -> Dict[str, Any]:
        """
        Update review progress for a word
        
        Args:
            word (str): Word that was reviewed
            success (bool): Whether recall was successful
            
        Returns:
            Dict[str, Any]: Updated word data
        """
        word_key = word.lower().strip()
        
        if word_key not in self.vocabulary:
            logger.warning(f"Word not found: {word}")
            return {}
        
        # Get current word data
        word_data = self.vocabulary[word_key]
        current_level = word_data.get("level", 0)
        
        # Update level based on success
        if success:
            # Increment level (max 6)
            new_level = min(current_level + 1, 6)
        else:
            # Decrement level (min 0)
            new_level = max(current_level - 1, 0)
        
        # Calculate next review date based on spaced repetition
        days_until_next_review = self._calculate_review_interval(new_level, success)
        next_review = (datetime.datetime.now() + timedelta(days=days_until_next_review)).isoformat()
        
        # Update word data
        word_data["level"] = new_level
        word_data["last_review"] = datetime.datetime.now().isoformat()
        word_data["next_review"] = next_review
        
        # Add to review history
        review_entry = {
            "date": datetime.datetime.now().isoformat(),
            "success": success,
            "old_level": current_level,
            "new_level": new_level
        }
        word_data["review_history"].append(review_entry)
        
        self.vocabulary[word_key] = word_data
        self._update_stats()
        self._save_vocabulary()
        
        return word_data
    
    def _calculate_review_interval(self, level: int, success: bool) -> int:
        """
        Calculate days until next review using spaced repetition
        
        Args:
            level (int): Current knowledge level
            success (bool): Whether recall was successful
            
        Returns:
            int: Days until next review
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
    
    def get_words_to_review(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get words due for review
        
        Args:
            limit (int): Maximum number of words to return
            
        Returns:
            List[Dict[str, Any]]: Words due for review
        """
        now = datetime.datetime.now().isoformat()
        
        # Find words due for review
        due_words = []
        for word_key, word_data in self.vocabulary.items():
            if word_data.get("next_review", now) <= now:
                due_words.append(word_data)
        
        # Sort by level (prioritize lower levels)
        due_words.sort(key=lambda x: (x.get("level", 0), x.get("next_review", "")))
        
        return due_words[:limit]
    
    def get_words_by_category(self, category: str) -> List[Dict[str, Any]]:
        """
        Get words by category
        
        Args:
            category (str): Category to filter by
            
        Returns:
            List[Dict[str, Any]]: Words in the category
        """
        return [word for word in self.vocabulary.values() 
               if word.get("category", "").lower() == category.lower()]
    
    def get_words_by_level(self, level: int) -> List[Dict[str, Any]]:
        """
        Get words by proficiency level
        
        Args:
            level (int): Level to filter by (0-6)
            
        Returns:
            List[Dict[str, Any]]: Words at the specified level
        """
        return [word for word in self.vocabulary.values() 
               if word.get("level", 0) == level]
    
    def get_progress_stats(self) -> Dict[str, Any]:
        """
        Get vocabulary progress statistics
        
        Returns:
            Dict[str, Any]: Progress statistics
        """
        # Update stats before returning
        self._update_stats()
        
        # Add level distribution
        level_counts = defaultdict(int)
        for word in self.vocabulary.values():
            level_counts[word.get("level", 0)] += 1
        
        # Add category distribution
        category_counts = defaultdict(int)
        for word in self.vocabulary.values():
            category = word.get("category", "uncategorized")
            category_counts[category] += 1
        
        # Calculate daily review counts for last 30 days
        daily_reviews = {}
        today = datetime.datetime.now().date()
        
        for i in range(30):
            date = (today - timedelta(days=i)).isoformat()
            daily_reviews[date] = 0
        
        for word in self.vocabulary.values():
            for review in word.get("review_history", []):
                review_date = datetime.datetime.fromisoformat(review["date"]).date().isoformat()
                if review_date in daily_reviews:
                    daily_reviews[review_date] += 1
        
        # Add to stats
        stats = self.stats.copy()
        stats["level_distribution"] = dict(level_counts)
        stats["category_distribution"] = dict(category_counts)
        stats["daily_reviews"] = daily_reviews
        
        # Calculate mastery percentage
        if stats["total_words"] > 0:
            stats["mastery_percentage"] = (stats["mastered_words"] / stats["total_words"]) * 100
        else:
            stats["mastery_percentage"] = 0
        
        return stats
    
    def generate_progress_chart(self, output_path: Optional[str] = None) -> Optional[Figure]:
        """
        Generate vocabulary progress chart
        
        Args:
            output_path (str, optional): Path to save chart image
            
        Returns:
            Figure: Matplotlib figure object (if no output path provided)
        """
        stats = self.get_progress_stats()
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Level distribution
        levels = MASTERY_LEVELS
        level_counts = [stats["level_distribution"].get(level, 0) for level in levels]
        
        ax1.bar(levels, level_counts, color='skyblue')
        ax1.set_xlabel('Proficiency Level')
        ax1.set_ylabel('Number of Words')
        ax1.set_title('Vocabulary by Proficiency Level')
        ax1.set_xticks(levels)
        
        # Plot 2: Daily reviews
        dates = list(stats["daily_reviews"].keys())
        dates.sort()  # Sort chronologically
        reviews = [stats["daily_reviews"][date] for date in dates]
        
        # Convert to nicer date format for display
        display_dates = [datetime.date.fromisoformat(date).strftime('%b %d') for date in dates]
        
        ax2.plot(display_dates, reviews, marker='o', linestyle='-', color='green')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Reviews')
        ax2.set_title('Daily Reviews (Last 30 Days)')
        
        # Only show every 5th date label to avoid crowding
        ax2.set_xticks(range(0, len(display_dates), 5))
        ax2.set_xticklabels([display_dates[i] for i in range(0, len(display_dates), 5)])
        
        # Rotate date labels for better readability
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
        
        # Add overall stats as text
        plt.figtext(0.5, 0.01, 
                  f"Total Words: {stats['total_words']} | "
                  f"Mastered: {stats['mastered_words']} | "
                  f"Learning: {stats['learning_words']} | "
                  f"Mastery: {stats['mastery_percentage']:.1f}%",
                  ha='center', fontsize=12)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save or return figure
        if output_path:
            plt.savefig(output_path)
            plt.close()
            return None
        
        return fig


class GrammarTracker:
    """Tracks grammar learning progress"""
    
    def __init__(self, user_id: str, language: str, data_dir: Union[str, Path] = DEFAULT_DATA_DIR):
        """
        Initialize grammar tracker
        
        Args:
            user_id (str): User identifier
            language (str): Language code
            data_dir (str): Directory to store progress data
        """
        self.user_id = user_id
        self.language = language
        self.data_dir = Path(data_dir)
        # Create data directory if it doesn't exist
        (self.data_dir / user_id).mkdir(parents=True, exist_ok=True)
        # Load grammar data
        self.grammar_file = self.data_dir / user_id / f"{language}_grammar.json"
        self.grammar_topics = self._load_grammar()
        
        # Learning stats
        self.stats = {
            "total_topics": 0,
            "learning_topics": 0,
            "mastered_topics": 0,
            "last_update": None
        }
        
        self._update_stats()
    
    def _load_grammar(self) -> Dict[str, Dict[str, Any]]:
        """
        Load grammar data from file
        
        Returns:
            Dict[str, Dict[str, Any]]: Grammar data
        """
        if self.grammar_file.exists():
            try:
                with open(self.grammar_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading grammar data: {e}")
                return {}
        return {}
    
    def _save_grammar(self) -> bool:
        """
        Save grammar data to file
        
        Returns:
            bool: Success status
        """
        try:
            with open(self.grammar_file, 'w', encoding='utf-8') as f:
                json.dump(self.grammar_topics, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error saving grammar data: {e}")
            return False
    
    def _update_stats(self) -> None:
        """Update grammar statistics"""
        self.stats["total_topics"] = len(self.grammar_topics)
        self.stats["mastered_topics"] = sum(1 for topic in self.grammar_topics.values() 
                                         if topic.get("mastery_level", 0) >= 0.8)
        self.stats["learning_topics"] = self.stats["total_topics"] - self.stats["mastered_topics"]
        self.stats["last_update"] = datetime.datetime.now().isoformat()
    
    def add_topic(self, 
                topic_id: str, 
                name: str, 
                description: str,
                category: str,
                difficulty: str = "beginner") -> bool:
        """
        Add a new grammar topic
        
        Args:
            topic_id (str): Unique topic identifier
            name (str): Topic name
            description (str): Topic description
            category (str): Grammar category
            difficulty (str): Difficulty level
            
        Returns:
            bool: Success status
        """
        # Check if topic already exists
        if topic_id in self.grammar_topics:
            # Update existing topic
            self.grammar_topics[topic_id].update({
                "name": name,
                "description": description,
                "category": category,
                "difficulty": difficulty
            })
            
            logger.info(f"Updated grammar topic: {name}")
        else:
            # Add new topic
            self.grammar_topics[topic_id] = {
                "topic_id": topic_id,
                "name": name,
                "description": description,
                "category": category,
                "difficulty": difficulty,
                "mastery_level": 0.0,
                "practice_count": 0,
                "last_practice": None,
                "practice_history": []
            }
            
            logger.info(f"Added new grammar topic: {name}")
        
        self._update_stats()
        return self._save_grammar()
    
    def update_topic_mastery(self, 
                           topic_id: str, 
                           score: float,
                           details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Update mastery level for a grammar topic
        
        Args:
            topic_id (str): Topic identifier
            score (float): Practice score (0.0-1.0)
            details (Dict, optional): Additional practice details
            
        Returns:
            Dict[str, Any]: Updated topic data
        """
        if topic_id not in self.grammar_topics:
            logger.warning(f"Grammar topic not found: {topic_id}")
            return {}
        
        # Get current topic data
        topic_data = self.grammar_topics[topic_id]
        current_mastery = topic_data.get("mastery_level", 0.0)
        practice_count = topic_data.get("practice_count", 0)
        
        # Calculate new mastery level with exponential moving average
        # Give more weight to recent practice sessions
        alpha = 0.3  # Weight for new score
        new_mastery = (alpha * score) + ((1 - alpha) * current_mastery)
        
        # Update topic data
        topic_data["mastery_level"] = new_mastery
        topic_data["practice_count"] = practice_count + 1
        topic_data["last_practice"] = datetime.datetime.now().isoformat()
        
        # Add to practice history
        practice_entry = {
            "date": datetime.datetime.now().isoformat(),
            "score": score,
            "old_mastery": current_mastery,
            "new_mastery": new_mastery
        }
        
        # Add details if provided
        if details:
            practice_entry["details"] = details
        
        topic_data["practice_history"].append(practice_entry)
        
        self.grammar_topics[topic_id] = topic_data
        self._update_stats()
        self._save_grammar()
        
        return topic_data
    
    def get_topics_by_category(self, category: str) -> List[Dict[str, Any]]:
        """
        Get grammar topics by category
        
        Args:
            category (str): Category to filter by
            
        Returns:
            List[Dict[str, Any]]: Topics in the category
        """
        return [topic for topic in self.grammar_topics.values() 
               if topic.get("category", "").lower() == category.lower()]
    
    def get_topics_by_difficulty(self, difficulty: str) -> List[Dict[str, Any]]:
        """
        Get grammar topics by difficulty level
        
        Args:
            difficulty (str): Difficulty level
            
        Returns:
            List[Dict[str, Any]]: Topics at the specified difficulty
        """
        return [topic for topic in self.grammar_topics.values() 
               if topic.get("difficulty", "").lower() == difficulty.lower()]
    
    def get_recommended_topics(self, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get recommended grammar topics for practice
        
        Args:
            limit (int): Maximum number of topics to return
            
        Returns:
            List[Dict[str, Any]]: Recommended topics
        """
        # Sort topics by mastery level (ascending)
        sorted_topics = sorted(
            self.grammar_topics.values(),
            key=lambda x: (x.get("mastery_level", 0.0), 
                          -datetime.datetime.fromisoformat(x.get("last_practice", "2000-01-01T00:00:00")).timestamp() 
                          if x.get("last_practice") else 0)
        )
        
        return sorted_topics[:limit]
    
    def get_progress_stats(self) -> Dict[str, Any]:
        """
        Get grammar progress statistics
        
        Returns:
            Dict[str, Any]: Progress statistics
        """
        # Update stats before returning
        self._update_stats()
        
        # Add category distribution
        category_counts = defaultdict(int)
        category_mastery = defaultdict(list)
        for topic in self.grammar_topics.values():
            category = topic.get("category", "uncategorized")
            category_counts[category] += 1
            category_mastery[category].append(topic.get("mastery_level", 0.0))
        
        # Calculate average mastery by category
        category_avg_mastery = {}
        for category, mastery_levels in category_mastery.items():
            if mastery_levels:
                category_avg_mastery[category] = sum(mastery_levels) / len(mastery_levels)
            else:
                category_avg_mastery[category] = 0.0
        
        # Calculate difficulty distribution
        difficulty_counts = defaultdict(int)
        for topic in self.grammar_topics.values():
            difficulty = topic.get("difficulty", "beginner")
            difficulty_counts[difficulty] += 1
        
        # Calculate practice counts over time
        monthly_practice = defaultdict(int)
        
        for topic in self.grammar_topics.values():
            for practice in topic.get("practice_history", []):
                practice_date = datetime.datetime.fromisoformat(practice["date"])
                month_key = f"{practice_date.year}-{practice_date.month:02d}"
                monthly_practice[month_key] += 1
        
        # Add to stats
        stats = self.stats.copy()
        stats["category_distribution"] = dict(category_counts)
        stats["category_mastery"] = category_avg_mastery
        stats["difficulty_distribution"] = dict(difficulty_counts)
        stats["monthly_practice"] = dict(monthly_practice)
        
        # Calculate overall mastery percentage
        if self.grammar_topics:
            mastery_levels = [topic.get("mastery_level", 0.0) for topic in self.grammar_topics.values()]
            stats["overall_mastery"] = sum(mastery_levels) / len(mastery_levels)
        else:
            stats["overall_mastery"] = 0.0
        
        return stats
    
    def generate_progress_chart(self, output_path: Optional[str] = None) -> Optional[Figure]:
        """
        Generate grammar progress chart
        
        Args:
            output_path (str, optional): Path to save chart image
            
        Returns:
            Figure: Matplotlib figure object (if no output path provided)
        """
        stats = self.get_progress_stats()
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Category mastery
        categories = list(stats["category_mastery"].keys())
        mastery_levels = [stats["category_mastery"][cat] for cat in categories]
        
        # Sort by mastery level
        sorted_data = sorted(zip(categories, mastery_levels), key=lambda x: x[1], reverse=True)
        categories = [item[0] for item in sorted_data]
        mastery_levels = [item[1] for item in sorted_data]
        
        bars = ax1.barh(categories, mastery_levels, color='purple')
        ax1.set_xlabel('Mastery Level')
        ax1.set_title('Grammar Mastery by Category')
        ax1.set_xlim(0, 1)
        
        # Add percentage labels
        for bar in bars:
            width = bar.get_width()
            ax1.text(max(0.05, width - 0.1), 
                   bar.get_y() + bar.get_height()/2, 
                   f"{width:.0%}", 
                   va='center', color='white')
        
        # Plot 2: Monthly practice
        months = list(stats["monthly_practice"].keys())
        months.sort()  # Sort chronologically
        practice_counts = [stats["monthly_practice"][month] for month in months]
        
        # Convert to readable month format
        display_months = []
        for month in months:
            year, month_num = month.split('-')
            display_months.append(f"{month_num}/{year[2:]}")
        
        ax2.plot(display_months, practice_counts, marker='o', linestyle='-', color='green')
        ax2.set_xlabel('Month')
        ax2.set_ylabel('Practice Sessions')
        ax2.set_title('Monthly Grammar Practice')
        
        # Rotate month labels for better readability
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
        
        # Add overall stats as text
        plt.figtext(0.5, 0.01, 
                  f"Total Topics: {stats['total_topics']} | "
                  f"Mastered: {stats['mastered_topics']} | "
                  f"Learning: {stats['learning_topics']} | "
                  f"Overall Mastery: {stats['overall_mastery']:.1%}",
                  ha='center', fontsize=12)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save or return figure
        if output_path:
            plt.savefig(output_path)
            plt.close()
            return None
        
        return fig


class UserProgress:
    """Main class for tracking overall language learning progress"""
    
    def __init__(self, user_id: str, language: str, data_dir: Union[str, Path] = DEFAULT_DATA_DIR):
        """
        Initialize user progress tracker
        
        Args:
            user_id (str): User identifier
            language (str): Language code
            data_dir (str): Directory to store progress data
        """
        self.user_id = user_id
        self.language = language
        self.data_dir = Path(data_dir)
        # Create component trackers
        self.vocabulary = VocabularyTracker(user_id, language, self.data_dir)
        self.grammar = GrammarTracker(user_id, language, self.data_dir)
        # Additional tracking data
        self.activities_file = self.data_dir / user_id / f"{language}_activities.json"
        self.activities = self._load_activities()
        # Learning goals
        self.goals_file = self.data_dir / user_id / f"{language}_goals.json"
        self.goals = self._load_goals()
        # Session streak tracking
        self.streak_data = {
            "current_streak": 0,
            "longest_streak": 0,
            "last_activity_date": None
        }
        self._update_streak()
    
    def _load_activities(self) -> List[Dict[str, Any]]:
        """
        Load activity data from file
        
        Returns:
            List[Dict[str, Any]]: Activity data
        """
        if self.activities_file.exists():
            try:
                with open(self.activities_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading activity data: {e}")
                return []
        return []
    
    def _save_activities(self) -> bool:
        """
        Save activity data to file
        
        Returns:
            bool: Success status
        """
        try:
            with open(self.activities_file, 'w', encoding='utf-8') as f:
                json.dump(self.activities, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error saving activity data: {e}")
            return False
    
    def _load_goals(self) -> Dict[str, Any]:
        """
        Load goals data from file
        
        Returns:
            Dict[str, Any]: Goals data
        """
        if self.goals_file.exists():
            try:
                with open(self.goals_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading goals data: {e}")
                return {}
        # Default goals structure
        return {
            "daily_goals": {
                "vocabulary": 10,
                "grammar": 2,
                "time_spent": 20  # minutes
            },
            "long_term_goals": []
        }
    
    def _save_goals(self) -> bool:
        """
        Save goals data to file
        
        Returns:
            bool: Success status
        """
        try:
            with open(self.goals_file, 'w', encoding='utf-8') as f:
                json.dump(self.goals, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error saving goals data: {e}")
            return False
    
    def _update_streak(self) -> None:
        """Update streak data based on activity history"""
        if not self.activities:
            self.streak_data = {
                "current_streak": 0,
                "longest_streak": 0,
                "last_activity_date": None
            }
            return
        
        # Get all activity dates
        activity_dates = [datetime.datetime.fromisoformat(activity["timestamp"]).date() 
                        for activity in self.activities]
        
        # Get unique dates
        unique_dates = set(activity_dates)
        
        # Sort dates
        sorted_dates = sorted(unique_dates)
        
        # Get most recent date
        last_date = sorted_dates[-1]
        
        # Calculate streak
        current_streak = 1
        longest_streak = 1
        
        # Check if streak is still active (activity today or yesterday)
        today = datetime.datetime.now().date()
        yesterday = today - timedelta(days=1)
        
        if last_date < yesterday:
            # Streak broken
            current_streak = 0
        else:
            # Count consecutive days backwards
            for i in range(1, len(sorted_dates)):
                prev_date = sorted_dates[-i-1]
                curr_date = sorted_dates[-i]
                
                # Check if dates are consecutive
                if (curr_date - prev_date).days == 1:
                    current_streak += 1
                else:
                    break
            
            # Update longest streak
            longest_streak = max(current_streak, longest_streak)
        
        self.streak_data = {
            "current_streak": current_streak,
            "longest_streak": longest_streak,
            "last_activity_date": last_date.isoformat()
        }
    
    def log_activity(self, 
                   activity_type: str, 
                   duration: int,
                   details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Log a learning activity
        
        Args:
            activity_type (str): Type of activity
            duration (int): Duration in minutes
            details (Dict, optional): Additional activity details
            
        Returns:
            Dict[str, Any]: Activity data
        """
        # Create activity entry
        activity = {
            "activity_id": f"{len(self.activities) + 1}",
            "activity_type": activity_type,
            "duration": duration,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        # Add details if provided
        if details:
            activity["details"] = details
        
        # Add to activities list
        self.activities.append(activity)
        
        # Save activities
        self._save_activities()
        
        # Update streak
        self._update_streak()
        
        return activity
    
    def set_daily_goal(self, goal_type: str, value: int) -> bool:
        """
        Set a daily learning goal
        
        Args:
            goal_type (str): Type of goal (vocabulary, grammar, time_spent)
            value (int): Goal value
            
        Returns:
            bool: Success status
        """
        if goal_type not in self.goals["daily_goals"]:
            logger.warning(f"Invalid goal type: {goal_type}")
            return False
        
        self.goals["daily_goals"][goal_type] = value
        return self._save_goals()
    
    def add_long_term_goal(self, 
                         description: str, 
                         target_date: str,
                         metrics: Dict[str, Any]) -> str:
        """
        Add a long-term learning goal
        
        Args:
            description (str): Goal description
            target_date (str): Target completion date
            metrics (Dict): Measurable metrics for the goal
            
        Returns:
            str: Goal ID
        """
        goal_id = f"goal_{len(self.goals['long_term_goals']) + 1}"
        
        goal = {
            "goal_id": goal_id,
            "description": description,
            "target_date": target_date,
            "metrics": metrics,
            "created_date": datetime.datetime.now().isoformat(),
            "completed": False,
            "progress": 0.0
        }
        
        self.goals["long_term_goals"].append(goal)
        self._save_goals()
        
        return goal_id
    
    def update_goal_progress(self, goal_id: str, progress: float, completed: bool = False) -> bool:
        """
        Update progress for a long-term goal
        
        Args:
            goal_id (str): Goal identifier
            progress (float): Progress value (0.0-1.0)
            completed (bool): Whether the goal is completed
            
        Returns:
            bool: Success status
        """
        for goal in self.goals["long_term_goals"]:
            if goal["goal_id"] == goal_id:
                goal["progress"] = progress
                goal["completed"] = completed
                return self._save_goals()
        
        logger.warning(f"Goal not found: {goal_id}")
        return False
    
    def get_daily_goal_progress(self) -> Dict[str, Any]:
        """
        Get progress towards daily goals
        
        Returns:
            Dict[str, Any]: Daily goal progress
        """
        # Get today's date
        today = datetime.datetime.now().date().isoformat()
        
        # Get today's activities
        today_activities = [activity for activity in self.activities 
                          if datetime.datetime.fromisoformat(activity["timestamp"]).date().isoformat() == today]
        
        # Calculate vocabulary progress
        vocab_goal = self.goals["daily_goals"].get("vocabulary", 0)
        vocab_count = 0
        
        for activity in today_activities:
            if activity["activity_type"] == "vocabulary":
                if "details" in activity and "words_reviewed" in activity["details"]:
                    vocab_count += activity["details"]["words_reviewed"]
        
        # Calculate grammar progress
        grammar_goal = self.goals["daily_goals"].get("grammar", 0)
        grammar_count = 0
        
        for activity in today_activities:
            if activity["activity_type"] == "grammar":
                if "details" in activity and "topics_practiced" in activity["details"]:
                    grammar_count += activity["details"]["topics_practiced"]
        
        # Calculate time spent
        time_goal = self.goals["daily_goals"].get("time_spent", 0)
        time_spent = sum(activity["duration"] for activity in today_activities)
        
        # Create progress data
        progress = {
            "date": today,
            "vocabulary": {
                "goal": vocab_goal,
                "current": vocab_count,
                "percentage": (vocab_count / vocab_goal * 100) if vocab_goal > 0 else 0
            },
            "grammar": {
                "goal": grammar_goal,
                "current": grammar_count,
                "percentage": (grammar_count / grammar_goal * 100) if grammar_goal > 0 else 0
            },
            "time_spent": {
                "goal": time_goal,
                "current": time_spent,
                "percentage": (time_spent / time_goal * 100) if time_goal > 0 else 0
            }
        }
        
        # Calculate overall progress
        if vocab_goal > 0 and grammar_goal > 0 and time_goal > 0:
            progress["overall_percentage"] = (
                progress["vocabulary"]["percentage"] +
                progress["grammar"]["percentage"] +
                progress["time_spent"]["percentage"]
            ) / 3
        else:
            # Only count goals that are set
            divisor = sum(1 for goal in [vocab_goal, grammar_goal, time_goal] if goal > 0)
            if divisor > 0:
                progress["overall_percentage"] = (
                    (progress["vocabulary"]["percentage"] if vocab_goal > 0 else 0) +
                    (progress["grammar"]["percentage"] if grammar_goal > 0 else 0) +
                    (progress["time_spent"]["percentage"] if time_goal > 0 else 0)
                ) / divisor
            else:
                progress["overall_percentage"] = 0
        
        return progress
    
    def get_activity_summary(self, days: int = 30) -> Dict[str, Any]:
        """
        Get summary of learning activities
        
        Args:
            days (int): Number of days to include
            
        Returns:
            Dict[str, Any]: Activity summary
        """
        # Calculate date range
        end_date = datetime.datetime.now().date()
        start_date = end_date - timedelta(days=days-1)
        
        # Filter activities by date range
        filtered_activities = [
            activity for activity in self.activities
            if start_date <= datetime.datetime.fromisoformat(activity["timestamp"]).date() <= end_date
        ]
        
        # Count activities by type
        activity_counts = Counter([activity["activity_type"] for activity in filtered_activities])
        
        # Calculate total time by type
        activity_time = defaultdict(int)
        for activity in filtered_activities:
            activity_time[activity["activity_type"]] += activity["duration"]
        
        # Create daily activity data
        daily_activity = defaultdict(int)
        for activity in filtered_activities:
            date = datetime.datetime.fromisoformat(activity["timestamp"]).date().isoformat()
            daily_activity[date] += activity["duration"]
        
        # Fill in missing dates
        date_range = [(start_date + timedelta(days=i)).isoformat() for i in range(days)]
        for date in date_range:
            if date not in daily_activity:
                daily_activity[date] = 0
        
        # Create summary
        summary = {
            "period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "days": days
            },
            "total_activities": len(filtered_activities),
            "total_time": sum(activity["duration"] for activity in filtered_activities),
            "activity_counts": dict(activity_counts),
            "activity_time": dict(activity_time),
            "daily_activity": dict(sorted(daily_activity.items())),
            "streak_data": self.streak_data
        }
        
        # Add average daily time
        if days > 0:
            summary["average_daily_time"] = summary["total_time"] / days
        else:
            summary["average_daily_time"] = 0
        
        return summary
    
    def generate_progress_dashboard(self, output_dir: str) -> Dict[str, str]:
        """
        Generate comprehensive progress dashboard
        
        Args:
            output_dir (str): Directory to save dashboard charts
            
        Returns:
            Dict[str, str]: Paths to generated charts
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        # Generate vocabulary chart
        vocab_chart_path = output_dir / f"{self.user_id}_{self.language}_vocabulary.png"
        self.vocabulary.generate_progress_chart(str(vocab_chart_path))
        # Generate grammar chart
        grammar_chart_path = output_dir / f"{self.user_id}_{self.language}_grammar.png"
        self.grammar.generate_progress_chart(str(grammar_chart_path))
        # Generate overall activity chart
        activity_chart_path = output_dir / f"{self.user_id}_{self.language}_activity.png"
        self._generate_activity_chart(str(activity_chart_path))
        # Generate goal progress chart
        goals_chart_path = output_dir / f"{self.user_id}_{self.language}_goals.png"
        self._generate_goals_chart(str(goals_chart_path))
        return {
            "vocabulary": str(vocab_chart_path),
            "grammar": str(grammar_chart_path),
            "activity": str(activity_chart_path),
            "goals": str(goals_chart_path)
        }
    
    def _generate_activity_chart(self, output_path: str) -> None:
        """
        Generate activity summary chart
        
        Args:
            output_path (str): Path to save chart image
        """
        # Get activity summary
        summary = self.get_activity_summary(days=30)
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Activity time by type
        activity_types = list(summary["activity_time"].keys())
        activity_times = [summary["activity_time"][activity] for activity in activity_types]
        
        # Sort by time (descending)
        sorted_data = sorted(zip(activity_types, activity_times), key=lambda x: x[1], reverse=True)
        activity_types = [item[0] for item in sorted_data]
        activity_times = [item[1] for item in sorted_data]
        
        ax1.pie(activity_times, labels=activity_types, autopct='%1.1f%%', startangle=90)
        ax1.axis('equal')
        ax1.set_title('Learning Time by Activity Type')
        
        # Plot 2: Daily activity time
        dates = list(summary["daily_activity"].keys())
        dates.sort()  # Sort chronologically
        times = [summary["daily_activity"][date] for date in dates]
        
        # Convert to nicer date format for display
        display_dates = [datetime.date.fromisoformat(date).strftime('%b %d') for date in dates]
        
        ax2.bar(display_dates, times, color='green')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Minutes')
        ax2.set_title('Daily Learning Time (Last 30 Days)')
        
        # Only show every 5th date label to avoid crowding
        ax2.set_xticks(range(0, len(display_dates), 5))
        ax2.set_xticklabels([display_dates[i] for i in range(0, len(display_dates), 5)])
        
        # Rotate date labels for better readability
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
        
        # Add streak and summary info
        streak_text = (
            f"Current Streak: {summary['streak_data']['current_streak']} days | "
            f"Longest Streak: {summary['streak_data']['longest_streak']} days | "
            f"Average Daily Time: {summary['average_daily_time']:.1f} minutes"
        )
        plt.figtext(0.5, 0.01, streak_text, ha='center', fontsize=12)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(output_path)
        plt.close()
    
    def _generate_goals_chart(self, output_path: str) -> None:
        """
        Generate goals progress chart
        
        Args:
            output_path (str): Path to save chart image
        """
        # Get daily goal progress
        daily_progress = self.get_daily_goal_progress()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Daily goals data
        goal_types = ['Vocabulary', 'Grammar', 'Time Spent']
        percentages = [
            daily_progress["vocabulary"]["percentage"],
            daily_progress["grammar"]["percentage"],
            daily_progress["time_spent"]["percentage"]
        ]
        
        # Progress bars
        colors = ['#4CAF50', '#2196F3', '#FFC107']
        bars = ax.barh(goal_types, percentages, color=colors)
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            goal_type = goal_types[i].lower().replace(' ', '_')
            current = daily_progress[goal_type]["current"]
            goal = daily_progress[goal_type]["goal"]
            
            ax.text(max(5, min(width - 10, 80)), 
                  bar.get_y() + bar.get_height()/2, 
                  f"{current}/{goal} ({width:.1f}%)", 
                  va='center', 
                  color='white' if width > 20 else 'black',
                  fontweight='bold')
        
        # Chart formatting
        ax.set_title('Daily Learning Goals Progress')
        ax.set_xlim(0, 110)  # Allow room for labels beyond 100%
        ax.set_xlabel('Percentage Complete')
        ax.axvline(x=100, color='gray', linestyle='--')
        
        # Long-term goals table
        if self.goals["long_term_goals"]:
            # Create a table for long-term goals
            goal_data = []
            for goal in self.goals["long_term_goals"]:
                target_date = datetime.datetime.fromisoformat(goal["target_date"]).date().strftime('%Y-%m-%d')
                status = "Completed" if goal["completed"] else f"{goal['progress']:.0%} Complete"
                goal_data.append([goal["description"], target_date, status])
            
            # Add table
            plt.figtext(0.5, 0.40, "Long-Term Goals", ha="center", fontsize=12, fontweight="bold")
            ax.table(
                cellText=goal_data,
                colLabels=["Description", "Target Date", "Status"],
                loc="bottom",
                cellLoc="left",
                bbox=[0, -0.55, 1, 0.3]
            )
        
        # Add overall progress
        plt.figtext(0.5, 0.01, 
                  f"Overall Daily Progress: {daily_progress['overall_percentage']:.1f}%",
                  ha='center', fontsize=12, fontweight='bold')
        
        plt.tight_layout(rect=[0, 0.15, 1, 0.95])
        plt.savefig(output_path)
        plt.close()


# CLI entrypoint
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", type=str, default="user123")
    parser.add_argument("--lang", type=str, default="es")
    parser.add_argument("--output", type=str, default="./charts")
    args = parser.parse_args()

    progress = UserProgress(args.user, args.lang)
    chart_paths = progress.generate_progress_dashboard(args.output)
    print(f"Charts saved to {args.output}")