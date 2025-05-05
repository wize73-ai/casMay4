"""
Content Manager for CasaLingua
Manages language learning content persistence
"""

import os
import json
import logging
import time
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple, Union, Callable, Set
from datetime import datetime
from .database import DatabaseManager

logger = logging.getLogger(__name__)

class ContentManager(DatabaseManager):
    """Manages language learning content persistence"""
    
    def __init__(self, db_path: str):
        """
        Initialize content manager
        
        Args:
            db_path (str): Path to database file
        """
        super().__init__(db_path)
        self._create_tables()
    
    def _create_tables(self) -> None:
        """Create necessary database tables"""
        script = """
        CREATE TABLE IF NOT EXISTS content_items (
            content_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            content_type TEXT NOT NULL,
            language_code TEXT NOT NULL,
            difficulty_level TEXT,
            content TEXT NOT NULL,
            metadata TEXT,
            tags TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS content_categories (
            category_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT,
            parent_id TEXT,
            language_code TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (parent_id) REFERENCES content_categories (category_id)
        );

        CREATE TABLE IF NOT EXISTS content_category_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            category_id TEXT NOT NULL,
            content_id TEXT NOT NULL,
            position INTEGER,
            FOREIGN KEY (category_id) REFERENCES content_categories (category_id),
            FOREIGN KEY (content_id) REFERENCES content_items (content_id),
            UNIQUE (category_id, content_id)
        );

        CREATE TABLE IF NOT EXISTS content_embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content_id TEXT NOT NULL,
            chunk_id TEXT NOT NULL,
            chunk_text TEXT NOT NULL,
            embedding BLOB,
            model_name TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (content_id) REFERENCES content_items (content_id),
            UNIQUE (content_id, chunk_id)
        );

        CREATE INDEX IF NOT EXISTS idx_content_language ON content_items (language_code);
        CREATE INDEX IF NOT EXISTS idx_content_type ON content_items (content_type);
        CREATE INDEX IF NOT EXISTS idx_content_difficulty ON content_items (difficulty_level);
        CREATE INDEX IF NOT EXISTS idx_category_language ON content_categories (language_code);
        CREATE INDEX IF NOT EXISTS idx_category_parent ON content_categories (parent_id);
        """
        self.execute_script(script)
    
    def create_content(self, content_data: Dict[str, Any]) -> str:
        """
        Create a new content item
        
        Args:
            content_data (Dict[str, Any]): Content data
            
        Returns:
            str: Content ID
            
        TypeScript equivalent:
            async createContent(contentData: ContentData): Promise<string>
        """
        now = datetime.now().isoformat()
        
        # Generate content ID if not provided
        if 'content_id' not in content_data:
            content_data['content_id'] = f"content_{int(time.time())}_{os.urandom(4).hex()}"
        
        # Process metadata and tags
        metadata = json.dumps(content_data.get('metadata', {}))
        tags = ','.join(content_data.get('tags', []))
        
        # Format content data
        content = {
            'content_id': content_data['content_id'],
            'title': content_data['title'],
            'content_type': content_data['content_type'],
            'language_code': content_data['language_code'],
            'difficulty_level': content_data.get('difficulty_level', 'intermediate'),
            'content': content_data['content'],
            'metadata': metadata,
            'tags': tags,
            'created_at': now,
            'updated_at': now
        }
        
        # Insert content
        query = """
        INSERT INTO content_items 
        (content_id, title, content_type, language_code, difficulty_level, 
         content, metadata, tags, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            content['content_id'], content['title'], content['content_type'],
            content['language_code'], content['difficulty_level'], content['content'],
            content['metadata'], content['tags'], content['created_at'], content['updated_at']
        )
        
        try:
            self.execute_query(query, params)
            logger.info(f"Created content: {content['title']} ({content['content_id']})")
            # Add to categories if provided
            if 'categories' in content_data:
                for category_id in content_data['categories']:
                    self.add_content_to_category(content['content_id'], category_id)
            return content['content_id']
        except Exception as e:
            logger.error(f"Error creating content: {e}", exc_info=True)
            raise
    
    def get_content(self, content_id: str) -> Optional[Dict[str, Any]]:
        """
        Get content by ID
        
        Args:
            content_id (str): Content ID
            
        Returns:
            Dict[str, Any]: Content data or None if not found
            
        TypeScript equivalent:
            async getContent(contentId: string): Promise<ContentData | null>
        """
        query = "SELECT * FROM content_items WHERE content_id = ?"
        results = self.execute_query(query, (content_id,))
        
        if not results:
            return None
        
        content = dict(results[0])
        
        # Parse JSON fields
        try:
            content['metadata'] = json.loads(content['metadata']) if content['metadata'] else {}
        except:
            content['metadata'] = {}
            
        # Parse tags
        content['tags'] = content['tags'].split(',') if content['tags'] else []
        
        # Get categories
        categories_query = """
        SELECT c.* FROM content_categories c
        JOIN content_category_items ci ON c.category_id = ci.category_id
        WHERE ci.content_id = ?
        """
        categories = self.execute_query(categories_query, (content_id,))
        content['categories'] = categories
        
        return content
    
    def update_content(self, content_id: str, content_data: Dict[str, Any]) -> bool:
        """
        Update content data
        
        Args:
            content_id (str): Content ID
            content_data (Dict[str, Any]): Content data to update
            
        Returns:
            bool: Success status
            
        TypeScript equivalent:
            async updateContent(contentId: string, contentData: Partial<ContentData>): Promise<boolean>
        """
        # Get current content data
        current_content = self.get_content(content_id)
        if not current_content:
            logger.error(f"Content not found: {content_id}")
            return False
        
        # Update content fields
        updates = []
        params = []
        
        if 'title' in content_data:
            updates.append("title = ?")
            params.append(content_data['title'])
            
        if 'content_type' in content_data:
            updates.append("content_type = ?")
            params.append(content_data['content_type'])
            
        if 'language_code' in content_data:
            updates.append("language_code = ?")
            params.append(content_data['language_code'])
            
        if 'difficulty_level' in content_data:
            updates.append("difficulty_level = ?")
            params.append(content_data['difficulty_level'])
            
        if 'content' in content_data:
            updates.append("content = ?")
            params.append(content_data['content'])
            
        if 'metadata' in content_data:
            updates.append("metadata = ?")
            params.append(json.dumps(content_data['metadata']))
            
        if 'tags' in content_data:
            updates.append("tags = ?")
            params.append(','.join(content_data['tags']))
        
        # Always update the updated_at timestamp
        updates.append("updated_at = ?")
        params.append(datetime.now().isoformat())
        
        # Add content_id to params
        params.append(content_id)
        
        if not updates:
            return True  # Nothing to update
        
        # Execute update query
        query = f"UPDATE content_items SET {', '.join(updates)} WHERE content_id = ?"
        self.execute_query(query, tuple(params))
        
        # Update categories if provided
        if 'categories' in content_data:
            # Clear existing categories
            self.execute_query("DELETE FROM content_category_items WHERE content_id = ?", (content_id,))
            
            # Add new categories
            for category_id in content_data['categories']:
                self.add_content_to_category(content_id, category_id)
        
        return True
    
    def delete_content(self, content_id: str) -> bool:
        """
        Delete a content item
        
        Args:
            content_id (str): Content ID
            
        Returns:
            bool: Success status
            
        TypeScript equivalent:
            async deleteContent(contentId: string): Promise<boolean>
        """
        # Delete related records first
        self.execute_query("DELETE FROM content_category_items WHERE content_id = ?", (content_id,))
        self.execute_query("DELETE FROM content_embeddings WHERE content_id = ?", (content_id,))
        
        # Delete content
        self.execute_query("DELETE FROM content_items WHERE content_id = ?", (content_id,))
        
        logger.info(f"Deleted content: {content_id}")
        return True
    
    def create_category(self, category_data: Dict[str, Any]) -> str:
        """
        Create a new content category
        
        Args:
            category_data (Dict[str, Any]): Category data
            
        Returns:
            str: Category ID
            
        TypeScript equivalent:
            async createCategory(categoryData: CategoryData): Promise<string>
        """
        now = datetime.now().isoformat()
        
        # Generate category ID if not provided
        if 'category_id' not in category_data:
            category_data['category_id'] = f"category_{int(time.time())}_{os.urandom(4).hex()}"
        
        # Format category data
        category = {
            'category_id': category_data['category_id'],
            'name': category_data['name'],
            'description': category_data.get('description', ''),
            'parent_id': category_data.get('parent_id'),
            'language_code': category_data.get('language_code'),
            'created_at': now
        }
        
        # Insert category
        query = """
        INSERT INTO content_categories 
        (category_id, name, description, parent_id, language_code, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """
        params = (
            category['category_id'], category['name'], category['description'],
            category['parent_id'], category['language_code'], category['created_at']
        )
        
        try:
            self.execute_query(query, params)
            logger.info(f"Created category: {category['name']} ({category['category_id']})")
            return category['category_id']
        except Exception as e:
            logger.error(f"Error creating category: {e}", exc_info=True)
            raise
    
    def get_category(self, category_id: str) -> Optional[Dict[str, Any]]:
        """
        Get category by ID
        
        Args:
            category_id (str): Category ID
            
        Returns:
            Dict[str, Any]: Category data or None if not found
            
        TypeScript equivalent:
            async getCategory(categoryId: string): Promise<CategoryData | null>
        """
        query = "SELECT * FROM content_categories WHERE category_id = ?"
        results = self.execute_query(query, (category_id,))
        
        if not results:
            return None
        
        category = dict(results[0])
        
        # Get subcategories
        subcategories_query = "SELECT * FROM content_categories WHERE parent_id = ?"
        subcategories = self.execute_query(subcategories_query, (category_id,))
        category['subcategories'] = subcategories
        
        # Get contents
        contents_query = """
        SELECT ci.* FROM content_items ci
        JOIN content_category_items cci ON ci.content_id = cci.content_id
        WHERE cci.category_id = ?
        """
        contents = self.execute_query(contents_query, (category_id,))
        category['contents'] = contents
        
        return category
    
    def get_categories_by_language(self, language_code: str, parent_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get categories by language
        
        Args:
            language_code (str): Language code
            parent_id (str, optional): Parent category ID
            
        Returns:
            List[Dict[str, Any]]: List of categories
            
        TypeScript equivalent:
            async getCategoriesByLanguage(languageCode: string, parentId?: string): Promise<CategoryData[]>
        """
        if parent_id:
            query = """
            SELECT * FROM content_categories 
            WHERE language_code = ? AND parent_id = ?
            ORDER BY name
            """
            return self.execute_query(query, (language_code, parent_id))
        else:
            query = """
            SELECT * FROM content_categories 
            WHERE language_code = ? AND parent_id IS NULL
            ORDER BY name
            """
            return self.execute_query(query, (language_code,))
    
    def add_content_to_category(self, content_id: str, category_id: str, position: Optional[int] = None) -> bool:
        """
        Add content to a category
        
        Args:
            content_id (str): Content ID
            category_id (str): Category ID
            position (int, optional): Position within category
            
        Returns:
            bool: Success status
            
        TypeScript equivalent:
            async addContentToCategory(contentId: string, categoryId: string, position?: number): Promise<boolean>
        """
        query = """
        INSERT OR REPLACE INTO content_category_items
        (content_id, category_id, position)
        VALUES (?, ?, ?)
        """
        
        self.execute_query(query, (content_id, category_id, position))
        return True
    
    def get_content_by_type(self, content_type: str, language_code: str, 
                         difficulty_level: Optional[str] = None, 
                         limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get content by type and language
        
        Args:
            content_type (str): Content type
            language_code (str): Language code
            difficulty_level (str, optional): Difficulty level
            limit (int): Maximum number of results
            
        Returns:
            List[Dict[str, Any]]: List of content items
            
        TypeScript equivalent:
            async getContentByType(contentType: string, languageCode: string, 
                                 difficultyLevel?: string, limit?: number): Promise<ContentData[]>
        """
        if difficulty_level:
            query = """
            SELECT * FROM content_items 
            WHERE content_type = ? AND language_code = ? AND difficulty_level = ?
            ORDER BY created_at DESC
            LIMIT ?
            """
            return self.execute_query(query, (content_type, language_code, difficulty_level, limit))
        else:
            query = """
            SELECT * FROM content_items 
            WHERE content_type = ? AND language_code = ?
            ORDER BY created_at DESC
            LIMIT ?
            """
            return self.execute_query(query, (content_type, language_code, limit))
    
    def search_content(self, search_term: str, language_code: Optional[str] = None, 
                    content_type: Optional[str] = None, 
                    limit: int = 100) -> List[Dict[str, Any]]:
        """
        Search content by term
        
        Args:
            search_term (str): Search term
            language_code (str, optional): Language code
            content_type (str, optional): Content type
            limit (int): Maximum number of results
            
        Returns:
            List[Dict[str, Any]]: List of matching content items
            
        TypeScript equivalent:
            async searchContent(searchTerm: string, languageCode?: string, 
                              contentType?: string, limit?: number): Promise<ContentData[]>
        """
        search_param = f"%{search_term}%"
        
        # Build query based on filters
        query = "SELECT * FROM content_items WHERE (title LIKE ? OR content LIKE ? OR tags LIKE ?)"
        params = [search_param, search_param, search_param]
        
        if language_code:
            query += " AND language_code = ?"
            params.append(language_code)
        
        if content_type:
            query += " AND content_type = ?"
            params.append(content_type)
        
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        
        return self.execute_query(query, tuple(params))
    
    def store_embedding(self, content_id: str, chunk_id: str, chunk_text: str, 
                      embedding: np.ndarray, model_name: str) -> int:
        """
        Store a content embedding
        
        Args:
            content_id (str): Content ID
            chunk_id (str): Chunk ID
            chunk_text (str): Text chunk
            embedding (np.ndarray): Embedding vector
            model_name (str): Embedding model name
            
        Returns:
            int: Embedding record ID
            
        TypeScript equivalent:
            async storeEmbedding(contentId: string, chunkId: string, chunkText: string,
                               embedding: number[], modelName: string): Promise<number>
        """
        now = datetime.now().isoformat()
        
        # Convert embedding to binary
        embedding_binary = embedding.tobytes()
        
        query = """
        INSERT OR REPLACE INTO content_embeddings
        (content_id, chunk_id, chunk_text, embedding, model_name, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """
        
        params = (content_id, chunk_id, chunk_text, embedding_binary, model_name, now)
        
        self.execute_query(query, params)
        
        # Get the ID of the inserted record
        result = self.execute_query(
            "SELECT id FROM content_embeddings WHERE content_id = ? AND chunk_id = ?",
            (content_id, chunk_id)
        )
        
        return result[0]['id'] if result else -1
    
    def get_embedding(self, content_id: str, chunk_id: str) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        """
        Get an embedding by content ID and chunk ID
        
        Args:
            content_id (str): Content ID
            chunk_id (str): Chunk ID
            
        Returns:
            Tuple[np.ndarray, Dict[str, Any]]: (embedding, metadata) or None if not found
            
        TypeScript equivalent:
            async getEmbedding(contentId: string, chunkId: string): Promise<[number[], Record<string, any>] | null>
        """
        query = "SELECT * FROM content_embeddings WHERE content_id = ? AND chunk_id = ?"
        results = self.execute_query(query, (content_id, chunk_id))
        
        if not results:
            return None
        
        record = dict(results[0])
        
        # Convert binary to numpy array
        embedding_binary = record['embedding']
        embedding = np.frombuffer(embedding_binary, dtype=np.float32)
        
        metadata = {
            'id': record['id'],
            'content_id': record['content_id'],
            'chunk_id': record['chunk_id'],
            'chunk_text': record['chunk_text'],
            'model_name': record['model_name'],
            'created_at': record['created_at']
        }
        
        return embedding, metadata
    
    def get_all_embeddings(self, model_name: Optional[str] = None) -> List[Tuple[np.ndarray, Dict[str, Any]]]:
        """
        Get all embeddings
        
        Args:
            model_name (str, optional): Filter by model name
            
        Returns:
            List[Tuple[np.ndarray, Dict[str, Any]]]: List of (embedding, metadata) tuples
            
        TypeScript equivalent:
            async getAllEmbeddings(modelName?: string): Promise<[number[], Record<string, any>][]>
        """
        if model_name:
            query = "SELECT * FROM content_embeddings WHERE model_name = ?"
            results = self.execute_query(query, (model_name,))
        else:
            query = "SELECT * FROM content_embeddings"
            results = self.execute_query(query)
        
        embeddings = []
        
        for record in results:
            record = dict(record)
            
            # Convert binary to numpy array
            embedding_binary = record['embedding']
            embedding = np.frombuffer(embedding_binary, dtype=np.float32)
            
            metadata = {
                'id': record['id'],
                'content_id': record['content_id'],
                'chunk_id': record['chunk_id'],
                'chunk_text': record['chunk_text'],
                'model_name': record['model_name'],
                'created_at': record['created_at']
            }
            
            embeddings.append((embedding, metadata))
        
        return embeddings