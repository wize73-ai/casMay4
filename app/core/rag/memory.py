"""
Conversation Memory Manager for CasaLingua

Handles chat session context tracking for conversational AI interactions.

Author: Exygy Development Team
Version: 1.0.0
License: MIT
"""

import logging
import time
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import json
import os
from pathlib import Path

# CasaLingua imports
from app.core.pipeline.tokenizer import TokenizerPipeline
from app.services.models.loader import ModelRegistry
from app.utils.logging import get_logger

logger = get_logger("casalingua.core.memory")

class ConversationMemory:
    """
    Manages conversation history for multiple chat sessions.
    Each session is identified by a unique session ID and stores a list of messages.
    """

    def __init__(self, 
                max_session_messages: int = 50,
                max_token_limit: int = 4000,
                session_ttl_hours: int = 24,
                storage_dir: Optional[str] = None) -> None:
        """
        Initialize the conversation memory manager.
        
        Args:
            max_session_messages: Maximum number of messages to store per session
            max_token_limit: Maximum number of tokens to maintain in conversation history
            session_ttl_hours: Time-to-live for inactive sessions in hours
            storage_dir: Optional directory for persistent storage
        """
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.max_session_messages = max_session_messages
        self.max_token_limit = max_token_limit
        self.session_ttl_hours = session_ttl_hours
        
        # Setup persistent storage if provided
        self.storage_dir = Path(storage_dir) if storage_dir else None
        if self.storage_dir:
            self.storage_dir.mkdir(parents=True, exist_ok=True)
            
        # Load tokenizer dynamically from registry
        try:
            registry = ModelRegistry()
            _, tokenizer_name = registry.get_model_and_tokenizer("rag_generator")
            self.tokenizer = TokenizerPipeline(model_name=tokenizer_name, task_type="rag_generation")
            logger.info(f"Loaded tokenizer: {tokenizer_name}")
        except Exception as e:
            logger.warning(f"Failed to load tokenizer: {e}. Token counting disabled.")
            self.tokenizer = None
    
    def get_history(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve conversation history for a session.
        
        Args:
            session_id: Unique session identifier
            
        Returns:
            List of message dictionaries with 'role' and 'content' keys
        """
        # Check if session exists
        if session_id not in self.sessions:
            # Try to load from persistent storage
            if self._load_session(session_id):
                logger.debug(f"Loaded session {session_id} from persistent storage")
            else:
                logger.debug(f"No history found for session {session_id}")
                return []
                
        # Update last accessed time
        self.sessions[session_id]["last_accessed"] = datetime.now()
        
        # Return messages
        return self.sessions[session_id]["messages"]
    
    def add_message(self, 
                  session_id: str, 
                  role: str, 
                  content: str,
                  metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a message to the session history.
        
        Args:
            session_id: Unique session identifier
            role: Message role (e.g., 'user', 'assistant')
            content: Message content text
            metadata: Optional additional message metadata
        """
        # Initialize session if it doesn't exist
        if session_id not in self.sessions:
            self._init_session(session_id)
            
        # Create message object
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        
        # Add token count if tokenizer available
        if self.tokenizer:
            token_ids = self.tokenizer.encode(content)
            message["token_count"] = len(token_ids)
        
        # Add metadata if provided
        if metadata:
            message["metadata"] = metadata
            
        # Add message to session
        self.sessions[session_id]["messages"].append(message)
        
        # Update token count
        if self.tokenizer:
            self.sessions[session_id]["total_tokens"] += message.get("token_count", 0)
            
        # Update last modified time
        self.sessions[session_id]["last_modified"] = datetime.now()
        self.sessions[session_id]["last_accessed"] = datetime.now()
        
        # Trim history if needed
        self._trim_history(session_id)
        
        # Save to persistent storage if enabled
        self._save_session(session_id)
        
        logger.debug(f"Added {role} message to session {session_id}")
    
    def get_last_message(self, session_id: str, role: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get the most recent message from the session history, optionally filtered by role.
        
        Args:
            session_id: Unique session identifier
            role: Optional role filter
            
        Returns:
            Most recent message dictionary or None if not found
        """
        history = self.get_history(session_id)
        
        if not history:
            return None
            
        # If role specified, filter by role
        if role:
            filtered = [msg for msg in history if msg.get("role") == role]
            return filtered[-1] if filtered else None
            
        # Otherwise return most recent message
        return history[-1]
    
    def clear_history(self, session_id: str) -> None:
        """
        Clear all history for a given session.
        
        Args:
            session_id: Unique session identifier
        """
        if session_id in self.sessions:
            del self.sessions[session_id]
            
            # Also remove from persistent storage if enabled
            if self.storage_dir:
                storage_path = self.storage_dir / f"{session_id}.json"
                if storage_path.exists():
                    storage_path.unlink()
                    
            logger.info(f"Cleared history for session {session_id}")
    
    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """
        Get summary information about a session.
        
        Args:
            session_id: Unique session identifier
            
        Returns:
            Dictionary with session summary information
        """
        if session_id not in self.sessions:
            if not self._load_session(session_id):
                return {"exists": False}
        
        session = self.sessions[session_id]
        
        return {
            "exists": True,
            "message_count": len(session["messages"]),
            "total_tokens": session.get("total_tokens", 0),
            "created_at": session["created_at"].isoformat(),
            "last_modified": session["last_modified"].isoformat(),
            "last_accessed": session["last_accessed"].isoformat(),
            "user_message_count": sum(1 for msg in session["messages"] if msg.get("role") == "user"),
            "assistant_message_count": sum(1 for msg in session["messages"] if msg.get("role") == "assistant")
        }
    
    def get_all_sessions(self) -> List[str]:
        """
        Get a list of all active session IDs.
        
        Returns:
            List of session IDs
        """
        return list(self.sessions.keys())
    
    def cleanup_expired_sessions(self) -> int:
        """
        Remove expired sessions based on TTL.
        
        Returns:
            Number of sessions removed
        """
        now = datetime.now()
        expired_cutoff = now - timedelta(hours=self.session_ttl_hours)
        
        expired_sessions = [
            sid for sid, session in self.sessions.items()
            if session["last_accessed"] < expired_cutoff
        ]
        
        for session_id in expired_sessions:
            self.clear_history(session_id)
            
        logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
        return len(expired_sessions)
    
    def _init_session(self, session_id: str) -> None:
        """Initialize a new session."""
        now = datetime.now()
        self.sessions[session_id] = {
            "messages": [],
            "created_at": now,
            "last_modified": now,
            "last_accessed": now,
            "total_tokens": 0
        }
    
    def _trim_history(self, session_id: str) -> None:
        """
        Trim session history to stay within limits.
        Removes oldest messages first.
        """
        if session_id not in self.sessions:
            return
            
        session = self.sessions[session_id]
        messages = session["messages"]
        
        # Trim by max messages
        if len(messages) > self.max_session_messages:
            # Calculate how many messages to remove
            excess = len(messages) - self.max_session_messages
            
            # Remove oldest messages
            removed_messages = messages[:excess]
            messages = messages[excess:]
            
            # Update token count
            if self.tokenizer:
                session["total_tokens"] -= sum(msg.get("token_count", 0) for msg in removed_messages)
                
            session["messages"] = messages
            logger.debug(f"Trimmed {excess} messages from session {session_id} based on message limit")
            
        # Trim by token limit
        if self.tokenizer and session.get("total_tokens", 0) > self.max_token_limit:
            # Remove oldest messages until under token limit
            while session.get("total_tokens", 0) > self.max_token_limit and messages:
                oldest = messages.pop(0)
                token_count = oldest.get("token_count", 0)
                session["total_tokens"] -= token_count
                logger.debug(f"Removed message with {token_count} tokens from session {session_id}")
    
    def _save_session(self, session_id: str) -> bool:
        """Save session to persistent storage if enabled."""
        if not self.storage_dir:
            return False
            
        try:
            session_path = self.storage_dir / f"{session_id}.json"
            session_data = self.sessions[session_id].copy()
            
            # Convert datetime objects to strings
            session_data["created_at"] = session_data["created_at"].isoformat()
            session_data["last_modified"] = session_data["last_modified"].isoformat()
            session_data["last_accessed"] = session_data["last_accessed"].isoformat()
            
            with open(session_path, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, ensure_ascii=False, indent=2)
                
            return True
        except Exception as e:
            logger.error(f"Error saving session {session_id}: {e}")
            return False
    
    def _load_session(self, session_id: str) -> bool:
        """Load session from persistent storage if available."""
        if not self.storage_dir:
            return False
            
        session_path = self.storage_dir / f"{session_id}.json"
        
        if not session_path.exists():
            return False
            
        try:
            with open(session_path, 'r', encoding='utf-8') as f:
                session_data = json.load(f)
                
            # Convert string timestamps to datetime objects
            session_data["created_at"] = datetime.fromisoformat(session_data["created_at"])
            session_data["last_modified"] = datetime.fromisoformat(session_data["last_modified"])
            session_data["last_accessed"] = datetime.fromisoformat(session_data["last_accessed"])
            
            self.sessions[session_id] = session_data
            return True
        except Exception as e:
            logger.error(f"Error loading session {session_id}: {e}")
            return False
    
    def get_session_context(self, 
                          session_id: str, 
                          max_context_tokens: Optional[int] = None,
                          include_roles: Optional[List[str]] = None) -> str:
        """
        Get formatted conversation context for a session.
        
        Args:
            session_id: Unique session identifier
            max_context_tokens: Maximum token count for context
            include_roles: Optional list of roles to include
            
        Returns:
            Formatted conversation context
        """
        history = self.get_history(session_id)
        
        if not history:
            return ""
            
        # Filter by roles if specified
        if include_roles:
            history = [msg for msg in history if msg.get("role") in include_roles]
            
        # If token limit specified and tokenizer available, trim history
        if max_context_tokens and self.tokenizer:
            # Trim from oldest messages first
            token_count = 0
            messages = []
            
            for msg in reversed(history):
                msg_tokens = msg.get("token_count", 0)
                if token_count + msg_tokens > max_context_tokens:
                    break
                    
                token_count += msg_tokens
                messages.insert(0, msg)
                
            history = messages
            
        # Format conversation context
        context_lines = []
        for msg in history:
            role = msg.get("role", "").upper()
            content = msg.get("content", "")
            context_lines.append(f"{role}: {content}")
            
        return "\n".join(context_lines)
        
    def __repr__(self) -> str:
        return f"<ConversationMemory sessions={len(self.sessions)}>"


class ConversationMemoryManager:
    """
    Manages conversation memories with automated cleanup.
    """
    
    def __init__(self, 
                memory: Optional[ConversationMemory] = None,
                cleanup_interval_hours: int = 6) -> None:
        """
        Initialize the conversation memory manager.
        
        Args:
            memory: Optional existing ConversationMemory instance
            cleanup_interval_hours: Interval for cleanup of expired sessions
        """
        self.memory = memory or ConversationMemory()
        self.cleanup_interval_hours = cleanup_interval_hours
        self.last_cleanup = datetime.now()
    
    def get_memory(self) -> ConversationMemory:
        """Get the conversation memory instance."""
        self._check_cleanup()
        return self.memory
    
    def _check_cleanup(self) -> None:
        """Check if cleanup is needed and perform if necessary."""
        now = datetime.now()
        elapsed = now - self.last_cleanup
        
        if elapsed > timedelta(hours=self.cleanup_interval_hours):
            self.memory.cleanup_expired_sessions()
            self.last_cleanup = now


# Example usage
if __name__ == "__main__":
    # Create a memory manager with persistent storage
    memory_manager = ConversationMemoryManager(
        memory=ConversationMemory(
            max_session_messages=100,
            max_token_limit=8000,
            storage_dir="./data/conversation_memory"
        )
    )
    
    # Get memory instance
    memory = memory_manager.get_memory()
    
    # Example: Add conversation messages
    session_id = "example_session_123"
    memory.add_message(session_id, "user", "¿Cómo puedo aprender español rápidamente?")
    memory.add_message(session_id, "assistant", "Para aprender español rápidamente, te recomiendo practicar todos los días, escuchar podcasts en español, y utilizar aplicaciones de aprendizaje de idiomas.")
    memory.add_message(session_id, "user", "What about grammar resources?")
    
    # Get conversation history
    history = memory.get_history(session_id)
    
    # Get formatted context
    context = memory.get_session_context(session_id)
    
    # Get session summary
    summary = memory.get_session_summary(session_id)
    
    print(f"Session {session_id} has {summary['message_count']} messages")
    print(f"Conversation context:\n{context}")