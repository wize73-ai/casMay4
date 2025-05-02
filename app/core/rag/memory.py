"""
Conversation Memory Manager for CasaLingua

Handles chat session context tracking for conversational AI interactions.

Author: Exygy Development Team
Version: 1.0.0
License: MIT
"""

import logging
from typing import List, Dict, Any, Optional
from app.core.pipeline.tokenizer import TokenizerPipeline
from app.services.models.loader import ModelRegistry

logger = logging.getLogger("casalingua.core.rag.memory")


class ConversationMemory:
    """
    Manages in-memory conversation history for multiple chat sessions.
    Each session is identified by a unique session ID and stores a list of messages.
    """

    def __init__(self) -> None:
        """Initialize in-memory session store."""
        self.sessions: Dict[str, List[Dict[str, Any]]] = {}

        # Dynamically load tokenizer
        registry = ModelRegistry()
        _, tokenizer_name = registry.get_model_and_tokenizer("rag_generator")
        self.tokenizer = TokenizerPipeline(model_name=tokenizer_name, task_type="rag_generation")

    def get_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Retrieve conversation history for a session."""
        logger.debug(f"Retrieving history for session {session_id}")
        return self.sessions.get(session_id, [])

    def add_message(self, session_id: str, role: str, content: str) -> None:
        """Add a message to the session history."""
        logger.debug(f"Adding message to session {session_id}: {role} -> {content}")
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        if self.tokenizer:
            token_ids = self.tokenizer.encode(content)
        else:
            token_ids = []
        self.sessions[session_id].append({
            "role": role,
            "content": content,
            "tokens": token_ids
        })

    def get_last_message(self, session_id: str) -> Dict[str, Any] | None:
        """Get the most recent message from the session history."""
        history = self.sessions.get(session_id, [])
        if history:
            return history[-1]
        return None

    def clear_history(self, session_id: str) -> None:
        """Clear all history for a given session."""
        logger.info(f"Clearing history for session {session_id}")
        if session_id in self.sessions:
            del self.sessions[session_id]

    def __repr__(self) -> str:
        return f"<ConversationMemory sessions={len(self.sessions)}>"