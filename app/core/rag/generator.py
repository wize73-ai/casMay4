"""
Augmented Generator for CasaLingua RAG Pipeline

This module handles retrieval-augmented translation and chat generation
by combining model responses with contextual data from relevant documents.

Author: Exygy Development Team
Version: 1.0.0
License: MIT
"""

import logging
from typing import List, Optional
from dataclasses import dataclass
from app.core.pipeline.tokenizer import TokenizerPipeline
from app.services.models.loader import ModelRegistry


logger = logging.getLogger("casalingua.core.rag.generator")


@dataclass
class TranslationResult:
    """
    Represents the result of a translation operation.
    """
    text: str
    confidence: float
    model_id: str
    timestamp: str


class AugmentedGenerator:
    def __init__(self, model_manager):
        self.model_manager = model_manager

        # Load tokenizer dynamically
        registry = ModelRegistry()
        _, tokenizer_name = registry.get_model_and_tokenizer("rag_generator")
        self.tokenizer = TokenizerPipeline(model_name=tokenizer_name, task_type="rag_generation")

    async def translate(
        self,
        text: str,
        source_language: str,
        target_language: str,
        reference_documents: Optional[List] = None,
        model_id: Optional[str] = None,
    ) -> TranslationResult:
        """
        Generate a context-aware translation using available models and reference documents.

        Args:
            text (str): The text to translate.
            source_language (str): The language code of the source text.
            target_language (str): The language code to translate into.
            reference_documents (Optional[List]): List of reference documents for context.
            model_id (Optional[str]): Optional model identifier.

        Returns:
            TranslationResult: The result of the translation, including text, confidence, model_id, and timestamp.
        """
        logger.info("Generating RAG-enhanced translation")

        # Combine text and reference context safely
        if reference_documents:
            try:
                context = "\n\n".join(getattr(doc, "content", "") for doc in reference_documents if hasattr(doc, "content"))
            except Exception as e:
                logger.warning(f"Malformed reference_documents: {e}")
                context = ""
        else:
            context = ""
        input_text = f"{context}\n\n{text}" if context else text

        if self.tokenizer:
            token_ids = self.tokenizer.encode(input_text)
            logger.debug(f"Tokenized input (translation): {token_ids}")

        model = self.model_manager.get_model(model_id, task="translation")
        if not model:
            logger.error("Translation model not found.")
            raise ValueError("Translation model could not be loaded.")
        try:
            output = await model.translate(input_text, source_language, target_language)
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            raise

        return TranslationResult(
            text=output.text,
            confidence=output.confidence,
            model_id=model.model_id,
            timestamp=output.timestamp,
        )

    async def generate_chat_response(
        self,
        message: str,
        conversation_history: List[dict],
        reference_documents: Optional[List] = None,
        language: str = "en",
    ) -> str:
        """
        Generate a chat response incorporating conversation history and contextual documents.

        Args:
            message (str): The latest user message.
            conversation_history (List[dict]): List of previous messages with 'role' and 'content'.
            reference_documents (Optional[List]): List of reference documents for context.
            language (str): Language code for the chat model.

        Returns:
            str: The generated assistant response.
        """
        logger.info("Generating RAG-enhanced chat response")

        history = "\n".join(f"{msg['role'].upper()}: {msg['content']}" for msg in conversation_history)
        # Safely combine reference context
        if reference_documents:
            try:
                context = "\n\n".join(getattr(doc, "content", "") for doc in reference_documents if hasattr(doc, "content"))
            except Exception as e:
                logger.warning(f"Malformed reference_documents: {e}")
                context = ""
        else:
            context = ""
        prompt = f"{context}\n\n{history}\nUSER: {message}\nASSISTANT:"

        if self.tokenizer:
            token_ids = self.tokenizer.encode(prompt)
            logger.debug(f"Tokenized input (chat): {token_ids}")

        model = self.model_manager.get_model(task="chat", language=language)
        if not model:
            logger.error("Chat model not found.")
            raise ValueError("Chat model could not be loaded.")
        try:
            response = await model.generate(prompt)
        except Exception as e:
            logger.error(f"Chat generation failed: {e}")
            raise

        return response.text
