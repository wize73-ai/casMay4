"""
RAG Expert Module for CasaLingua

This module implements Retrieval-Augmented Generation functionality for enhanced
language processing, providing context-aware translation and generation.

Author: Exygy Development Team
Version: 1.0.0
License: MIT
"""

import os
import json
from typing import List, Dict, Optional, Any
import numpy as np
import torch
from pathlib import Path

from app.core.pipeline.tokenizer import TokenizerPipeline

# Import ModelRegistry for dynamic tokenizer loading
from app.services.models.loader import ModelRegistry

# Import sentence transformers if available
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

from app.utils.logging import get_logger

logger = get_logger(__name__)

class RAGExpert:
    """
    RAG Expert class for implementing Retrieval-Augmented Generation
    in the CasaLingua language processing application.
    
    This class handles:
    - Document embedding and retrieval
    - Context-aware processing
    - Knowledge base management
    """
    
    def __init__(
        self,
        model_manager: Any,
        config: Optional[Dict[str, Any]] = None,
        registry_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the RAG Expert system.
        
        Args:
            model_manager: Model manager for accessing models
            config: Configuration dictionary
            registry_config: Model registry configuration dictionary
        """
        self.model_manager = model_manager
        self.config: Dict[str, Any] = config or {}
        self.initialized: bool = False
        # Load tokenizer dynamically from provided registry_config
        registry_config = registry_config or {}
        tokenizer_name = registry_config["rag_retriever"]["tokenizer_name"]
        self.tokenizer = TokenizerPipeline(model_name=tokenizer_name, task_type="rag_retrieval")

        # Initialize parameters from config
        self.embedding_model_key: str = self.config.get("rag_embedding_model", "embedding_model")
        self.use_gpu: bool = self.config.get("use_gpu", torch.cuda.is_available())
        self.top_k: int = self.config.get("rag_top_k", 5)

        # Knowledge base settings
        kb_dir = self.config.get("knowledge_base_dir", "knowledge_base")
        self.knowledge_base_dir: Path = Path(kb_dir)
        self.knowledge_base: List[Dict[str, Any]] = []

        # Index settings
        self.index_dir: Path = Path(self.config.get("index_dir", "indexes"))
        self.index_path: Path = self.index_dir / "rag_index.faiss"
        self.index: Optional[Any] = None
        self.embedding_dim: Optional[int] = None

        # Fallback to sparse retrieval if needed
        self.use_sparse_fallback: bool = self.config.get("use_sparse_fallback", True)

        logger.info("RAG Expert initialized (not yet loaded)")
    
    async def initialize(self) -> None:
        """
        Initialize the RAG Expert system.
        
        This loads necessary models and prepares the knowledge base.
        """
        if self.initialized:
            logger.warning("RAG Expert already initialized")
            return

        logger.info("Initializing RAG Expert")

        os.makedirs(self.knowledge_base_dir, exist_ok=True)
        os.makedirs(self.index_dir, exist_ok=True)

        try:
            # Load embedding model
            self.embedding_model = None
            if self.model_manager is not None:
                try:
                    model_info = await self.model_manager.load_model(self.embedding_model_key)
                    self.embedding_model = model_info.get("model") if model_info else None
                    # Get embedding dimension if available
                    if self.embedding_model and hasattr(self.embedding_model, "get_sentence_embedding_dimension"):
                        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
                    elif self.embedding_model and hasattr(self.embedding_model, "config") and hasattr(self.embedding_model.config, "hidden_size"):
                        self.embedding_dim = self.embedding_model.config.hidden_size
                    else:
                        self.embedding_dim = 768
                except Exception as e:
                    logger.warning(f"Could not load embedding model: {e}, using fallback")
                    self.embedding_dim = 768
            elif SENTENCE_TRANSFORMERS_AVAILABLE:
                model_name = self.config.get(
                    "rag_embedding_model_name", "paraphrase-multilingual-mpnet-base-v2"
                )
                device = "cuda" if self.use_gpu and torch.cuda.is_available() else "cpu"
                self.embedding_model = SentenceTransformer(model_name, device=device)
                self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
            else:
                logger.warning("Sentence Transformers not available, RAG will use fallback methods")
                self.embedding_model = None

            logger.info(f"RAG embedding model loaded with dimension {self.embedding_dim}")

            await self._load_knowledge_base()

            # Load or build index
            if self.index_path.exists() and FAISS_AVAILABLE:
                await self._load_index()
            elif len(self.knowledge_base) > 0:
                await self._build_index()

            self.initialized = True
            logger.info("RAG Expert initialization complete")
        except Exception as e:
            logger.error(f"Error initializing RAG Expert: {str(e)}", exc_info=True)
            raise
    
    async def _load_knowledge_base(self) -> None:
        """Load the knowledge base from files."""
        try:
            self.knowledge_base = []
            if not self.knowledge_base_dir.exists():
                logger.warning(f"Knowledge base directory not found: {self.knowledge_base_dir}")
                return

            # Load from JSON files
            for file_path in self.knowledge_base_dir.glob("**/*.json"):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    if isinstance(data, list):
                        self.knowledge_base.extend(data)
                    else:
                        self.knowledge_base.append(data)
                except Exception as e:
                    logger.error(f"Error loading {file_path}: {str(e)}")

            # Load from text files
            for file_path in self.knowledge_base_dir.glob("**/*.txt"):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                self.knowledge_base.append({
                                    "text": line,
                                    "source": file_path.name,
                                    "metadata": {
                                        "type": "text",
                                        "language": self._detect_language(line)
                                    }
                                })
                                if self.tokenizer:
                                    tokens = self.tokenizer.encode(line)
                                    self.knowledge_base[-1]["tokens"] = tokens
                except Exception as e:
                    logger.error(f"Error loading {file_path}: {str(e)}")

            logger.info(f"Loaded {len(self.knowledge_base)} documents into knowledge base")
        except Exception as e:
            logger.error(f"Error loading knowledge base: {str(e)}", exc_info=True)
            self.knowledge_base = []
    
    async def _build_index(self) -> None:
        """Build the search index from the knowledge base."""
        if not self.knowledge_base:
            logger.warning("Knowledge base is empty, cannot build index")
            return

        try:
            texts = [item.get("text", "") for item in self.knowledge_base]
            if self.embedding_model:
                logger.info("Generating embeddings for knowledge base...")
                if hasattr(self.embedding_model, "encode"):
                    embeddings = self.embedding_model.encode(
                        texts,
                        show_progress_bar=True,
                        convert_to_numpy=True,
                    )
                else:
                    embeddings = await self.model_manager.create_embeddings(texts)

                if FAISS_AVAILABLE and self.embedding_dim:
                    self.index = faiss.IndexFlatL2(self.embedding_dim)
                    self.index.add(np.array(embeddings).astype('float32'))
                    faiss.write_index(self.index, str(self.index_path))
                    logger.info(f"Built and saved FAISS index to {self.index_path}")
                else:
                    logger.warning("FAISS not available, using fallback retrieval methods")
            else:
                logger.warning("No embedding model available, using fallback retrieval methods")
        except Exception as e:
            logger.error(f"Error building index: {str(e)}", exc_info=True)
    
    async def _load_index(self) -> None:
        """Load the FAISS index from disk."""
        try:
            if FAISS_AVAILABLE and self.index_path.exists():
                self.index = faiss.read_index(str(self.index_path))
                logger.info(f"Loaded FAISS index from {self.index_path} with {self.index.ntotal} vectors")
            else:
                logger.warning("FAISS index not found, will build new index if needed")
        except Exception as e:
            logger.error(f"Error loading index: {str(e)}", exc_info=True)
            self.index = None
    
    async def get_context(
        self,
        query: str,
        source_language: str,
        target_language: str,
        options: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get relevant context for a query from the knowledge base.
        
        Args:
            query: The query text
            source_language: Source language code
            target_language: Target language code
            options: Additional options
            
        Returns:
            List of relevant context documents
        """
        if not self.initialized:
            await self.initialize()
        if not self.knowledge_base:
            logger.warning("Knowledge base is empty, no context available")
            return []
        options = options or {}
        max_results = options.get("max_results", self.top_k)
        try:
            results = await self._retrieve(
                query,
                source_language,
                max_results=max_results,
            )
            grade_level = options.get("grade_level")
            if grade_level is not None and isinstance(grade_level, (int, float)):
                results = self._filter_by_grade_level(results, grade_level)
            if target_language != source_language:
                target_lang_results = [
                    r for r in results
                    if r.get("metadata", {}).get("language") == target_language
                ]
                if len(target_lang_results) >= max_results // 2:
                    results = target_lang_results[:max_results]
            if not results:
                logger.info("No relevant context found for query")
            return results[:max_results]
        except Exception as e:
            logger.error(f"Error getting context: {str(e)}", exc_info=True)
            return []
    
    async def _retrieve(
        self,
        query: str,
        language: str,
        max_results: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: The query text
            language: Language code
            max_results: Maximum number of results to return
            
        Returns:
            List of retrieved documents with relevance scores
        """
        try:
            # Vector-based retrieval
            if self.embedding_model is not None and self.index is not None:
                if hasattr(self.embedding_model, "encode"):
                    query_embedding = self.embedding_model.encode([query])[0]
                else:
                    embeddings = await self.model_manager.create_embeddings([query])
                    query_embedding = embeddings[0]
                query_embedding = np.array([query_embedding]).astype('float32')
                distances, indices = self.index.search(
                    query_embedding,
                    k=min(max_results, len(self.knowledge_base))
                )
                results = []
                for i, idx in enumerate(indices[0]):
                    if 0 <= idx < len(self.knowledge_base):
                        doc = self.knowledge_base[idx]
                        score = float(1.0 / (1.0 + distances[0][i]))
                        results.append({
                            "text": doc.get("text", ""),
                            "score": score,
                            "source": doc.get("source", "unknown"),
                            "metadata": doc.get("metadata", {})
                        })
                return results
            # Fallback to keyword-based retrieval
            if self.use_sparse_fallback:
                return self._fallback_retrieval(query, language, max_results)
            logger.warning("No retrieval method available")
            return []
        except Exception as e:
            logger.error(f"Error in retrieval: {str(e)}", exc_info=True)
            if self.use_sparse_fallback:
                logger.info("Using fallback retrieval after error")
                return self._fallback_retrieval(query, language, max_results)
            return []
    
    def _fallback_retrieval(
        self,
        query: str,
        language: str,
        max_results: int
    ) -> List[Dict[str, Any]]:
        """
        Fallback keyword-based retrieval when vector search is unavailable.
        
        Args:
            query: The query text
            language: Language code
            max_results: Maximum number of results
            
        Returns:
            List of retrieved documents
        """
        try:
            query_words = set(query.lower().split())
            scored_docs = []
            for doc in self.knowledge_base:
                doc_text = doc.get("text", "").lower()
                doc_words = set(doc_text.split())
                common_words = query_words.intersection(doc_words)
                if not common_words:
                    continue
                score = len(common_words) / (len(query_words) + len(doc_words) - len(common_words))
                doc_lang = doc.get("metadata", {}).get("language")
                if doc_lang and doc_lang == language:
                    score *= 1.5
                scored_docs.append({
                    "text": doc.get("text", ""),
                    "score": score,
                    "source": doc.get("source", "unknown"),
                    "metadata": doc.get("metadata", {})
                })
            scored_docs.sort(key=lambda x: x["score"], reverse=True)
            return scored_docs[:max_results]
        except Exception as e:
            logger.error(f"Error in fallback retrieval: {str(e)}", exc_info=True)
            return []
    
    def _filter_by_grade_level(
        self,
        results: List[Dict[str, Any]],
        target_grade: int,
    ) -> List[Dict[str, Any]]:
        """
        Filter results by grade level.
        
        Args:
            results: List of retrieved documents
            target_grade: Target grade level (1-12)
            
        Returns:
            Filtered list of documents
        """
        filtered_results: List[Dict[str, Any]] = []
        for result in results:
            doc_grade = result.get("metadata", {}).get("grade_level")
            if doc_grade is None:
                filtered_results.append(result)
                continue
            if isinstance(doc_grade, (int, float)) and doc_grade <= target_grade:
                filtered_results.append(result)
        if not filtered_results and results:
            return results[:min(3, len(results))]
        return filtered_results
    
    def _detect_language(self, text: str) -> str:
        """
        Simple language detection based on common words.
        
        Args:
            text: Text to analyze
            
        Returns:
            Detected language code
        """
        markers = {
            'en': ['the', 'and', 'of', 'to', 'in', 'is', 'you', 'that', 'it', 'for'],
            'es': ['el', 'la', 'de', 'que', 'y', 'a', 'en', 'un', 'ser', 'se'],
            'fr': ['le', 'la', 'de', 'et', 'est', 'en', 'que', 'un', 'une', 'du'],
            'de': ['der', 'die', 'das', 'und', 'ist', 'in', 'zu', 'den', 'mit', 'nicht']
        }
        text = text.lower()
        scores = {lang: 0 for lang in markers}
        for lang, words in markers.items():
            for word in words:
                if f" {word} " in f" {text} ":
                    scores[lang] += 1
        max_score = 0
        detected_lang = "en"
        for lang, score in scores.items():
            if score > max_score:
                max_score = score
                detected_lang = lang
        return detected_lang
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        self.knowledge_base = []
        self.index = None
        self.embedding_model = None
        logger.info("RAG Expert resources cleaned up")