import requests
from typing import List, Dict

def crawl_github_repo_for_docs(repo_url: str, max_depth: int = 1) -> List[Dict[str, str]]:
    """
    Crawl a GitHub repository's README and Markdown content to build document records.

    Args:
        repo_url (str): The base GitHub repository URL (e.g., https://github.com/bloom-housing/bloom)
        max_depth (int): Currently unused. For future expansion into recursive crawling.

    Returns:
        List of document dicts with 'title' and 'content'
    """
    docs = []
    if "github.com" not in repo_url:
        raise ValueError("Invalid GitHub URL")

    base_raw = repo_url.replace("github.com", "raw.githubusercontent.com") + "/main/"
    readme_urls = ["README.md", "docs/README.md", "docs/index.md"]

    for path in readme_urls:
        try:
            full_url = base_raw + path
            response = requests.get(full_url, timeout=5)
            if response.status_code == 200:
                content = response.text
                docs.append({
                    "title": path,
                    "content": content
                })
        except Exception as e:
            logger.warning(f"Error fetching {path}: {str(e)}")

    return docs


def ingest_sources_from_config(config_path: str = "config/rag_sources.json") -> List[Dict[str, str]]:
    """
    Load RAG sources from a config file and crawl GitHub repos for content.

    Args:
        config_path (str): Path to JSON file with list of repo URLs.

    Returns:
        List of document dictionaries from all sources.
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"RAG sources config not found: {config_path}")

    with open(config_file, "r") as f:
        sources = json.load(f)

    all_docs = []
    for repo_url in sources.get("github_repos", []):
        try:
            docs = crawl_github_repo_for_docs(repo_url)
            all_docs.extend(docs)
        except Exception as e:
            logger.warning(f"Skipping repo {repo_url} due to error: {str(e)}")

    return all_docs
"""
Retriever Module for CasaLingua
Implements advanced retrieval methods for language learning content
"""

import os
import json
import faiss
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple, Union, Callable
from sentence_transformers import SentenceTransformer, util
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
import pickle
from pathlib import Path
# Add TokenizerPipeline import
from app.core.pipeline.tokenizer import TokenizerPipeline
# Import ModelRegistry for dynamic tokenizer loading
from app.services.models.loader import ModelRegistry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BaseRetriever:
    """Base class for all retriever implementations."""

    def __init__(self) -> None:
        """Initialize the base retriever."""
        self.documents: List[Dict[str, Any]] = []

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Add documents to the retriever.

        Args:
            documents (List[Dict[str, Any]]): List of document dictionaries.
        """
        self.documents.extend(documents)

    def retrieve(self, query: str, top_k: int = 5, **kwargs) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.

        Args:
            query (str): Query string.
            top_k (int): Number of documents to retrieve.
            **kwargs: Additional parameters for specific retrievers.

        Returns:
            List[Dict[str, Any]]: List of retrieved documents with relevance scores.
        """
        raise NotImplementedError("Subclasses must implement retrieve()")

    def save(self, path: str) -> bool:
        """
        Save the retriever state to disk.

        Args:
            path (str): Path to save the retriever.

        Returns:
            bool: Success status.
        """
        raise NotImplementedError("Subclasses must implement save()")

    @classmethod
    def load(cls, path: str) -> "BaseRetriever":
        """
        Load a retriever from disk.

        Args:
            path (str): Path to the saved retriever.

        Returns:
            BaseRetriever: Loaded retriever instance.
        """
        raise NotImplementedError("Subclasses must implement load()")


class TfidfRetriever(BaseRetriever):
    """TF-IDF based retriever using scikit-learn"""
    
    def __init__(self) -> None:
        """Initialize the TF-IDF retriever."""
        super().__init__()
        self.vectorizer: TfidfVectorizer = TfidfVectorizer()
        self.index = None
        self.doc_texts: List[str] = []
        
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Add documents to the retriever and update the index
        
        Args:
            documents (List[Dict]): List of document dictionaries
        """
        # Extract text from documents
        texts = [doc.get("text", "") for doc in documents]
        
        # Update document store
        current_doc_count = len(self.documents)
        super().add_documents(documents)
        
        # If we already have an index, update it
        if self.index is not None:
            # Add new documents to the doc_texts
            self.doc_texts.extend(texts)
            
            # Rebuild the vectorizer with all texts
            self.vectorizer = TfidfVectorizer()
            self.index = self.vectorizer.fit_transform(self.doc_texts)
            
        # Otherwise, build a new index if we have documents
        elif len(self.documents) > 0:
            self.doc_texts = [doc.get("text", "") for doc in self.documents]
            self.index = self.vectorizer.fit_transform(self.doc_texts)
            
        logger.info(f"TF-IDF index updated with {len(documents)} new documents. Total: {len(self.documents)}")
        
    def retrieve(self, query: str, top_k: int = 5, **kwargs) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents using TF-IDF
        
        Args:
            query (str): Query string
            top_k (int): Number of documents to retrieve
            
        Returns:
            List[Dict]: List of retrieved documents with relevance scores
        """
        if self.index is None or len(self.documents) == 0:
            logger.warning("No documents indexed for retrieval")
            return []
            
        # Transform query to TF-IDF space
        query_vec = self.vectorizer.transform([query])
        
        # Compute similarities
        similarities = (query_vec * self.index.T).toarray()[0]
        
        # Get top-k indices
        top_indices = similarities.argsort()[::-1][:top_k]
        
        # Gather results
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:  # Only include if there's some similarity
                doc = self.documents[idx].copy()
                doc["score"] = float(similarities[idx])
                results.append(doc)
                
        return results
    
    def save(self, path: str) -> bool:
        """
        Save the TF-IDF retriever to disk
        
        Args:
            path (str): Path to save the retriever
            
        Returns:
            bool: Success status
        """
        try:
            save_path = Path(path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            data = {
                "documents": self.documents,
                "doc_texts": self.doc_texts,
                "vectorizer": self.vectorizer,
            }
            with save_path.open('wb') as f:
                pickle.dump(data, f)
            logger.info(f"Saved TF-IDF retriever to {save_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save TF-IDF retriever to {path}: {e}")
            return False
    
    @classmethod
    def load(cls, path: str) -> "TfidfRetriever":
        """
        Load a TF-IDF retriever from disk
        
        Args:
            path (str): Path to the saved retriever
            
        Returns:
            TfidfRetriever: Loaded retriever instance
        """
        try:
            load_path = Path(path)
            with load_path.open('rb') as f:
                data = pickle.load(f)
            retriever = cls()
            retriever.documents = data["documents"]
            retriever.doc_texts = data["doc_texts"]
            retriever.vectorizer = data["vectorizer"]
            if retriever.doc_texts:
                retriever.index = retriever.vectorizer.transform(retriever.doc_texts)
            logger.info(f"Loaded TF-IDF retriever from {load_path} with {len(retriever.documents)} documents")
            return retriever
        except Exception as e:
            logger.error(f"Failed to load TF-IDF retriever from {path}: {e}")
            # Return an empty retriever instead of reinitializing twice
            empty = cls()
            return empty


class DenseRetriever(BaseRetriever):
    """Dense retriever using sentence transformers and FAISS"""
    
    def __init__(
        self,
        model_name: str = "paraphrase-multilingual-mpnet-base-v2",
        use_gpu: bool = torch.cuda.is_available(),
    ) -> None:
        """
        Initialize the dense retriever.

        Args:
            model_name (str): Name of the SentenceTransformer model.
            use_gpu (bool): Whether to use GPU for embeddings.
        """
        super().__init__()
        self.device: str = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        self.model_name: str = model_name
        try:
            self.model = SentenceTransformer(model_name, device=self.device)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"Loaded embedding model: {model_name} (dim={self.embedding_dim})")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise RuntimeError(f"Failed to load embedding model: {e}")
        self.index = None
        
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Add documents to the retriever and update the index
        
        Args:
            documents (List[Dict]): List of document dictionaries
        """
        if not documents:
            return
            
        # Extract text from documents
        texts = [doc.get("text", "") for doc in documents]
        
        # Compute embeddings
        embeddings = self.model.encode(texts, show_progress_bar=True, 
                                       convert_to_numpy=True)
        embeddings = embeddings.astype(np.float32)
        
        # Update document store
        current_doc_count = len(self.documents)
        super().add_documents(documents)
        
        # If we already have an index, update it
        if self.index is not None:
            self.index.add(embeddings)
        # Otherwise, build a new index
        else:
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            self.index.add(embeddings)
            
        logger.info(f"Dense index updated with {len(documents)} new documents. Total: {len(self.documents)}")
        
    def retrieve(self, query: str, top_k: int = 5, **kwargs) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents using dense embeddings
        
        Args:
            query (str): Query string
            top_k (int): Number of documents to retrieve
            
        Returns:
            List[Dict]: List of retrieved documents with relevance scores
        """
        if self.index is None or self.index.ntotal == 0:
            logger.warning("No documents indexed for retrieval")
            return []
            
        # Encode query
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        query_embedding = query_embedding.astype(np.float32)
        
        # Search index
        distances, indices = self.index.search(query_embedding, min(top_k, len(self.documents)))
        
        # Gather results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents) and idx >= 0:
                doc = self.documents[idx].copy()
                # Convert distance to similarity score (inverting and normalizing)
                doc["score"] = float(1.0 / (1.0 + distances[0][i]))
                results.append(doc)
                
        return results
    
    def save(self, path: str) -> bool:
        """
        Save the dense retriever to disk
        
        Args:
            path (str): Path to save the retriever
            
        Returns:
            bool: Success status
        """
        try:
            save_path = Path(path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            index_path = str(save_path.with_suffix(save_path.suffix + ".index"))
            if self.index is not None:
                faiss.write_index(self.index, index_path)
            data = {
                "documents": self.documents,
                "model_name": self.model_name,
                "embedding_dim": self.embedding_dim,
                "index_path": index_path,
            }
            with save_path.open('wb') as f:
                pickle.dump(data, f)
            logger.info(f"Saved dense retriever to {save_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save dense retriever to {path}: {e}")
            return False
    
    @classmethod
    def load(cls, path: str) -> "DenseRetriever":
        """
        Load a dense retriever from disk
        
        Args:
            path (str): Path to the saved retriever
            
        Returns:
            DenseRetriever: Loaded retriever instance
        """
        try:
            load_path = Path(path)
            with load_path.open('rb') as f:
                data = pickle.load(f)
            retriever = cls(model_name=data["model_name"])
            retriever.documents = data["documents"]
            index_path = data["index_path"]
            if Path(index_path).exists():
                retriever.index = faiss.read_index(index_path)
            logger.info(f"Loaded dense retriever from {load_path} with {len(retriever.documents)} documents")
            return retriever
        except Exception as e:
            logger.error(f"Failed to load dense retriever from {path}: {e}")
            raise


class HybridRetriever(BaseRetriever):
    """Hybrid retriever combining sparse and dense approaches"""
    
    def __init__(self, 
                dense_model: str = "paraphrase-multilingual-mpnet-base-v2",
                use_gpu: bool = torch.cuda.is_available(),
                alpha: float = 0.5):
        """
        Initialize the hybrid retriever
        
        Args:
            dense_model (str): Name of the SentenceTransformer model
            use_gpu (bool): Whether to use GPU for embeddings
            alpha (float): Weight of dense scores (0.0 to 1.0)
        """
        super().__init__()
        self.dense_retriever = DenseRetriever(dense_model, use_gpu)
        self.sparse_retriever = TfidfRetriever()
        self.alpha = alpha  # Weight for dense scores (1-alpha for sparse)
        
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Add documents to both retrievers
        
        Args:
            documents (List[Dict]): List of document dictionaries
        """
        # Add to document store
        super().add_documents(documents)
        
        # Add to both retrievers
        self.dense_retriever.add_documents(documents)
        self.sparse_retriever.add_documents(documents)
        
        logger.info(f"Hybrid index updated with {len(documents)} new documents. Total: {len(self.documents)}")
        
    def retrieve(self, 
               query: str, 
               top_k: int = 5, 
               alpha: Optional[float] = None,
               rerank: bool = True,
               **kwargs) -> List[Dict[str, Any]]:
        """
        Retrieve documents using both dense and sparse retrievers
        
        Args:
            query (str): Query string
            top_k (int): Number of documents to retrieve
            alpha (float, optional): Weight for dense scores (0.0 to 1.0)
            rerank (bool): Whether to rerank results
            
        Returns:
            List[Dict]: List of retrieved documents with relevance scores
        """
        if len(self.documents) == 0:
            logger.warning("No documents indexed for retrieval")
            return []
            
        # Use provided alpha or default
        alpha = alpha if alpha is not None else self.alpha
        
        # Get extended results from both retrievers
        # We retrieve more documents than needed to ensure good coverage for reranking
        sparse_factor = 2 if rerank else 1
        dense_results = self.dense_retriever.retrieve(query, top_k=top_k * sparse_factor)
        sparse_results = self.sparse_retriever.retrieve(query, top_k=top_k * sparse_factor)
        
        # Combine results
        result_map = {}
        
        # Add sparse results with weight (1-alpha)
        for doc in sparse_results:
            doc_id = doc.get("id")
            if doc_id:
                result_map[doc_id] = {
                    "doc": doc,
                    "sparse_score": doc.get("score", 0) * (1 - alpha),
                    "dense_score": 0
                }
        
        # Add or update with dense results
        for doc in dense_results:
            doc_id = doc.get("id")
            if doc_id:
                if doc_id in result_map:
                    # Update existing entry
                    result_map[doc_id]["dense_score"] = doc.get("score", 0) * alpha
                else:
                    # Add new entry
                    result_map[doc_id] = {
                        "doc": doc,
                        "sparse_score": 0,
                        "dense_score": doc.get("score", 0) * alpha
                    }
        
        # Compute combined scores
        combined_results = []
        for doc_id, data in result_map.items():
            doc = data["doc"].copy()
            doc["score"] = data["sparse_score"] + data["dense_score"]
            doc["sparse_score"] = data["sparse_score"] / (1 - alpha) if alpha < 1 else 0
            doc["dense_score"] = data["dense_score"] / alpha if alpha > 0 else 0
            combined_results.append(doc)
            
        # Sort by combined score
        combined_results.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        # Rerank if requested and possible
        if rerank and len(combined_results) > top_k:
            # Get top documents for reranking
            rerank_docs = combined_results[:top_k * 2]
            reranked = self._rerank(query, rerank_docs)
            return reranked[:top_k]
            
        # Return top-k results
        return combined_results[:top_k]
    
    def _rerank(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rerank documents using cross-attention
        
        Args:
            query (str): Query string
            documents (List[Dict]): Documents to rerank
            
        Returns:
            List[Dict]: Reranked documents
        """
        try:
            # Extract texts from documents
            texts = [doc.get("text", "") for doc in documents]
            
            # Use cross-encoder for more accurate scoring
            # This uses the same sentence transformer model in cross-attention mode
            # For a production system, consider using a dedicated cross-encoder
            scores = self.dense_retriever.model.cross_encode(
                [[query, text] for text in texts], 
                convert_to_numpy=True
            )
            
            # Update document scores
            for i, doc in enumerate(documents):
                doc["rerank_score"] = float(scores[i])
                doc["score"] = float(scores[i])  # Override with rerank score
                
            # Sort by rerank score
            documents.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
            
            return documents
                
        except Exception as e:
            logger.warning(f"Reranking failed: {e}. Using original scores.")
            return documents
    
    def save(self, path: str) -> bool:
        """
        Save the hybrid retriever to disk
        
        Args:
            path (str): Path to save the retriever
            
        Returns:
            bool: Success status
        """
        try:
            save_path = Path(path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            dense_path = str(save_path.with_suffix(save_path.suffix + ".dense"))
            sparse_path = str(save_path.with_suffix(save_path.suffix + ".sparse"))
            self.dense_retriever.save(dense_path)
            self.sparse_retriever.save(sparse_path)
            data = {
                "documents": self.documents,
                "alpha": self.alpha,
                "dense_path": dense_path,
                "sparse_path": sparse_path
            }
            with save_path.open('wb') as f:
                pickle.dump(data, f)
            logger.info(f"Saved hybrid retriever to {save_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save hybrid retriever to {path}: {e}")
            return False
    
    @classmethod
    def load(cls, path: str) -> "HybridRetriever":
        """
        Load a hybrid retriever from disk
        
        Args:
            path (str): Path to the saved retriever
            
        Returns:
            HybridRetriever: Loaded retriever instance
        """
        try:
            load_path = Path(path)
            with load_path.open('rb') as f:
                data = pickle.load(f)
            dense_path = data["dense_path"]
            sparse_path = data["sparse_path"]
            dense_retriever = DenseRetriever.load(dense_path)
            sparse_retriever = TfidfRetriever.load(sparse_path)
            retriever = cls(alpha=data["alpha"])
            retriever.documents = data["documents"]
            retriever.dense_retriever = dense_retriever
            retriever.sparse_retriever = sparse_retriever
            logger.info(f"Loaded hybrid retriever from {load_path} with {len(retriever.documents)} documents")
            return retriever
        except Exception as e:
            logger.error(f"Failed to load hybrid retriever from {path}: {e}")
            raise


class MultilingualRetriever(BaseRetriever):
    """Retriever optimized for multilingual content"""
    
    def __init__(
        self, 
        hybrid_retriever: Optional[HybridRetriever] = None,
        model_name: str = "paraphrase-multilingual-mpnet-base-v2",
        use_gpu: bool = torch.cuda.is_available(),
    ):
        """
        Initialize the multilingual retriever
        
        Args:
            hybrid_retriever (HybridRetriever, optional): Existing hybrid retriever
            model_name (str): Name of the SentenceTransformer model
            use_gpu (bool): Whether to use GPU for embeddings
        """
        super().__init__()
        
        # Use provided hybrid retriever or create new one
        if hybrid_retriever:
            self.retriever = hybrid_retriever
        else:
            self.retriever = HybridRetriever(model_name, use_gpu)
            
        # Language specific data
        self.language_map = {}  # Maps language codes to document indices
        # Dynamically load tokenizer from registry
        registry = ModelRegistry()
        _, tokenizer_name = registry.get_model_and_tokenizer("rag_retriever")
        self.tokenizer = TokenizerPipeline(model_name=tokenizer_name, task_type="rag_retrieval")
        
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Add documents to the retriever with language tracking
        
        Args:
            documents (List[Dict]): List of document dictionaries
        """
        # Add to base document store
        current_index = len(self.documents)
        super().add_documents(documents)
        
        # Add to hybrid retriever
        self.retriever.add_documents(documents)
        
        # Update language map, and optionally tokenize
        for i, doc in enumerate(documents):
            doc_index = current_index + i
            
            # Get language from metadata if available
            language = None
            if "metadata" in doc and isinstance(doc["metadata"], dict):
                language = doc["metadata"].get("language")
                
            if language:
                if language not in self.language_map:
                    self.language_map[language] = []
                self.language_map[language].append(doc_index)
            # Tokenize if tokenizer is provided
            if self.tokenizer:
                doc["tokens"] = self.tokenizer.encode(doc.get("text", ""))
                
        logger.info(f"Multilingual index updated with {len(documents)} new documents. Total: {len(self.documents)}")
        
    def retrieve(self, 
               query: str, 
               language: Optional[str] = None,
               top_k: int = 5, 
               **kwargs) -> List[Dict[str, Any]]:
        """
        Retrieve documents with language filtering
        
        Args:
            query (str): Query string
            language (str, optional): Language code to filter by
            top_k (int): Number of documents to retrieve
            
        Returns:
            List[Dict]: List of retrieved documents with relevance scores
        """
        # If language specified and we have documents in that language
        if language and language in self.language_map and self.language_map[language]:
            # Get more results than needed to allow for filtering
            results = self.retriever.retrieve(query, top_k=top_k * 3, **kwargs)
            
            # Get indices of documents in the specified language
            language_indices = set(self.language_map[language])
            
            # Filter results by language
            language_results = []
            for i, doc in enumerate(self.documents):
                if i in language_indices and doc.get("id") in [r.get("id") for r in results]:
                    # Find the corresponding result
                    for r in results:
                        if r.get("id") == doc.get("id"):
                            language_results.append(r)
                            break
                            
            # If we have enough language-specific results, return them
            if len(language_results) >= top_k:
                return language_results[:top_k]
                
            # Otherwise, append general results to fill up to top_k
            general_results = [r for r in results if r.get("id") not in [d.get("id") for d in language_results]]
            return language_results + general_results[:top_k - len(language_results)]
            
        # If no language specified or no documents in that language, use standard retrieval
        return self.retriever.retrieve(query, top_k=top_k, **kwargs)
    
    def retrieve_multi_language(self, 
                              query: str, 
                              languages: List[str],
                              top_k: int = 5, 
                              **kwargs) -> Dict[str, List[Dict[str, Any]]]:
        """
        Retrieve documents across multiple languages
        
        Args:
            query (str): Query string
            languages (List[str]): List of language codes
            top_k (int): Number of documents per language
            
        Returns:
            Dict[str, List[Dict]]: Dictionary mapping language codes to results
        """
        results = {}
        
        # Get results for each language
        for language in languages:
            lang_results = self.retrieve(query, language=language, top_k=top_k, **kwargs)
            results[language] = lang_results
            
        return results
    
    def save(self, path: str) -> bool:
        """
        Save the multilingual retriever to disk
        
        Args:
            path (str): Path to save the retriever
            
        Returns:
            bool: Success status
        """
        try:
            save_path = Path(path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            retriever_path = str(save_path.with_suffix(save_path.suffix + ".hybrid"))
            self.retriever.save(retriever_path)
            data = {
                "documents": self.documents,
                "language_map": self.language_map,
                "retriever_path": retriever_path
            }
            with save_path.open('wb') as f:
                pickle.dump(data, f)
            logger.info(f"Saved multilingual retriever to {save_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save multilingual retriever to {path}: {e}")
            return False
    
    @classmethod
    def load(cls, path: str) -> "MultilingualRetriever":
        """
        Load a multilingual retriever from disk
        
        Args:
            path (str): Path to the saved retriever
            
        Returns:
            MultilingualRetriever: Loaded retriever instance
        """
        try:
            load_path = Path(path)
            with load_path.open('rb') as f:
                data = pickle.load(f)
            retriever_path = data["retriever_path"]
            hybrid_retriever = HybridRetriever.load(retriever_path)
            retriever = cls(hybrid_retriever=hybrid_retriever)
            retriever.documents = data["documents"]
            retriever.language_map = data["language_map"]
            logger.info(f"Loaded multilingual retriever from {load_path} with {len(retriever.documents)} documents")
            return retriever
        except Exception as e:
            logger.error(f"Failed to load multilingual retriever from {path}: {e}")
            raise


class RetrieverFactory:
    """Factory class to create different types of retrievers"""
    
    @staticmethod
    def create_retriever(retriever_type: str, **kwargs) -> BaseRetriever:
        """
        Create a retriever instance
        
        Args:
            retriever_type (str): Type of retriever to create
            **kwargs: Arguments for the retriever
            
        Returns:
            BaseRetriever: Retriever instance
        """
        if retriever_type.lower() == "tfidf":
            return TfidfRetriever()
        elif retriever_type.lower() == "dense":
            model_name = kwargs.get("model_name", "paraphrase-multilingual-mpnet-base-v2")
            use_gpu = kwargs.get("use_gpu", torch.cuda.is_available())
            return DenseRetriever(model_name, use_gpu)
        elif retriever_type.lower() == "hybrid":
            model_name = kwargs.get("model_name", "paraphrase-multilingual-mpnet-base-v2")
            use_gpu = kwargs.get("use_gpu", torch.cuda.is_available())
            alpha = kwargs.get("alpha", 0.5)
            return HybridRetriever(model_name, use_gpu, alpha)
        elif retriever_type.lower() == "multilingual":
            model_name = kwargs.get("model_name", "paraphrase-multilingual-mpnet-base-v2")
            use_gpu = kwargs.get("use_gpu", torch.cuda.is_available())
            hybrid = kwargs.get("hybrid_retriever")
            if hybrid:
                return MultilingualRetriever(hybrid)
            return MultilingualRetriever(None, model_name, use_gpu)
        else:
            raise ValueError(f"Unknown retriever type: {retriever_type}")
            

# Example usage
if __name__ == "__main__":
    # Create a multilingual retriever
    retriever = RetrieverFactory.create_retriever("multilingual")

    # Example documents with different languages
    documents = [
        {
            "id": "es_1",
            "text": "Hola, ¿cómo estás? Esta es una frase en español.",
            "metadata": {"language": "es", "type": "greeting"}
        },
        {
            "id": "en_1",
            "text": "Hello, how are you? This is a phrase in English.",
            "metadata": {"language": "en", "type": "greeting"}
        },
        {
            "id": "fr_1",
            "text": "Bonjour, comment ça va? C'est une phrase en français.",
            "metadata": {"language": "fr", "type": "greeting"}
        },
        {
            "id": "es_2",
            "text": "El español es una lengua romance que se habla en España y América Latina.",
            "metadata": {"language": "es", "type": "language_info"}
        }
    ]

    # Index documents
    retriever.add_documents(documents)

    # Retrieve across all languages
    query = "greeting in Spanish"
    results = retriever.retrieve(query, top_k=2)

    logger.info(f"Query: {query}")
    logger.info("All results:")
    for result in results:
        logger.info(f"- {result['text']} (Score: {result['score']:.4f}, Language: {result['metadata']['language']})")

    # Retrieve specific language
    spanish_results = retriever.retrieve(query, language="es", top_k=2)

    logger.info("Spanish results:")
    for result in spanish_results:
        logger.info(f"- {result['text']} (Score: {result['score']:.4f})")

    # Multi-language retrieval
    multi_results = retriever.retrieve_multi_language(query, languages=["es", "en", "fr"], top_k=1)

    logger.info("Multi-language results:")
    for lang, results in multi_results.items():
        logger.info(f"{lang.upper()} results:")
        for result in results:
            logger.info(f"- {result['text']} (Score: {result['score']:.4f})")

    # Save retriever
    # retriever.save("./retriever_data/multilingual_retriever.pkl")

    # Load retriever
    # loaded_retriever = MultilingualRetriever.load("./retriever_data/multilingual_retriever.pkl")