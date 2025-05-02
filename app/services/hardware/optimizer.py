"""
Optimizer Module for CasaLingua
Provides performance optimization for RAG components based on hardware capabilities

Optimized and verified for Apple Silicon M4 with 48GB RAM (2025)
"""

import os
import json
import logging
import time
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
import numpy as np
import torch
from tqdm import tqdm
import faiss
import threading
from concurrent.futures import ThreadPoolExecutor

# Local import
from app.services.hardware.detector import HardwareDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RAGOptimizer:
    """Optimizer for RAG system components"""
    
    def __init__(self, 
                config: Optional[Dict[str, Any]] = None,
                auto_detect: bool = True):
        """
        Initialize the RAG optimizer
        
        Args:
            config (Dict[str, Any], optional): Configuration parameters
            auto_detect (bool): Whether to auto-detect hardware
        """
        # Default configuration
        self.config = {
            "device": "cpu",
            "precision": "float32",
            "batch_size": 16,
            "num_workers": 4,
            "use_threads": True,
            "num_threads": 4,
            "cache_dir": "./.cache",
            "vector_db": "faiss",
            "rag": {
                "embedding_model": "medium",
                "chunk_size": 512,
                "chunk_overlap": 50,
                "retriever": "hybrid",
                "hybrid_alpha": 0.7
            }
        }
        
        # Update with provided config
        if config:
            self._update_config(config)
        
        # Auto-detect hardware and update config
        if auto_detect:
            self._auto_configure()
        
        # Initialize metrics tracking
        self.metrics = {
            "embedding_time": [],
            "retrieval_time": [],
            "rerank_time": [],
            "total_time": []
        }
    
    def _update_config(self, config: Dict[str, Any]) -> None:
        """
        Update configuration with new values
        
        Args:
            config (Dict[str, Any]): New configuration parameters
        """
        # Helper function for recursive update
        def recursive_update(target, source):
            for key, value in source.items():
                if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                    recursive_update(target[key], value)
                else:
                    target[key] = value
        
        recursive_update(self.config, config)
    
    def _auto_configure(self) -> None:
        """
        Auto-configure based on hardware detection
        """
        try:
            detector = HardwareDetector()
            recommendations = detector.recommend_config()
            if not recommendations.get("memory", {}).get("total_gb"):
                logger.warning("Hardware detection returned no memory info. Using default 32GB.")
                recommendations["memory"] = {"total_gb": 32}

            # Heuristic override for known Apple M4 Max class machines
            system = recommendations.get("system", "").lower()
            ram = recommendations.get("memory", {}).get("total_gb", 0)
            gpu = recommendations.get("gpu", "").lower()

            if "apple" in system and "mps" in gpu and ram >= 32:
                logger.info("Apple M4 Max with MPS GPU detected. Using medium model profile.")
                recommendations["model_profile"] = "medium"

            self._update_config(recommendations)
            logger.info("Auto-configured RAG optimizer based on hardware detection")
        except Exception as e:
            logger.warning(f"Hardware auto-detection failed: {e}. Using default configuration.")
    
    def optimize_embedding_model(self, embedding_model: Any) -> Any:
        """
        Optimize embedding model based on hardware capabilities
        
        Args:
            embedding_model: The embedding model to optimize
            
        Returns:
            Any: Optimized embedding model
        """
        # Skip if no model provided
        if embedding_model is None:
            return None
            
        try:
            # Move to appropriate device
            device = self.config.get("device", "cpu")
            
            if hasattr(embedding_model, "to") and callable(embedding_model.to):
                embedding_model = embedding_model.to(device)
                logger.info(f"Moved embedding model to {device}")
                
            # Set precision
            precision = self.config.get("precision", "float32")
            
            if precision == "float16":
                # Clarify float16 enablement for MPS and CUDA
                if device in ("cuda", "mps") and torch.backends.mps.is_available():
                    if hasattr(embedding_model, "half") and callable(embedding_model.half):
                        embedding_model = embedding_model.half()
                        logger.info("Set embedding model precision to float16 for MPS")
            elif precision == "int8":
                try:
                    # Try quantization if possible
                    import torch.quantization
                    
                    if hasattr(embedding_model, "qconfig") and hasattr(torch.quantization, "quantize_dynamic"):
                        embedding_model = torch.quantization.quantize_dynamic(
                            embedding_model, {torch.nn.Linear}, dtype=torch.qint8
                        )
                        logger.info("Quantized embedding model to int8")
                except (ImportError, AttributeError):
                    logger.warning("Int8 quantization not supported, using default precision")
            
            # Enable eval mode for inference
            if hasattr(embedding_model, "eval") and callable(embedding_model.eval):
                embedding_model.eval()
                logger.info("Set embedding model to eval mode")
                
            # Apply gradient checkpointing if large model
            if hasattr(embedding_model, "config") and hasattr(embedding_model.config, "hidden_size"):
                hidden_size = getattr(embedding_model.config, "hidden_size")
                if hidden_size > 768 and hasattr(embedding_model, "gradient_checkpointing_enable"):
                    embedding_model.gradient_checkpointing_enable()
                    logger.info("Enabled gradient checkpointing for large model")
            
            return embedding_model
                
        except Exception as e:
            logger.error(f"Error optimizing embedding model: {e}")
            return embedding_model
    
    def optimize_batch_size(self, 
                          embedding_model: Any, 
                          sample_texts: List[str],
                          min_batch_size: int = 1,
                          max_batch_size: int = 64,
                          target_latency_ms: int = 100) -> int:
        """
        Find optimal batch size for the embedding model
        
        Args:
            embedding_model: The embedding model
            sample_texts (List[str]): Sample texts for benchmarking
            min_batch_size (int): Minimum batch size to try
            max_batch_size (int): Maximum batch size to try
            target_latency_ms (int): Target latency per item in milliseconds
            
        Returns:
            int: Optimal batch size
        """
        # Skip if no model or samples provided
        if embedding_model is None or not sample_texts:
            return self.config.get("batch_size", 16)
            
        # Ensure we have enough sample texts
        if len(sample_texts) < max_batch_size:
            sample_texts = sample_texts * (max_batch_size // len(sample_texts) + 1)
            sample_texts = sample_texts[:max_batch_size]
            
        try:
            # Move to appropriate device
            device = self.config.get("device", "cpu")
            
            # Try different batch sizes
            batch_sizes = [2**i for i in range(max(0, min_batch_size.bit_length() - 1), 
                                              max_batch_size.bit_length())]
            if batch_sizes[0] < min_batch_size:
                batch_sizes[0] = min_batch_size
            if batch_sizes[-1] > max_batch_size:
                batch_sizes[-1] = max_batch_size
                
            results = []
            
            logger.info(f"Finding optimal batch size between {min_batch_size} and {max_batch_size}...")
            
            # Test each batch size
            for batch_size in batch_sizes:
                # Warm-up
                with torch.no_grad():
                    _ = embedding_model.encode(sample_texts[:batch_size])
                
                # Benchmark
                start_time = time.time()
                with torch.no_grad():
                    _ = embedding_model.encode(sample_texts[:batch_size])
                end_time = time.time()
                
                # Calculate metrics
                total_time = end_time - start_time
                per_item_time = (total_time / batch_size) * 1000  # ms
                
                results.append({
                    "batch_size": batch_size,
                    "total_time": total_time,
                    "per_item_time": per_item_time
                })
                
                logger.info(f"Batch size {batch_size}: {per_item_time:.2f} ms per item")
                
                # Stop if we exceed target latency by too much
                if per_item_time > target_latency_ms * 3:
                    break
            
            # Find batch size closest to target latency
            selected_batch_size = min_batch_size
            min_diff = float('inf')
            
            for result in results:
                diff = abs(result["per_item_time"] - target_latency_ms)
                if diff < min_diff:
                    min_diff = diff
                    selected_batch_size = result["batch_size"]
            
            logger.info(f"Selected optimal batch size: {selected_batch_size}")
            
            # Update config
            self.config["batch_size"] = selected_batch_size
            
            return selected_batch_size
                
        except Exception as e:
            logger.error(f"Error optimizing batch size: {e}")
            return self.config.get("batch_size", 16)
    
    def optimize_faiss_index(self, 
                           index: Any, 
                           embeddings: Optional[np.ndarray] = None) -> Any:
        """
        Optimize FAISS index for better performance
        
        Args:
            index: FAISS index object
            embeddings (np.ndarray, optional): Embeddings for training
            
        Returns:
            Any: Optimized FAISS index
        """
        # Skip if no index provided
        if index is None:
            return None
            
        try:
            # Check if we're using GPU
            use_gpu = self.config.get("device") == "cuda"
            
            # Get vector dimension
            if hasattr(index, "d"):
                dim = index.d
            else:
                dim = index.ntotal
            
            # Optimize based on vector count
            vector_count = index.ntotal if hasattr(index, "ntotal") else 0
            
            # For small indexes, just return the original
            if vector_count < 1000:
                return index
                
            # Create optimized index based on size
            if vector_count >= 1000000:  # Large index (1M+ vectors)
                # Use hierarchical index for large datasets
                new_index = faiss.IndexHNSWFlat(dim, 32)  # 32 neighbors
                
                # Copy vectors if possible
                if hasattr(index, "reconstruct_batch") and hasattr(new_index, "add"):
                    vectors = index.reconstruct_batch(range(vector_count))
                    new_index.add(vectors)
                
            elif vector_count >= 100000:  # Medium index (100K+ vectors)
                # Use IVF index for medium datasets
                nlist = min(4096, vector_count // 100)  # Rule of thumb
                quantizer = faiss.IndexFlatL2(dim)
                new_index = faiss.IndexIVFFlat(quantizer, dim, nlist)
                
                # Train if we have embeddings
                if embeddings is not None and len(embeddings) > 0:
                    new_index.train(embeddings)
                elif hasattr(index, "reconstruct_batch"):
                    # Sample vectors for training
                    sample_size = min(50000, vector_count)
                    sample_indices = np.random.choice(vector_count, sample_size, replace=False)
                    vectors = index.reconstruct_batch(sample_indices)
                    new_index.train(vectors)
                
                # Copy vectors if possible
                if hasattr(index, "reconstruct_batch") and hasattr(new_index, "add"):
                    vectors = index.reconstruct_batch(range(vector_count))
                    new_index.add(vectors)
                
            else:  # Small index (1K-100K vectors)
                # Keep flat index for small datasets
                new_index = index
            
            # Use GPU if available and configured
            if use_gpu and hasattr(faiss, "StandardGpuResources"):
                try:
                    res = faiss.StandardGpuResources()
                    gpu_index = faiss.index_cpu_to_gpu(res, 0, new_index)
                    logger.info(f"Moved FAISS index to GPU")
                    return gpu_index
                except Exception as e:
                    logger.warning(f"Failed to move index to GPU: {e}")
                    
            return new_index
                
        except Exception as e:
            logger.error(f"Error optimizing FAISS index: {e}")
            return index
    
    def optimize_embeddings(self, 
                          embedding_func: Callable[[List[str]], np.ndarray],
                          texts: List[str]) -> np.ndarray:
        """
        Optimize the embedding process for a list of texts
        
        Args:
            embedding_func (Callable): Function to embed texts
            texts (List[str]): Texts to embed
            
        Returns:
            np.ndarray: Text embeddings
        """
        # Skip if no function or texts provided
        if embedding_func is None or not texts:
            return np.array([])
            
        try:
            # Start timer
            start_time = time.time()
            
            # Get batch size
            batch_size = self.config.get("batch_size", 16)
            
            # Check if function supports batch processing
            supports_batching = True
            
            # Process in batches
            all_embeddings = []
            
            for i in tqdm(range(0, len(texts), batch_size), desc="Computing embeddings"):
                batch_texts = texts[i:i+batch_size]
                
                # Compute embeddings for batch
                batch_embeddings = embedding_func(batch_texts)
                
                # Check if returned as numpy array
                if not isinstance(batch_embeddings, np.ndarray):
                    try:
                        batch_embeddings = np.array(batch_embeddings)
                    except:
                        batch_embeddings = batch_embeddings.cpu().numpy()
                
                all_embeddings.append(batch_embeddings)
            
            # Concatenate all embeddings
            embeddings = np.vstack(all_embeddings)
            
            # Log metrics
            end_time = time.time()
            total_time = end_time - start_time
            per_item_time = (total_time / len(texts)) * 1000  # ms
            
            self.metrics["embedding_time"].append({
                "count": len(texts),
                "total_time": total_time,
                "per_item_time": per_item_time
            })
            
            logger.info(f"Embedded {len(texts)} texts in {total_time:.2f}s ({per_item_time:.2f}ms per item)")
            
            return embeddings
                
        except Exception as e:
            logger.error(f"Error optimizing embeddings: {e}")
            
            # Try processing one by one if batching failed
            try:
                all_embeddings = []
                for text in tqdm(texts, desc="Computing embeddings (fallback)"):
                    embedding = embedding_func([text])
                    all_embeddings.append(embedding[0])
                return np.vstack(all_embeddings)
            except:
                return np.array([])
    
    def optimize_retrieval(self, 
                         retriever: Any,
                         query: str,
                         top_k: int = 5,
                         **kwargs) -> List[Dict[str, Any]]:
        """
        Optimize retrieval process
        
        Args:
            retriever: Retriever object
            query (str): Query string
            top_k (int): Number of results to retrieve
            
        Returns:
            List[Dict[str, Any]]: Retrieved results
        """
        # Skip if no retriever provided
        if retriever is None:
            return []
            
        try:
            # Start timer
            start_time = time.time()
            
            # Ensure retriever has retrieve method
            if not hasattr(retriever, "retrieve") or not callable(retriever.retrieve):
                logger.error("Retriever does not have a retrieve method")
                return []
            
            # Call retrieve method
            results = retriever.retrieve(query, top_k=top_k, **kwargs)
            
            # Log metrics
            end_time = time.time()
            total_time = end_time - start_time
            
            self.metrics["retrieval_time"].append({
                "query": query,
                "total_time": total_time
            })
            
            logger.info(f"Retrieved {len(results)} results in {total_time:.2f}s")
            
            return results
                
        except Exception as e:
            logger.error(f"Error optimizing retrieval: {e}")
            return []
    
    def optimize_chunking(self, 
                        text: str,
                        chunk_size: Optional[int] = None,
                        chunk_overlap: Optional[int] = None) -> List[str]:
        """
        Optimize text chunking strategy
        
        Args:
            text (str): Text to chunk
            chunk_size (int, optional): Size of chunks
            chunk_overlap (int, optional): Overlap between chunks
            
        Returns:
            List[str]: Optimized text chunks
        """
        # Use config values if not provided
        if chunk_size is None:
            chunk_size = self.config.get("rag", {}).get("chunk_size", 512)
        if chunk_overlap is None:
            chunk_overlap = self.config.get("rag", {}).get("chunk_overlap", 50)
            
        # Validate inputs
        if chunk_size <= 0:
            chunk_size = 512
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            chunk_overlap = min(50, chunk_size // 10)
            
        # Skip if no text provided
        if not text:
            return []
            
        try:
            # Split text into paragraphs
            paragraphs = [p for p in text.split("\n") if p.strip()]
            
            # If text is short, return as single chunk
            if len(text) <= chunk_size:
                return [text]
                
            chunks = []
            current_chunk = []
            current_size = 0
            
            for paragraph in paragraphs:
                # If paragraph is too long, split it into sentences
                if len(paragraph) > chunk_size:
                    sentences = self._split_into_sentences(paragraph)
                    
                    # Process each sentence
                    for sentence in sentences:
                        sentence_len = len(sentence)
                        
                        # If adding this sentence would exceed chunk size, 
                        # save current chunk and start a new one
                        if current_size + sentence_len > chunk_size and current_size > 0:
                            chunks.append(" ".join(current_chunk))
                            
                            # Keep overlap from previous chunk
                            overlap_size = 0
                            overlap_chunks = []
                            
                            while overlap_size < chunk_overlap and current_chunk:
                                overlap_text = current_chunk.pop()
                                overlap_size += len(overlap_text)
                                overlap_chunks.insert(0, overlap_text)
                                
                            current_chunk = overlap_chunks
                            current_size = overlap_size
                        
                        # Add sentence to current chunk
                        current_chunk.append(sentence)
                        current_size += sentence_len
                else:
                    # If adding this paragraph would exceed chunk size, 
                    # save current chunk and start a new one
                    if current_size + len(paragraph) > chunk_size and current_size > 0:
                        chunks.append(" ".join(current_chunk))
                        
                        # Keep overlap from previous chunk
                        overlap_size = 0
                        overlap_chunks = []
                        
                        while overlap_size < chunk_overlap and current_chunk:
                            overlap_text = current_chunk.pop()
                            overlap_size += len(overlap_text)
                            overlap_chunks.insert(0, overlap_text)
                            
                        current_chunk = overlap_chunks
                        current_size = overlap_size
                    
                    # Add paragraph to current chunk
                    current_chunk.append(paragraph)
                    current_size += len(paragraph)
            
            # Add final chunk
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            
            logger.info(f"Split text into {len(chunks)} chunks")
            
            return chunks
                
        except Exception as e:
            logger.error(f"Error optimizing chunking: {e}")
            
            # Fallback to simple chunking
            chunks = []
            for i in range(0, len(text), chunk_size - chunk_overlap):
                chunk = text[i:i+chunk_size]
                if chunk:
                    chunks.append(chunk)
            
            return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences
        
        Args:
            text (str): Text to split
            
        Returns:
            List[str]: List of sentences
        """
        # Simple sentence splitting logic
        delimiters = [".", "!", "?", ";", ":", "\n"]
        sentences = []
        current_sentence = ""
        
        for char in text:
            current_sentence += char
            
            if char in delimiters and current_sentence.strip():
                sentences.append(current_sentence.strip())
                current_sentence = ""
        
        # Add any remaining text
        if current_sentence.strip():
            sentences.append(current_sentence.strip())
        
        return sentences
    
    def optimize_parallel_processing(self, 
                                  func: Callable,
                                  items: List[Any],
                                  **kwargs) -> List[Any]:
        """
        Optimize parallel processing of items
        
        Args:
            func (Callable): Function to apply to each item
            items (List[Any]): Items to process
            
        Returns:
            List[Any]: Processed items
        """
        # Skip if no items provided
        if not items:
            return []
            
        # Get threading configuration
        use_threads = self.config.get("use_threads", True)
        num_threads = self.config.get("num_threads", 4)
        
        # If only one item or threading disabled, process sequentially
        if len(items) == 1 or not use_threads or num_threads <= 1:
            return [func(item, **kwargs) for item in items]
            
        try:
            # Process in parallel
            results = []
            
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                # Submit all tasks
                futures = [executor.submit(func, item, **kwargs) for item in items]
                
                # Collect results
                for future in futures:
                    results.append(future.result())
            
            return results
                
        except Exception as e:
            logger.error(f"Error in parallel processing: {e}")
            
            # Fallback to sequential processing
            return [func(item, **kwargs) for item in items]
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics
        
        Returns:
            Dict[str, Any]: Performance metrics
        """
        # Calculate average metrics
        metrics = {
            "config": self.config
        }
        
        for metric_name, values in self.metrics.items():
            if values:
                if metric_name == "embedding_time":
                    # Calculate per-item metrics
                    total_items = sum(m.get("count", 0) for m in values)
                    total_time = sum(m.get("total_time", 0) for m in values)
                    metrics[metric_name] = {
                        "total_items": total_items,
                        "total_time": total_time,
                        "average_per_item_ms": (total_time / total_items * 1000) if total_items > 0 else 0
                    }
                else:
                    # Calculate average times
                    total_time = sum(m.get("total_time", 0) for m in values)
                    metrics[metric_name] = {
                        "count": len(values),
                        "total_time": total_time,
                        "average_time": total_time / len(values) if values else 0
                    }
        
        return metrics
    
    def optimize_memory_usage(self, obj: Any) -> Any:
        """
        Optimize memory usage of an object
        
        Args:
            obj: Object to optimize
            
        Returns:
            Any: Memory-optimized object
        """
        try:
            # Handle NumPy arrays
            if isinstance(obj, np.ndarray):
                # If float64, convert to float32
                if obj.dtype == np.float64:
                    return obj.astype(np.float32)
                    
                # If int64 with small values, convert to int32
                if obj.dtype == np.int64:
                    max_val = np.max(obj)
                    min_val = np.min(obj)
                    if max_val < 2**31 and min_val > -2**31:
                        return obj.astype(np.int32)
                
                return obj
                
            # Handle torch tensors
            elif torch.is_tensor(obj):
                # If float64, convert to float32
                if obj.dtype == torch.float64:
                    return obj.to(torch.float32)
                    
                # If int64 with small values, convert to int32
                if obj.dtype == torch.int64:
                    max_val = torch.max(obj).item()
                    min_val = torch.min(obj).item()
                    if max_val < 2**31 and min_val > -2**31:
                        return obj.to(torch.int32)
                
                return obj
                
            # Handle lists of NumPy arrays or torch tensors
            elif isinstance(obj, list) and len(obj) > 0:
                if all(isinstance(item, np.ndarray) for item in obj):
                    return [self.optimize_memory_usage(item) for item in obj]
                elif all(torch.is_tensor(item) for item in obj):
                    return [self.optimize_memory_usage(item) for item in obj]
            
            # Return original object if no optimization applied
            return obj
                
        except Exception as e:
            logger.error(f"Error optimizing memory usage: {e}")
            return obj
    
    def export_config(self, filepath: str) -> bool:
        """
        Export optimizer configuration to a JSON file
        
        Args:
            filepath (str): Path to save the JSON file
            
        Returns:
            bool: Success status
        """
        try:
            # Export to JSON file
            with open(filepath, 'w') as f:
                json.dump(self.config, f, indent=2)
                
            logger.info(f"Exported optimizer configuration to {filepath}")
            return True
                
        except Exception as e:
            logger.error(f"Error exporting configuration: {e}")
            return False


# Utility functions for specific optimizations

def optimize_sentence_transformers(model: Any) -> Any:
    """
    Apply optimizations specific to SentenceTransformers models
    
    Args:
        model: SentenceTransformer model
        
    Returns:
        Any: Optimized model
    """
    try:
        # Check if we're working with SentenceTransformers
        if not hasattr(model, 'encode') or not callable(model.encode):
            return model
            
        # Create optimizer
        optimizer = RAGOptimizer()
        
        # Apply general optimizations
        model = optimizer.optimize_embedding_model(model)
        
        # Specific SentenceTransformers optimizations
        if hasattr(model, '_first_module') and hasattr(model._first_module, 'auto_model'):
            # Enable better memory usage in transformer
            if hasattr(model._first_module.auto_model.config, 'update'):
                model._first_module.auto_model.config.update({
                    'output_hidden_states': False,
                    'output_attentions': False,
                    'use_cache': True
                })
                
            # Apply pooling optimization if possible
            if hasattr(model, '_modules') and 'pooling' in model._modules:
                pooling = model._modules['pooling']
                if hasattr(pooling, 'pooling_mode_mean_tokens'):
                    # Mean pooling is more efficient
                    pooling.pooling_mode_mean_tokens = True
                    pooling.pooling_mode_cls_token = False
                    pooling.pooling_mode_max_tokens = False
        
        logger.info("Applied SentenceTransformers specific optimizations")
        return model
            
    except Exception as e:
        logger.error(f"Error optimizing SentenceTransformers model: {e}")
        return model


def optimize_retriever_class(retriever_cls: Any, config: Optional[Dict[str, Any]] = None) -> Any:
    """
    Factory function to create an optimized retriever based on configuration
    
    Args:
        retriever_cls: Base retriever class to optimize
        config (Dict[str, Any], optional): Configuration parameters
        
    Returns:
        Any: Optimized retriever class
    """
    # Create optimizer with given config or default
    optimizer = RAGOptimizer(config)
    
    # Get retriever type from config
    retriever_type = optimizer.config.get("rag", {}).get("retriever", "hybrid")
    
    try:
        # Depending on the retriever type, apply different optimizations
        if retriever_type == "dense":
            # For dense retriever, optimize the embedding model
            class OptimizedDenseRetriever(retriever_cls):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    # Optimize embedding model
                    if hasattr(self, 'embedding_model'):
                        self.embedding_model = optimizer.optimize_embedding_model(self.embedding_model)
                    
                    # Optimize FAISS index if present
                    if hasattr(self, 'index'):
                        self.index = optimizer.optimize_faiss_index(self.index)
                
                def retrieve(self, query, top_k=5, **kwargs):
                    # Use optimized retrieval
                    return optimizer.optimize_retrieval(self, query, top_k, **kwargs)
                
                def add_documents(self, documents):
                    # Optimize batch processing of embeddings
                    result = super().add_documents(documents)
                    
                    # Optimize FAISS index after adding documents
                    if hasattr(self, 'index'):
                        self.index = optimizer.optimize_faiss_index(self.index)
                        
                    return result
            
            return OptimizedDenseRetriever
            
        elif retriever_type == "sparse":
            # For sparse retriever, optimize TF-IDF operations
            class OptimizedSparseRetriever(retriever_cls):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                
                def retrieve(self, query, top_k=5, **kwargs):
                    # Use optimized retrieval
                    return optimizer.optimize_retrieval(self, query, top_k, **kwargs)
            
            return OptimizedSparseRetriever
            
        elif retriever_type == "hybrid":
            # For hybrid retriever, optimize both dense and sparse components
            class OptimizedHybridRetriever(retriever_cls):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    
                    # Optimize dense retriever
                    if hasattr(self, 'dense_retriever'):
                        # Optimize embedding model
                        if hasattr(self.dense_retriever, 'embedding_model'):
                            self.dense_retriever.embedding_model = optimizer.optimize_embedding_model(
                                self.dense_retriever.embedding_model
                            )
                        
                        # Optimize FAISS index
                        if hasattr(self.dense_retriever, 'index'):
                            self.dense_retriever.index = optimizer.optimize_faiss_index(
                                self.dense_retriever.index
                            )
                
                def retrieve(self, query, top_k=5, alpha=None, rerank=True, **kwargs):
                    # Use optimized retrieval logic
                    return optimizer.optimize_retrieval(
                        self, query, top_k, alpha=alpha, rerank=rerank, **kwargs
                    )
            
            return OptimizedHybridRetriever
            
        else:
            # Default optimization wrapper
            class OptimizedRetriever(retriever_cls):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                
                def retrieve(self, query, top_k=5, **kwargs):
                    # Use optimized retrieval
                    return optimizer.optimize_retrieval(self, query, top_k, **kwargs)
            
            return OptimizedRetriever
                
    except Exception as e:
        logger.error(f"Error creating optimized retriever: {e}")
        return retriever_cls


# Example usage
if __name__ == "__main__":
    # Create optimizer with auto hardware detection
    optimizer = RAGOptimizer(auto_detect=True)
    
    # Print configuration
    print("Optimized Configuration:")
    print(json.dumps(optimizer.config, indent=2))
    
    # Export configuration to file
    optimizer.export_config("rag_optimizer_config.json")