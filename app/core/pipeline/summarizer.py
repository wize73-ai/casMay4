"""
Summarization Pipeline for CasaLingua

This module provides text summarization capabilities, turning long documents
into concise summaries with controlled length and quality.

Author: Exygy Development Team
Version: 1.0.0
License: MIT
"""

import logging
from typing import Dict, Any, List, Optional, Union, Tuple

from app.utils.logging import get_logger

logger = get_logger("casalingua.summarizer")


class SummarizationPipeline:
    """
    Text summarization pipeline.
    
    Features:
    - Controllable summary length
    - Multiple summarization strategies
    - Context-aware summarization
    """
    
    def __init__(
        self,
        model_manager,
        config: Dict[str, Any] = None,
        registry_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the summarization pipeline.
        
        Args:
            model_manager: Model manager for accessing models
            config: Configuration dictionary
            registry_config: Registry configuration
        """
        self.model_manager = model_manager
        self.config = config or {}
        self.registry_config = registry_config or {}
        self.initialized = False
        self.model_type = "rag_generator"  # Default model type for summarization
        
        logger.info("Summarization pipeline created (not yet initialized)")
    
    async def initialize(self) -> None:
        """
        Initialize the summarization pipeline.
        
        This loads necessary models through the model manager.
        """
        if self.initialized:
            logger.warning("Summarization pipeline already initialized")
            return
        
        logger.info("Initializing summarization pipeline")
        
        try:
            # Load model through model manager
            model_info = await self.model_manager.load_model(self.model_type)
            
            # Verify model was loaded
            if not model_info or "model" not in model_info:
                raise ValueError(f"Failed to load {self.model_type} model")
                
            logger.info(f"Summarization model loaded successfully: {self.model_type}")
            self.initialized = True
            
        except Exception as e:
            logger.error(f"Failed to initialize summarization pipeline: {str(e)}")
            raise
    
    async def summarize(
        self,
        text: str,
        language: str = "en",
        max_length: int = 100,
        min_length: int = 30,
        model_id: Optional[str] = None,
        user_id: Optional[str] = None,
        request_id: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Summarize text content.
        
        Args:
            text: Text to summarize
            language: Language code
            max_length: Maximum summary length
            min_length: Minimum summary length
            model_id: Optional specific model ID to use
            user_id: Optional user ID for tracking
            request_id: Optional request ID for tracking
            **kwargs: Additional parameters
            
        Returns:
            Dict containing:
            - summary: Summarized text
            - model_used: Name of model used
            - metrics: Performance metrics (if available)
        """
        if not self.initialized:
            await self.initialize()
        
        # Use specified model_id if provided, otherwise use default model type
        model_type = model_id or self.model_type
        
        try:
            # Prepare input for the model
            input_data = {
                "text": text,
                "source_language": language,
                "parameters": {
                    "max_length": max_length,
                    "min_length": min_length,
                    "user_id": user_id,
                    "request_id": request_id,
                    **kwargs
                }
            }
            
            # Run the model through model manager
            result = await self.model_manager.run_model(
                model_type,
                "process",  # Call the wrapper's process method
                input_data
            )
            
            # Extract result and metadata
            if isinstance(result, dict):
                summary = result.get("result", "")
                metrics = result.get("metrics", {})
                metadata = result.get("metadata", {})
            else:
                # Handle case where result is not a dict
                summary = str(result)
                metrics = {}
                metadata = {}
            
            logger.info(f"Successfully generated summary (length: {len(summary)})")
            
            return {
                "summary": summary,
                "model_used": model_type,
                "language": language,
                "metrics": metrics,
                "metadata": metadata
            }
            
        except Exception as e:
            logger.error(f"Summarization error: {str(e)}", exc_info=True)
            return {
                "error": str(e),
                "model_used": model_type,
                "language": language
            }