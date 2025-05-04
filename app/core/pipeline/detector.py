"""
Input Type Detection for CasaLingua

This module provides functionality to automatically detect input types
and determine the appropriate processing pipeline.

Author: Exygy Development Team
Version: 1.0.0
License: MIT
"""

import os
import re
import io
import mimetypes
import logging
from typing import Dict, Any, Optional, List, Union, Tuple

from app.utils.logging import get_logger

logger = get_logger("casalingua.core.detector")

class InputDetector:
    """
    Detects input type and determines appropriate processing pipeline.
    
    This detector can handle various input types:
    - Text (normal text, legal text, chat)
    - Documents (PDF, DOCX, etc.)
    - Images (for OCR)
    - Audio (for speech-to-text)
    """
    
    def __init__(
        self,
        model_manager,
        config: Dict[str, Any] = None,
        registry_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the input detector.
        
        Args:
            model_manager: Model manager for accessing detection models
            config: Configuration dictionary
            registry_config: Registry configuration for models
        """
        self.model_manager = model_manager
        self.config = config or {}
        self.registry_config = registry_config or {}
        self.model_type = "ner_detection"
        
        # Initialize mime type mapping
        self.mime_types = {
            "text": ["text/plain", "text/html", "text/markdown", "text/csv"],
            "document": [
                "application/pdf", 
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                "application/msword", 
                "application/rtf",
                "application/vnd.oasis.opendocument.text"
            ],
            "image": [
                "image/jpeg", "image/png", "image/gif", "image/tiff", 
                "image/webp", "image/bmp", "image/svg+xml"
            ],
            "audio": [
                "audio/mpeg", "audio/mp3", "audio/wav", "audio/ogg", 
                "audio/flac", "audio/aac", "audio/webm"
            ]
        }
        
        # Initialize language detector patterns
        self.language_patterns = {
            "en": re.compile(r'\b(the|and|is|to|a|in|that|of|for|it|with|as|was|on)\b', re.IGNORECASE),
            "es": re.compile(r'\b(el|la|los|las|y|es|de|en|que|por|con|para|un|una)\b', re.IGNORECASE),
            "fr": re.compile(r'\b(le|la|les|des|et|est|de|en|que|pour|avec|dans|un|une)\b', re.IGNORECASE),
            "de": re.compile(r'\b(der|die|das|und|ist|zu|von|in|den|für|mit|dem|ein|eine)\b', re.IGNORECASE),
        }
        
        logger.info("Input detector initialized")
    
    async def detect(self, 
                    content: Union[str, bytes], 
                    metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Detect the input type and determine processing pipeline.
        
        Args:
            content: Input content (text or binary data)
            metadata: Additional metadata about the input
            
        Returns:
            Dict containing:
            - input_type: Detected type (text, document, image, audio)
            - confidence: Detection confidence (0.0-1.0)
            - pipeline: Suggested processing pipeline
            - metadata: Enhanced metadata
        """
        metadata = metadata or {}
        logger.debug(f"Detecting input type for content of length: {len(content)}")
        
        # Check if content type is provided in metadata
        if "content_type" in metadata:
            return self._detect_from_mime_type(metadata["content_type"], content, metadata)
        
        # Check if filename is provided
        if "filename" in metadata:
            mime_type, _ = mimetypes.guess_type(metadata["filename"])
            if mime_type:
                return self._detect_from_mime_type(mime_type, content, metadata)
        
        # Detect based on content
        if isinstance(content, str):
            logger.debug("Content is string, detecting text type")
            return await self._detect_text_type(content, metadata)
        else:
            logger.debug("Content is binary, detecting binary type")
            return self._detect_binary_type(content, metadata)
    
    def _detect_from_mime_type(self, 
                              mime_type: str, 
                              content: Union[str, bytes], 
                              metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect input type based on MIME type.
        
        Args:
            mime_type: MIME type of the content
            content: Input content
            metadata: Additional metadata
            
        Returns:
            Detection result
        """
        logger.debug(f"Detecting input type from MIME type: {mime_type}")
        
        # Find the high-level type
        input_type = None
        for type_name, mime_list in self.mime_types.items():
            if mime_type in mime_list or any(mime_type.startswith(m.split('/')[0]) for m in mime_list):
                input_type = type_name
                break
        
        # If no match, use a default
        if not input_type:
            if isinstance(content, str):
                input_type = "text"
            else:
                # Analyze first few bytes for magic numbers
                input_type = self._detect_from_magic_bytes(content[:32])
        
        # Determine appropriate pipeline
        pipeline = self._get_pipeline_for_type(input_type, mime_type, metadata)
        
        # Enhance metadata
        metadata["detected_mime_type"] = mime_type
        
        return {
            "input_type": input_type,
            "confidence": 0.9,  # High confidence with explicit MIME type
            "pipeline": pipeline,
            "metadata": metadata
        }
    
    async def _detect_text_type(self, 
                         text: str, 
                         metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect the type of text content.
        
        Args:
            text: Text content
            metadata: Additional metadata
            
        Returns:
            Detection result
        """
        logger.debug("Detecting text type")
        
        # Default values
        input_type = "text"
        confidence = 0.8
        pipeline = "text"
        
        # Check for chat-like content
        if self._is_chat_like(text):
            pipeline = "chat"
            metadata["is_chat"] = True
            logger.debug("Detected chat-like content")
        
        # Check for legal content
        elif self._is_legal_text(text):
            metadata["is_legal"] = True
            metadata["domain"] = "legal"
            logger.debug("Detected legal content")
        
        # Detect language if not provided
        if "source_language" not in metadata:
            detected_lang = await self._detect_language(text)
            if detected_lang:
                metadata["detected_language"] = detected_lang
                metadata["source_language"] = detected_lang
                logger.debug(f"Detected language: {detected_lang}")

        # Use NER model to enhance detection if available
        try:
            # Prepare input for NER model
            input_data = {
                "text": text[:500],  # Use limited text for performance
                "source_language": metadata.get("source_language", "en")
            }
            
            # Run NER model through model manager
            result = await self.model_manager.run_model(
                self.model_type,
                "process",
                input_data
            )
            
            # Extract entities and enhance metadata
            if isinstance(result, dict) and "result" in result:
                entities = result["result"]
                if entities:
                    # Count entity types
                    entity_types = {}
                    for entity in entities:
                        entity_type = entity.get("type", "")
                        entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
                    
                    metadata["detected_entities"] = entity_types
                    logger.debug(f"Detected entities: {entity_types}")
            
        except Exception as e:
            logger.debug(f"NER detection failed: {str(e)}")

        return {
            "input_type": input_type,
            "confidence": confidence,
            "pipeline": pipeline,
            "metadata": metadata
        }
    
    def _detect_binary_type(self, 
                           content: bytes, 
                           metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect the type of binary content.
        
        Args:
            content: Binary content
            metadata: Additional metadata
            
        Returns:
            Detection result
        """
        logger.debug("Detecting binary type")
        
        # Check for known file headers (magic bytes)
        detected_type = self._detect_from_magic_bytes(content[:32])
        confidence = 0.7
        
        # Set appropriate pipeline based on type
        pipeline = detected_type
        
        if detected_type == "image":
            logger.debug("Detected image content, using OCR pipeline")
            pipeline = "ocr"
        
        # Enhance metadata
        if detected_type == "document" and len(content) > 500_000:
            metadata["large_document"] = True
            logger.debug("Detected large document")
        
        return {
            "input_type": detected_type,
            "confidence": confidence,
            "pipeline": pipeline,
            "metadata": metadata
        }
    
    def _detect_from_magic_bytes(self, header_bytes: bytes) -> str:
        """
        Detect file type from magic bytes at the start of the file.
        
        Args:
            header_bytes: First few bytes of the file
            
        Returns:
            Detected input type
        """
        # Check for PDF
        if header_bytes.startswith(b'%PDF'):
            return "document"
        
        # Check for common image formats
        if (header_bytes.startswith(b'\xff\xd8\xff') or  # JPEG
            header_bytes.startswith(b'\x89PNG\r\n\x1a\n') or  # PNG
            header_bytes.startswith(b'GIF87a') or  # GIF
            header_bytes.startswith(b'GIF89a')):  # GIF
            return "image"
        
        # Check for common audio formats
        if (header_bytes.startswith(b'ID3') or  # MP3 with ID3
            header_bytes.startswith(b'RIFF') or  # WAV
            header_bytes.startswith(b'OggS')):  # OGG
            return "audio"
        
        # Check for Office documents
        if (header_bytes.startswith(b'PK\x03\x04') or  # ZIP (Office Open XML)
            header_bytes.startswith(b'\xd0\xcf\x11\xe0')):  # OLE (Old Office format)
            return "document"
        
        # Default to document for unknown binary
        return "document"
    
    def _get_pipeline_for_type(self, 
                              input_type: str, 
                              mime_type: str, 
                              metadata: Dict[str, Any]) -> str:
        """
        Determine the appropriate processing pipeline.
        
        Args:
            input_type: Detected input type
            mime_type: MIME type if available
            metadata: Additional metadata
            
        Returns:
            Pipeline name
        """
        # Handle special cases
        if metadata.get("is_chat", False):
            return "chat"
        
        # Standard mapping
        pipeline_map = {
            "text": "text",
            "document": "document",
            "image": "ocr",
            "audio": "audio"
        }
        
        return pipeline_map.get(input_type, "text")
    
    def _is_chat_like(self, text: str) -> bool:
        """
        Determine if text is chat-like.
        
        Args:
            text: Text to analyze
            
        Returns:
            True if chat-like, False otherwise
        """
        # Check if it's a short message
        if len(text.split()) < 15:
            return True
        
        # Check if it ends with a question mark
        if text.rstrip().endswith('?'):
            return True
        
        # Check for conversational patterns
        chat_patterns = [
            r'^\s*hi\b',
            r'^\s*hello\b',
            r'^\s*hey\b',
            r'^\s*thanks\b',
            r'\bcan you\b',
            r'\bcould you\b',
            r'\bwould you\b',
            r'\?$'
        ]
        
        for pattern in chat_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        return False
    
    def _is_legal_text(self, text: str) -> bool:
        """
        Determine if text is legal content.
        
        Args:
            text: Text to analyze
            
        Returns:
            True if legal content, False otherwise
        """
        # Check for common legal terms
        legal_terms = [
            r'\bhereby\b', r'\bwhereas\b', r'\bshall\b', r'\bpursuant to\b',
            r'\bin accordance with\b', r'\bterms and conditions\b',
            r'\bliability\b', r'\bindemnity\b', r'\bwarranty\b',
            r'\bagreement\b', r'\bcontract\b', r'\bparty\b', r'\bclause\b',
            r'\bsection\b', r'\barticle\b', r'\bexecution\b', r'\bsignature\b',
            r'\bin witness whereof\b', r'\bhereunto\b', r'\bcontractual\b',
            r'\bobligation\b', r'\bgoverning law\b', r'\bjurisdiction\b',
            r'\bdefault\b', r'\bdispute\b', r'\bremedy\b', r'\btermination\b'
        ]
        
        legal_term_count = 0
        for term in legal_terms:
            if re.search(term, text, re.IGNORECASE):
                legal_term_count += 1
        
        # If more than 3 legal terms, consider it legal text
        return legal_term_count >= 3
    
    async def _detect_language(self, text: str) -> Optional[str]:
        """
        Detect the language of text content using model manager.
        
        Args:
            text: Text to analyze
            
        Returns:
            Language code or None if unable to detect
        """
        if not text or len(text) < 10:
            return None
        
        try:
            # Prepare input for language detection
            input_data = {
                "text": text[:500],  # Use limited text for performance
                "parameters": {"detailed": False}
            }
            
            # Try to use language detection model through model manager
            result = await self.model_manager.run_model(
                "language_detection",
                "process",
                input_data
            )
            
            # Extract language from result
            if isinstance(result, dict) and "result" in result:
                detection_result = result["result"]
                
                if isinstance(detection_result, dict):
                    return detection_result.get("language")
                elif isinstance(detection_result, list) and detection_result:
                    return detection_result[0].get("language")
            
            logger.debug("Language detection model gave unexpected result, using pattern fallback")
            
        except Exception as e:
            logger.debug(f"Language detection model failed: {str(e)}, using pattern fallback")
        
        # Fall back to pattern-based detection
        best_match = None
        highest_count = 0
        
        for lang, pattern in self.language_patterns.items():
            matches = len(pattern.findall(text))
            if matches > highest_count:
                highest_count = matches
                best_match = lang
        
        # Require a minimum number of matches
        if highest_count < 2:
            return None
        
        return best_match