"""
Bloom Housing Integration Schemas for CasaLingua

This module defines schemas for integration with Bloom Housing's API structure
and provides compatibility transformations for seamless integration.

Author: Exygy Development Team
Version: 1.0.0
License: MIT
"""

from typing import Dict, List, Optional, Any, Union
from enum import Enum
from pydantic import BaseModel, Field, validator, root_validator

# Base Bloom Housing schema definitions for compatibility

class BloomLanguage(str, Enum):
    """Supported languages in Bloom Housing ecosystem."""
    ENGLISH = "en"
    SPANISH = "es"
    CHINESE = "zh"
    CHINESE_TRADITIONAL = "zh-TW"
    VIETNAMESE = "vi"
    TAGALOG = "tl"
    ARABIC = "ar"
    KOREAN = "ko"
    RUSSIAN = "ru"
    HMONG = "hmn"
    KHMER = "km"
    HEBREW = "he"
    
    @classmethod
    def to_casa_lingua(cls, code: str) -> str:
        """Convert Bloom Housing language code to CasaLingua code."""
        if code == "zh-TW":
            return "zh-TW"
        return code
    
    @classmethod
    def from_casa_lingua(cls, code: str) -> str:
        """Convert CasaLingua language code to Bloom Housing code."""
        # Bloom Housing defaults to using the "DEFAULT" code for anything not recognized
        if code not in [lang.value for lang in BloomLanguage]:
            return "en"
        return code

class BloomTranslationRequest(BaseModel):
    """
    Standard Bloom Housing translation request format.
    This matches the structure used across Bloom Housing systems.
    """
    text: str = Field(..., description="Text to be translated")
    sourceLanguage: str = Field(..., description="Source language code")
    targetLanguage: str = Field(..., description="Target language code")
    projectId: Optional[str] = Field(None, description="Bloom Housing project identifier")
    formatPreservation: Optional[bool] = Field(True, description="Whether to preserve formatting")
    documentType: Optional[str] = Field(None, description="Type of document being translated")
    domain: Optional[str] = Field("housing", description="Domain context for translation")
    apiKey: Optional[str] = Field(None, description="Optional API key for authentication")
    
    @validator('sourceLanguage', 'targetLanguage')
    def validate_language_code(cls, v):
        """Validate the language code against supported Bloom Housing languages."""
        # Convert to lowercase
        v = v.lower()
        # Handle special cases
        if v == "zh-tw":
            return "zh-TW"
        # Default value for unsupported languages
        if v not in [lang.value for lang in BloomLanguage]:
            return "en"
        return v
    
    def to_casa_lingua_format(self) -> Dict[str, Any]:
        """Convert to CasaLingua format for processing."""
        return {
            "text": self.text,
            "source_language": BloomLanguage.to_casa_lingua(self.sourceLanguage),
            "target_language": BloomLanguage.to_casa_lingua(self.targetLanguage),
            "preserve_formatting": self.formatPreservation,
            "domain": self.domain,
            "glossary_id": self.projectId,  # Map project ID to glossary
            "model_id": None  # Let CasaLingua determine best model
        }

class BloomTranslationResponse(BaseModel):
    """
    Standard Bloom Housing translation response format.
    """
    translatedText: str = Field(..., description="Translated text")
    sourceLanguage: str = Field(..., description="Source language code")
    targetLanguage: str = Field(..., description="Target language code")
    confidenceScore: float = Field(..., description="Confidence score (0-1)")
    projectId: Optional[str] = Field(None, description="Bloom Housing project identifier")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    @classmethod
    def from_casa_lingua_response(cls, casa_response: Dict[str, Any]):
        """Create from CasaLingua response format."""
        # Extract the inner data from CasaLingua BaseResponse wrapper
        if "data" in casa_response:
            data = casa_response["data"]
        else:
            data = casa_response
            
        # Extract required fields
        translated_text = data.get("translated_text", "")
        source_language = data.get("source_language", "en")
        target_language = data.get("target_language", "en")
        confidence = data.get("confidence", 0.0)
        
        # Create metadata dictionary
        metadata = {
            "processTime": data.get("process_time", 0.0),
            "wordCount": data.get("word_count", 0),
            "characterCount": data.get("character_count", 0),
            "modelId": data.get("model_id", "unknown"),
            "cached": casa_response.get("metadata", {}).get("cached", False)
        }
        
        # If there's verification data, add it
        if data.get("verified", False):
            metadata["verified"] = data.get("verified", False)
            metadata["verificationScore"] = data.get("verification_score", None)
        
        return cls(
            translatedText=translated_text,
            sourceLanguage=BloomLanguage.from_casa_lingua(source_language),
            targetLanguage=BloomLanguage.from_casa_lingua(target_language),
            confidenceScore=confidence,
            projectId=None,  # No project ID in CasaLingua response
            metadata=metadata
        )

class BloomLanguageDetectionRequest(BaseModel):
    """
    Standard Bloom Housing language detection request format.
    """
    text: str = Field(..., description="Text to detect language in")
    detailed: Optional[bool] = Field(False, description="Whether to return detailed results")
    
    def to_casa_lingua_format(self) -> Dict[str, Any]:
        """Convert to CasaLingua format for processing."""
        return {
            "text": self.text,
            "detailed": self.detailed
        }

class BloomLanguageDetails(BaseModel):
    """Details for a detected language."""
    languageCode: str = Field(..., description="Language code")
    confidence: float = Field(..., description="Confidence score (0-1)")

class BloomLanguageDetectionResponse(BaseModel):
    """
    Standard Bloom Housing language detection response format.
    """
    detectedLanguage: str = Field(..., description="Detected primary language code")
    confidence: float = Field(..., description="Confidence score (0-1)")
    alternatives: Optional[List[BloomLanguageDetails]] = Field(None, description="Alternative language possibilities")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    @classmethod
    def from_casa_lingua_response(cls, casa_response: Dict[str, Any]):
        """Create from CasaLingua response format."""
        # Extract the inner data from CasaLingua BaseResponse wrapper
        if "data" in casa_response:
            data = casa_response["data"]
        else:
            data = casa_response
            
        # Extract required fields
        detected_language = data.get("detected_language", "en")
        confidence = data.get("confidence", 0.0)
        
        # Create metadata dictionary
        metadata = {
            "processTime": data.get("process_time", 0.0),
            "cached": casa_response.get("metadata", {}).get("cached", False)
        }
        
        # Process alternatives if available
        alternatives = None
        if data.get("alternatives"):
            alternatives = [
                BloomLanguageDetails(
                    languageCode=BloomLanguage.from_casa_lingua(alt.get("language", "en")),
                    confidence=alt.get("confidence", 0.0)
                )
                for alt in data.get("alternatives", [])
            ]
        
        return cls(
            detectedLanguage=BloomLanguage.from_casa_lingua(detected_language),
            confidence=confidence,
            alternatives=alternatives,
            metadata=metadata
        )

class BloomAnalysisType(str, Enum):
    """Types of text analysis supported in Bloom Housing ecosystem."""
    SENTIMENT = "sentiment"
    ENTITIES = "entities"
    TOPICS = "topics"
    SUMMARY = "summary"
    ALL = "all"

class BloomTextAnalysisRequest(BaseModel):
    """
    Standard Bloom Housing text analysis request format.
    """
    text: str = Field(..., description="Text to analyze")
    language: Optional[str] = Field(None, description="Language code (will auto-detect if not provided)")
    analysisTypes: List[str] = Field(default_factory=lambda: ["all"], description="Types of analysis to perform")
    
    def to_casa_lingua_format(self) -> Dict[str, Any]:
        """Convert to CasaLingua format for processing."""
        return {
            "text": self.text,
            "language": BloomLanguage.to_casa_lingua(self.language) if self.language else None,
            "analyses": self.analysisTypes,
            "model_id": None  # Let CasaLingua determine best model
        }

class BloomEntityMention(BaseModel):
    """Entity mention in text analysis."""
    text: str = Field(..., description="Text of entity mention")
    startOffset: int = Field(..., description="Start offset in original text")
    endOffset: int = Field(..., description="End offset in original text")
    probability: float = Field(..., description="Confidence of entity detection")

class BloomEntity(BaseModel):
    """Named entity in text analysis."""
    name: str = Field(..., description="Entity name")
    type: str = Field(..., description="Entity type")
    mentions: List[BloomEntityMention] = Field(default_factory=list, description="Mentions in text")

class BloomTopic(BaseModel):
    """Topic in text analysis."""
    name: str = Field(..., description="Topic name")
    confidence: float = Field(..., description="Confidence score")

class BloomSentiment(BaseModel):
    """Sentiment analysis result."""
    score: float = Field(..., description="Sentiment score (-1 to 1)")
    magnitude: float = Field(..., description="Magnitude of sentiment")
    classification: str = Field(..., description="Sentiment classification (positive/negative/neutral)")

class BloomTextAnalysisResponse(BaseModel):
    """
    Standard Bloom Housing text analysis response format.
    """
    language: str = Field(..., description="Language of analyzed text")
    sentiment: Optional[BloomSentiment] = Field(None, description="Sentiment analysis result")
    entities: Optional[List[BloomEntity]] = Field(None, description="Named entities found in text")
    topics: Optional[List[BloomTopic]] = Field(None, description="Topics detected in text")
    summary: Optional[str] = Field(None, description="Text summary if requested")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    @classmethod
    def from_casa_lingua_response(cls, casa_response: Dict[str, Any]):
        """Create from CasaLingua response format."""
        # Extract the inner data from CasaLingua BaseResponse wrapper
        if "data" in casa_response:
            data = casa_response["data"]
        else:
            data = casa_response
            
        # Extract language
        language = data.get("language", "en")
        
        # Process sentiment if available
        sentiment = None
        if data.get("sentiment"):
            sent_data = data["sentiment"]
            # Convert score to range expected by Bloom Housing
            score = sent_data.get("score", 0.0)
            magnitude = sent_data.get("magnitude", 0.0)
            
            # Determine classification based on score
            if score > 0.25:
                classification = "positive"
            elif score < -0.25:
                classification = "negative"
            else:
                classification = "neutral"
                
            sentiment = BloomSentiment(
                score=score,
                magnitude=magnitude,
                classification=classification
            )
        
        # Process entities if available
        entities = None
        if data.get("entities"):
            entities = []
            for entity_data in data["entities"]:
                mentions = []
                for mention in entity_data.get("mentions", []):
                    mentions.append(BloomEntityMention(
                        text=mention.get("text", ""),
                        startOffset=mention.get("start", 0),
                        endOffset=mention.get("end", 0),
                        probability=mention.get("probability", 0.0)
                    ))
                
                entities.append(BloomEntity(
                    name=entity_data.get("name", ""),
                    type=entity_data.get("type", "OTHER"),
                    mentions=mentions
                ))
        
        # Process topics if available
        topics = None
        if data.get("topics"):
            topics = []
            for topic_data in data["topics"]:
                topics.append(BloomTopic(
                    name=topic_data.get("name", ""),
                    confidence=topic_data.get("confidence", 0.0)
                ))
        
        # Extract summary if available
        summary = data.get("summary")
        
        # Create metadata dictionary
        metadata = {
            "processTime": data.get("process_time", 0.0),
            "wordCount": data.get("word_count", 0),
            "sentenceCount": data.get("sentence_count", 0),
            "cached": casa_response.get("metadata", {}).get("cached", False)
        }
        
        return cls(
            language=BloomLanguage.from_casa_lingua(language),
            sentiment=sentiment,
            entities=entities,
            topics=topics,
            summary=summary,
            metadata=metadata
        )

# Document processing schemas compatible with Bloom Housing

class BloomDocumentType(str, Enum):
    """Document types supported in Bloom Housing ecosystem."""
    PDF = "pdf"
    DOCX = "docx"
    DOC = "doc"
    TEXT = "txt"
    HTML = "html"
    MARKDOWN = "markdown"
    RTF = "rtf"

class BloomDocumentTranslationRequest(BaseModel):
    """
    Standard Bloom Housing document translation request format.
    This schema is used for file-based translation requests.
    """
    sourceLanguage: str = Field(..., description="Source language code")
    targetLanguage: str = Field(..., description="Target language code")
    documentType: Optional[BloomDocumentType] = Field(None, description="Document type")
    preserveLayout: Optional[bool] = Field(True, description="Whether to preserve document layout")
    projectId: Optional[str] = Field(None, description="Bloom Housing project identifier")
    callbackUrl: Optional[str] = Field(None, description="URL to call when processing completes")
    
    @validator('sourceLanguage', 'targetLanguage')
    def validate_language_code(cls, v):
        """Validate the language code against supported Bloom Housing languages."""
        # Convert to lowercase
        v = v.lower()
        # Handle special cases
        if v == "zh-tw":
            return "zh-TW"
        # Default value for unsupported languages
        if v not in [lang.value for lang in BloomLanguage]:
            return "en"
        return v
    
    def to_casa_lingua_format(self) -> Dict[str, Any]:
        """Convert to CasaLingua format for processing."""
        return {
            "source_language": BloomLanguage.to_casa_lingua(self.sourceLanguage),
            "target_language": BloomLanguage.to_casa_lingua(self.targetLanguage),
            "output_format": None,  # Use original format
            "preserve_layout": self.preserveLayout,
            "glossary_id": self.projectId,  # Map project ID to glossary
            "callback_url": self.callbackUrl,
            "translate_tracked_changes": False,
            "translate_comments": False
        }

class BloomDocumentTranslationResponse(BaseModel):
    """
    Standard Bloom Housing document translation response format.
    """
    documentId: str = Field(..., description="Document processing ID")
    status: str = Field(..., description="Processing status")
    progress: float = Field(..., description="Processing progress (0-1)")
    sourceLanguage: str = Field(..., description="Source language code")
    targetLanguage: str = Field(..., description="Target language code")
    fileName: Optional[str] = Field(None, description="Original filename")
    translatedFileName: Optional[str] = Field(None, description="Translated filename")
    downloadUrl: Optional[str] = Field(None, description="URL to download translated document")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    @classmethod
    def from_casa_lingua_response(cls, casa_response: Dict[str, Any]):
        """Create from CasaLingua response format."""
        # Extract the inner data from CasaLingua BaseResponse wrapper
        if "data" in casa_response:
            data = casa_response["data"]
        else:
            data = casa_response
            
        # Extract required fields
        document_id = data.get("document_id", "")
        status = data.get("status", "error")
        progress = data.get("progress", 0.0)
        source_language = data.get("source_language", "en")
        target_language = data.get("target_language", "en")
        
        # Extract optional fields
        filename = data.get("filename")
        translated_filename = data.get("translated_filename")
        download_url = data.get("download_url")
        
        # Create metadata dictionary
        metadata = {
            "pageCount": data.get("page_count"),
            "wordCount": data.get("word_count"),
            "startTime": data.get("start_time"),
            "endTime": data.get("end_time"),
            "processTime": data.get("process_time")
        }
        
        return cls(
            documentId=document_id,
            status=status,
            progress=progress,
            sourceLanguage=BloomLanguage.from_casa_lingua(source_language),
            targetLanguage=BloomLanguage.from_casa_lingua(target_language),
            fileName=filename,
            translatedFileName=translated_filename,
            downloadUrl=download_url,
            metadata=metadata
        )