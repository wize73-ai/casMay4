from typing import Optional, List
from datetime import datetime
from pydantic import BaseModel
from .base import BaseResponse


# TranslationRequest schema for translation endpoint
class TranslationRequest(BaseModel):
    text: str
    source_language: Optional[str] = None
    target_language: str
    context: Optional[List[str]] = None
    preserve_formatting: bool = True
    model_name: Optional[str] = None
    domain: Optional[str] = None
    glossary_id: Optional[str] = None
    verify: Optional[bool] = False
    formality: Optional[str] = None
    parameters: Optional[dict] = None  # Additional parameters for the translation model


# BatchTranslationRequest schema for batch translation endpoint
class BatchTranslationRequest(BaseModel):
    texts: List[str]
    source_language: Optional[str] = None
    target_language: str
    model_id: Optional[str] = None
    glossary_id: Optional[str] = None
    preserve_formatting: bool = True


class DocumentTranslationRequest(BaseModel):
    document_id: str
    source_language: Optional[str] = None
    target_language: str
    model_id: Optional[str] = None
    priority: Optional[str] = "normal"
    preserve_layout: bool = True


class TranslationResult(BaseModel):
    source_text: str = ""  # Default value to avoid validation issues
    translated_text: str
    source_language: str = "auto"  # Default value
    target_language: str
    confidence: Optional[float] = 0.0
    model_id: str = "default"
    process_time: float = 0.0
    word_count: int = 0
    character_count: int = 0
    detected_language: Optional[str] = None
    verified: Optional[bool] = False
    verification_score: Optional[float] = None
    model_used: Optional[str] = "translation"
    used_fallback: Optional[bool] = False
    fallback_model: Optional[str] = None

class TranslationResponse(BaseResponse[TranslationResult]):
    pass

class DocumentTranslationResult(BaseModel):
    document_id: str
    filename: str
    source_language: str
    target_language: str
    status: str
    progress: float
    translated_filename: Optional[str]
    page_count: Optional[int]
    word_count: Optional[int]
    start_time: datetime
    end_time: Optional[datetime]
    process_time: Optional[float]
    download_url: Optional[str]

class DocumentTranslationResponse(BaseResponse[DocumentTranslationResult]):
    pass