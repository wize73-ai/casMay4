"""
Request Schemas for CasaLingua API
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class TranslationRequest(BaseModel):
    text: str = Field(..., example="Hola, ¿cómo estás?")
    source_language: str = Field(..., example="es")
    target_language: str = Field(..., example="en")
    options: Optional[Dict[str, Any]] = Field(default_factory=dict)


class LanguageDetectionRequest(BaseModel):
    text: str = Field(..., example="Bonjour, comment allez-vous?")


class DocumentUploadRequest(BaseModel):
    filename: str
    content_type: str
    document_bytes: bytes


class RAGQueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    context_docs: Optional[List[str]] = None


class TextSimplificationRequest(BaseModel):
    text: str
    grade_level: Optional[int] = 8
