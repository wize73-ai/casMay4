from typing import Dict, List, Optional
from pydantic import BaseModel
from .base import BaseResponse

class LanguageSupportInfo(BaseModel):
    language_code: str
    language_name: str
    supported_operations: List[str]
    translation_pairs: Optional[List[Dict[str, str]]]
    models: Optional[List[str]]

class LanguageSupportListResponse(BaseResponse[List[LanguageSupportInfo]]):
    pass

class LanguageDetectionResult(BaseModel):
    text: str
    detected_language: str
    confidence: float
    alternatives: Optional[List[Dict[str, float]]]
    process_time: float

class LanguageDetectionResponse(BaseResponse[LanguageDetectionResult]):
    pass

class LanguageDetectionRequest(BaseModel):
    text: str
    detailed: Optional[bool] = False
    model_id: Optional[str] = None


# Add SupportedLanguage class
class SupportedLanguage(BaseModel):
    language_code: str
    language_name: str
    is_enabled: bool


# Request body model for /upload route
class UploadDocumentRequest(BaseModel):
    language: str
    metadata: Optional[dict] = None