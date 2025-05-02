from typing import Dict, List, Any, Optional, Generic, TypeVar
from enum import Enum
from datetime import datetime
from pydantic import BaseModel, Field, validator
from app.api.schemas.base import StatusEnum, ErrorDetail, MetadataModel, BaseResponse

# Generic type for response data
T = TypeVar('T')

class ErrorCode(str, Enum):
    VALIDATION_ERROR = "validation_error"
    AUTHENTICATION_ERROR = "authentication_error"
    AUTHORIZATION_ERROR = "authorization_error"
    RESOURCE_NOT_FOUND = "resource_not_found"
    MODEL_NOT_FOUND = "model_not_found"
    LANGUAGE_NOT_SUPPORTED = "language_not_supported"
    PROCESSING_ERROR = "processing_error"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    SERVER_ERROR = "server_error"
    INVALID_REQUEST = "invalid_request"
    MODEL_LOADING_ERROR = "model_loading_error"

class ModelInfo(BaseModel):
    id: str
    name: str
    version: str
    type: str
    languages: List[str]
    capabilities: List[str]
    size: Optional[float]
    status: str
    loaded: bool
    last_used: Optional[datetime]
    performance_metrics: Optional[Dict[str, Any]]

class ModelInfoResponse(BaseResponse[ModelInfo]):
    pass

class ModelListResponse(BaseResponse[List[ModelInfo]]):
    pass

class LanguageSupportInfo(BaseModel):
    language_code: str
    language_name: str
    supported_operations: List[str]
    translation_pairs: Optional[List[Dict[str, str]]]
    models: Optional[List[str]]

class LanguageSupportListResponse(BaseResponse[List[LanguageSupportInfo]]):
    pass

class AdminMetrics(BaseModel):
    system_metrics: Dict[str, Any]
    request_metrics: Dict[str, Any]
    model_metrics: Dict[str, Any]
    language_metrics: Dict[str, Any]
    user_metrics: Dict[str, Any]

class AdminMetricsResponse(BaseResponse[AdminMetrics]):
    pass

class UserInfo(BaseModel):
    id: str
    username: str
    email: str
    role: str
    created_at: datetime
    last_login: Optional[datetime]
    status: str

class TranslationResult(BaseModel):
    source_text: str
    translated_text: str
    source_language: str
    target_language: str
    confidence: Optional[float]
    model_id: str
    process_time: float
    word_count: int
    character_count: int
    detected_language: Optional[str]
    verified: Optional[bool]
    verification_score: Optional[float]

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

class LanguageDetectionResult(BaseModel):
    text: str
    detected_language: str
    confidence: float
    alternatives: Optional[List[Dict[str, float]]]
    process_time: float

class LanguageDetectionResponse(BaseResponse[LanguageDetectionResult]):
    pass

class TextAnalysisResult(BaseModel):
    text: str
    language: str
    sentiment: Optional[Dict[str, float]]
    entities: Optional[List[Dict[str, Any]]]
    topics: Optional[List[Dict[str, float]]]
    summary: Optional[str]
    word_count: int
    sentence_count: int
    process_time: float

class TextAnalysisResponse(BaseResponse[TextAnalysisResult]):
    pass

class QueueStatus(BaseModel):
    queue_id: str
    task_id: str
    status: str
    position: Optional[int]
    estimated_start_time: Optional[datetime]
    estimated_completion_time: Optional[datetime]
    progress: float
    result_url: Optional[str]

class QueueStatusResponse(BaseResponse[QueueStatus]):
    pass

class VerificationResult(BaseModel):
    verified: bool
    score: float
    confidence: float
    issues: Optional[List[Dict[str, Any]]]
    source_text: str
    translated_text: str
    source_language: str
    target_language: str
    metrics: Optional[Dict[str, Any]]
    process_time: float

class VerificationResponse(BaseResponse[VerificationResult]):
    pass