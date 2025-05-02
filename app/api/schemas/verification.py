from typing import Any, Dict, List, Optional
from pydantic import BaseModel
from .base import BaseResponse

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
