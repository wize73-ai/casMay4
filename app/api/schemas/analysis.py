from typing import Dict, List, Any, Optional
from pydantic import BaseModel
from .base import BaseResponse

class TextAnalysisRequest(BaseModel):
    text: str
    language: Optional[str] = None
    include_sentiment: bool = True
    include_entities: bool = True
    include_topics: bool = False
    include_summary: bool = False

class TextAnalysisResult(BaseModel):
    text: str
    language: str
    sentiment: Optional[Dict[str, float]] = None
    entities: Optional[List[Dict[str, Any]]] = None
    topics: Optional[List[Dict[str, float]]] = None
    summary: Optional[str] = None
    word_count: Optional[int] = None
    sentence_count: Optional[int] = None
    process_time: Optional[float] = None

class TextAnalysisResponse(BaseResponse[TextAnalysisResult]):
    pass