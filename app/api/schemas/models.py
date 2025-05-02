from datetime import datetime
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from .base import BaseResponse

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