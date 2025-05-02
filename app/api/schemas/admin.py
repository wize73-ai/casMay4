from typing import Any, Dict
from pydantic import BaseModel
from .base import BaseResponse

class AdminMetrics(BaseModel):
    system_metrics: Dict[str, Any]
    request_metrics: Dict[str, Any]
    model_metrics: Dict[str, Any]
    language_metrics: Dict[str, Any]
    user_metrics: Dict[str, Any]

class AdminMetricsResponse(BaseResponse[AdminMetrics]):
    pass