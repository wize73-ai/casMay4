from typing import Optional
from datetime import datetime
from pydantic import BaseModel
from .base import BaseResponse

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