from typing import Any, Optional
from pydantic import BaseModel, Field

class SystemConfigUpdateRequest(BaseModel):
    key: str = Field(..., description="Configuration key to update")
    value: Any = Field(..., description="New value for the configuration key")
    reason: Optional[str] = Field(None, description="Reason for the update")

class SystemSettingsConfig(BaseModel):
    logging_level: Optional[str] = "INFO"
    debug_mode: Optional[bool] = False
    max_concurrent_tasks: Optional[int] = 5
    maintenance_mode: Optional[bool] = False