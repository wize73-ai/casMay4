

from pydantic import BaseModel, Field
from typing import Optional, Any

class SystemConfigUpdateRequest(BaseModel):
    """Request model for updating a single system configuration key-value pair."""
    key: str = Field(..., description="Configuration key to update")
    value: Any = Field(..., description="New value for the configuration key")
    reason: Optional[str] = Field(None, description="Reason for the update")

class SystemSettingsConfig(BaseModel):
    """Full configuration snapshot for internal usage."""
    logging_level: Optional[str] = "INFO"
    debug_mode: Optional[bool] = False
    max_concurrent_tasks: Optional[int] = 5
    maintenance_mode: Optional[bool] = False