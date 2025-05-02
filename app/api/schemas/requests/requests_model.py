

"""
Model Load Request Schema for CasaLingua

This schema defines the expected format for requests related to loading models.

Author: CasaLingua Development Team
Version: 1.0.0
"""

from pydantic import BaseModel, Field

class ModelLoadRequest(BaseModel):
    model_id: str = Field(..., description="Unique identifier of the model to load")
    task: str = Field(..., description="Task category this model supports, e.g. translation or detection")
    path: str = Field(..., description="Filesystem path to the model directory")


# API Key Creation Request Schema
from typing import Optional

class ApiKeyCreateRequest(BaseModel):
    """Request model for creating a new API key."""
    name: str = Field(..., description="Human-friendly name for the API key")
    scopes: list[str] = Field(default_factory=list, description="List of authorized scopes")
    expires_in_days: Optional[int] = Field(365, description="Expiration duration in days (default: 365)")