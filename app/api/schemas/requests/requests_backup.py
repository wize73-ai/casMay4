"""
Request schema module routing for CasaLingua
"""

# Avoid circular import: do not re-export here.
# Import directly in modules where needed.
# Example:
# from app.api.schemas.requests_admin import SystemConfigUpdateRequest

# NOTE: Avoid import-time failures. These should be imported where used.
# from app.api.schemas.requests_admin import (
#     ApiKeyCreateRequest,
#     ApiKeyUpdateRequest,
#     ModelLoadRequest,
#     ModelUnloadRequest,
#     ModelReloadRequest
# )

from pydantic import BaseModel, Field
from typing import List, Optional, Any

class TranslationRequest(BaseModel):
    text: str = Field(..., description="Text to be translated")
    source_language: str = Field(..., description="Language code of the input text (e.g. 'en', 'fr')")
    target_language: str = Field(..., description="Language code for the translated output (e.g. 'es', 'de')")


# Additional request schemas
class SystemConfigUpdateRequest(BaseModel):
    key: str = Field(..., description="Configuration key to update")
    value: Any = Field(..., description="New value for the configuration key")
    reason: Optional[str] = Field(None, description="Reason for the update")

class ApiKeyCreateRequest(BaseModel):
    name: str
    scopes: List[str]
    expires_in_days: Optional[int] = 365

class ModelLoadRequest(BaseModel):
    device: Optional[str] = None
    low_memory_mode: Optional[bool] = False