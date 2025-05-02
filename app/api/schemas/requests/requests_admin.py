from pydantic import BaseModel, Field

class AdminTranslationRequest(BaseModel):
    """Schema for admin translation requests."""
    text: str = Field(..., description="The input text to be translated")
    source_language: str = Field(..., description="Source language code (e.g., 'en')")
    target_language: str = Field(..., description="Target language code (e.g., 'es')")

class ApiKeyCreateRequest(BaseModel):
    """Request model for creating a new API key."""
    name: str = Field(..., description="A friendly name for the API key")
    scopes: list[str] = Field(default_factory=list, description="List of scopes assigned to the key")