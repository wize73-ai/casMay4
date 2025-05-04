# No changes to the provided file `requests_translation.py` as the instructions specify edits to `app/api/schemas/requests/__init__.py` only.
# Add TranslationTextRequest at the top
from pydantic import BaseModel, Field
from typing import Optional, List

class TranslationTextRequest(BaseModel):
    text: str = Field(..., description="The text to be translated")
    source_language: Optional[str] = Field(None, description="Source language code")
    target_language: str = Field(..., description="Target language code")
    context: Optional[List[str]] = Field(None, description="Optional context")
    preserve_formatting: bool = Field(default=True, description="Preserve formatting")
    model_name: Optional[str] = Field(None, description="Translation model to use")
    domain: Optional[str] = Field(None, description="Domain for translation (e.g. legal, medical)")
    
class TranslationRequest(BaseModel):
    text: str = Field(..., description="The text to be translated")
    source_language: Optional[str] = Field(None, description="Source language code (auto-detect if not provided)")
    target_language: str = Field(..., description="Target language code")
    model_id: Optional[str] = Field(None, description="Specific model ID to use for translation")