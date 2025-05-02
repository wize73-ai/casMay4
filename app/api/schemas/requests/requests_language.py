from pydantic import BaseModel, Field
from typing import Optional

class LanguageDetectionRequest(BaseModel):
    text: str = Field(..., description="Text input to detect the language of")
    top_k: Optional[int] = Field(1, description="Return the top K most likely language predictions")
    include_confidence: bool = Field(default=True, description="Include confidence scores in the result")
