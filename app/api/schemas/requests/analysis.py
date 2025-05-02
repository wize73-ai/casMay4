

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class TextAnalysisRequest(BaseModel):
    text: str = Field(..., description="Text to analyze")
    language: Optional[str] = Field(None, description="Optional language code override")
    include_sentiment: bool = Field(default=True, description="Whether to perform sentiment analysis")
    include_entities: bool = Field(default=True, description="Whether to extract named entities")
    include_topics: bool = Field(default=False, description="Whether to classify topics")
    include_summary: bool = Field(default=False, description="Whether to generate a summary")

class TextAnalysisResult(BaseModel):
    text: str = Field(..., description="Input text")
    language: str = Field(..., description="Detected or specified language")
    sentiment: Optional[Dict[str, float]] = Field(None, description="Sentiment scores")
    entities: Optional[List[Dict[str, Any]]] = Field(None, description="Extracted named entities")
    topics: Optional[List[Dict[str, float]]] = Field(None, description="Topic classification scores")
    summary: Optional[str] = Field(None, description="Generated summary")
    word_count: int = Field(..., description="Word count of the input")
    sentence_count: int = Field(..., description="Sentence count of the input")
    process_time: float = Field(..., description="Processing time in seconds")