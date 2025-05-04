"""
Streaming API endpoints for large document processing.
These endpoints provide streaming responses for operations on large documents.
"""

import os
import time
import uuid
import json
import logging
import asyncio
from typing import Dict, List, Any, Optional, AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Request, Body, Query, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.api.middleware.auth import get_current_user
from app.api.schemas.requests.requests_translation import TranslationRequest
from app.api.schemas.requests.analysis import TextAnalysisRequest
from app.utils.logging import get_logger
from app.audit.metrics import track_request_metrics

# Initialize logger
logger = get_logger(__name__)

# Create router
router = APIRouter(
    prefix="/streaming",
    tags=["streaming"],
    responses={404: {"description": "Not found"}},
)

async def stream_translation(
    translation_request: TranslationRequest, 
    current_user: Dict[str, Any]
) -> AsyncIterator[str]:
    """Process translation request as a stream, yielding partial results."""
    from app.core.pipeline.processor import get_pipeline_processor
    from app.core.pipeline.translator import get_translator
    
    # Set up processor and translator
    processor = await get_pipeline_processor()
    translator = await get_translator(
        source_lang=translation_request.source_language, 
        target_lang=translation_request.target_language,
        model_id=translation_request.model_id
    )
    
    # Get or detect source language
    source_lang = translation_request.source_language
    if source_lang == "auto" or not source_lang:
        # Detect language from first chunk
        first_chunk = translation_request.text[:1000]  # Use limited text for faster detection
        detection_result = await processor.detect_language(text=first_chunk, detailed=False)
        source_lang = detection_result.get("detected_language", "en")
        # Yield language detection event
        yield json.dumps({
            "event": "language_detected",
            "language": source_lang,
            "confidence": detection_result.get("confidence", 0.0)
        }) + "\n"
    
    # Split text into chunks for streaming
    # For simplicity, we'll use sentence boundaries or maximum chunk size
    text = translation_request.text
    chunks = []
    
    # Simple chunk splitting by sentence boundaries
    import re
    sentence_boundaries = re.compile(r'(?<=[.!?])\s+')
    sentences = sentence_boundaries.split(text)
    
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) > 500:  # Max chunk size
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = sentence
        else:
            current_chunk += (" " if current_chunk else "") + sentence
    
    if current_chunk:  # Add the last chunk
        chunks.append(current_chunk)
    
    # Process chunks
    translated_text = ""
    chunk_count = len(chunks)
    
    for i, chunk in enumerate(chunks):
        # Translate chunk
        translation_result = await translator.translate(
            text=chunk,
            source_lang=source_lang,
            target_lang=translation_request.target_language
        )
        
        translated_chunk = translation_result.get("translated_text", "")
        translated_text += translated_chunk
        
        # Yield progress event with the translated chunk
        yield json.dumps({
            "event": "chunk_translated",
            "chunk_index": i,
            "total_chunks": chunk_count,
            "progress": (i + 1) / chunk_count,
            "translated_chunk": translated_chunk
        }) + "\n"
        
        # Small delay to prevent overwhelming the client
        await asyncio.sleep(0.05)
    
    # Yield completion event with full translated text
    yield json.dumps({
        "event": "translation_completed",
        "translated_text": translated_text,
        "source_language": source_lang,
        "target_language": translation_request.target_language
    }) + "\n"

@router.post(
    "/translate",
    summary="Stream translation of text",
    description="Translates text in chunks and returns a stream of partial results."
)
async def stream_translate_text(
    request: Request,
    background_tasks: BackgroundTasks,
    translation_request: TranslationRequest = Body(...),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Endpoint for streaming translation of text."""
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    # Track metrics in the background
    background_tasks.add_task(
        track_request_metrics,
        endpoint="/streaming/translate",
        request_id=request_id,
        user_id=current_user.get("id", "anonymous"),
        request_data={
            "text_length": len(translation_request.text),
            "source_language": translation_request.source_language,
            "target_language": translation_request.target_language
        },
        start_time=start_time
    )
    
    # Return streaming response
    return StreamingResponse(
        stream_translation(translation_request, current_user),
        media_type="text/event-stream"
    )

async def stream_analysis(
    analysis_request: TextAnalysisRequest, 
    current_user: Dict[str, Any]
) -> AsyncIterator[str]:
    """Process analysis request as a stream, yielding partial results."""
    from app.core.pipeline.processor import get_pipeline_processor
    
    # Set up processor
    processor = await get_pipeline_processor()
    
    # Get or detect language
    language = analysis_request.language
    if language == "auto" or not language:
        # Detect language from first chunk
        first_chunk = analysis_request.text[:1000]  # Use limited text for faster detection
        detection_result = await processor.detect_language(text=first_chunk, detailed=False)
        language = detection_result.get("detected_language", "en")
        # Yield language detection event
        yield json.dumps({
            "event": "language_detected",
            "language": language,
            "confidence": detection_result.get("confidence", 0.0)
        }) + "\n"
    
    # Execute requested analysis tasks in sequence, yielding results as they complete
    analysis_types = analysis_request.analysis_types or ["sentiment", "entities", "topics", "summary"]
    total_tasks = len(analysis_types)
    completed_tasks = 0
    
    # Process each analysis type and stream results
    for analysis_type in analysis_types:
        # Yield processing start event
        yield json.dumps({
            "event": "analysis_started",
            "analysis_type": analysis_type,
            "progress": completed_tasks / total_tasks
        }) + "\n"
        
        try:
            # Process the analysis
            result = {}
            if analysis_type == "sentiment":
                result = await processor.analyze_sentiment(analysis_request.text, language)
            elif analysis_type == "entities":
                result = await processor.extract_entities(analysis_request.text, language)
            elif analysis_type == "topics":
                result = await processor.extract_topics(analysis_request.text, language)
            elif analysis_type == "summary":
                result = await processor.generate_summary(
                    analysis_request.text, 
                    language, 
                    max_length=analysis_request.max_length
                )
            else:
                # Skip unknown analysis type
                logger.warning(f"Unknown analysis type requested: {analysis_type}")
                continue
                
            # Yield result for this analysis type
            completed_tasks += 1
            yield json.dumps({
                "event": "analysis_completed",
                "analysis_type": analysis_type,
                "result": result,
                "progress": completed_tasks / total_tasks
            }) + "\n"
            
        except Exception as e:
            # Yield error for this analysis type but continue with others
            logger.error(f"Error in {analysis_type} analysis: {str(e)}")
            yield json.dumps({
                "event": "analysis_error",
                "analysis_type": analysis_type,
                "error": str(e)
            }) + "\n"
            
            # Still count as completed for progress tracking
            completed_tasks += 1
            
        # Small delay to prevent overwhelming the client
        await asyncio.sleep(0.05)
    
    # Yield completion event
    yield json.dumps({
        "event": "all_analysis_completed",
        "analysis_types": analysis_types
    }) + "\n"

@router.post(
    "/analyze",
    summary="Stream text analysis",
    description="Analyzes text and returns a stream of partial results for different analysis types."
)
async def stream_analyze_text(
    request: Request,
    background_tasks: BackgroundTasks,
    analysis_request: TextAnalysisRequest = Body(...),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Endpoint for streaming analysis of text."""
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    # Track metrics in the background
    background_tasks.add_task(
        track_request_metrics,
        endpoint="/streaming/analyze",
        request_id=request_id,
        user_id=current_user.get("id", "anonymous"),
        request_data={
            "text_length": len(analysis_request.text),
            "language": analysis_request.language,
            "analysis_types": analysis_request.analysis_types
        },
        start_time=start_time
    )
    
    # Return streaming response
    return StreamingResponse(
        stream_analysis(analysis_request, current_user),
        media_type="text/event-stream"
    )