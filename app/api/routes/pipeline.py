"""
Pipeline API Routes for CasaLingua

This module defines API endpoints for the unified language processing
pipeline, handling tasks such as translation, language detection,
text analysis, and processing monitoring.

Author: Exygy Development Team
Version: 1.0.0
License: MIT
"""

import time
import json
import uuid
import logging
import os
import asyncio
from typing import Dict, List, Any, Optional, Union
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query, Path, Body, Request, status, File, UploadFile, Form
from pydantic import BaseModel, Field, validator

from app.api.schemas.translation import TranslationRequest, BatchTranslationRequest, DocumentTranslationRequest
import app.api.schemas.language
from app.api.schemas.analysis import TextAnalysisRequest
from app.api.schemas.base import BaseResponse, StatusEnum, MetadataModel, ErrorDetail
from app.api.schemas.translation import TranslationResult, TranslationResponse, DocumentTranslationResult, DocumentTranslationResponse
from app.api.schemas.language import LanguageDetectionResult, LanguageDetectionResponse, LanguageDetectionRequest
from app.api.schemas.analysis import TextAnalysisResult, TextAnalysisResponse
from app.api.schemas.queue import QueueStatus, QueueStatusResponse
from app.api.schemas.verification import VerificationResult, VerificationResponse
from app.utils.auth import verify_api_key
from app.api.middleware.auth import get_current_user
from app.utils.logging import get_logger

logger = get_logger(__name__)

# Create router
router = APIRouter()

# ----- Text Translation Endpoints -----

@router.post(
    "/translate",
    response_model=TranslationResponse,
    status_code=status.HTTP_200_OK,
    summary="Translate text",
    description="Translates text from one language to another."
)
async def translate_text(
    request: Request,
    background_tasks: BackgroundTasks,
    translation_request: TranslationRequest = Body(...),
    verification: bool = Query(False, description="Whether to verify translation quality"),
    current_user: Dict[str, Any] = Depends(get_current_user),
    use_cache: bool = Query(True, description="Whether to use request-level caching")
):
    """
    Translate text from one language to another.
    
    This endpoint handles single text translation requests and supports
    automatic language detection, custom glossaries, and quality verification.
    """
    start_time = time.time()
    request_id = str(uuid.uuid4())
    
    try:
        # Get application components from state
        processor = request.app.state.processor
        if processor is None:
            raise HTTPException(status_code=503, detail="Processor not initialized")
        metrics = request.app.state.metrics
        audit_logger = request.app.state.audit_logger
        
        # Check request cache first if enabled
        cached_response = None
        if use_cache and hasattr(request.app.state, "route_cache"):
            from app.services.storage.route_cache import RouteCacheManager, RouteCache
            
            # Get translation cache instance
            translation_cache = await RouteCacheManager.get_cache(
                name="translation",
                max_size=1000,
                ttl_seconds=3600,  # 1 hour by default
                bloom_compatible=True
            )
            
            # Generate cache key
            cache_params = {
                "text": translation_request.text,
                "source_language": translation_request.source_language,
                "target_language": translation_request.target_language,
                "model_id": translation_request.model_name,  # Use model_name from TranslationRequest
                "preserve_formatting": translation_request.preserve_formatting,
                "formality": translation_request.formality,
                "verify": verification or translation_request.verify
            }
            cache_key = RouteCache.bloom_compatible_key("/pipeline/translate", cache_params)
            
            # Try to get from cache
            cached_response = await translation_cache.get(cache_key)
            
            if cached_response:
                logger.info(f"Cache hit for translation request {request_id}")
                
                # Record cache hit metrics in background
                # Extract translated_text safely from cached response
                if isinstance(cached_response, dict) and "data" in cached_response:
                    if hasattr(cached_response["data"], "translated_text"):
                        output_size = len(cached_response["data"].translated_text)
                    elif isinstance(cached_response["data"], dict) and "translated_text" in cached_response["data"]:
                        output_size = len(cached_response["data"]["translated_text"])
                    else:
                        output_size = 0
                        logger.warning(f"Could not determine output size from cached response: {type(cached_response)}")
                else:
                    output_size = 0
                    logger.warning(f"Cached response structure unexpected: {type(cached_response)}")
                
                background_tasks.add_task(
                    metrics.record_pipeline_execution,
                    pipeline_id="translation",
                    operation="translate_cached",
                    duration=time.time() - start_time,
                    input_size=len(translation_request.text),
                    output_size=output_size,
                    success=True,
                    metadata={
                        "source_language": translation_request.source_language,
                        "target_language": translation_request.target_language,
                        "cache_hit": True
                    }
                )
                
                # Log cache hit to audit log
                background_tasks.add_task(
                    audit_logger.log_api_request,
                    endpoint="/pipeline/translate",
                    method="POST",
                    user_id=current_user["id"],
                    source_ip=request.client.host,
                    request_id=request_id,
                    request_params={
                        "source_language": translation_request.source_language,
                        "target_language": translation_request.target_language,
                        "text_length": len(translation_request.text),
                        "cached": True
                    },
                    status_code=200
                )
                
                # Update cached response metadata safely
                if isinstance(cached_response, dict) and "metadata" in cached_response:
                    # Check if metadata is a dict or an object with attributes
                    if isinstance(cached_response["metadata"], dict):
                        cached_response["metadata"]["request_id"] = request_id
                        cached_response["metadata"]["timestamp"] = time.time()
                        cached_response["metadata"]["process_time"] = time.time() - start_time
                        cached_response["metadata"]["cached"] = True
                    elif hasattr(cached_response["metadata"], "request_id"):
                        # It's an object with attributes (like MetadataModel)
                        cached_response["metadata"].request_id = request_id
                        cached_response["metadata"].timestamp = time.time()
                        cached_response["metadata"].process_time = time.time() - start_time
                        cached_response["metadata"].cached = True
                    else:
                        # Create new metadata
                        logger.warning(f"Creating new metadata for cached response")
                        cached_response["metadata"] = MetadataModel(
                            request_id=request_id,
                            timestamp=time.time(),
                            version=request.app.state.config.get("version", "1.0.0"),
                            process_time=time.time() - start_time,
                            cached=True
                        )
                else:
                    logger.warning(f"Cached response has no metadata field: {type(cached_response)}")
                
                # Return cached response
                return cached_response
        
        # Log request to audit log
        await audit_logger.log_api_request(
            endpoint="/pipeline/translate",
            method="POST",
            user_id=current_user["id"],
            source_ip=request.client.host,
            request_id=request_id,
            request_params={
                "source_language": translation_request.source_language,
                "target_language": translation_request.target_language,
                "text_length": len(translation_request.text),
                "model_id": getattr(translation_request, "model_id", None),
                "verification": verification
            }
        )
        
        # Process translation request
        translation_result = await processor.process_translation(
            text=translation_request.text,
            source_language=translation_request.source_language,
            target_language=translation_request.target_language,
            model_id=getattr(translation_request, "model_name", None),  # Use model_name instead of model_id
            glossary_id=translation_request.glossary_id,
            preserve_formatting=translation_request.preserve_formatting,
            formality=translation_request.formality,
            verify=verification or translation_request.verify,
            user_id=current_user["id"],
            request_id=request_id
        )
        
        # Ensure source_text is included in the result
        if "source_text" not in translation_result:
            translation_result["source_text"] = translation_request.text
        
        # Calculate process time
        process_time = time.time() - start_time
        
        # Record metrics in background
        background_tasks.add_task(
            metrics.record_pipeline_execution,
            pipeline_id="translation",
            operation="translate",
            duration=process_time,
            input_size=len(translation_request.text),
            output_size=len(translation_result["translated_text"]),
            success=True,
            metadata={
                "source_language": translation_request.source_language,
                "target_language": translation_request.target_language,
                "model_id": translation_result.get("model_id", getattr(translation_request, "model_id", "unknown"))
            }
        )
        
        # Record language operation metrics
        background_tasks.add_task(
            metrics.record_language_operation,
            source_lang=translation_request.source_language,
            target_lang=translation_request.target_language,
            operation="translation",
            duration=process_time,
            input_size=len(translation_request.text),
            output_size=len(translation_result["translated_text"]),
            success=True
        )
        
        # Log the received result for debugging
        logger.debug(f"Raw translation result: {translation_result}")
        
        # Create result dictionary manually with all required fields
        result_dict = {
            "source_text": translation_request.text,
            "translated_text": translation_result.get("translated_text", ""),
            "source_language": translation_result.get("source_language") or translation_request.source_language or "auto",
            "target_language": translation_request.target_language,
            "confidence": translation_result.get("confidence", 0.0),
            "model_id": translation_result.get("model_id", "default"),
            "process_time": process_time,
            "word_count": len(translation_request.text.split()),
            "character_count": len(translation_request.text),
            "detected_language": translation_result.get("detected_language"),
            "verified": translation_result.get("verified", False),
            "verification_score": translation_result.get("verification_score", None),
            "model_used": translation_result.get("model_used", "translation")
        }
        
        # Create result model from dictionary
        result = TranslationResult(**result_dict)
        
        # Create response
        response = BaseResponse(
            status=StatusEnum.SUCCESS,
            message="Translation completed successfully",
            data=result,
            metadata=MetadataModel(
                request_id=request_id,
                timestamp=time.time(),
                version=request.app.state.config.get("version", "1.0.0"),
                process_time=process_time,
                cached=False
            ),
            errors=None,
            pagination=None
        )
        
        # Store response in cache if caching is enabled
        if use_cache and hasattr(request.app.state, "route_cache") and 'translation_cache' in locals() and 'cache_key' in locals():
            await translation_cache.set(cache_key, response)
            logger.debug(f"Stored translation in cache with key {cache_key[:8]}...")
        
        return response
        
    except Exception as e:
        logger.error(f"Translation error: {str(e)}", exc_info=True)

        # Robust access to Pydantic model fields
        source_language = getattr(translation_request, "source_language", "unknown")
        target_language = getattr(translation_request, "target_language", "unknown")
        text = getattr(translation_request, "text", "")

        # Record error metrics in background
        background_tasks.add_task(
            metrics.record_pipeline_execution,
            pipeline_id="translation",
            operation="translate",
            duration=time.time() - start_time,
            input_size=len(text),
            output_size=0,
            success=False,
            metadata={
                "error": str(e),
                "source_language": source_language,
                "target_language": target_language
            }
        )

        # Log error to audit log
        background_tasks.add_task(
            audit_logger.log_api_request,
            endpoint="/pipeline/translate",
            method="POST",
            user_id=current_user["id"],
            source_ip=request.client.host,
            request_id=request_id,
            request_params={
                "source_language": source_language,
                "target_language": target_language,
                "text_length": len(text)
            },
            status_code=500,
            error_message=str(e)
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Translation error: {str(e)}"
        )

# ----- Language Detection Endpoints -----

@router.post(
    "/detect",
    response_model=LanguageDetectionResponse,
    status_code=status.HTTP_200_OK,
    summary="Detect language",
    description="Detects the language of provided text."
)
async def detect_language(
    request: Request,
    background_tasks: BackgroundTasks,
    detection_request: LanguageDetectionRequest = Body(...),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Detect the language of provided text.
    
    This endpoint analyzes text to determine its language, providing
    confidence scores and alternative language possibilities.
    """
    start_time = time.time()
    request_id = str(uuid.uuid4())
    
    try:
        # Get application components from state
        processor = request.app.state.processor
        if processor is None:
            raise HTTPException(status_code=503, detail="Processor not initialized")
        metrics = request.app.state.metrics
        audit_logger = request.app.state.audit_logger
        
        # Log request to audit log
        await audit_logger.log_api_request(
            endpoint="/pipeline/detect",
            method="POST",
            user_id=current_user["id"],
            source_ip=request.client.host,
            request_id=request_id,
            request_params={
                "text_length": len(detection_request.text),
                "detailed": detection_request.detailed,
                "model_id": detection_request.model_id
            }
        )
        
        # Process language detection
        detection_result = await processor.detect_language(
            text=detection_request.text,
            detailed=detection_request.detailed,
            model_id=detection_request.model_id
        )
        
        # Calculate process time
        process_time = time.time() - start_time
        
        # Record metrics in background
        background_tasks.add_task(
            metrics.record_pipeline_execution,
            pipeline_id="language_detection",
            operation="detect",
            duration=process_time,
            input_size=len(detection_request.text),
            output_size=0,
            success=True,
            metadata={
                "detected_language": detection_result["detected_language"],
                "confidence": detection_result["confidence"],
                "model_id": detection_result.get("model_id", "default")
            }
        )
        
        # Create result model
        result = LanguageDetectionResult(
            text=detection_request.text,
            detected_language=detection_result["detected_language"],
            confidence=detection_result["confidence"],
            alternatives=detection_result.get("alternatives") if detection_request.detailed else None,
            process_time=process_time
        )
        
        # Create response
        response = BaseResponse(
            status=StatusEnum.SUCCESS,
            message="Language detection completed successfully",
            data=result,
            metadata=MetadataModel(
                request_id=request_id,
                timestamp=time.time(),
                version=request.app.state.config.get("version", "1.0.0"),
                process_time=process_time
            ),
            errors=None,
            pagination=None
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Language detection error: {str(e)}", exc_info=True)
        
        # Record error metrics in background
        background_tasks.add_task(
            metrics.record_pipeline_execution,
            pipeline_id="language_detection",
            operation="detect",
            duration=time.time() - start_time,
            input_size=len(detection_request.text),
            output_size=0,
            success=False,
            metadata={
                "error": str(e)
            }
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Language detection error: {str(e)}"
        )

# Alias route for language detection
@router.post(
    "/detect-language",
    response_model=LanguageDetectionResponse,
    status_code=status.HTTP_200_OK,
    summary="Detect language (alias)",
    description="Alias route for detecting the language of provided text."
)
async def detect_language_alias(
    request: Request,
    background_tasks: BackgroundTasks,
    detection_request: LanguageDetectionRequest = Body(...),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    return await detect_language(
        request=request,
        background_tasks=background_tasks,
        detection_request=detection_request,
        current_user=current_user
    )

# ----- Text Simplification Endpoint -----

class SimplificationRequest(BaseModel):
    text: str
    language: str = "en"
    target_level: str = "simple"
    model_id: Optional[str] = None

@router.post(
    "/simplify",
    response_model=BaseResponse,
    status_code=status.HTTP_200_OK,
    summary="Simplify text",
    description="Simplifies complex text to make it more readable."
)
async def simplify_text(
    request: Request,
    background_tasks: BackgroundTasks,
    simplification_request: SimplificationRequest = Body(...),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Simplify complex text to make it more readable.
    
    This endpoint processes text to reduce complexity while maintaining meaning,
    making content more accessible to a wider audience.
    """
    start_time = time.time()
    request_id = str(uuid.uuid4())
    
    try:
        # Get application components from state
        processor = request.app.state.processor
        if processor is None:
            raise HTTPException(status_code=503, detail="Processor not initialized")
        metrics = request.app.state.metrics
        audit_logger = request.app.state.audit_logger
        
        # Log request to audit log
        await audit_logger.log_api_request(
            endpoint="/pipeline/simplify",
            method="POST",
            user_id=current_user["id"],
            source_ip=request.client.host,
            request_id=request_id,
            request_params={
                "text_length": len(simplification_request.text),
                "language": simplification_request.language,
                "target_level": simplification_request.target_level
            }
        )
        
        # Process simplification
        if hasattr(processor, "simplify_text"):
            simplification_result = await processor.simplify_text(
                text=simplification_request.text,
                target_level=simplification_request.target_level,
                language=simplification_request.language
            )
        elif hasattr(processor, "process_simplification"):
            simplification_result = await processor.process_simplification(
                text=simplification_request.text,
                level=simplification_request.target_level,
                language=simplification_request.language
            )
        else:
            # Try to use existing methods in the UnifiedProcessor
            # that may be used in the test_direct_model_calls.py
            async def run_simplification():
                # Create a structured input similar to what the processor might expect
                input_data = {
                    "text": simplification_request.text,
                    "target_level": simplification_request.target_level,
                    "language": simplification_request.language
                }
                
                # Try to use the simplification pipeline directly if available
                if hasattr(processor, "simplification_pipeline") and processor.simplification_pipeline:
                    return await processor.simplification_pipeline.simplify(
                        text=simplification_request.text,
                        language=simplification_request.language,
                        level=int(simplification_request.target_level) if simplification_request.target_level.isdigit() else 1,
                        target_grade_level=None
                    )
                
                # Fallback to generic processing
                return await processor._process_text(
                    text=simplification_request.text, 
                    options={
                        "simplify": True, 
                        "target_level": simplification_request.target_level,
                        "source_language": simplification_request.language
                    }, 
                    metadata={}
                )
            
            # Attempt to run simplification
            simplification_result = await run_simplification()
            
            # Handle different response formats
            if isinstance(simplification_result, dict):
                if "simplified_text" in simplification_result:
                    simplified_text = simplification_result["simplified_text"]
                elif "processed_text" in simplification_result:
                    simplified_text = simplification_result["processed_text"]
                else:
                    simplified_text = str(simplification_result)
            else:
                simplified_text = str(simplification_result)
        
        # Calculate process time
        process_time = time.time() - start_time
        
        # Record metrics in background
        background_tasks.add_task(
            metrics.record_pipeline_execution,
            pipeline_id="simplification",
            operation="simplify",
            duration=process_time,
            input_size=len(simplification_request.text),
            output_size=len(simplified_text) if isinstance(simplified_text, str) else 0,
            success=True,
            metadata={
                "language": simplification_request.language,
                "target_level": simplification_request.target_level
            }
        )
        
        # Create response
        response = BaseResponse(
            status=StatusEnum.SUCCESS,
            message="Text simplification completed successfully",
            data={
                "original_text": simplification_request.text,
                "simplified_text": simplified_text,
                "language": simplification_request.language,
                "target_level": simplification_request.target_level,
                "process_time": process_time
            },
            metadata=MetadataModel(
                request_id=request_id,
                timestamp=time.time(),
                version=request.app.state.config.get("version", "1.0.0"),
                process_time=process_time
            ),
            errors=None,
            pagination=None
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Text simplification error: {str(e)}", exc_info=True)
        
        # Record error metrics in background
        background_tasks.add_task(
            metrics.record_pipeline_execution,
            pipeline_id="simplification",
            operation="simplify",
            duration=time.time() - start_time,
            input_size=len(simplification_request.text),
            output_size=0,
            success=False,
            metadata={
                "error": str(e),
                "language": simplification_request.language,
                "target_level": simplification_request.target_level
            }
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Text simplification error: {str(e)}"
        )

# ----- Text Anonymization Endpoint -----

class AnonymizationRequest(BaseModel):
    text: str
    language: str = "en"
    strategy: str = "mask"
    model_id: Optional[str] = None

@router.post(
    "/anonymize",
    response_model=BaseResponse,
    status_code=status.HTTP_200_OK,
    summary="Anonymize text",
    description="Anonymizes personally identifiable information (PII) in text."
)
async def anonymize_text(
    request: Request,
    background_tasks: BackgroundTasks,
    anonymization_request: AnonymizationRequest = Body(...),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Anonymize personally identifiable information (PII) in text.
    
    This endpoint detects and masks/replaces personal information like names,
    email addresses, phone numbers, and other sensitive data.
    """
    start_time = time.time()
    request_id = str(uuid.uuid4())
    
    try:
        # Get application components from state
        processor = request.app.state.processor
        if processor is None:
            raise HTTPException(status_code=503, detail="Processor not initialized")
        metrics = request.app.state.metrics
        audit_logger = request.app.state.audit_logger
        
        # Log request to audit log
        await audit_logger.log_api_request(
            endpoint="/pipeline/anonymize",
            method="POST",
            user_id=current_user["id"],
            source_ip=request.client.host,
            request_id=request_id,
            request_params={
                "text_length": len(anonymization_request.text),
                "language": anonymization_request.language,
                "strategy": anonymization_request.strategy
            }
        )
        
        # Process anonymization
        if hasattr(processor, "anonymize_text"):
            anonymized_text = await processor.anonymize_text(
                text=anonymization_request.text,
                language=anonymization_request.language
            )
        elif hasattr(processor, "process_anonymization"):
            anonymized_text = await processor.process_anonymization(
                text=anonymization_request.text,
                language=anonymization_request.language
            )
        else:
            # Try to use existing methods in the UnifiedProcessor
            async def run_anonymization():
                # Create a structured input similar to what the processor might expect
                input_data = {
                    "text": anonymization_request.text,
                    "language": anonymization_request.language,
                    "strategy": anonymization_request.strategy
                }
                
                # Try to use the anonymization pipeline directly if available
                if hasattr(processor, "anonymization_pipeline") and processor.anonymization_pipeline:
                    return await processor.anonymization_pipeline.process(
                        text=anonymization_request.text,
                        language=anonymization_request.language,
                        options={"strategy": anonymization_request.strategy}
                    )
                
                # Fallback to generic processing
                return await processor._process_text(
                    text=anonymization_request.text, 
                    options={
                        "anonymize": True, 
                        "anonymization_strategy": anonymization_request.strategy,
                        "source_language": anonymization_request.language
                    }, 
                    metadata={}
                )
            
            # Attempt to run anonymization
            anonymization_result = await run_anonymization()
            
            # Handle different response formats
            if isinstance(anonymization_result, dict):
                if "anonymized_text" in anonymization_result:
                    anonymized_text = anonymization_result["anonymized_text"]
                elif "processed_text" in anonymization_result:
                    anonymized_text = anonymization_result["processed_text"]
                else:
                    anonymized_text = str(anonymization_result)
            elif isinstance(anonymization_result, tuple) and len(anonymization_result) == 2:
                # Some anonymization methods return (text, entities)
                anonymized_text = anonymization_result[0]
            else:
                anonymized_text = str(anonymization_result)
        
        # Calculate process time
        process_time = time.time() - start_time
        
        # Record metrics in background
        background_tasks.add_task(
            metrics.record_pipeline_execution,
            pipeline_id="anonymization",
            operation="anonymize",
            duration=process_time,
            input_size=len(anonymization_request.text),
            output_size=len(anonymized_text) if isinstance(anonymized_text, str) else 0,
            success=True,
            metadata={
                "language": anonymization_request.language,
                "strategy": anonymization_request.strategy
            }
        )
        
        # Create response
        response = BaseResponse(
            status=StatusEnum.SUCCESS,
            message="Text anonymization completed successfully",
            data={
                "original_text": anonymization_request.text,
                "anonymized_text": anonymized_text,
                "language": anonymization_request.language,
                "strategy": anonymization_request.strategy,
                "process_time": process_time
            },
            metadata=MetadataModel(
                request_id=request_id,
                timestamp=time.time(),
                version=request.app.state.config.get("version", "1.0.0"),
                process_time=process_time
            ),
            errors=None,
            pagination=None
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Text anonymization error: {str(e)}", exc_info=True)
        
        # Record error metrics in background
        background_tasks.add_task(
            metrics.record_pipeline_execution,
            pipeline_id="anonymization",
            operation="anonymize",
            duration=time.time() - start_time,
            input_size=len(anonymization_request.text),
            output_size=0,
            success=False,
            metadata={
                "error": str(e),
                "language": anonymization_request.language,
                "strategy": anonymization_request.strategy
            }
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Text anonymization error: {str(e)}"
        )

# ----- Text Analysis Endpoints -----

@router.post(
    "/analyze",
    response_model=TextAnalysisResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyze text",
    description="Performs analysis on text such as sentiment, entity detection, and more."
)
async def analyze_text(
    request: Request,
    background_tasks: BackgroundTasks,
    analysis_request: TextAnalysisRequest = Body(...),
    current_user: Dict[str, Any] = Depends(get_current_user),
    use_cache: bool = Query(True, description="Whether to use request-level caching")
):
    """
    Analyze text for various linguistic features.
    
    This endpoint processes text analysis requests for sentiment analysis,
    entity recognition, topic classification, summarization, and more.
    """
    start_time = time.time()
    request_id = str(uuid.uuid4())
    
    try:
        # Get application components from state
        processor = request.app.state.processor
        if processor is None:
            raise HTTPException(status_code=503, detail="Processor not initialized")
        metrics = request.app.state.metrics
        audit_logger = request.app.state.audit_logger
        
        # Check request cache first if enabled
        cached_response = None
        if use_cache and hasattr(request.app.state, "route_cache"):
            from app.services.storage.route_cache import RouteCacheManager, RouteCache
            
            # Get analysis cache instance
            analysis_cache = await RouteCacheManager.get_cache(
                name="analysis",
                max_size=500,
                ttl_seconds=3600,  # 1 hour by default
                bloom_compatible=True
            )
            
            # Generate cache key
            cache_params = {
                "text": analysis_request.text,
                "language": analysis_request.language,
                "analyses": analysis_request.analyses,
                "model_id": analysis_request.model_id
            }
            cache_key = RouteCache.bloom_compatible_key("/pipeline/analyze", cache_params)
            
            # Try to get from cache
            cached_response = await analysis_cache.get(cache_key)
            
            if cached_response:
                logger.info(f"Cache hit for analysis request {request_id}")
                
                # Record cache hit metrics in background
                background_tasks.add_task(
                    metrics.record_pipeline_execution,
                    pipeline_id="text_analysis",
                    operation="analyze_cached",
                    duration=time.time() - start_time,
                    input_size=len(analysis_request.text),
                    output_size=0,
                    success=True,
                    metadata={
                        "cache_hit": True,
                        "analyses": analysis_request.analyses,
                    }
                )
                
                # Update cached response metadata safely
                if isinstance(cached_response, dict) and "metadata" in cached_response:
                    # Check if metadata is a dict or an object with attributes
                    if isinstance(cached_response["metadata"], dict):
                        cached_response["metadata"]["request_id"] = request_id
                        cached_response["metadata"]["timestamp"] = time.time()
                        cached_response["metadata"]["process_time"] = time.time() - start_time
                        cached_response["metadata"]["cached"] = True
                    elif hasattr(cached_response["metadata"], "request_id"):
                        # It's an object with attributes (like MetadataModel)
                        cached_response["metadata"].request_id = request_id
                        cached_response["metadata"].timestamp = time.time()
                        cached_response["metadata"].process_time = time.time() - start_time
                        cached_response["metadata"].cached = True
                    else:
                        # Create new metadata
                        logger.warning(f"Creating new metadata for cached response")
                        cached_response["metadata"] = MetadataModel(
                            request_id=request_id,
                            timestamp=time.time(),
                            version=request.app.state.config.get("version", "1.0.0"),
                            process_time=time.time() - start_time,
                            cached=True
                        )
                else:
                    logger.warning(f"Cached response has no metadata field: {type(cached_response)}")
                
                # Return cached response
                return cached_response
        
        # Log request to audit log
        await audit_logger.log_api_request(
            endpoint="/pipeline/analyze",
            method="POST",
            user_id=current_user["id"],
            source_ip=request.client.host,
            request_id=request_id,
            request_params={
                "text_length": len(analysis_request.text),
                "language": analysis_request.language,
                "analyses": analysis_request.analyses,
                "model_id": analysis_request.model_id
            }
        )
        
        # Set up parallel task for language detection if needed
        preprocessing_tasks = {}
        detected_language = None
        
        # Add language detection task if no language specified
        if not analysis_request.language or analysis_request.language == "auto":
            preprocessing_tasks["language_detection"] = processor.detect_language(
                text=analysis_request.text[:1000],  # Use limited text for faster detection
                detailed=False
            )
        
        # Execute preprocessing tasks in parallel if any
        if preprocessing_tasks:
            # Run all tasks concurrently
            preprocessing_results = {}
            results = await asyncio.gather(*preprocessing_tasks.values(), return_exceptions=True)
            preprocessing_results = dict(zip(preprocessing_tasks.keys(), results))
            
            # Process language detection result
            if "language_detection" in preprocessing_results:
                detection_result = preprocessing_results["language_detection"]
                # Check for exceptions
                if isinstance(detection_result, Exception):
                    logger.warning(f"Language detection failed: {detection_result}")
                else:
                    detected_language = detection_result.get("detected_language", "en")
                    logger.debug(f"Detected language in analyze endpoint: {detected_language}")
        
        # Process text analysis with detected language if needed
        analysis_result = await processor.analyze_text(
            text=analysis_request.text,
            language=detected_language or analysis_request.language,
            analyses=analysis_request.analyses,
            model_id=analysis_request.model_id,
            user_id=current_user["id"],
            request_id=request_id
        )
        
        # Calculate process time
        process_time = time.time() - start_time
        
        # Record metrics in background
        background_tasks.add_task(
            metrics.record_pipeline_execution,
            pipeline_id="text_analysis",
            operation="analyze",
            duration=process_time,
            input_size=len(analysis_request.text),
            output_size=0,
            success=True,
            metadata={
                "language": analysis_result.get("language", analysis_request.language),
                "analyses": analysis_request.analyses,
                "model_id": analysis_result.get("model_id", "default")
            }
        )
        
        # Create result model
        result = TextAnalysisResult(
            text=analysis_request.text,
            language=analysis_result.get("language", analysis_request.language),
            sentiment=analysis_result.get("sentiment"),
            entities=analysis_result.get("entities"),
            topics=analysis_result.get("topics"),
            summary=analysis_result.get("summary"),
            word_count=analysis_result.get("word_count", 0),
            sentence_count=analysis_result.get("sentence_count", 0),
            process_time=process_time
        )
        
        # Create response
        response = BaseResponse(
            status=StatusEnum.SUCCESS,
            message="Text analysis completed successfully",
            data=result,
            metadata=MetadataModel(
                request_id=request_id,
                timestamp=time.time(),
                version=request.app.state.config.get("version", "1.0.0"),
                process_time=process_time,
                cached=False
            ),
            errors=None,
            pagination=None
        )
        
        # Store response in cache if caching is enabled
        if use_cache and hasattr(request.app.state, "route_cache") and 'analysis_cache' in locals() and 'cache_key' in locals():
            await analysis_cache.set(cache_key, response)
            logger.debug(f"Stored analysis result in cache with key {cache_key[:8]}...")
        
        return response
        
    except Exception as e:
        logger.error(f"Text analysis error: {str(e)}", exc_info=True)
        
        # Record error metrics in background
        background_tasks.add_task(
            metrics.record_pipeline_execution,
            pipeline_id="text_analysis",
            operation="analyze",
            duration=time.time() - start_time,
            input_size=len(analysis_request.text),
            output_size=0,
            success=False,
            metadata={
                "error": str(e),
                "language": analysis_request.language,
                "analyses": analysis_request.analyses
            }
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Text analysis error: {str(e)}"
        )


# ----- Summarization Endpoint -----

from typing import Optional

class SummarizationRequest(BaseModel):
    text: str
    language: str
    model_id: Optional[str] = None

@router.post(
    "/summarize",
    response_model=BaseResponse,
    status_code=status.HTTP_200_OK,
    summary="Summarize text",
    description="Generates a summary for the given text using a summarization-capable model."
)
async def summarize_text(
    request: Request,
    background_tasks: BackgroundTasks,
    analysis_request: SummarizationRequest = Body(...),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Summarize the given text using a multipurpose model.
    """
    start_time = time.time()
    request_id = str(uuid.uuid4())

    try:
        processor = request.app.state.processor
        if processor is None:
            raise HTTPException(status_code=503, detail="Processor not initialized")

        metrics = request.app.state.metrics
        audit_logger = request.app.state.audit_logger

        await audit_logger.log_api_request(
            endpoint="/pipeline/summarize",
            method="POST",
            user_id=current_user["id"],
            source_ip=request.client.host,
            request_id=request_id,
            request_params={
                "text_length": len(analysis_request.text),
                "language": analysis_request.language,
                "model_id": analysis_request.model_id
            }
        )

        summary_result = await processor.process_summarization(
            text=analysis_request.text,
            language=analysis_request.language,
            user_id=current_user["id"],
            request_id=request_id
        )

        process_time = time.time() - start_time

        background_tasks.add_task(
            metrics.record_pipeline_execution,
            pipeline_id="summarization",
            operation="summarize",
            duration=process_time,
            input_size=len(analysis_request.text),
            output_size=len(summary_result.get("summary", "")),
            success=True,
            metadata={
                "language": summary_result.get("language", analysis_request.language),
                "model_id": summary_result.get("model_id", "default")
            }
        )

        result = TextAnalysisResult(
            text=analysis_request.text,
            language=summary_result.get("language", analysis_request.language),
            summary=summary_result.get("summary", ""),
            word_count=summary_result.get("word_count", 0),
            sentence_count=summary_result.get("sentence_count", 0),
            process_time=process_time
        )

        return BaseResponse(
            status=StatusEnum.SUCCESS,
            message="Text summarization completed successfully",
            data=result,
            metadata=MetadataModel(
                request_id=request_id,
                timestamp=time.time(),
                version=request.app.state.config.get("version", "1.0.0"),
                process_time=process_time
            ),
            errors=None,
            pagination=None
        )

    except Exception as e:
        logger.error(f"Summarization error: {str(e)}", exc_info=True)

        background_tasks.add_task(
            metrics.record_pipeline_execution,
            pipeline_id="summarization",
            operation="summarize",
            duration=time.time() - start_time,
            input_size=len(analysis_request.text),
            output_size=0,
            success=False,
            metadata={"error": str(e)}
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Summarization error: {str(e)}"
        )