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
    current_user: Dict[str, Any] = Depends(get_current_user)
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
            model_id=getattr(translation_request, "model_id", None),
            glossary_id=translation_request.glossary_id,
            preserve_formatting=translation_request.preserve_formatting,
            formality=translation_request.formality,
            verify=verification or translation_request.verify,
            user_id=current_user["id"],
            request_id=request_id
        )
        
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
        
        # Create result model
        result = TranslationResult(
            source_text=translation_request.text,
            translated_text=translation_result["translated_text"],
            source_language=translation_result.get("detected_language", translation_request.source_language),
            target_language=translation_request.target_language,
            confidence=translation_result.get("confidence", 0.0),
            model_id=translation_result.get("model_id", "default"),
            process_time=process_time,
            word_count=translation_result.get("word_count", 0),
            character_count=len(translation_request.text),
            detected_language=translation_result.get("detected_language") if translation_request.source_language == "auto" else None,
            verified=translation_result.get("verified", False),
            verification_score=translation_result.get("verification_score", None)
        )
        
        # Create response
        response = BaseResponse(
            status=StatusEnum.SUCCESS,
            message="Translation completed successfully",
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

@router.post(
    "/translate/batch",
    response_model=BaseResponse[List[TranslationResult]],
    status_code=status.HTTP_200_OK,
    summary="Batch translate texts",
    description="Translates multiple texts in a single request."
)
async def batch_translate(
    request: Request,
    background_tasks: BackgroundTasks,
    batch_request: BatchTranslationRequest = Body(...),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Translate multiple texts in a single request.
    
    This endpoint processes multiple translation requests simultaneously,
    improving efficiency for batch translation needs.
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
            endpoint="/pipeline/translate/batch",
            method="POST",
            user_id=current_user["id"],
            source_ip=request.client.host,
            request_id=request_id,
            request_params={
                "source_language": batch_request.source_language,
                "target_language": batch_request.target_language,
                "text_count": len(batch_request.texts),
                "model_id": batch_request.model_id
            }
        )
        
        # Process batch translation
        batch_results = await processor.process_batch_translation(
            texts=batch_request.texts,
            source_language=batch_request.source_language,
            target_language=batch_request.target_language,
            model_id=batch_request.model_id,
            glossary_id=batch_request.glossary_id,
            preserve_formatting=batch_request.preserve_formatting,
            user_id=current_user["id"],
            request_id=request_id
        )
        
        # Calculate process time
        process_time = time.time() - start_time
        
        # Prepare results
        results = []
        total_input_size = sum(len(text) for text in batch_request.texts)
        total_output_size = sum(len(result["translated_text"]) for result in batch_results)
        
        for i, result in enumerate(batch_results):
            results.append(TranslationResult(
                source_text=batch_request.texts[i],
                translated_text=result["translated_text"],
                source_language=result.get("detected_language", batch_request.source_language),
                target_language=batch_request.target_language,
                confidence=result.get("confidence", 0.0),
                model_id=result.get("model_id", "default"),
                process_time=result.get("process_time", 0.0),
                word_count=result.get("word_count", 0),
                character_count=len(batch_request.texts[i]),
                detected_language=result.get("detected_language") if batch_request.source_language == "auto" else None,
                verified=False,
                verification_score=None
            ))
            
        # Record metrics in background
        background_tasks.add_task(
            metrics.record_pipeline_execution,
            pipeline_id="translation",
            operation="batch_translate",
            duration=process_time,
            input_size=total_input_size,
            output_size=total_output_size,
            success=True,
            metadata={
                "source_language": batch_request.source_language,
                "target_language": batch_request.target_language,
                "batch_size": len(batch_request.texts)
            }
        )
        
        # Create response
        response = BaseResponse(
            status=StatusEnum.SUCCESS,
            message=f"Batch translation of {len(batch_request.texts)} texts completed successfully",
            data=results,
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
        logger.error(f"Batch translation error: {str(e)}", exc_info=True)
        
        # Record error metrics in background
        background_tasks.add_task(
            metrics.record_pipeline_execution,
            pipeline_id="translation",
            operation="batch_translate",
            duration=time.time() - start_time,
            input_size=sum(len(text) for text in batch_request.texts),
            output_size=0,
            success=False,
            metadata={
                "error": str(e),
                "source_language": batch_request.source_language,
                "target_language": batch_request.target_language,
                "batch_size": len(batch_request.texts)
            }
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch translation error: {str(e)}"
        )

# ----- Document Translation Endpoints -----

@router.post(
    "/translate/document",
    response_model=DocumentTranslationResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Translate document",
    description="Uploads and translates a document."
)
async def translate_document(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    source_language: str = Form(...),
    target_language: str = Form(...),
    model_id: Optional[str] = Form(None),
    glossary_id: Optional[str] = Form(None),
    preserve_layout: bool = Form(True),
    callback_url: Optional[str] = Form(None),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Upload and translate a document.
    
    This endpoint handles document translation requests and processes
    documents asynchronously, providing a task ID for status tracking.
    """
    start_time = time.time()
    request_id = str(uuid.uuid4())
    task_id = str(uuid.uuid4())
    
    try:
        # Get application components from state
        processor = request.app.state.processor
        if processor is None:
            raise HTTPException(status_code=503, detail="Processor not initialized")
        metrics = request.app.state.metrics
        audit_logger = request.app.state.audit_logger
        
        # Read file content
        file_content = await file.read()
        file_size = len(file_content)
        
        # Log request to audit log
        await audit_logger.log_api_request(
            endpoint="/pipeline/translate/document",
            method="POST",
            user_id=current_user["id"],
            source_ip=request.client.host,
            request_id=request_id,
            request_params={
                "filename": file.filename,
                "file_size": file_size,
                "source_language": source_language,
                "target_language": target_language,
                "model_id": model_id
            }
        )
        
        # Queue document for translation
        doc_request = DocumentTranslationRequest(
            document_id=task_id,  # Use task_id as document_id
            source_language=source_language,
            target_language=target_language,
            model_id=model_id,
            glossary_id=glossary_id,
            output_format=None,  # Use original format
            callback_url=callback_url,
            preserve_layout=preserve_layout,
            translate_tracked_changes=False,
            translate_comments=False
        )
        
        # Start processing in background
        background_tasks.add_task(
            processor.process_document_translation,
            document=file_content,
            filename=file.filename,
            request=doc_request,
            user_id=current_user["id"],
            task_id=task_id,
            request_id=request_id
        )
        
        # Prepare initial result
        result = DocumentTranslationResult(
            document_id=task_id,
            filename=file.filename,
            source_language=source_language,
            target_language=target_language,
            status="processing",
            progress=0.0,
            translated_filename=None,
            page_count=None,
            word_count=None,
            start_time=time.time(),
            end_time=None,
            process_time=None,
            download_url=None
        )
        
        # Record request metrics in background
        background_tasks.add_task(
            metrics.record_pipeline_execution,
            pipeline_id="document_translation",
            operation="start",
            duration=time.time() - start_time,
            input_size=file_size,
            output_size=0,
            success=True,
            metadata={
                "filename": file.filename,
                "source_language": source_language,
                "target_language": target_language,
                "task_id": task_id
            }
        )
        
        # Create response
        response = BaseResponse(
            status=StatusEnum.SUCCESS,
            message="Document translation started",
            data=result,
            metadata=MetadataModel(
                request_id=request_id,
                timestamp=time.time(),
                version=request.app.state.config.get("version", "1.0.0"),
                process_time=time.time() - start_time
            ),
            errors=None,
            pagination=None
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Document translation error: {str(e)}", exc_info=True)
        
        # Record error metrics in background
        background_tasks.add_task(
            metrics.record_pipeline_execution,
            pipeline_id="document_translation",
            operation="start",
            duration=time.time() - start_time,
            input_size=len(await file.read()),
            output_size=0,
            success=False,
            metadata={
                "error": str(e),
                "filename": file.filename,
                "source_language": source_language,
                "target_language": target_language
            }
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document translation error: {str(e)}"
        )

@router.get(
    "/translate/document/{task_id}",
    response_model=DocumentTranslationResponse,
    status_code=status.HTTP_200_OK,
    summary="Get document translation status",
    description="Retrieves the status of a document translation task."
)
async def get_document_translation_status(
    request: Request,
    task_id: str = Path(..., description="Task identifier"),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Get status of a document translation task.
    
    This endpoint allows clients to check the status of an ongoing
    document translation task and retrieve results when complete.
    """
    start_time = time.time()
    
    try:
        # Get application components from state
        processor = request.app.state.processor
        if processor is None:
            raise HTTPException(status_code=503, detail="Processor not initialized")
        
        # Get task status
        task_status = await processor.get_task_status(task_id)
        
        if not task_status:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Translation task not found: {task_id}"
            )
            
        # Prepare result
        result = DocumentTranslationResult(
            document_id=task_id,
            filename=task_status.get("filename", ""),
            source_language=task_status.get("source_language", ""),
            target_language=task_status.get("target_language", ""),
            status=task_status.get("status", "unknown"),
            progress=task_status.get("progress", 0.0),
            translated_filename=task_status.get("translated_filename"),
            page_count=task_status.get("page_count"),
            word_count=task_status.get("word_count"),
            start_time=task_status.get("start_time"),
            end_time=task_status.get("end_time"),
            process_time=task_status.get("process_time"),
            download_url=task_status.get("download_url")
        )
        
        # Create response
        response = BaseResponse(
            status=StatusEnum.SUCCESS,
            message=f"Document translation status: {result.status}",
            data=result,
            metadata=MetadataModel(
                request_id=str(uuid.uuid4()),
                timestamp=time.time(),
                version=request.app.state.config.get("version", "1.0.0"),
                process_time=time.time() - start_time
            ),
            errors=None,
            pagination=None
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving document translation status: {str(e)}", exc_info=True)
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving translation status: {str(e)}"
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
    current_user: Dict[str, Any] = Depends(get_current_user)
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
        
        # Process text analysis
        analysis_result = await processor.analyze_text(
            text=analysis_request.text,
            language=analysis_request.language,
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
                process_time=process_time
            ),
            errors=None,
            pagination=None
        )
        
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

# ----- Verification Endpoints -----

@router.post(
    "/verify",
    response_model=VerificationResponse,
    status_code=status.HTTP_200_OK,
    summary="Verify translation",
    description="Verifies the quality of a translation."
)
async def verify_translation(
    request: Request,
    background_tasks: BackgroundTasks,
    source_text: str = Body(..., embed=True),
    translated_text: str = Body(..., embed=True),
    source_language: str = Body(..., embed=True),
    target_language: str = Body(..., embed=True),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Verify the quality of a translation.
    
    This endpoint evaluates translations for accuracy, completeness,
    and other quality metrics, providing detailed feedback.
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
            endpoint="/pipeline/verify",
            method="POST",
            user_id=current_user["id"],
            source_ip=request.client.host,
            request_id=request_id,
            request_params={
                "source_text_length": len(source_text),
                "translated_text_length": len(translated_text),
                "source_language": source_language,
                "target_language": target_language
            }
        )
        
        # Process verification
        verification_result = await processor.verify_translation(
            source_text=source_text,
            translated_text=translated_text,
            source_language=source_language,
            target_language=target_language,
            user_id=current_user["id"],
            request_id=request_id
        )
        
        # Calculate process time
        process_time = time.time() - start_time
        
        # Record metrics in background
        background_tasks.add_task(
            metrics.record_pipeline_execution,
            pipeline_id="verification",
            operation="verify",
            duration=process_time,
            input_size=len(source_text) + len(translated_text),
            output_size=0,
            success=True,
            metadata={
                "source_language": source_language,
                "target_language": target_language,
                "verified": verification_result["verified"],
                "score": verification_result["score"]
            }
        )
        
        # Create result model
        result = VerificationResult(
            verified=verification_result["verified"],
            score=verification_result["score"],
            confidence=verification_result["confidence"],
            issues=verification_result.get("issues", []),
            source_text=source_text,
            translated_text=translated_text,
            source_language=source_language,
            target_language=target_language,
            metrics=verification_result.get("metrics", {}),
            process_time=process_time
        )
        
        # Create response
        response = BaseResponse(
            status=StatusEnum.SUCCESS,
            message="Translation verification completed successfully",
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
        logger.error(f"Translation verification error: {str(e)}", exc_info=True)
        
        # Record error metrics in background
        background_tasks.add_task(
            metrics.record_pipeline_execution,
            pipeline_id="verification",
            operation="verify",
            duration=time.time() - start_time,
            input_size=len(source_text) + len(translated_text),
            output_size=0,
            success=False,
            metadata={
                "error": str(e),
                "source_language": source_language,
                "target_language": target_language
            }
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Translation verification error: {str(e)}"
        )

# ----- Task Status Endpoints -----

@router.get(
    "/tasks/{task_id}",
    response_model=QueueStatusResponse,
    status_code=status.HTTP_200_OK,
    summary="Get task status",
    description="Retrieves the status of an asynchronous processing task."
)
async def get_task_status(
    request: Request,
    task_id: str = Path(..., description="Task identifier"),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Get status of an asynchronous processing task.
    
    This endpoint retrieves the current status of any asynchronous
    pipeline task, including progress information and results.
    """
    start_time = time.time()
    
    try:
        # Get application components from state
        processor = request.app.state.processor
        if processor is None:
            raise HTTPException(status_code=503, detail="Processor not initialized")
        
        # Get task status
        task_status = await processor.get_task_status(task_id)
        
        if not task_status:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Task not found: {task_id}"
            )
            
        # Prepare result
        result = QueueStatus(
            queue_id=task_status.get("queue_id", "default"),
            task_id=task_id,
            status=task_status.get("status", "unknown"),
            position=task_status.get("position"),
            estimated_start_time=task_status.get("estimated_start_time"),
            estimated_completion_time=task_status.get("estimated_completion_time"),
            progress=task_status.get("progress", 0.0),
            result_url=task_status.get("result_url")
        )
        
        # Create response
        response = BaseResponse(
            status=StatusEnum.SUCCESS,
            message=f"Task status: {result.status}",
            data=result,
            metadata=MetadataModel(
                request_id=str(uuid.uuid4()),
                timestamp=time.time(),
                version=request.app.state.config.get("version", "1.0.0"),
                process_time=time.time() - start_time
            ),
            errors=None,
            pagination=None
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving task status: {str(e)}", exc_info=True)
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving task status: {str(e)}"
        )