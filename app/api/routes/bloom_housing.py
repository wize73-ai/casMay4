"""
Bloom Housing API compatibility routes for CasaLingua

This module provides API endpoints compatible with Bloom Housing's translation
and language processing standards, ensuring seamless integration.

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
import asyncio
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from fastapi import Request, status, File, UploadFile, Form, Body
from pydantic import BaseModel, Field, validator

from app.api.middleware.auth import get_current_user
from app.utils.logging import get_logger
from app.api.schemas.bloom_housing import (
    BloomTranslationRequest, BloomTranslationResponse,
    BloomLanguageDetectionRequest, BloomLanguageDetectionResponse,
    BloomTextAnalysisRequest, BloomTextAnalysisResponse,
    BloomDocumentTranslationRequest, BloomDocumentTranslationResponse
)

logger = get_logger(__name__)

# Create router with Bloom Housing compatible prefix
router = APIRouter(prefix="/bloom-housing", tags=["Bloom Housing"])

@router.post(
    "/translate",
    response_model=BloomTranslationResponse,
    status_code=status.HTTP_200_OK,
    summary="Translate text (Bloom Housing format)",
    description="Translates text using Bloom Housing compatible format."
)
async def bloom_translate_text(
    request: Request,
    background_tasks: BackgroundTasks,
    translation_request: BloomTranslationRequest = Body(...),
    cache: bool = Query(True, description="Whether to use request-level caching"),
    verify: bool = Query(False, description="Whether to verify translation quality"),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Translate text using Bloom Housing compatible request and response formats.
    
    This endpoint wraps the standard CasaLingua translation endpoint but converts
    to and from the Bloom Housing API format.
    """
    # Convert to CasaLingua format
    casa_request = translation_request.to_casa_lingua_format()
    
    # Invoke the main translation endpoint using internal API call
    # First, get the pipeline router
    from app.api.routes.pipeline import translate_text
    
    # Create a TranslationRequest using Pydantic model
    from app.api.schemas.translation import TranslationRequest
    casa_translation_request = TranslationRequest(
        text=casa_request["text"],
        source_language=casa_request["source_language"],
        target_language=casa_request["target_language"],
        preserve_formatting=casa_request["preserve_formatting"],
        domain=casa_request.get("domain", "housing"),
        glossary_id=casa_request.get("glossary_id"),
        model_id=casa_request.get("model_id"),
        verify=verify
    )
    
    # Call the internal endpoint
    casa_response = await translate_text(
        request=request,
        background_tasks=background_tasks,
        translation_request=casa_translation_request,
        verification=verify,
        current_user=current_user,
        use_cache=cache
    )
    
    # Convert response to Bloom Housing format
    bloom_response = BloomTranslationResponse.from_casa_lingua_response(casa_response)
    bloom_response.projectId = translation_request.projectId
    
    return bloom_response

@router.post(
    "/detect-language",
    response_model=BloomLanguageDetectionResponse,
    status_code=status.HTTP_200_OK,
    summary="Detect language (Bloom Housing format)",
    description="Detects language using Bloom Housing compatible format."
)
async def bloom_detect_language(
    request: Request,
    background_tasks: BackgroundTasks,
    detection_request: BloomLanguageDetectionRequest = Body(...),
    cache: bool = Query(True, description="Whether to use request-level caching"),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Detect language using Bloom Housing compatible request and response formats.
    
    This endpoint wraps the standard CasaLingua language detection endpoint but
    converts to and from the Bloom Housing API format.
    """
    # Convert to CasaLingua format
    casa_request = detection_request.to_casa_lingua_format()
    
    # Invoke the main detection endpoint using internal API call
    from app.api.routes.pipeline import detect_language
    
    # Create a LanguageDetectionRequest using Pydantic model
    from app.api.schemas.language import LanguageDetectionRequest
    casa_detection_request = LanguageDetectionRequest(
        text=casa_request["text"],
        detailed=casa_request["detailed"],
        model_id=None  # Let CasaLingua choose best model
    )
    
    # Call the internal endpoint
    casa_response = await detect_language(
        request=request,
        background_tasks=background_tasks,
        detection_request=casa_detection_request,
        current_user=current_user
    )
    
    # Convert response to Bloom Housing format
    bloom_response = BloomLanguageDetectionResponse.from_casa_lingua_response(casa_response)
    
    return bloom_response

@router.post(
    "/analyze",
    response_model=BloomTextAnalysisResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyze text (Bloom Housing format)",
    description="Analyzes text using Bloom Housing compatible format."
)
async def bloom_analyze_text(
    request: Request,
    background_tasks: BackgroundTasks,
    analysis_request: BloomTextAnalysisRequest = Body(...),
    cache: bool = Query(True, description="Whether to use request-level caching"),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Analyze text using Bloom Housing compatible request and response formats.
    
    This endpoint wraps the standard CasaLingua text analysis endpoint but
    converts to and from the Bloom Housing API format.
    """
    # Convert to CasaLingua format
    casa_request = analysis_request.to_casa_lingua_format()
    
    # Invoke the main analysis endpoint using internal API call
    from app.api.routes.pipeline import analyze_text
    
    # Create a TextAnalysisRequest using Pydantic model
    from app.api.schemas.analysis import TextAnalysisRequest
    casa_analysis_request = TextAnalysisRequest(
        text=casa_request["text"],
        language=casa_request["language"] or "auto",
        analyses=casa_request["analyses"],
        model_id=casa_request.get("model_id")
    )
    
    # Call the internal endpoint
    casa_response = await analyze_text(
        request=request,
        background_tasks=background_tasks,
        analysis_request=casa_analysis_request,
        current_user=current_user,
        use_cache=cache
    )
    
    # Convert response to Bloom Housing format
    bloom_response = BloomTextAnalysisResponse.from_casa_lingua_response(casa_response)
    
    return bloom_response

@router.post(
    "/translate-document",
    response_model=BloomDocumentTranslationResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Translate document (Bloom Housing format)",
    description="Uploads and translates a document using Bloom Housing compatible format."
)
async def bloom_translate_document(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    source_language: str = Form(...),
    target_language: str = Form(...),
    preserve_layout: bool = Form(True),
    project_id: Optional[str] = Form(None),
    callback_url: Optional[str] = Form(None),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Upload and translate a document using Bloom Housing compatible format.
    
    This endpoint wraps the standard CasaLingua document translation endpoint but
    converts to and from the Bloom Housing API format.
    """
    # Create a complete Bloom request object
    bloom_request = BloomDocumentTranslationRequest(
        sourceLanguage=source_language,
        targetLanguage=target_language,
        preserveLayout=preserve_layout,
        projectId=project_id,
        callbackUrl=callback_url
    )
    
    # Convert to CasaLingua format
    casa_request = bloom_request.to_casa_lingua_format()
    
    # Invoke the main document translation endpoint using internal API call
    from app.api.routes.pipeline import translate_document
    
    # Call the internal endpoint
    casa_response = await translate_document(
        request=request,
        background_tasks=background_tasks,
        file=file,
        source_language=casa_request["source_language"],
        target_language=casa_request["target_language"],
        model_id=None,  # Let CasaLingua choose best model
        glossary_id=casa_request["glossary_id"],
        preserve_layout=casa_request["preserve_layout"],
        callback_url=casa_request["callback_url"],
        current_user=current_user
    )
    
    # Convert response to Bloom Housing format
    bloom_response = BloomDocumentTranslationResponse.from_casa_lingua_response(casa_response)
    
    return bloom_response

@router.get(
    "/document-status/{document_id}",
    response_model=BloomDocumentTranslationResponse,
    status_code=status.HTTP_200_OK,
    summary="Get document translation status (Bloom Housing format)",
    description="Retrieves the status of a document translation task using Bloom Housing compatible format."
)
async def bloom_document_status(
    request: Request,
    document_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Get the status of a document translation task using Bloom Housing compatible format.
    
    This endpoint wraps the standard CasaLingua document status endpoint but
    converts to and from the Bloom Housing API format.
    """
    # Invoke the main document status endpoint using internal API call
    from app.api.routes.pipeline import get_document_translation_status
    
    # Call the internal endpoint
    casa_response = await get_document_translation_status(
        request=request,
        task_id=document_id,
        current_user=current_user
    )
    
    # Convert response to Bloom Housing format
    bloom_response = BloomDocumentTranslationResponse.from_casa_lingua_response(casa_response)
    
    return bloom_response