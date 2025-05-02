"""
Admin API Routes for CasaLingua

This module defines administrative API endpoints for managing the
CasaLingua language processing platform, including system configuration,
model management, metrics, and user administration.
"""

import time
import json
import uuid
import logging
from typing import Dict, List, Any, Optional, Union
from fastapi import (
    APIRouter, Depends, HTTPException, BackgroundTasks, 
    Query, Path, Body, Request, status, Security
)
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field, validator
from datetime import datetime, timedelta

from app.api.schemas.requests.requests_system import SystemConfigUpdateRequest
from app.api.schemas.requests.requests_model import ApiKeyCreateRequest, ModelLoadRequest
from app.api.schemas.requests.responses import (
    BaseResponse,
    StatusEnum,
    MetadataModel,
    ErrorDetail,
    ErrorCode,
    ModelInfoResponse,
    ModelInfo,
    ModelListResponse,
    LanguageSupportListResponse,
    LanguageSupportInfo,
    AdminMetricsResponse,
    AdminMetrics,
    UserInfo
)
from app.utils.auth import verify_admin_role, get_current_user, verify_api_key
from app.utils.logging import get_logger

logger = get_logger(__name__)

# Create router with admin tag
router = APIRouter(tags=["Admin"])

# ----- System Information Endpoints -----

@router.get(
    "/system/info",
    response_model=BaseResponse[Dict[str, Any]],
    status_code=status.HTTP_200_OK,
    summary="Get system information",
    description="Retrieves general system information and configuration."
)
async def get_system_info(
    request: Request,
    current_user: Dict[str, Any] = Depends(verify_admin_role)
):
    """
    Get system information and configuration.
    
    This endpoint provides details about the system configuration,
    environment, and operational parameters.
    """
    start_time = time.time()
    
    try:
        # Get application components from state
        config = request.app.state.config
        hardware_info = request.app.state.hardware_info
        
        # Get basic system info
        system_info = {
            "version": config.get("version", "1.0.0"),
            "environment": config.get("environment", "development"),
            "start_time": datetime.fromtimestamp(request.app.state.start_time).isoformat(),
            "uptime_seconds": time.time() - request.app.state.start_time,
            "python_version": hardware_info.get("python_version", "unknown"),
            "server_host": config.get("server_host", "0.0.0.0"),
            "server_port": config.get("server_port", 8000),
            "workers": config.get("worker_threads", 4),
            "debug_mode": config.get("debug", False),
            "log_level": config.get("log_level", "INFO"),
            "hardware": {
                "cpu_count": hardware_info.get("cpu_count", 0),
                "gpu_count": hardware_info.get("gpu_count", 0),
                "memory_gb": hardware_info.get("system", {}).get("memory", 0),
                "gpus": hardware_info.get("gpus", [])
            }
        }
        
        # Create response
        response = BaseResponse(
            status=StatusEnum.SUCCESS,
            message="System information retrieved successfully",
            data=system_info,
            metadata=MetadataModel(
                request_id=str(uuid.uuid4()),
                timestamp=time.time(),
                version=config.get("version", "1.0.0"),
                process_time=time.time() - start_time
            ),
            errors=None,
            pagination=None
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error retrieving system info: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving system information: {str(e)}"
        )

@router.get(
    "/system/config",
    response_model=BaseResponse[Dict[str, Any]],
    status_code=status.HTTP_200_OK,
    summary="Get system configuration",
    description="Retrieves the current system configuration."
)
async def get_system_config(
    request: Request,
    include_secrets: bool = Query(False, description="Whether to include secret values"),
    current_user: Dict[str, Any] = Depends(verify_admin_role)
):
    """
    Get system configuration.
    
    This endpoint provides the current system configuration settings,
    optionally including secret values for authenticated administrators.
    """
    start_time = time.time()
    
    try:
        # Get application components from state
        config = request.app.state.config
        
        # Create a copy of the config
        config_copy = config.copy()
        
        # Remove sensitive information if not explicitly requested
        if not include_secrets:
            sensitive_keys = [
                "database_password", 
                "api_keys", 
                "secret_key", 
                "jwt_secret",
                "encryption_key",
                "auth_secret",
                "credentials"
            ]
            
            for key in sensitive_keys:
                if key in config_copy:
                    config_copy[key] = "[REDACTED]"
                    
            # Also check nested dictionaries
            for section_key, section in config_copy.items():
                if isinstance(section, dict):
                    for key in sensitive_keys:
                        if key in section:
                            config_copy[section_key][key] = "[REDACTED]"
                            
        # Create response
        response = BaseResponse(
            status=StatusEnum.SUCCESS,
            message="System configuration retrieved successfully",
            data=config_copy,
            metadata=MetadataModel(
                request_id=str(uuid.uuid4()),
                timestamp=time.time(),
                version=config.get("version", "1.0.0"),
                process_time=time.time() - start_time
            ),
            errors=None,
            pagination=None
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error retrieving system config: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving system configuration: {str(e)}"
        )

@router.post(
    "/system/config",
    response_model=BaseResponse[Dict[str, Any]],
    status_code=status.HTTP_200_OK,
    summary="Update system configuration",
    description="Updates a specific system configuration value."
)
async def update_system_config(
    request: Request,
    background_tasks: BackgroundTasks,
    config_update: SystemConfigUpdateRequest = Body(...),
    current_user: Dict[str, Any] = Depends(verify_admin_role)
):
    """
    Update system configuration.
    
    This endpoint allows administrators to update specific configuration
    values, with changes logged for audit purposes.
    """
    start_time = time.time()
    
    try:
        # Get application components from state
        config = request.app.state.config
        audit_logger = request.app.state.audit_logger
        
        # Check if key exists
        current_value = None
        key_parts = config_update.key.split(".")
        
        if len(key_parts) == 1:
            if key_parts[0] in config:
                current_value = config[key_parts[0]]
            else:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Configuration key not found: {config_update.key}"
                )
        elif len(key_parts) == 2:
            if key_parts[0] in config and isinstance(config[key_parts[0]], dict):
                if key_parts[1] in config[key_parts[0]]:
                    current_value = config[key_parts[0]][key_parts[1]]
                else:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"Configuration key not found: {config_update.key}"
                    )
            else:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Configuration section not found: {key_parts[0]}"
                )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid configuration key format. Use 'section.key' or 'key'"
            )
            
        # Check if key is protected
        protected_keys = [
            "version", 
            "environment", 
            "database_password", 
            "secret_key", 
            "jwt_secret",
            "encryption_key",
            "auth_secret"
        ]
        
        if key_parts[-1] in protected_keys:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Configuration key is protected: {config_update.key}"
            )
            
        # Update config
        if len(key_parts) == 1:
            config[key_parts[0]] = config_update.value
        else:
            config[key_parts[0]][key_parts[1]] = config_update.value
            
        # Log change to audit log
        background_tasks.add_task(
            audit_logger.log_config_change,
            user_id=current_user["id"],
            component="system",
            setting=config_update.key,
            old_value=current_value,
            new_value=config_update.value,
            reason=config_update.reason
        )
        
        # Prepare response data
        update_result = {
            "key": config_update.key,
            "old_value": current_value,
            "new_value": config_update.value,
            "updated_at": datetime.now().isoformat(),
            "updated_by": current_user["username"]
        }
        
        # Create response
        response = BaseResponse(
            status=StatusEnum.SUCCESS,
            message=f"Configuration '{config_update.key}' updated successfully",
            data=update_result,
            metadata=MetadataModel(
                request_id=str(uuid.uuid4()),
                timestamp=time.time(),
                version=config.get("version", "1.0.0"),
                process_time=time.time() - start_time
            ),
            errors=None,
            pagination=None
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating system config: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating system configuration: {str(e)}"
        )

# ----- Model Management Endpoints -----

@router.get(
    "/models",
    response_model=ModelListResponse,
    status_code=status.HTTP_200_OK,
    summary="List available models",
    description="Lists all available language models."
)
async def list_models(
    request: Request,
    status: Optional[str] = Query(None, description="Filter by model status"),
    type: Optional[str] = Query(None, description="Filter by model type"),
    language: Optional[str] = Query(None, description="Filter by supported language"),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    List available models.
    
    This endpoint lists all available language models in the system,
    with optional filtering by status, type, and supported language.
    """
    start_time = time.time()
    
    try:
        # Get application components from state
        model_registry = request.app.state.model_registry
        model_manager = request.app.state.model_manager
        
        if not model_registry:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model registry not available"
            )
            
        # Get loaded models information
        manager_stats = await model_manager.get_manager_stats()
        loaded_models = set(manager_stats.get("active_models", []))
        
        # Find models based on criteria
        models = model_registry.find_models(
            task=None,
            language=language,
            requires_gpu=None
        )
        
        # Apply additional filters
        if status:
            if status == "loaded":
                models = [m for m in models if m["id"] in loaded_models]
            elif status == "unloaded":
                models = [m for m in models if m["id"] not in loaded_models]
                
        if type:
            models = [m for m in models if m.get("model_type") == type]
            
        # Convert to ModelInfo objects
        model_infos = []
        for model in models:
            model_id = model["id"]
            model_info = ModelInfo(
                id=model_id,
                name=model.get("name", model_id),
                version=model.get("version", "1.0.0"),
                type=model.get("model_type", "unknown"),
                languages=model.get("languages", []),
                capabilities=model.get("tasks", []),
                size=model.get("size_gb"),
                status="loaded" if model_id in loaded_models else "available",
                loaded=model_id in loaded_models,
                last_used=None,
                performance_metrics=None
            )
            
            # Add performance metrics if available
            if model_id in loaded_models and "performance_stats" in manager_stats:
                if model_id in manager_stats["performance_stats"]:
                    model_info.performance_metrics = manager_stats["performance_stats"][model_id]
                    
            model_infos.append(model_info)
            
        # Create response
        response = BaseResponse(
            status=StatusEnum.SUCCESS,
            message=f"Retrieved {len(model_infos)} models",
            data=model_infos,
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
        logger.error(f"Error listing models: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving models: {str(e)}"
        )

@router.get(
    "/models/{model_id}",
    response_model=ModelInfoResponse,
    status_code=status.HTTP_200_OK,
    summary="Get model details",
    description="Retrieves detailed information about a specific model."
)
async def get_model_details(
    request: Request,
    model_id: str = Path(..., description="Model identifier"),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Get model details.
    
    This endpoint provides detailed information about a specific
    language model, including its capabilities and status.
    """
    start_time = time.time()
    
    try:
        # Get application components from state
        model_registry = request.app.state.model_registry
        model_manager = request.app.state.model_manager
        
        if not model_registry:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model registry not available"
            )
            
        # Get model information
        model_info = model_registry.get_model_info(model_id)
        
        if not model_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model not found: {model_id}"
            )
            
        # Get loaded models information
        manager_stats = await model_manager.get_manager_stats()
        loaded_models = set(manager_stats.get("active_models", []))
        
        # Create ModelInfo
        model_detail = ModelInfo(
            id=model_id,
            name=model_info.get("name", model_id),
            version=model_info.get("version", "1.0.0"),
            type=model_info.get("model_type", "unknown"),
            languages=model_info.get("languages", []),
            capabilities=model_info.get("tasks", []),
            size=model_info.get("size_gb"),
            status="loaded" if model_id in loaded_models else "available",
            loaded=model_id in loaded_models,
            last_used=None,
            performance_metrics=None
        )
        
        # Add performance metrics if available
        if model_id in loaded_models and "performance_stats" in manager_stats:
            if model_id in manager_stats["performance_stats"]:
                model_detail.performance_metrics = manager_stats["performance_stats"][model_id]
                
        # Add usage statistics if available
        if "model_usage_stats" in manager_stats and model_id in manager_stats["model_usage_stats"]:
            usage_stats = manager_stats["model_usage_stats"][model_id]
            if "last_used_at" in usage_stats:
                model_detail.last_used = datetime.fromtimestamp(usage_stats["last_used_at"])
                
        # Create response
        response = BaseResponse(
            status=StatusEnum.SUCCESS,
            message=f"Retrieved model: {model_id}",
            data=model_detail,
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
        logger.error(f"Error retrieving model details: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving model details: {str(e)}"
        )

@router.post(
    "/models/{model_id}/load",
    response_model=BaseResponse[ModelInfo],
    status_code=status.HTTP_200_OK,
    summary="Load model",
    description="Loads a model into memory for use."
)
async def load_model(
    request: Request,
    background_tasks: BackgroundTasks,
    model_id: str = Path(..., description="Model identifier"),
    load_request: Optional[ModelLoadRequest] = Body(None),
    current_user: Dict[str, Any] = Depends(verify_admin_role)
):
    """
    Load a model into memory.
    
    This endpoint loads a specific model into memory for use in the system,
    with optional device and memory mode configuration.
    """
    start_time = time.time()
    
    try:
        # Get application components from state
        model_manager = request.app.state.model_manager
        model_registry = request.app.state.model_registry
        audit_logger = request.app.state.audit_logger
        
        if not model_registry:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model registry not available"
            )
            
        # Check if model exists
        model_info = model_registry.get_model_info(model_id)
        
        if not model_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model not found: {model_id}"
            )
            
        # Prepare load options
        device = None
        low_memory_mode = None
        
        if load_request:
            device = load_request.device
            low_memory_mode = load_request.low_memory_mode
            
        # Load model
        try:
            await model_manager.get_model(model_id)
            
            # Log to audit logger
            background_tasks.add_task(
                audit_logger.log_model_operation,
                model_id=model_id,
                operation="load",
                user_id=current_user["id"],
                input_metadata={
                    "device": device,
                    "low_memory_mode": low_memory_mode
                },
                status="success"
            )
            
            # Get updated model information
            manager_stats = await model_manager.get_manager_stats()
            loaded_models = set(manager_stats.get("active_models", []))
            
            # Create ModelInfo response
            model_detail = ModelInfo(
                id=model_id,
                name=model_info.get("name", model_id),
                version=model_info.get("version", "1.0.0"),
                type=model_info.get("model_type", "unknown"),
                languages=model_info.get("languages", []),
                capabilities=model_info.get("tasks", []),
                size=model_info.get("size_gb"),
                status="loaded" if model_id in loaded_models else "available",
                loaded=model_id in loaded_models,
                last_used=datetime.now(),
                performance_metrics=None
            )
            
            # Create response
            response = BaseResponse(
                status=StatusEnum.SUCCESS,
                message=f"Model {model_id} loaded successfully",
                data=model_detail,
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
            
        except Exception as e:
            # Log error to audit logger
            background_tasks.add_task(
                audit_logger.log_model_operation,
                model_id=model_id,
                operation="load",
                user_id=current_user["id"],
                input_metadata={
                    "device": device,
                    "low_memory_mode": low_memory_mode
                },
                status="failure",
                error_message=str(e)
            )
            
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error loading model: {str(e)}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error loading model: {str(e)}"
        )

@router.post(
    "/models/{model_id}/unload",
    response_model=BaseResponse[Dict[str, Any]],
    status_code=status.HTTP_200_OK,
    summary="Unload model",
    description="Unloads a model from memory."
)
async def unload_model(
    request: Request,
    background_tasks: BackgroundTasks,
    model_id: str = Path(..., description="Model identifier"),
    current_user: Dict[str, Any] = Depends(verify_admin_role)
):
    """
    Unload a model from memory.
    
    This endpoint unloads a specific model from memory to free resources.
    """
    start_time = time.time()
    
    try:
        # Get application components from state
        model_manager = request.app.state.model_manager
        audit_logger = request.app.state.audit_logger
        
        # Unload model
        success = await model_manager.unload_model(model_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model not loaded or not found: {model_id}"
            )
            
        # Log to audit logger
        background_tasks.add_task(
            audit_logger.log_model_operation,
            model_id=model_id,
            operation="unload",
            user_id=current_user["id"],
            status="success"
        )
        
        # Create response
        response = BaseResponse(
            status=StatusEnum.SUCCESS,
            message=f"Model {model_id} unloaded successfully",
            data={"model_id": model_id, "unloaded": True},
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
        logger.error(f"Error unloading model: {str(e)}", exc_info=True)
        
        # Log error to audit logger
        background_tasks.add_task(
            audit_logger.log_model_operation,
            model_id=model_id,
            operation="unload",
            user_id=current_user["id"],
            status="failure",
            error_message=str(e)
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error unloading model: {str(e)}"
        )

@router.get(
    "/languages",
    response_model=LanguageSupportListResponse,
    status_code=status.HTTP_200_OK,
    summary="List supported languages",
    description="Lists all supported languages and their capabilities."
)
async def list_supported_languages(
    request: Request,
    operation: Optional[str] = Query(None, description="Filter by supported operation"),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    List supported languages.
    
    This endpoint lists all languages supported by the system,
    along with their available operations and models.
    """
    start_time = time.time()
    
    try:
        # Get application components from state
        model_registry = request.app.state.model_registry
        
        if not model_registry:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model registry not available"
            )
            
        # Get all supported languages
        languages = []
        
        if operation:
            # Filter by operation
            language_codes = model_registry.get_supported_languages(operation)
            
            for lang_code in language_codes:
                # Get models supporting this language and operation
                models = model_registry.find_models(task=operation, language=lang_code)
                model_ids = [model["id"] for model in models]
                
                # Get translation pairs for this language
                translation_pairs = []
                for target_lang in language_codes:
                    if target_lang != lang_code:
                        if model_registry.get_best_model_for_task(
                            task="translation", language=f"{lang_code}-{target_lang}"
                        ):
                            translation_pairs.append({
                                "source": lang_code,
                                "target": target_lang
                            })
                            
                # Create language support info
                languages.append(LanguageSupportInfo(
                    language_code=lang_code,
                    language_name=get_language_name(lang_code),
                    supported_operations=[operation],
                    translation_pairs=translation_pairs if operation == "translation" else None,
                    models=model_ids
                ))
        else:
            # Get all languages
            language_codes = model_registry.get_supported_languages()
            
            for lang_code in language_codes:
                # Get operations supported for this language
                operations = model_registry.get_supported_tasks(lang_code)
                
                # Get models supporting this language
                models = model_registry.find_models(language=lang_code)
                model_ids = [model["id"] for model in models]
                
                # Get translation pairs if translation is supported
                translation_pairs = []
                if "translation" in operations:
                    for target_lang in language_codes:
                        if target_lang != lang_code:
                            if model_registry.get_best_model_for_task(
                                task="translation", language=f"{lang_code}-{target_lang}"
                            ):
                                translation_pairs.append({
                                    "source": lang_code,
                                    "target": target_lang
                                })
                                
                # Create language support info
                languages.append(LanguageSupportInfo(
                    language_code=lang_code,
                    language_name=get_language_name(lang_code),
                    supported_operations=operations,
                    translation_pairs=translation_pairs if translation_pairs else None,
                    models=model_ids
                ))
                
        # Create response
        response = BaseResponse(
            status=StatusEnum.SUCCESS,
            message=f"Retrieved {len(languages)} supported languages",
            data=languages,
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
        logger.error(f"Error listing supported languages: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving supported languages: {str(e)}"
        )

# ----- Metrics and Logging Endpoints -----

@router.get(
    "/metrics",
    response_model=AdminMetricsResponse,
    status_code=status.HTTP_200_OK,
    summary="Get system metrics",
    description="Retrieves system performance metrics."
)
async def get_system_metrics(
    request: Request,
    current_user: Dict[str, Any] = Depends(verify_admin_role)
):
    """
    Get system metrics.
    
    This endpoint provides comprehensive metrics about system performance,
    including API usage, model performance, and resource utilization.
    """
    start_time = time.time()
    
    try:
        # Get application components from state
        metrics = request.app.state.metrics
        
        if not metrics:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Metrics collector not available"
            )
            
        # Get all metrics
        all_metrics = metrics.get_all_metrics()
        
        # Create AdminMetrics
        admin_metrics = AdminMetrics(
            system_metrics=all_metrics.get("system", {}),
            request_metrics=all_metrics.get("api", {}),
            model_metrics=all_metrics.get("model", {}),
            language_metrics=all_metrics.get("language", {}),
            user_metrics={"total_users": 0}  # Placeholder - implement user metrics as needed
        )
        
        # Create response
        response = BaseResponse(
            status=StatusEnum.SUCCESS,
            message="System metrics retrieved successfully",
            data=admin_metrics,
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
        logger.error(f"Error retrieving system metrics: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving system metrics: {str(e)}"
        )

@router.get(
    "/metrics/time-series/{series_name}",
    response_model=BaseResponse[List[Dict[str, Any]]],
    status_code=status.HTTP_200_OK,
    summary="Get time series metrics",
    description="Retrieves time series metrics for visualization."
)
async def get_time_series_metrics(
    request: Request,
    series_name: str = Path(..., description="Time series name"),
    start_time: Optional[float] = Query(None, description="Start time (UNIX timestamp)"),
    end_time: Optional[float] = Query(None, description="End time (UNIX timestamp)"),
    limit: Optional[int] = Query(100, description="Maximum number of data points"),
    current_user: Dict[str, Any] = Depends(verify_admin_role)
):
    """
    Get time series metrics.
    
    This endpoint provides time series metrics for visualization and analysis.
    """
    request_start_time = time.time()
    
    try:
        # Get application components from state
        metrics = request.app.state.metrics
        
        if not metrics:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Metrics collector not available"
            )
            
        # Get time series data
        time_series = metrics.get_time_series(
            series_name=series_name,
            start_time=start_time,
            end_time=end_time,
            limit=limit
        )
        
        # Create response
        response = BaseResponse(
            status=StatusEnum.SUCCESS,
            message=f"Retrieved {len(time_series)} data points for {series_name}",
            data=time_series,
            metadata=MetadataModel(
                request_id=str(uuid.uuid4()),
                timestamp=time.time(),
                version=request.app.state.config.get("version", "1.0.0"),
                process_time=time.time() - request_start_time
            ),
            errors=None,
            pagination=None
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving time series metrics: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving time series metrics: {str(e)}"
        )

@router.get(
    "/logs",
    response_model=BaseResponse[List[Dict[str, Any]]],
    status_code=status.HTTP_200_OK,
    summary="Search audit logs",
    description="Searches audit logs based on criteria."
)
async def search_audit_logs(
    request: Request,
    start_date: Optional[str] = Query(None, description="Start date (ISO format)"),
    end_date: Optional[str] = Query(None, description="End date (ISO format)"),
    log_type: Optional[str] = Query(None, description="Type of log entries"),
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    status: Optional[str] = Query(None, description="Filter by status"),
    correlation_id: Optional[str] = Query(None, description="Filter by correlation ID"),
    limit: int = Query(100, description="Maximum number of entries"),
    current_user: Dict[str, Any] = Depends(verify_admin_role)
):
    """
    Search audit logs.
    
    This endpoint allows administrators to search and filter audit logs
    for security, compliance, and troubleshooting purposes.
    """
    search_start_time = time.time()
    
    try:
        # Get application components from state
        audit_logger = request.app.state.audit_logger
        
        if not audit_logger:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Audit logger not available"
            )
            
        # Convert date strings to datetime
        start_datetime = None
        end_datetime = None
        
        if start_date:
            start_datetime = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
            
        if end_date:
            end_datetime = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
            
        # Search logs
        logs = await audit_logger.search_logs(
            start_time=start_datetime,
            end_time=end_datetime,
            log_type=log_type,
            user_id=user_id,
            status=status,
            correlation_id=correlation_id,
            limit=limit
        )
        
        # Create response
        response = BaseResponse(
            status=StatusEnum.SUCCESS,
            message=f"Retrieved {len(logs)} audit log entries",
            data=logs,
            metadata=MetadataModel(
                request_id=str(uuid.uuid4()),
                timestamp=time.time(),
                version=request.app.state.config.get("version", "1.0.0"),
                process_time=time.time() - search_start_time
            ),
            errors=None,
            pagination=None
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error searching audit logs: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error searching audit logs: {str(e)}"
        )

@router.get(
    "/logs/export",
    status_code=status.HTTP_200_OK,
    summary="Export audit logs",
    description="Exports audit logs to a file for download."
)
async def export_audit_logs(
    request: Request,
    start_date: str = Query(..., description="Start date (ISO format)"),
    end_date: str = Query(..., description="End date (ISO format)"),
    log_type: Optional[str] = Query(None, description="Type of log entries"),
    format: str = Query("json", description="Export format (json or csv)"),
    current_user: Dict[str, Any] = Depends(verify_admin_role)
):
    """
    Export audit logs.
    
    This endpoint allows administrators to export audit logs to a file
    for offline analysis or archiving.
    """
    try:
        # Get application components from state
        audit_logger = request.app.state.audit_logger
        
        if not audit_logger:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Audit logger not available"
            )
            
        # Convert date strings to datetime
        start_datetime = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
        end_datetime = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
        
        # Search logs with high limit for export
        logs = await audit_logger.search_logs(
            start_time=start_datetime,
            end_time=end_datetime,
            log_type=log_type,
            limit=10000  # High limit for export
        )
        
        # Create export file
        export_filename = f"audit_logs_{start_date.split('T')[0]}_{end_date.split('T')[0]}"
        
        if format.lower() == "csv":
            # Convert to CSV
            import csv
            import io
            
            output = io.StringIO()
            
            # Get all possible fields from logs
            fields = set()
            for log in logs:
                fields.update(log.keys())
                
            # Sort fields for consistent output
            sorted_fields = sorted(list(fields))
            
            # Write CSV
            writer = csv.DictWriter(output, fieldnames=sorted_fields)
            writer.writeheader()
            
            for log in logs:
                writer.writerow(log)
                
            # Prepare response
            export_content = output.getvalue()
            media_type = "text/csv"
            export_filename += ".csv"
            
        else:
            # Default to JSON
            export_content = json.dumps(logs, indent=2)
            media_type = "application/json"
            export_filename += ".json"
            
        # Log export to audit log
        await audit_logger.log_data_access(
            data_type="audit_logs",
            action="export",
            user_id=current_user["id"],
            access_reason="Admin export",
            metadata={
                "start_date": start_date,
                "end_date": end_date,
                "log_type": log_type,
                "format": format,
                "record_count": len(logs)
            }
        )
        
        # Return file response
        return JSONResponse(
            content={"logs": logs},
            media_type=media_type,
            headers={
                "Content-Disposition": f"attachment; filename={export_filename}"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exporting audit logs: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error exporting audit logs: {str(e)}"
        )

# ----- API Key Management Endpoints -----

@router.post(
    "/api-keys",
    response_model=BaseResponse[Dict[str, Any]],
    status_code=status.HTTP_201_CREATED,
    summary="Create API key",
    description="Creates a new API key for application access."
)
async def create_api_key(
    request: Request,
    background_tasks: BackgroundTasks,
    key_request: ApiKeyCreateRequest = Body(...),
    current_user: Dict[str, Any] = Depends(verify_admin_role)
):
    """
    Create a new API key.
    
    This endpoint allows administrators to create new API keys
    for application access, with specified scopes and expiration.
    """
    start_time = time.time()
    
    try:
        # Get application components from state
        config = request.app.state.config
        audit_logger = request.app.state.audit_logger
        
        # Generate API key
        api_key = f"cslg_{uuid.uuid4().hex}"
        
        # Calculate expiration
        expires_in_days = key_request.expires_in_days or 365
        expiration_date = datetime.now() + timedelta(days=expires_in_days)
        
        # Create key data
        key_data = {
            "key": api_key,
            "name": key_request.name,
            "scopes": key_request.scopes,
            "created_by": current_user["id"],
            "created_at": datetime.now().isoformat(),
            "expires_at": expiration_date.isoformat(),
            "active": True
        }
        
        # Get current API keys
        api_keys = config.get("api_keys", {})
        
        # Add new key
        api_keys[api_key] = key_data
        config["api_keys"] = api_keys
        
        # Log to audit log
        background_tasks.add_task(
            audit_logger.log_security_event,
            event_type="api_key_created",
            severity="info",
            user_id=current_user["id"],
            details={
                "key_name": key_request.name,
                "scopes": key_request.scopes,
                "expires_in_days": expires_in_days
            }
        )
        
        # Prepare response data (exclude the full key from the stored data)
        response_data = key_data.copy()
        
        # Create response
        response = BaseResponse(
            status=StatusEnum.SUCCESS,
            message=f"API key '{key_request.name}' created successfully",
            data=response_data,
            metadata=MetadataModel(
                request_id=str(uuid.uuid4()),
                timestamp=time.time(),
                version=config.get("version", "1.0.0"),
                process_time=time.time() - start_time
            ),
            errors=None,
            pagination=None
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error creating API key: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating API key: {str(e)}"
        )

@router.get(
    "/api-keys",
    response_model=BaseResponse[List[Dict[str, Any]]],
    status_code=status.HTTP_200_OK,
    summary="List API keys",
    description="Lists all API keys."
)
async def list_api_keys(
    request: Request,
    include_inactive: bool = Query(False, description="Whether to include inactive keys"),
    current_user: Dict[str, Any] = Depends(verify_admin_role)
):
    """
    List API keys.
    
    This endpoint lists all API keys in the system, optionally
    including inactive keys.
    """
    start_time = time.time()
    
    try:
        # Get application components from state
        config = request.app.state.config
        
        # Get API keys
        api_keys = config.get("api_keys", {})
        
        # Filter inactive keys if requested
        key_list = []
        for key_id, key_data in api_keys.items():
            # Skip inactive keys if not requested
            if not include_inactive and not key_data.get("active", True):
                continue
                
            # Check expiration
            if "expires_at" in key_data:
                expires_at = datetime.fromisoformat(key_data["expires_at"].replace("Z", "+00:00"))
                if expires_at < datetime.now():
                    key_data["active"] = False
                    
            # Add to list without the actual key
            key_info = key_data.copy()
            key_info["key_id"] = key_id[:8]  # First 8 characters of key ID
            key_info.pop("key", None)  # Remove the actual key
            key_list.append(key_info)
            
        # Create response
        response = BaseResponse(
            status=StatusEnum.SUCCESS,
            message=f"Retrieved {len(key_list)} API keys",
            data=key_list,
            metadata=MetadataModel(
                request_id=str(uuid.uuid4()),
                timestamp=time.time(),
                version=config.get("version", "1.0.0"),
                process_time=time.time() - start_time
            ),
            errors=None,
            pagination=None
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error listing API keys: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing API keys: {str(e)}"
        )

@router.delete(
    "/api-keys/{key_id}",
    response_model=BaseResponse[Dict[str, Any]],
    status_code=status.HTTP_200_OK,
    summary="Revoke API key",
    description="Revokes an API key."
)
async def revoke_api_key(
    request: Request,
    background_tasks: BackgroundTasks,
    key_id: str = Path(..., description="API key identifier"),
    current_user: Dict[str, Any] = Depends(verify_admin_role)
):
    """
    Revoke an API key.
    
    This endpoint allows administrators to revoke API keys,
    preventing further use for authentication.
    """
    start_time = time.time()
    
    try:
        # Get application components from state
        config = request.app.state.config
        audit_logger = request.app.state.audit_logger
        
        # Get API keys
        api_keys = config.get("api_keys", {})
        
        # Find matching key
        found_key = None
        for key, key_data in api_keys.items():
            if key.startswith(key_id):
                found_key = key
                break
                
        if not found_key:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"API key not found: {key_id}"
            )
            
        # Revoke key
        api_keys[found_key]["active"] = False
        api_keys[found_key]["revoked_at"] = datetime.now().isoformat()
        api_keys[found_key]["revoked_by"] = current_user["id"]
        
        # Update config
        config["api_keys"] = api_keys
        
        # Log to audit log
        background_tasks.add_task(
            audit_logger.log_security_event,
            event_type="api_key_revoked",
            severity="info",
            user_id=current_user["id"],
            details={
                "key_name": api_keys[found_key].get("name"),
                "key_id": key_id
            }
        )
        
        # Create response
        response = BaseResponse(
            status=StatusEnum.SUCCESS,
            message=f"API key revoked successfully",
            data={
                "key_id": key_id,
                "revoked": True,
                "revoked_at": api_keys[found_key]["revoked_at"]
            },
            metadata=MetadataModel(
                request_id=str(uuid.uuid4()),
                timestamp=time.time(),
                version=config.get("version", "1.0.0"),
                process_time=time.time() - start_time
            ),
            errors=None,
            pagination=None
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error revoking API key: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error revoking API key: {str(e)}"
        )

# ----- Helper Functions -----

def get_language_name(language_code: str) -> str:
    """
    Get language name from language code.
    
    Args:
        language_code: ISO language code
        
    Returns:
        Language name
    """
    language_names = {
        "en": "English",
        "es": "Spanish",
        "fr": "French",
        "de": "German",
        "it": "Italian",
        "pt": "Portuguese",
        "ru": "Russian",
        "zh": "Chinese",
        "ja": "Japanese",
        "ko": "Korean",
        "ar": "Arabic",
        "hi": "Hindi",
        "vi": "Vietnamese",
        "tl": "Tagalog",
        "th": "Thai",
        "auto": "Auto-detect"
    }
    
    return language_names.get(language_code, language_code)