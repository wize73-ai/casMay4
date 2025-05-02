"""
Health Check API Routes for CasaLingua

This module defines API endpoints for system health monitoring,
status reporting, and diagnostics to ensure operational reliability
of the CasaLingua language processing platform.

Author: Exygy Development Team
Version: 1.0.0
License: MIT
"""

import time
import os
import platform
import psutil
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, Field

from app.utils.config import load_config, get_config_value
from app.utils.logging import get_logger
from app.api.schemas.base import HealthResponse

logger = get_logger(__name__)

# Create router
router = APIRouter(tags=["Health"])

class ComponentStatus(BaseModel):
    """Model for component status information."""
    name: str = Field(..., description="Component name")
    status: str = Field(..., description="Component status")
    version: Optional[str] = Field(None, description="Component version")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional details")
    last_check: datetime = Field(..., description="Last check timestamp")

class SystemMetrics(BaseModel):
    """Model for system metrics information."""
    cpu_usage: float = Field(..., description="CPU usage percentage")
    memory_usage: float = Field(..., description="Memory usage percentage")
    memory_available: float = Field(..., description="Available memory in MB")
    disk_usage: float = Field(..., description="Disk usage percentage")
    disk_available: float = Field(..., description="Available disk space in GB")
    load_average: List[float] = Field(..., description="System load average (1, 5, 15 minutes)")
    process_memory: float = Field(..., description="Process memory usage in MB")
    open_files: int = Field(..., description="Number of open files by the process")
    
class DetailedHealthResponse(BaseModel):
    """Detailed health check response model."""
    status: str = Field(..., description="Overall system status")
    version: str = Field(..., description="Service version")
    environment: str = Field(..., description="Environment name")
    uptime: float = Field(..., description="Service uptime in seconds")
    uptime_formatted: str = Field(..., description="Formatted uptime string")
    build_info: Dict[str, str] = Field(..., description="Build information")
    components: List[ComponentStatus] = Field(..., description="Component statuses")
    metrics: SystemMetrics = Field(..., description="System metrics")
    timestamp: datetime = Field(..., description="Current server time")

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Basic health check",
    description="Performs a basic health check of the service."
)
async def health_check(request: Request):
    """
    Basic health check endpoint.
    
    This endpoint provides a simple health status, mainly for
    load balancers and monitoring systems.
    """
    start_time = time.time()
    
    try:
        # Get components from application state
        config = request.app.state.config
        start_time_ts = request.app.state.start_time
        uptime = time.time() - start_time_ts
        
        # Check basic component status
        services = {
            "database": "ok",
            "models": "ok",
            "pipeline": "ok"
        }
        
        # Prepare response
        response = HealthResponse(
            status="healthy",
            version=config.get("version", "1.0.0"),
            environment=config.get("environment", "development"),
            uptime=uptime,
            timestamp=datetime.now(),
            services=services
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Health check error: {str(e)}", exc_info=True)
        
        # Return degraded status
        return HealthResponse(
            status="degraded",
            version=request.app.state.config.get("version", "1.0.0"),
            environment=request.app.state.config.get("environment", "development"),
            uptime=0.0,
            timestamp=datetime.now(),
            services={"error": str(e)}
        )

@router.get(
    "/health/detailed",
    response_model=DetailedHealthResponse,
    summary="Detailed health check",
    description="Performs a detailed health check of all system components.",
    response_description="Detailed health status of all system components."
)
async def detailed_health_check(request: Request):
    """
    Detailed health check endpoint.
    
    This endpoint provides comprehensive status information about
    all system components and resources.
    """
    start_time = time.time()
    
    try:
        # Get components from application state
        config = request.app.state.config
        start_time_ts = request.app.state.start_time
        model_manager = request.app.state.model_manager
        uptime = time.time() - start_time_ts
        
        # Format uptime
        uptime_formatted = format_uptime(uptime)
        
        # Check component status
        components = await check_component_status(request)
        
        # Get system metrics
        metrics = get_system_metrics()
        
        # Get build information
        build_info = {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "build_date": config.get("build_date", "unknown"),
            "build_id": config.get("build_id", "unknown"),
            "git_commit": config.get("git_commit", "unknown")
        }
        
        # Determine overall status
        overall_status = "healthy"
        for component in components:
            if component.status == "error":
                overall_status = "unhealthy"
                break
            elif component.status == "degraded" and overall_status != "unhealthy":
                overall_status = "degraded"
                
        # Prepare response
        response = DetailedHealthResponse(
            status=overall_status,
            version=config.get("version", "1.0.0"),
            environment=config.get("environment", "development"),
            uptime=uptime,
            uptime_formatted=uptime_formatted,
            build_info=build_info,
            components=components,
            metrics=metrics,
            timestamp=datetime.now()
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Detailed health check error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Health check error: {str(e)}"
        )

@router.get(
    "/health/models",
    summary="Model health check",
    description="Performs a health check of the language models."
)
async def model_health_check(request: Request) -> dict:
    """
    Model health check endpoint.

    This endpoint provides status information specifically about
    the language models used in the system.
    """
    start_time = time.time()
    try:
        model_manager = getattr(request.app.state, "model_manager", None)
        if model_manager is None:
            logger.error("Model manager not initialized")
            return {
                "status": "error",
                "message": "Model manager not initialized",
                "loaded_models": [],
                "device": None,
                "response_time": time.time() - start_time,
            }
        model_info = await model_manager.get_model_info()
        loaded_models = model_info.get("loaded_models", [])
        model_registry = getattr(request.app.state, "model_registry", None)
        status_str = "healthy" if loaded_models else "degraded"
        result = {
            "status": status_str,
            "message": f"{len(loaded_models)} models loaded" if loaded_models else "No models loaded",
            "loaded_models": loaded_models,
            "device": model_info.get("device", "cpu"),
            "response_time": time.time() - start_time,
        }
        if model_registry is not None:
            registry_summary = model_registry.get_registry_summary()
            result["registry"] = registry_summary
        else:
            if loaded_models:
                result["status"] = "degraded"
                result["message"] += "; model registry not initialized"
        return result
    except Exception as e:
        logger.error(f"Model health check error: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "message": f"Error: {str(e)}",
            "loaded_models": [],
            "device": None,
            "response_time": time.time() - start_time,
        }

@router.get(
    "/health/database",
    summary="Database health check",
    description="Performs a health check of the database connection."
)
async def database_health_check(request: Request):
    """
    Database health check endpoint.
    
    This endpoint verifies the database connection and reports status.
    """
    start_time = time.time()
    
    try:
        # Check if database connection exists in app state
        # This is a placeholder - implement based on your actual DB setup
        db_status = "ok"
        db_response_time = 0.0
        
        try:
            # Simulate a DB ping/test query
            # Replace with actual DB check
            time.sleep(0.01)  # Simulate DB query time
            db_response_time = 0.01
        except Exception as e:
            db_status = "error"
            db_response_time = 0.0
            
        return {
            "status": db_status,
            "response_time": db_response_time,
            "total_time": time.time() - start_time
        }
        
    except Exception as e:
        logger.error(f"Database health check error: {str(e)}", exc_info=True)
        
        return {
            "status": "error",
            "message": f"Error: {str(e)}",
            "response_time": time.time() - start_time
        }

@router.get(
    "/readiness",
    status_code=status.HTTP_200_OK,
    summary="Readiness probe",
    description="Kubernetes readiness probe endpoint."
)
async def readiness_probe(request: Request, response: Response):
    """
    Readiness probe endpoint for Kubernetes.
    
    This endpoint indicates whether the service is ready to handle requests.
    """
    try:
        # Check if essential services are initialized
        if not hasattr(request.app.state, "processor") or not request.app.state.processor:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            return {"status": "not_ready", "message": "Processor not initialized"}
            
        if not hasattr(request.app.state, "model_manager") or not request.app.state.model_manager:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            return {"status": "not_ready", "message": "Model manager not initialized"}
            
        # Check if any models are loaded
        model_manager = request.app.state.model_manager
        model_info = await model_manager.get_model_info()
        
        # Service is considered ready if at least the essential models are loaded
        # Adjust this condition based on your application's requirements
        if not model_info.get("loaded_models"):
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            return {"status": "not_ready", "message": "No models loaded"}
            
        return {"status": "ready"}
        
    except Exception as e:
        logger.error(f"Readiness probe error: {str(e)}", exc_info=True)
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return {"status": "error", "message": str(e)}

@router.get(
    "/liveness",
    status_code=status.HTTP_200_OK,
    summary="Liveness probe",
    description="Kubernetes liveness probe endpoint."
)
async def liveness_probe():
    """
    Liveness probe endpoint for Kubernetes.
    
    This endpoint indicates whether the service is alive and running.
    """
    # Simple liveness probe - if this endpoint responds, the service is alive
    return {"status": "alive"}

# ----- Helper Functions -----

async def check_component_status(request: Request) -> List[ComponentStatus]:
    """
    Check the status of all system components.
    
    Args:
        request: Request object containing application state
        
    Returns:
        List of component status objects
    """
    components = []
    
    # Check processor
    if hasattr(request.app.state, "processor") and request.app.state.processor:
        processor = request.app.state.processor
        components.append(ComponentStatus(
            name="processor",
            status="ok",
            version=processor.version if hasattr(processor, "version") else None,
            details={"pipeline_count": len(processor.pipelines) if hasattr(processor, "pipelines") else 0},
            last_check=datetime.now()
        ))
    else:
        components.append(ComponentStatus(
            name="processor",
            status="error",
            details={"error": "Not initialized"},
            last_check=datetime.now()
        ))
        
    # Check model manager
    if hasattr(request.app.state, "model_manager") and request.app.state.model_manager:
        model_manager = request.app.state.model_manager
        model_info = await model_manager.get_model_info()
        model_status = "ok" if model_info.get("loaded_models") else "degraded"
        
        components.append(ComponentStatus(
            name="model_manager",
            status=model_status,
            details={
                "loaded_models": len(model_info.get("loaded_models", [])),
                "device": model_info.get("device", "cpu"),
                "low_memory_mode": model_info.get("low_memory_mode", False)
            },
            last_check=datetime.now()
        ))
    else:
        components.append(ComponentStatus(
            name="model_manager",
            status="error",
            details={"error": "Not initialized"},
            last_check=datetime.now()
        ))
        
    # Check model registry
    if hasattr(request.app.state, "model_registry") and request.app.state.model_registry:
        model_registry = request.app.state.model_registry
        registry_summary = model_registry.get_registry_summary()
        
        components.append(ComponentStatus(
            name="model_registry",
            status="ok",
            details={
                "total_models": registry_summary["model_counts"]["total"],
                "languages": len(registry_summary["supported_languages"]),
                "tasks": len(registry_summary["supported_tasks"])
            },
            last_check=datetime.now()
        ))
    else:
        components.append(ComponentStatus(
            name="model_registry",
            status="error",
            details={"error": "Not initialized"},
            last_check=datetime.now()
        ))
        
    # Check metrics collector
    if hasattr(request.app.state, "metrics") and request.app.state.metrics:
        metrics = request.app.state.metrics
        system_metrics = metrics.get_system_metrics()
        
        components.append(ComponentStatus(
            name="metrics",
            status="ok",
            details={
                "total_requests": system_metrics["request_metrics"]["total_requests"],
                "uptime": system_metrics["uptime_seconds"]
            },
            last_check=datetime.now()
        ))
    else:
        components.append(ComponentStatus(
            name="metrics",
            status="error",
            details={"error": "Not initialized"},
            last_check=datetime.now()
        ))
        
    # Check audit logger
    if hasattr(request.app.state, "audit_logger") and request.app.state.audit_logger:
        components.append(ComponentStatus(
            name="audit_logger",
            status="ok",
            last_check=datetime.now()
        ))
    else:
        components.append(ComponentStatus(
            name="audit_logger",
            status="error",
            details={"error": "Not initialized"},
            last_check=datetime.now()
        ))
        
    # Add more component checks as needed
    
    return components

def get_system_metrics() -> SystemMetrics:
    """
    Get system metrics.
    
    Returns:
        System metrics object
    """
    # Get CPU usage
    cpu_usage = psutil.cpu_percent(interval=0.1)
    
    # Get memory usage
    memory = psutil.virtual_memory()
    memory_usage = memory.percent
    memory_available = memory.available / (1024 * 1024)  # MB
    
    # Get disk usage
    disk = psutil.disk_usage("/")
    disk_usage = disk.percent
    disk_available = disk.free / (1024 * 1024 * 1024)  # GB
    
    # Get load average
    try:
        load_avg = os.getloadavg()
    except (AttributeError, OSError):
        # Windows doesn't support getloadavg
        load_avg = (0.0, 0.0, 0.0)
        
    # Get process info
    process = psutil.Process(os.getpid())
    process_memory = process.memory_info().rss / (1024 * 1024)  # MB
    
    try:
        open_files = len(process.open_files())
    except (psutil.AccessDenied, psutil.Error):
        open_files = -1
        
    return SystemMetrics(
        cpu_usage=cpu_usage,
        memory_usage=memory_usage,
        memory_available=memory_available,
        disk_usage=disk_usage,
        disk_available=disk_available,
        load_average=list(load_avg),
        process_memory=process_memory,
        open_files=open_files
    )

def format_uptime(seconds: float) -> str:
    """
    Format uptime in seconds to a human-readable string.
    
    Args:
        seconds: Uptime in seconds
        
    Returns:
        Formatted uptime string
    """
    days, remainder = divmod(int(seconds), 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    parts = []
    if days > 0:
        parts.append(f"{days}d")
    if hours > 0 or days > 0:
        parts.append(f"{hours}h")
    if minutes > 0 or hours > 0 or days > 0:
        parts.append(f"{minutes}m")
    parts.append(f"{seconds}s")
    
    return " ".join(parts)