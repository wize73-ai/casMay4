"""
Enhanced error handling system for CasaLingua API.
Provides categorized error handling, fallbacks, and standardized error responses.
"""

import os
import sys
import time
import traceback
import logging
from enum import Enum
from typing import Dict, Any, Optional, Callable, Type, List, Union
from functools import wraps

from fastapi import HTTPException, status, Request
from pydantic import BaseModel, Field

from app.utils.logging import get_logger

# Initialize logger
logger = get_logger(__name__)

class ErrorCategory(str, Enum):
    """Categories of errors for better client handling."""
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    VALIDATION = "validation"
    RESOURCE_NOT_FOUND = "resource_not_found"
    RESOURCE_CONFLICT = "resource_conflict"
    RATE_LIMIT = "rate_limit"
    SERVICE_UNAVAILABLE = "service_unavailable"
    MODEL_ERROR = "model_error"
    TRANSLATION_ERROR = "translation_error"
    PROCESSING_ERROR = "processing_error"
    INTERNAL_ERROR = "internal_error"
    NETWORK_ERROR = "network_error"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"

class ErrorResponse(BaseModel):
    """Standardized error response model."""
    status_code: int = Field(..., description="HTTP status code")
    error_code: str = Field(..., description="Machine-readable error code")
    category: ErrorCategory = Field(..., description="Error category")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    request_id: Optional[str] = Field(None, description="Request ID for tracking")
    timestamp: float = Field(default_factory=time.time, description="Error timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "status_code": 400,
                "error_code": "invalid_input",
                "category": "validation",
                "message": "Invalid input parameters",
                "details": {"field": "source_language", "issue": "Unsupported language code"},
                "request_id": "abcd1234-5678-efgh-ijkl",
                "timestamp": 1620000000.123
            }
        }

class APIError(Exception):
    """Base exception for API errors with categorization."""
    def __init__(
        self, 
        message: str, 
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        error_code: str = "internal_error",
        category: ErrorCategory = ErrorCategory.INTERNAL_ERROR,
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None
    ):
        self.message = message
        self.status_code = status_code
        self.error_code = error_code
        self.category = category
        self.details = details or {}
        self.request_id = request_id
        super().__init__(self.message)
    
    def to_response(self) -> ErrorResponse:
        """Convert to a standardized error response."""
        return ErrorResponse(
            status_code=self.status_code,
            error_code=self.error_code,
            category=self.category,
            message=self.message,
            details=self.details,
            request_id=self.request_id
        )

# Specific error types
class AuthenticationError(APIError):
    """Authentication errors."""
    def __init__(
        self, 
        message: str = "Authentication failed", 
        error_code: str = "authentication_failed",
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None
    ):
        super().__init__(
            message=message,
            status_code=status.HTTP_401_UNAUTHORIZED,
            error_code=error_code,
            category=ErrorCategory.AUTHENTICATION,
            details=details,
            request_id=request_id
        )

class AuthorizationError(APIError):
    """Authorization errors."""
    def __init__(
        self, 
        message: str = "Not authorized", 
        error_code: str = "not_authorized",
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None
    ):
        super().__init__(
            message=message,
            status_code=status.HTTP_403_FORBIDDEN,
            error_code=error_code,
            category=ErrorCategory.AUTHORIZATION,
            details=details,
            request_id=request_id
        )

class ValidationError(APIError):
    """Validation errors."""
    def __init__(
        self, 
        message: str = "Validation error", 
        error_code: str = "validation_error",
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None
    ):
        super().__init__(
            message=message,
            status_code=status.HTTP_400_BAD_REQUEST,
            error_code=error_code,
            category=ErrorCategory.VALIDATION,
            details=details,
            request_id=request_id
        )

class ResourceNotFoundError(APIError):
    """Resource not found errors."""
    def __init__(
        self, 
        message: str = "Resource not found", 
        error_code: str = "resource_not_found",
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None
    ):
        super().__init__(
            message=message,
            status_code=status.HTTP_404_NOT_FOUND,
            error_code=error_code,
            category=ErrorCategory.RESOURCE_NOT_FOUND,
            details=details,
            request_id=request_id
        )

class ResourceConflictError(APIError):
    """Resource conflict errors."""
    def __init__(
        self, 
        message: str = "Resource conflict", 
        error_code: str = "resource_conflict",
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None
    ):
        super().__init__(
            message=message,
            status_code=status.HTTP_409_CONFLICT,
            error_code=error_code,
            category=ErrorCategory.RESOURCE_CONFLICT,
            details=details,
            request_id=request_id
        )

class RateLimitError(APIError):
    """Rate limit errors."""
    def __init__(
        self, 
        message: str = "Rate limit exceeded", 
        error_code: str = "rate_limit_exceeded",
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None
    ):
        super().__init__(
            message=message,
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            error_code=error_code,
            category=ErrorCategory.RATE_LIMIT,
            details=details,
            request_id=request_id
        )

class ServiceUnavailableError(APIError):
    """Service unavailable errors."""
    def __init__(
        self, 
        message: str = "Service unavailable", 
        error_code: str = "service_unavailable",
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None
    ):
        super().__init__(
            message=message,
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            error_code=error_code,
            category=ErrorCategory.SERVICE_UNAVAILABLE,
            details=details,
            request_id=request_id
        )

class ModelError(APIError):
    """Model-related errors."""
    def __init__(
        self, 
        message: str = "Model error", 
        error_code: str = "model_error",
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None
    ):
        super().__init__(
            message=message,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_code=error_code,
            category=ErrorCategory.MODEL_ERROR,
            details=details,
            request_id=request_id
        )

class TranslationError(APIError):
    """Translation-specific errors."""
    def __init__(
        self, 
        message: str = "Translation error", 
        error_code: str = "translation_error",
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None
    ):
        super().__init__(
            message=message,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_code=error_code,
            category=ErrorCategory.TRANSLATION_ERROR,
            details=details,
            request_id=request_id
        )

class TimeoutError(APIError):
    """Timeout errors."""
    def __init__(
        self, 
        message: str = "Request timed out", 
        error_code: str = "request_timeout",
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None
    ):
        super().__init__(
            message=message,
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            error_code=error_code,
            category=ErrorCategory.TIMEOUT,
            details=details,
            request_id=request_id
        )

# Error handler decorator for route functions
def handle_errors(fallbacks: Optional[Dict[Type[Exception], Callable]] = None):
    """
    Decorator for handling errors in route functions with optional fallbacks.
    
    Args:
        fallbacks: Optional mapping of exception types to fallback functions.
            Each fallback function should accept the exception and request as arguments.
    
    Example:
        @handle_errors(fallbacks={
            TranslationError: lambda e, req: {"text": "Fallback translation", "error": str(e)}
        })
        async def translate_text(...):
            ...
    """
    fallbacks = fallbacks or {}
    
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            request = None
            request_id = None
            
            # Extract request object and request_id from args/kwargs if available
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break
            
            if not request and 'request' in kwargs:
                request = kwargs['request']
            
            if request:
                request_id = request.headers.get('X-Request-ID')
            
            try:
                return await func(*args, **kwargs)
            except APIError as e:
                # Set request_id if available and not already set
                if request_id and not e.request_id:
                    e.request_id = request_id
                
                # Log the error
                logger.error(
                    f"API Error ({e.category}): {e.message}", 
                    extra={
                        "status_code": e.status_code,
                        "error_code": e.error_code,
                        "category": e.category,
                        "request_id": e.request_id,
                        "details": e.details
                    }
                )
                
                # Return the standardized error response
                raise HTTPException(
                    status_code=e.status_code,
                    detail=e.to_response().dict()
                )
            except Exception as e:
                # Check for fallback handler
                for exc_type, fallback_func in fallbacks.items():
                    if isinstance(e, exc_type):
                        logger.warning(
                            f"Using fallback for {exc_type.__name__}: {str(e)}", 
                            extra={"request_id": request_id}
                        )
                        return await fallback_func(e, request)
                
                # Log the unexpected error
                logger.error(
                    f"Unexpected error: {str(e)}", 
                    extra={
                        "traceback": traceback.format_exc(),
                        "request_id": request_id
                    }
                )
                
                # Convert to APIError and return standardized response
                api_error = APIError(
                    message=f"An unexpected error occurred: {str(e)}",
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    error_code="internal_server_error",
                    category=ErrorCategory.INTERNAL_ERROR,
                    request_id=request_id,
                    details={"traceback": traceback.format_exc()}
                )
                
                raise HTTPException(
                    status_code=api_error.status_code,
                    detail=api_error.to_response().dict()
                )
        
        return wrapper
    
    return decorator