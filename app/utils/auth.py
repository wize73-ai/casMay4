"""
Authentication Utilities for CasaLingua

This module provides authentication and authorization utilities
for securing API endpoints and managing user access.
"""

import os
import time
import logging
import jwt
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timedelta
from fastapi import Request, HTTPException, Depends, Security, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, APIKeyHeader

# Configure logging
logger = logging.getLogger("casalingua.auth")

# Security schemes
bearer_scheme = HTTPBearer(auto_error=False)
api_key_scheme = APIKeyHeader(name="X-API-Key", auto_error=False)

# Token settings
TOKEN_EXPIRY = 24  # hours
JWT_SECRET_KEY = os.environ.get("CASALINGUA_JWT_SECRET", "dev_secret_key")
JWT_ALGORITHM = "HS256"

# API key storage (this would be a database in production)
API_KEYS = {
    os.environ.get("API_KEY", "dev-key"): {
        "client_id": "dev_client",
        "roles": ["admin"],
        "rate_limit": 1000
    }
}

# User roles and permissions
ROLE_PERMISSIONS = {
    "admin": ["read", "write", "delete", "manage_users"],
    "user": ["read", "write"],
    "reader": ["read"]
}

class AuthError(HTTPException):
    """Authentication or authorization error."""
    
    def __init__(self, detail: str, status_code: int = status.HTTP_401_UNAUTHORIZED):
        super().__init__(status_code=status_code, detail=detail)


def create_access_token(
    user_id: str, 
    roles: List[str] = None,
    expires_delta: Optional[timedelta] = None
) -> str:
    """
    Create a JWT access token.
    
    Args:
        user_id: User identifier
        roles: User roles
        expires_delta: Token expiration time
        
    Returns:
        str: Encoded JWT
    """
    roles = roles or ["user"]
    expires = datetime.utcnow() + (expires_delta or timedelta(hours=TOKEN_EXPIRY))
    
    claims = {
        "sub": user_id,
        "roles": roles,
        "exp": expires,
        "iat": datetime.utcnow()
    }
    
    try:
        return jwt.encode(claims, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    except Exception as e:
        logger.error(f"Token creation failed: {e}")
        raise AuthError("Token creation failed")


def decode_access_token(token: str) -> Dict[str, Any]:
    """
    Decode and validate a JWT access token.
    
    Args:
        token: JWT access token
        
    Returns:
        Dict[str, Any]: Token claims
    """
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        
        # Check if token has expired
        if payload.get("exp") < time.time():
            raise AuthError("Token has expired")
            
        return payload
    except jwt.PyJWTError as e:
        logger.warning(f"Token validation failed: {e}")
        raise AuthError("Invalid authentication token")


def verify_api_key(api_key: str) -> Dict[str, Any]:
    """
    Verify an API key and return client information.
    
    Args:
        api_key: API key
        
    Returns:
        Dict[str, Any]: Client information
    """
    if api_key not in API_KEYS:
        logger.warning(f"Invalid API key attempt: {api_key[:5]}***")
        raise AuthError("Invalid API key")
    
    return API_KEYS[api_key]


async def get_token_from_request(request: Request) -> Optional[str]:
    """
    Extract JWT token from request.
    
    Args:
        request: FastAPI request
        
    Returns:
        Optional[str]: JWT token if found, None otherwise
    """
    credentials = await bearer_scheme(request)
    
    if credentials:
        return credentials.credentials
    
    # Check for token in cookies (for web sessions)
    token = request.cookies.get("access_token")
    if token:
        return token
    
    return None


async def get_api_key_from_request(request: Request) -> Optional[str]:
    """
    Extract API key from request.
    
    Args:
        request: FastAPI request
        
    Returns:
        Optional[str]: API key if found, None otherwise
    """
    api_key = await api_key_scheme(request)
    return api_key


async def get_current_user(request: Request) -> Dict[str, Any]:
    """
    Get current user from request authentication.
    
    Args:
        request: FastAPI request
        
    Returns:
        Dict[str, Any]: User information
    """
    # Try JWT authentication first
    token = await get_token_from_request(request)
    if token:
        try:
            claims = decode_access_token(token)
            return {
                "user_id": claims["sub"],
                "roles": claims.get("roles", ["user"]),
                "auth_type": "jwt"
            }
        except AuthError as e:
            logger.info(
                f"JWT authentication failed for request to {request.url.path}: {e.detail}"
            )
            # Continue to API key fallback
    # Try API key authentication
    api_key = await get_api_key_from_request(request)
    if api_key:
        try:
            client_info = verify_api_key(api_key)
            return {
                "user_id": client_info["client_id"],
                "roles": client_info.get("roles", ["user"]),
                "auth_type": "api_key"
            }
        except AuthError as e:
            logger.info(
                f"API key authentication failed for request to {request.url.path}: {e.detail}"
            )
            # Continue to authentication failure
    # Authentication failed; both JWT and API key were invalid or missing.
    logger.warning(
        f"Authentication failed (no valid JWT or API key) for request to {request.url.path}"
    )
    raise AuthError("Authentication required")


async def check_permission(request: Request, required_permission: str) -> bool:
    """
    Check if the current user has the required permission.
    
    Args:
        request: FastAPI request
        required_permission: Required permission
        
    Returns:
        bool: True if user has permission
    """
    try:
        user = await get_current_user(request)
        
        # Check user roles for the required permission
        for role in user.get("roles", []):
            if role == "admin" or required_permission in ROLE_PERMISSIONS.get(role, []):
                return True
                
        # Permission denied
        logger.warning(f"Permission denied: {user['user_id']} lacks {required_permission}")
        raise AuthError(
            f"Permission denied: {required_permission} required", 
            status_code=status.HTTP_403_FORBIDDEN
        )
    except AuthError as e:
        if e.status_code == status.HTTP_401_UNAUTHORIZED:
            raise  # Re-raise authentication errors
        return False


async def verify_admin_role(request: Request) -> bool:
    """
    Verify the current user has admin role.
    
    Args:
        request: FastAPI request
        
    Returns:
        bool: True if user is admin
    """
    user = await get_current_user(request)
    if "admin" in user.get("roles", []):
        return True
        
    logger.warning(f"Admin access denied for user: {user['user_id']}")
    raise AuthError(
        "Admin privileges required", 
        status_code=status.HTTP_403_FORBIDDEN
    )


async def optional_auth(request: Request) -> Optional[Dict[str, Any]]:
    """
    Get current user if authenticated, but don't require authentication.
    
    Args:
        request: FastAPI request
        
    Returns:
        Optional[Dict[str, Any]]: User information if authenticated, None otherwise
    """
    try:
        return await get_current_user(request)
    except AuthError:
        return None


def rate_limit(limit_per_minute: int = 60):
    """
    Rate limiting decorator for API endpoints.
    
    Args:
        limit_per_minute: Maximum requests per minute
        
    Returns:
        Callable: Dependency for FastAPI
    """
    # This is a simplified rate limiter; production would use Redis or similar
    rate_limits = {}
    
    async def check_rate_limit(request: Request):
        client_ip = request.client.host
        user = await optional_auth(request)
        
        # Use user ID if authenticated, otherwise use IP
        client_id = user.get("user_id") if user else client_ip
        
        # Check for custom rate limits for API keys
        if user and user.get("auth_type") == "api_key":
            api_key = await get_api_key_from_request(request)
            if api_key and api_key in API_KEYS:
                custom_limit = API_KEYS[api_key].get("rate_limit")
                if custom_limit:
                    return
        
        # Apply rate limiting
        current_time = time.time()
        key = f"{client_id}:{int(current_time / 60)}"  # One minute bucket
        
        if key in rate_limits:
            if rate_limits[key] >= limit_per_minute:
                logger.warning(f"Rate limit exceeded for {client_id}")
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Rate limit exceeded. Please try again later."
                )
            rate_limits[key] += 1
        else:
            # Clean up old entries
            rate_limits.clear()
            rate_limits[key] = 1
    
    return check_rate_limit