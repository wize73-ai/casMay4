

"""
Authentication and Authorization Module for CasaLingua

This module provides authentication and authorization functionality
for the CasaLingua language processing platform, including API key
validation, JWT token handling, role-based access control, and
integration with FastAPI security dependencies.

Author: Exygy Development Team
Version: 1.0.0
License: MIT
"""

from dotenv import load_dotenv
load_dotenv()
import os
import time
import json
import logging
import secrets
import hashlib
import base64
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime, timedelta
from functools import wraps

import jwt
from fastapi import Depends, HTTPException, Request, Security, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, APIKeyHeader
from fastapi.security.api_key import APIKey
from pydantic import BaseModel

from app.utils.config import load_config, get_config_value
from app.utils.logging import get_logger

logger = get_logger(__name__)

# Security schemes
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
bearer_scheme = HTTPBearer(auto_error=False)

# User model
class User(BaseModel):
    """User model for authentication."""
    id: str
    username: str
    email: Optional[str] = None
    role: str
    permissions: List[str] = []
    metadata: Dict[str, Any] = {}

# Role definitions with associated permissions
DEFAULT_ROLES = {
    "admin": [
        "admin:read",
        "admin:write",
        "models:read",
        "models:write",
        "translation:read",
        "translation:write",
        "audit:read"
    ],
    "translator": [
        "models:read",
        "translation:read",
        "translation:write"
    ],
    "viewer": [
        "models:read",
        "translation:read"
    ],
    "api": [
        "models:read",
        "translation:read",
        "translation:write"
    ]
}

async def get_api_key(
    api_key: str = Depends(api_key_header),
    request: Request = None
) -> Optional[APIKey]:
    """
    Validate API key from header.
    
    Args:
        api_key: API key from header
        request: FastAPI request
        
    Returns:
        Validated API key or None
    """
    if api_key is None:
        return None
        
    # Get configuration
    config = load_config()
    api_keys = config.get("api_keys", {})
    
    # Check if API key exists and is active
    if api_key in api_keys and api_keys[api_key].get("active", True):
        # Check expiration
        if "expires_at" in api_keys[api_key]:
            expires_at = datetime.fromisoformat(api_keys[api_key]["expires_at"].replace("Z", "+00:00"))
            if expires_at < datetime.now():
                logger.warning(f"Expired API key used: {api_key[:8]}...")
                return None
                
        # Add API key info to request state if available
        if request:
            request.state.api_key_info = api_keys[api_key]
            request.state.auth_type = "api_key"
            
        return APIKey(api_key)
        
    return None

async def validate_jwt_token(
    token: str,
    secret_key: str,
    audience: Optional[str] = None
) -> Dict[str, Any]:
    """
    Validate JWT token and extract claims.
    
    Args:
        token: JWT token
        secret_key: Secret key for validation
        audience: Expected audience
        
    Returns:
        Token claims
    """
    try:
        # Decode and validate token
        payload = jwt.decode(
            token,
            secret_key,
            algorithms=["HS256"],
            audience=audience,
            options={"verify_signature": True}
        )
        
        # Check expiration
        if "exp" in payload and payload["exp"] < time.time():
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired"
            )
            
        return payload
        
    except jwt.PyJWTError as e:
        logger.warning(f"JWT validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token"
        )

async def get_jwt_token(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    request: Request = None
) -> Optional[Dict[str, Any]]:
    """
    Extract and validate JWT token from Authorization header.
    
    Args:
        credentials: Authorization credentials
        request: FastAPI request
        
    Returns:
        Decoded token payload or None
    """
    if credentials is None:
        return None
        
    # Get secret key from config
    config = load_config()
    secret_key = config.get("jwt_secret")
    
    if not secret_key:
        logger.error("JWT secret key not configured")
        return None
        
    # Validate token
    try:
        token = credentials.credentials
        payload = await validate_jwt_token(token, secret_key)
        
        # Add token payload to request state if available
        if request:
            request.state.token_payload = payload
            request.state.auth_type = "jwt"
            
        return payload
        
    except HTTPException:
        return None


async def get_user_from_api_key(
    api_key: Optional[APIKey] = Depends(get_api_key),
    request: Request = None
) -> Optional[Dict[str, Any]]:
    """
    Get user information from API key.
    
    Args:
        api_key: API key
        request: FastAPI request
        
    Returns:
        User information or None
    """
    if api_key is None:
        return None
        
    # Get API key info from request state
    if request and hasattr(request.state, "api_key_info"):
        api_key_info = request.state.api_key_info
        
        # Create user from API key info
        user = {
            "id": f"api:{api_key_info.get('name', str(api_key)[:8])}",
            "username": api_key_info.get("name", f"API User {str(api_key)[:8]}"),
            "role": "api",
            "permissions": api_key_info.get("scopes", []),
            "metadata": {
                "api_key_id": str(api_key)[:8],
                "created_at": api_key_info.get("created_at"),
                "expires_at": api_key_info.get("expires_at")
            }
        }
        
        return user
        
    # Fallback - create minimal user
    return {
        "id": f"api:{str(api_key)[:8]}",
        "username": f"API User {str(api_key)[:8]}",
        "role": "api",
        "permissions": ["translation:read", "translation:write", "models:read"]
    }

async def get_user_from_token(
    token_payload: Optional[Dict[str, Any]] = None
) -> Optional[Dict[str, Any]]:
    """
    Get user information from token payload.
    
    Args:
        token_payload: Decoded token payload
        
    Returns:
        User information or None
    """
    if token_payload is None:
        return None
        
    # Get user properties from token
    user_id = token_payload.get("sub")
    if not user_id:
        return None
        
    # Create user from token payload
    user = {
        "id": user_id,
        "username": token_payload.get("username", user_id),
        "email": token_payload.get("email"),
        "role": token_payload.get("role", "viewer"),
        "permissions": token_payload.get("permissions", [])
    }
    
    # If no permissions in token, use role to determine permissions
    if not user["permissions"]:
        user["permissions"] = get_permissions_for_role(user["role"])
        
    return user

async def get_current_user(
    request: Request,
    api_key: Optional[APIKey] = Depends(get_api_key),
    jwt_payload: Optional[Dict[str, Any]] = Depends(get_jwt_token),
) -> Dict[str, Any]:
    """
    Get current authenticated user from various auth methods.
    
    This dependency combines multiple authentication methods
    and returns the authenticated user or raises an exception.
    
    Args:
        request: FastAPI request
        api_key: API key
        jwt_payload: JWT token payload
        oauth2_payload: OAuth2 token payload
        
    Returns:
        Authenticated user information
        
    Raises:
        HTTPException: If authentication fails
    """

    # Fixed development mode bypass with clear logging
    env = os.getenv("CASALINGUA_ENV", "production").lower()
    logger.info(f"Auth check - Environment: {env}")
    
    if env == "development":
        logger.warning("🧪 Development mode active — Auth bypass enabled, returning dev user")
        # Create dev user with admin permissions
        dev_user = {
            "id": "dev-user",
            "username": "developer",
            "role": "admin",
            "permissions": ["*"],
            "metadata": {}
        }
        # Store in request state
        request.state.user = dev_user
        # Log to logger instead of console for clarity
        logger.warning(f"🔓 AUTH BYPASS: Dev user authenticated with admin role (ENV={env})")
        return dev_user

    user = None
    
    # Try all authentication methods
    if api_key:
        user = await get_user_from_api_key(api_key, request)
    elif jwt_payload:
        user = await get_user_from_token(jwt_payload)
        
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"}
        )
        
    # Store user in request state
    request.state.user = user
    
    # Log authentication event if audit logger available
    try:
        if hasattr(request.app.state, "audit_logger"):
            audit_logger = request.app.state.audit_logger
            await audit_logger.log_authentication_event(
                event_type="api_auth",
                user_id=user["id"],
                source_ip=request.client.host if request.client else None,
                success=True,
                details={
                    "auth_type": getattr(request.state, "auth_type", "unknown"),
                    "endpoint": request.url.path
                }
            )
    except Exception as e:
        logger.warning(f"Failed to log authentication event: {str(e)}")
        
    return user

async def get_optional_user(
    request: Request,
    api_key: Optional[APIKey] = Depends(get_api_key),
    jwt_payload: Optional[Dict[str, Any]] = Depends(get_jwt_token),
) -> Optional[Dict[str, Any]]:
    """
    Get current user if authenticated, otherwise None.
    
    Similar to get_current_user but doesn't raise an exception
    if no valid authentication is provided.
    
    Args:
        request: FastAPI request
        api_key: API key
        jwt_payload: JWT token payload
        oauth2_payload: OAuth2 token payload
        
    Returns:
        Authenticated user information or None
    """
    user = None
    
    # Try all authentication methods
    if api_key:
        user = await get_user_from_api_key(api_key, request)
    elif jwt_payload:
        user = await get_user_from_token(jwt_payload)
        
    # Store user in request state if authenticated
    if user:
        request.state.user = user
        
        # Log authentication event if audit logger available
        try:
            if hasattr(request.app.state, "audit_logger"):
                audit_logger = request.app.state.audit_logger
                await audit_logger.log_authentication_event(
                    event_type="api_auth",
                    user_id=user["id"],
                    source_ip=request.client.host if request.client else None,
                    success=True,
                    details={
                        "auth_type": getattr(request.state, "auth_type", "unknown"),
                        "endpoint": request.url.path
                    }
                )
        except Exception as e:
            logger.warning(f"Failed to log authentication event: {str(e)}")
            
    return user

def get_permissions_for_role(role: str) -> List[str]:
    """
    Get permissions for a specific role.
    
    Args:
        role: Role name
        
    Returns:
        List of permissions
    """
    # Get role definitions from config
    config = load_config()
    roles = config.get("roles", DEFAULT_ROLES)
    
    # Return permissions for role
    return roles.get(role, roles.get("viewer", []))

def verify_permission(
    permission: str,
    user: Dict[str, Any]
) -> bool:
    """
    Verify that a user has a specific permission.
    
    Args:
        permission: Permission to check
        user: User information
        
    Returns:
        True if user has permission
    """
    # Admin role has all permissions
    if user.get("role") == "admin":
        return True
        
    # Check user permissions
    user_permissions = user.get("permissions", [])
    if permission in user_permissions:
        return True
        
    # Check for wildcard permissions
    permission_parts = permission.split(":")
    if len(permission_parts) == 2:
        resource = permission_parts[0]
        wildcard_permission = f"{resource}:*"
        if wildcard_permission in user_permissions:
            return True
            
    return False

async def verify_admin_role(
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Verify that the current user has admin role.
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        User information if admin
        
    Raises:
        HTTPException: If user is not admin
    """
    if current_user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin role required"
        )
        
    return current_user

async def verify_api_key(
    request: Request,
    api_key: APIKey = Depends(get_api_key)
) -> APIKey:
    """
    Verify API key dependency.
    
    Args:
        request: FastAPI request
        api_key: API key
        
    Returns:
        Validated API key
        
    Raises:
        HTTPException: If API key is invalid
    """
    if api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key"
        )
        
    return api_key

def permission_required(permission: str):
    """
    Dependency factory to check for a specific permission.
    
    Args:
        permission: Required permission
        
    Returns:
        Dependency function
    """
    async def verify_permission_dependency(
        current_user: Dict[str, Any] = Depends(get_current_user)
    ) -> Dict[str, Any]:
        if not verify_permission(permission, current_user):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission required: {permission}"
            )
            
        return current_user
        
    return verify_permission_dependency

def role_required(allowed_roles: List[str]):
    """
    Dependency factory to check for specific roles.
    
    Args:
        allowed_roles: List of allowed roles
        
    Returns:
        Dependency function
    """
    async def verify_role_dependency(
        current_user: Dict[str, Any] = Depends(get_current_user)
    ) -> Dict[str, Any]:
        if current_user.get("role") not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Required role: {', '.join(allowed_roles)}"
            )
            
        return current_user
        
    return verify_role_dependency

def create_access_token(
    user_id: str,
    username: str,
    role: str,
    permissions: Optional[List[str]] = None,
    email: Optional[str] = None,
    expires_delta: Optional[timedelta] = None
) -> str:
    """
    Create a JWT access token.
    
    Args:
        user_id: User identifier
        username: Username
        role: User role
        permissions: User permissions
        email: User email
        expires_delta: Token expiration time
        
    Returns:
        JWT token string
    """
    # Get configuration
    config = load_config()
    secret_key = config.get("jwt_secret")
    
    if not secret_key:
        raise ValueError("JWT secret key not configured")
        
    # Set default expiration if not provided
    if expires_delta is None:
        expires_delta = timedelta(hours=24)
        
    # Create token data
    expire = datetime.utcnow() + expires_delta
    
    token_data = {
        "sub": user_id,
        "username": username,
        "role": role,
        "iat": datetime.utcnow(),
        "exp": expire,
        "aud": "casalingua-api"
    }
    
    # Add optional claims
    if permissions:
        token_data["permissions"] = permissions
    else:
        token_data["permissions"] = get_permissions_for_role(role)
        
    if email:
        token_data["email"] = email
        
    # Encode token
    encoded_token = jwt.encode(token_data, secret_key, algorithm="HS256")
    return encoded_token

def hash_password(password: str) -> str:
    """
    Hash a password securely.
    
    Args:
        password: Plain text password
        
    Returns:
        Hashed password
    """
    # Generate salt
    salt = secrets.token_hex(16)
    
    # Hash password
    hash_obj = hashlib.sha256()
    hash_obj.update(f"{salt}${password}".encode('utf-8'))
    hashed = hash_obj.hexdigest()
    
    # Return salted hash
    return f"{salt}${hashed}"

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a password against a hash.
    
    Args:
        plain_password: Plain text password
        hashed_password: Stored password hash
        
    Returns:
        True if password matches
    """
    # Split salt and hash
    try:
        salt, stored_hash = hashed_password.split('$', 1)
        
        # Hash password with salt
        hash_obj = hashlib.sha256()
        hash_obj.update(f"{salt}${plain_password}".encode('utf-8'))
        computed_hash = hash_obj.hexdigest()
        
        # Compare hashes
        return secrets.compare_digest(computed_hash, stored_hash)
    except ValueError:
        return False

class AuthorizationMiddleware:
    """
    Middleware for route-based authorization checks.
    
    This middleware checks permissions for specific routes
    based on configured route policies.
    """
    
    def __init__(self, route_policies: Dict[str, str] = None):
        """
        Initialize authorization middleware.
        
        Args:
            route_policies: Mapping of route patterns to required permissions
        """
        self.route_policies = route_policies or {}
        
    async def __call__(self, request: Request, call_next):
        """
        Process a request and check authorization.
        
        Args:
            request: FastAPI request
            call_next: Next middleware handler
            
        Returns:
            FastAPI response
        """
        # Skip auth check if no policies defined
        if not self.route_policies:
            return await call_next(request)
            
        # Get the route path
        path = request.url.path
        
        # Find matching policy
        required_permission = None
        for route_pattern, permission in self.route_policies.items():
            if self._match_route(path, route_pattern):
                required_permission = permission
                break
                
        # No policy for this route
        if required_permission is None:
            return await call_next(request)
            
        # Get current user from request state
        user = getattr(request.state, "user", None)
        
        # If no user and authentication is required
        if user is None:
            # Try to authenticate
            try:
                api_key = await get_api_key(request=request)
                if api_key:
                    user = await get_user_from_api_key(api_key, request)
                else:
                    credentials = await bearer_scheme(request)
                    if credentials:
                        jwt_payload = await get_jwt_token(credentials, request)
                        user = await get_user_from_token(jwt_payload)
            except HTTPException:
                pass
                
            # If still no user, authentication failed
            if user is None:
                return JSONResponse(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    content={"detail": "Authentication required"},
                    headers={"WWW-Authenticate": "Bearer"}
                )
                
            # Store user in request state
            request.state.user = user
            
        # Check permission
        if not verify_permission(required_permission, user):
            return JSONResponse(
                status_code=status.HTTP_403_FORBIDDEN,
                content={"detail": f"Permission required: {required_permission}"}
            )
            
        # Continue processing
        return await call_next(request)
        
    def _match_route(self, path: str, pattern: str) -> bool:
        """
        Check if a path matches a route pattern.
        
        Args:
            path: Request path
            pattern: Route pattern
            
        Returns:
            True if path matches pattern
        """
        # Simple exact match
        if path == pattern:
            return True
            
        # Pattern with wildcard
        if pattern.endswith("/*"):
            prefix = pattern[:-1]
            return path.startswith(prefix)
            
        # Pattern with parameter
        if "{" in pattern:
            # Convert to regex
            import re
            regex_pattern = pattern.replace("{", "(?P<").replace("}", ">[^/]+)")
            regex_pattern = f"^{regex_pattern}$"
            match = re.match(regex_pattern, path)
            return match is not None
            
        return False