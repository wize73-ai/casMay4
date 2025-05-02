"""
Audit Logging Module for CasaLingua

This module provides comprehensive audit logging capabilities for
tracking system activities, user actions, and security events
within the CasaLingua language processing pipeline.

Author: Exygy Development Team
Version: 1.0.0
License: MIT
"""

import os
import json
import time
import asyncio
import logging
import uuid
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
from datetime import datetime, timedelta
from collections import deque
import threading

from app.utils.config import load_config, get_config_value
from app.utils.logging import get_logger

logger = get_logger(__name__)

class AuditLogger:
    """
    Handles audit logging for security, compliance, and operational
    tracking within the CasaLingua system.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the audit logger.
        
        Args:
            config: Application configuration
        """
        self.config = config or load_config()
        self.audit_config = get_config_value(self.config, "audit", {})
        
        # Configure audit logging settings
        self.enabled = get_config_value(self.audit_config, "enabled", True)
        self.log_dir = Path(get_config_value(self.audit_config, "log_dir", "logs/audit"))
        self.retention_days = get_config_value(self.audit_config, "retention_days", 90)
        self.buffer_size = get_config_value(self.audit_config, "buffer_size", 1000)
        self.flush_interval = get_config_value(self.audit_config, "flush_interval", 60)
        self.sensitive_fields = set(get_config_value(self.audit_config, "sensitive_fields", [
            "password", "api_key", "token", "secret", "credential", "ssn", "credit_card"
        ]))
        
        # Initialize audit log buffer
        self.log_buffer = deque(maxlen=self.buffer_size)
        
        # Thread lock for thread safety
        self.lock = threading.RLock()
        
        # Flush task
        self.flush_task = None

        # Create log directory
        os.makedirs(self.log_dir, exist_ok=True)

        logger.info("Audit logger initialized")
        
    async def initialize(self) -> None:
        """Initialize the audit logger and start background tasks."""
        if not self.enabled:
            logger.info("Audit logging is disabled")
            return

        logger.info("Initializing audit logger...")

        # Start periodic flush task
        if self.flush_interval > 0:
            self.flush_task = asyncio.create_task(self._periodic_flush())

        # Log system startup event
        await self.log_system_event(
            event_type="system_startup",
            details={
                "timestamp": datetime.now().isoformat(),
                "environment": get_config_value(self.config, "environment", "development")
            }
        )

        # Start retention cleanup
        retention_interval = get_config_value(self.audit_config, "cleanup_interval", 86400)
        if retention_interval > 0:
            asyncio.create_task(self._periodic_cleanup(retention_interval))

        logger.info("Audit logger initialization complete")
        
    async def _periodic_flush(self) -> None:
        """Periodically flush the audit log buffer to disk."""
        try:
            while True:
                await asyncio.sleep(self.flush_interval)
                await self.flush()
        except asyncio.CancelledError:
            logger.info("Audit log flush task cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in audit log flush task: {str(e)}", exc_info=True)
            
    async def _periodic_cleanup(self, interval: int) -> None:
        """
        Periodically clean up old audit logs.
        
        Args:
            interval: Cleanup interval in seconds
        """
        try:
            while True:
                await asyncio.sleep(interval)
                await self._cleanup_old_logs()
        except asyncio.CancelledError:
            logger.info("Audit log cleanup task cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in audit log cleanup task: {str(e)}", exc_info=True)
            
    async def log_user_action(
        self,
        user_id: str,
        action: str,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        source_ip: Optional[str] = None,
        status: str = "success",
        correlation_id: Optional[str] = None
    ) -> str:
        """
        Log a user action.
        
        Args:
            user_id: User identifier
            action: Action performed
            resource_type: Type of resource acted upon
            resource_id: Identifier of resource acted upon
            details: Additional details about the action
            source_ip: Source IP address
            status: Action status (success, failure, etc.)
            correlation_id: Identifier for correlating related actions
            
        Returns:
            Audit log entry ID
        """
        if not self.enabled:
            return ""
            
        # Generate log entry ID
        entry_id = str(uuid.uuid4())
        
        # Create log entry
        log_entry = {
            "id": entry_id,
            "timestamp": datetime.now().isoformat(),
            "type": "user_action",
            "user_id": user_id,
            "action": action,
            "resource_type": resource_type,
            "resource_id": resource_id,
            "source_ip": source_ip,
            "status": status,
            "correlation_id": correlation_id or str(uuid.uuid4()),
            "details": self._sanitize_sensitive_data(details or {})
        }
        
        # Add to buffer
        with self.lock:
            self.log_buffer.append(log_entry)
            
        return entry_id
        
    async def log_system_event(
        self,
        event_type: str,
        details: Optional[Dict[str, Any]] = None,
        severity: str = "info",
        correlation_id: Optional[str] = None
    ) -> str:
        """
        Log a system event.
        
        Args:
            event_type: Type of system event
            details: Additional details about the event
            severity: Event severity (info, warning, error, etc.)
            correlation_id: Identifier for correlating related events
            
        Returns:
            Audit log entry ID
        """
        if not self.enabled:
            return ""
            
        # Generate log entry ID
        entry_id = str(uuid.uuid4())
        
        # Create log entry
        log_entry = {
            "id": entry_id,
            "timestamp": datetime.now().isoformat(),
            "type": "system_event",
            "event_type": event_type,
            "severity": severity,
            "correlation_id": correlation_id or str(uuid.uuid4()),
            "details": self._sanitize_sensitive_data(details or {})
        }
        
        # Add to buffer
        with self.lock:
            self.log_buffer.append(log_entry)
            
        return entry_id
        
    async def log_api_request(
        self,
        endpoint: str,
        method: str,
        user_id: Optional[str] = None,
        source_ip: Optional[str] = None,
        request_id: Optional[str] = None,
        request_params: Optional[Dict[str, Any]] = None,
        status_code: Optional[int] = None,
        response_time: Optional[float] = None,
        error_message: Optional[str] = None,
        correlation_id: Optional[str] = None
    ) -> str:
        """
        Log an API request.
        
        Args:
            endpoint: API endpoint path
            method: HTTP method
            user_id: User identifier
            source_ip: Source IP address
            request_id: Request identifier
            request_params: Request parameters
            status_code: HTTP status code
            response_time: Request processing time in seconds
            error_message: Error message if request failed
            correlation_id: Identifier for correlating related requests
            
        Returns:
            Audit log entry ID
        """
        if not self.enabled:
            return ""
            
        # Generate log entry ID
        entry_id = str(uuid.uuid4())
        
        # Determine status
        status = "success"
        if status_code and status_code >= 400:
            status = "failure"
            
        # Create log entry
        log_entry = {
            "id": entry_id,
            "timestamp": datetime.now().isoformat(),
            "type": "api_request",
            "endpoint": endpoint,
            "method": method,
            "user_id": user_id,
            "source_ip": source_ip,
            "request_id": request_id or str(uuid.uuid4()),
            "request_params": self._sanitize_sensitive_data(request_params or {}),
            "status_code": status_code,
            "status": status,
            "response_time": response_time,
            "error_message": error_message,
            "correlation_id": correlation_id or str(uuid.uuid4())
        }
        
        # Add to buffer
        with self.lock:
            self.log_buffer.append(log_entry)
            
        return entry_id
        
    async def log_authentication_event(
        self,
        event_type: str,
        user_id: str,
        source_ip: Optional[str] = None,
        success: bool = True,
        details: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None
    ) -> str:
        """
        Log an authentication event.
        
        Args:
            event_type: Type of authentication event
            user_id: User identifier
            source_ip: Source IP address
            success: Whether authentication was successful
            details: Additional details about the event
            correlation_id: Identifier for correlating related events
            
        Returns:
            Audit log entry ID
        """
        if not self.enabled:
            return ""
            
        # Generate log entry ID
        entry_id = str(uuid.uuid4())
        
        # Create log entry
        log_entry = {
            "id": entry_id,
            "timestamp": datetime.now().isoformat(),
            "type": "authentication",
            "event_type": event_type,
            "user_id": user_id,
            "source_ip": source_ip,
            "success": success,
            "status": "success" if success else "failure",
            "correlation_id": correlation_id or str(uuid.uuid4()),
            "details": self._sanitize_sensitive_data(details or {})
        }
        
        # Add to buffer
        with self.lock:
            self.log_buffer.append(log_entry)
            
        return entry_id
        
    async def log_model_operation(
        self,
        model_id: str,
        operation: str,
        user_id: Optional[str] = None,
        input_metadata: Optional[Dict[str, Any]] = None,
        output_metadata: Optional[Dict[str, Any]] = None,
        duration: Optional[float] = None,
        status: str = "success",
        error_message: Optional[str] = None,
        correlation_id: Optional[str] = None
    ) -> str:
        """
        Log a model operation.
        
        Args:
            model_id: Model identifier
            operation: Operation performed
            user_id: User identifier
            input_metadata: Metadata about input (not the actual input)
            output_metadata: Metadata about output (not the actual output)
            duration: Operation duration in seconds
            status: Operation status
            error_message: Error message if operation failed
            correlation_id: Identifier for correlating related operations
            
        Returns:
            Audit log entry ID
        """
        if not self.enabled:
            return ""
            
        # Generate log entry ID
        entry_id = str(uuid.uuid4())
        
        # Create log entry
        log_entry = {
            "id": entry_id,
            "timestamp": datetime.now().isoformat(),
            "type": "model_operation",
            "model_id": model_id,
            "operation": operation,
            "user_id": user_id,
            "duration": duration,
            "status": status,
            "error_message": error_message,
            "correlation_id": correlation_id or str(uuid.uuid4()),
            "input_metadata": self._sanitize_sensitive_data(input_metadata or {}),
            "output_metadata": self._sanitize_sensitive_data(output_metadata or {})
        }
        
        # Add to buffer
        with self.lock:
            self.log_buffer.append(log_entry)
            
        return entry_id
        
    async def log_data_access(
        self,
        data_type: str,
        action: str,
        user_id: str,
        data_id: Optional[str] = None,
        access_reason: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        status: str = "success",
        correlation_id: Optional[str] = None
    ) -> str:
        """
        Log a data access event.
        
        Args:
            data_type: Type of data accessed
            action: Action performed on data
            user_id: User identifier
            data_id: Identifier of accessed data
            access_reason: Reason for access
            metadata: Additional metadata
            status: Access status
            correlation_id: Identifier for correlating related events
            
        Returns:
            Audit log entry ID
        """
        if not self.enabled:
            return ""
            
        # Generate log entry ID
        entry_id = str(uuid.uuid4())
        
        # Create log entry
        log_entry = {
            "id": entry_id,
            "timestamp": datetime.now().isoformat(),
            "type": "data_access",
            "data_type": data_type,
            "action": action,
            "user_id": user_id,
            "data_id": data_id,
            "access_reason": access_reason,
            "status": status,
            "correlation_id": correlation_id or str(uuid.uuid4()),
            "metadata": self._sanitize_sensitive_data(metadata or {})
        }
        
        # Add to buffer
        with self.lock:
            self.log_buffer.append(log_entry)
            
        return entry_id
        
    async def log_config_change(
        self,
        user_id: str,
        component: str,
        setting: str,
        old_value: Any,
        new_value: Any,
        reason: Optional[str] = None,
        correlation_id: Optional[str] = None
    ) -> str:
        """
        Log a configuration change.
        
        Args:
            user_id: User who made the change
            component: Component being configured
            setting: Setting that was changed
            old_value: Previous value
            new_value: New value
            reason: Reason for the change
            correlation_id: Identifier for correlating related events
            
        Returns:
            Audit log entry ID
        """
        if not self.enabled:
            return ""
            
        # Generate log entry ID
        entry_id = str(uuid.uuid4())
        
        # Create log entry
        log_entry = {
            "id": entry_id,
            "timestamp": datetime.now().isoformat(),
            "type": "config_change",
            "user_id": user_id,
            "component": component,
            "setting": setting,
            "old_value": self._sanitize_sensitive_field(setting, old_value),
            "new_value": self._sanitize_sensitive_field(setting, new_value),
            "reason": reason,
            "correlation_id": correlation_id or str(uuid.uuid4())
        }
        
        # Add to buffer
        with self.lock:
            self.log_buffer.append(log_entry)
            
        return entry_id
        
    async def log_security_event(
        self,
        event_type: str,
        severity: str,
        user_id: Optional[str] = None,
        source_ip: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None
    ) -> str:
        """
        Log a security-related event.
        
        Args:
            event_type: Type of security event
            severity: Event severity
            user_id: Associated user identifier
            source_ip: Source IP address
            details: Additional details
            correlation_id: Identifier for correlating related events
            
        Returns:
            Audit log entry ID
        """
        if not self.enabled:
            return ""
            
        # Generate log entry ID
        entry_id = str(uuid.uuid4())
        
        # Create log entry
        log_entry = {
            "id": entry_id,
            "timestamp": datetime.now().isoformat(),
            "type": "security_event",
            "event_type": event_type,
            "severity": severity,
            "user_id": user_id,
            "source_ip": source_ip,
            "correlation_id": correlation_id or str(uuid.uuid4()),
            "details": self._sanitize_sensitive_data(details or {})
        }
        
        # Add to buffer
        with self.lock:
            self.log_buffer.append(log_entry)
            
        # Log critical security events to the standard logger as well
        if severity == "critical":
            logger.critical(
                f"Security event: {event_type} - " +
                f"User: {user_id}, IP: {source_ip}, Details: {details}"
            )
        elif severity == "high":
            logger.error(
                f"Security event: {event_type} - " +
                f"User: {user_id}, IP: {source_ip}"
            )
            
        return entry_id
        
    def _sanitize_sensitive_field(self, field_name: str, value: Any) -> Any:
        """
        Sanitize potentially sensitive field values.
        
        Args:
            field_name: Field name
            value: Field value
            
        Returns:
            Sanitized value
        """
        # Check if field name is sensitive
        if any(sensitive in field_name.lower() for sensitive in self.sensitive_fields):
            if isinstance(value, str):
                if len(value) > 0:
                    return "[REDACTED]"
            return "[REDACTED]"
            
        return value
        
    def _sanitize_sensitive_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize dictionary to remove sensitive data.
        
        Args:
            data: Dictionary to sanitize
            
        Returns:
            Sanitized dictionary
        """
        sanitized = {}
        
        for key, value in data.items():
            if isinstance(value, dict):
                sanitized[key] = self._sanitize_sensitive_data(value)
            else:
                sanitized[key] = self._sanitize_sensitive_field(key, value)
                
        return sanitized
        
    async def flush(self) -> None:
        """Flush the audit log buffer to disk."""
        if not self.enabled:
            return

        with self.lock:
            if not self.log_buffer:
                return

            # Get log entries to write
            entries = list(self.log_buffer)
            self.log_buffer.clear()

        try:
            # Organize entries by date
            entries_by_date = {}
            for entry in entries:
                try:
                    timestamp = datetime.fromisoformat(entry["timestamp"].replace("Z", "+00:00"))
                except Exception:
                    timestamp = datetime.now()
                date_str = timestamp.strftime("%Y-%m-%d")

                if date_str not in entries_by_date:
                    entries_by_date[date_str] = []

                entries_by_date[date_str].append(entry)

            # Write entries to date-specific files
            for date_str, date_entries in entries_by_date.items():
                log_file = self.log_dir / f"audit_{date_str}.jsonl"

                with open(log_file, "a", encoding="utf-8") as f:
                    for entry in date_entries:
                        f.write(json.dumps(entry) + "\n")

            logger.debug(f"Flushed {len(entries)} audit log entries")

        except Exception as e:
            logger.error(f"Error flushing audit logs: {str(e)}", exc_info=True)

            # Put entries back in buffer if write failed
            with self.lock:
                for entry in entries:
                    self.log_buffer.append(entry)
                    
    async def _cleanup_old_logs(self) -> None:
        """Clean up audit logs older than retention period."""
        if not self.enabled or self.retention_days <= 0:
            return
            
        try:
            # Calculate cutoff date
            cutoff_date = datetime.now() - timedelta(days=self.retention_days)
            logger.info(f"Cleaning up audit logs older than {cutoff_date.isoformat()}")
            
            deleted_count = 0
            for file_path in self.log_dir.glob("audit_*.jsonl"):
                if file_path.is_file():
                    # Extract date from filename
                    try:
                        date_str = file_path.name.split("_")[1].split(".")[0]
                        file_date = datetime.strptime(date_str, "%Y-%m-%d")
                        
                        if file_date < cutoff_date:
                            file_path.unlink()
                            deleted_count += 1
                    except (IndexError, ValueError):
                        logger.warning(f"Could not parse date from filename: {file_path.name}")
                        
            if deleted_count > 0:
                logger.info(f"Deleted {deleted_count} old audit log files")
                
        except Exception as e:
            logger.error(f"Error cleaning up old audit logs: {str(e)}", exc_info=True)
            
    async def search_logs(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        log_type: Optional[str] = None,
        user_id: Optional[str] = None,
        status: Optional[str] = None,
        correlation_id: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Search audit logs based on criteria.
        
        Args:
            start_time: Start of time range
            end_time: End of time range
            log_type: Type of log entries to return
            user_id: Filter by user ID
            status: Filter by status
            correlation_id: Filter by correlation ID
            limit: Maximum number of entries to return
            
        Returns:
            List of matching audit log entries
        """
        if not self.enabled:
            return []
            
        try:
            # Default time range if not specified
            if not start_time:
                start_time = datetime.now() - timedelta(days=7)
            if not end_time:
                end_time = datetime.now()
                
            # Generate list of dates to search
            current_date = start_time.date()
            end_date = end_time.date()
            dates_to_search = []
            
            while current_date <= end_date:
                dates_to_search.append(current_date.strftime("%Y-%m-%d"))
                current_date += timedelta(days=1)
                
            # Search through log files
            results = []
            for date_str in dates_to_search:
                log_file = self.log_dir / f"audit_{date_str}.jsonl"
                
                if not log_file.exists():
                    continue
                    
                with open(log_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            try:
                                entry = json.loads(line)
                                
                                # Check if entry matches search criteria
                                if self._matches_criteria(
                                    entry, start_time, end_time, log_type, 
                                    user_id, status, correlation_id
                                ):
                                    results.append(entry)
                                    
                                    # Check limit
                                    if len(results) >= limit:
                                        break
                            except json.JSONDecodeError:
                                logger.warning(f"Invalid JSON in audit log: {log_file}")
                                
                # Check limit
                if len(results) >= limit:
                    break
                    
            # Sort results by timestamp (newest first)
            results.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
            
            return results[:limit]
            
        except Exception as e:
            logger.error(f"Error searching audit logs: {str(e)}", exc_info=True)
            return []
            
    def _matches_criteria(
        self,
        entry: Dict[str, Any],
        start_time: datetime,
        end_time: datetime,
        log_type: Optional[str],
        user_id: Optional[str],
        status: Optional[str],
        correlation_id: Optional[str]
    ) -> bool:
        """
        Check if log entry matches search criteria.
        
        Args:
            entry: Log entry to check
            start_time: Start of time range
            end_time: End of time range
            log_type: Type of log entries to return
            user_id: Filter by user ID
            status: Filter by status
            correlation_id: Filter by correlation ID
            
        Returns:
            True if entry matches criteria
        """
        # Check timestamp
        try:
            timestamp = datetime.fromisoformat(entry["timestamp"].replace("Z", "+00:00"))
            if not (start_time <= timestamp <= end_time):
                return False
        except (ValueError, KeyError):
            return False
            
        # Check log type
        if log_type and entry.get("type") != log_type:
            return False
            
        # Check user ID
        if user_id and entry.get("user_id") != user_id:
            return False
            
        # Check status
        if status and entry.get("status") != status:
            return False
            
        # Check correlation ID
        if correlation_id and entry.get("correlation_id") != correlation_id:
            return False
            
        return True
        
    async def get_user_activity(
        self,
        user_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get activity for a specific user.
        
        Args:
            user_id: User identifier
            start_time: Start of time range
            end_time: End of time range
            limit: Maximum number of entries to return
            
        Returns:
            List of user activity log entries
        """
        return await self.search_logs(
            start_time=start_time,
            end_time=end_time,
            user_id=user_id,
            limit=limit
        )
        
    async def get_related_events(
        self,
        correlation_id: str,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get events related by correlation ID.
        
        Args:
            correlation_id: Correlation identifier
            limit: Maximum number of entries to return
            
        Returns:
            List of related log entries
        """
        return await self.search_logs(
            correlation_id=correlation_id,
            limit=limit
        )
        
    async def get_security_events(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        severity: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get security events.
        
        Args:
            start_time: Start of time range
            end_time: End of time range
            severity: Filter by severity
            limit: Maximum number of entries to return
            
        Returns:
            List of security event log entries
        """
        events = await self.search_logs(
            start_time=start_time,
            end_time=end_time,
            log_type="security_event",
            limit=limit
        )
        
        # Filter by severity if specified
        if severity:
            events = [e for e in events if e.get("severity") == severity]
            
        return events[:limit]