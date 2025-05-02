"""
Metrics Collection Module for CasaLingua

This module provides comprehensive metrics collection, aggregation,
and reporting capabilities for monitoring the performance and usage
of the CasaLingua language processing pipeline.

Author: Exygy Development Team
Version: 1.0.0
License: MIT
"""
import time
import os
import json
import threading
import math
import statistics
from collections import defaultdict
from pathlib import Path
from datetime import datetime, timedelta
import asyncio
from typing import Dict, List, Any, Optional, Union, Tuple

# Third-party imports (none)

# Local imports
from app.utils.config import load_config, get_config_value
from app.utils.logging import get_logger

logger = get_logger(__name__)

class MetricsCollector:
    """
    Collects, aggregates, and reports performance metrics for
    the CasaLingua language processing pipeline.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the metrics collector.
        
        Args:
            config: Application configuration
        """
        self.config = config or load_config()
        self.metrics_config = get_config_value(self.config, "metrics", {})
        
        # Initialize metrics storage
        self.metrics: Dict[str, Dict[str, Any]] = {}
        self.request_times: Dict[str, List[float]] = defaultdict(list)
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.status_codes: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self.pipeline_metrics: Dict[str, Dict[str, Any]] = {}
        self.model_metrics: Dict[str, Dict[str, Any]] = {}
        self.language_metrics: Dict[str, Dict[str, Any]] = {}
        
        # Initialize time series data
        self.time_series: Dict[str, List[Dict[str, Any]]] = {}
        
        # Configure metrics settings
        self.enabled = get_config_value(self.metrics_config, "enabled", True)
        self.save_interval = get_config_value(self.metrics_config, "save_interval", 3600)  # seconds
        self.metrics_dir = Path(get_config_value(self.metrics_config, "metrics_dir", "logs/metrics"))
        self.retention_days = get_config_value(self.metrics_config, "retention_days", 30)
        self.detailed_logging = get_config_value(self.metrics_config, "detailed_logging", False)
        
        # Performance threshold mappings
        self.thresholds = get_config_value(self.metrics_config, "thresholds", {
            "request_time_warning": 1.0,  # seconds
            "request_time_critical": 3.0,  # seconds
            "error_rate_warning": 0.05,    # 5%
            "error_rate_critical": 0.10,    # 10%
            "memory_usage_warning": 0.80,  # 80%
            "memory_usage_critical": 0.95   # 95%
        })
        
        # Memory and CPU metrics
        self.system_metrics: Dict[str, Any] = {
            "start_time": time.time(),
            "peak_memory_usage": 0,
            "current_memory_usage": 0,
            "total_requests": 0,
            "total_errors": 0,
            "uptime_seconds": 0
        }
        
        # Thread lock for thread safety
        self.lock = threading.RLock()
        
        # Ensure metrics directory exists
        os.makedirs(self.metrics_dir, exist_ok=True)
        
        # Start periodic saving background task if enabled
        self.save_task = None
        if self.enabled and self.save_interval > 0:
            self.save_task = asyncio.create_task(self._periodic_save())
            
        logger.info("Metrics collector initialized")
        
    async def _periodic_save(self) -> None:
        """
        Periodically save metrics to disk and clean up old files.
        """
        try:
            while True:
                await asyncio.sleep(self.save_interval)
                self.save_metrics()
                await self._cleanup_old_metrics()
        except asyncio.CancelledError:
            logger.info("Metrics save task cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in metrics save task: {str(e)}", exc_info=True)
            
    def record_request(
        self,
        endpoint: str,
        success: bool,
        duration: float,
        status_code: int = 200,
        payload_size: Optional[int] = None
    ) -> None:
        """
        Record API request metrics.

        Args:
            endpoint (str): API endpoint path.
            success (bool): Whether request was successful.
            duration (float): Request processing time in seconds.
            status_code (int, optional): HTTP status code.
            payload_size (Optional[int], optional): Size of request/response payload in bytes.
        Returns:
            None
        """
        if not self.enabled:
            return
            
        with self.lock:
            # Update request times
            self.request_times[endpoint].append(duration)
            
            # Limit the size of request_times lists to prevent memory issues
            max_samples = get_config_value(self.metrics_config, "max_samples", 1000)
            if len(self.request_times[endpoint]) > max_samples:
                self.request_times[endpoint] = self.request_times[endpoint][-max_samples:]
                
            # Update error counts
            if not success:
                self.error_counts[endpoint] += 1
                
            # Update status code counts
            self.status_codes[endpoint][status_code] += 1
            
            # Update system metrics
            self.system_metrics["total_requests"] += 1
            if not success:
                self.system_metrics["total_errors"] += 1
                
            # Record time series data
            self._record_time_series("requests", {
                "endpoint": endpoint,
                "duration": duration,
                "success": success,
                "status_code": status_code,
                "payload_size": payload_size
            })
            
            # Log detailed metrics if enabled
            if self.detailed_logging:
                logger.debug(
                    f"Request metrics: endpoint={endpoint}, success={success}, "
                    f"duration={duration:.4f}s, status={status_code}"
                )
                
    def record_pipeline_execution(
        self,
        pipeline_id: str,
        operation: str,
        duration: float,
        input_size: int,
        output_size: int,
        success: bool,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Record pipeline execution metrics.

        Args:
            pipeline_id (str): Pipeline identifier.
            operation (str): Operation performed (e.g., translation, transcription).
            duration (float): Execution time in seconds.
            input_size (int): Size of input in characters/tokens.
            output_size (int): Size of output in characters/tokens.
            success (bool): Whether execution was successful.
            metadata (Optional[Dict[str, Any]], optional): Additional execution metadata.
        Returns:
            None
        """
        if not self.enabled:
            return
            
        with self.lock:
            pipeline_key = f"{pipeline_id}:{operation}"
            
            if pipeline_key not in self.pipeline_metrics:
                now = datetime.utcnow().timestamp()
                self.pipeline_metrics[pipeline_key] = {
                    "total_executions": 0,
                    "successful_executions": 0,
                    "total_duration": 0.0,
                    "durations": [],
                    "input_sizes": [],
                    "output_sizes": [],
                    "first_execution": now,
                    "last_execution": now
                }
                
            metrics = self.pipeline_metrics[pipeline_key]
            
            # Update execution counts
            metrics["total_executions"] += 1
            if success:
                metrics["successful_executions"] += 1
                
            # Update timing metrics
            metrics["total_duration"] += duration
            metrics["durations"].append(duration)
            
            # Limit the size of durations list
            max_samples = get_config_value(self.metrics_config, "max_samples", 1000)
            if len(metrics["durations"]) > max_samples:
                metrics["durations"] = metrics["durations"][-max_samples:]
                
            # Update size metrics
            metrics["input_sizes"].append(input_size)
            metrics["output_sizes"].append(output_size)
            
            # Update timestamp
            metrics["last_execution"] = datetime.utcnow().timestamp()
            
            # Record time series data
            self._record_time_series("pipeline", {
                "pipeline_id": pipeline_id,
                "operation": operation,
                "duration": duration,
                "input_size": input_size,
                "output_size": output_size,
                "success": success,
                "metadata": metadata
            })
            
    def record_model_usage(
        self,
        model_id: str,
        operation: str,
        duration: float,
        input_tokens: int,
        output_tokens: int,
        success: bool
    ) -> None:
        """
        Record model usage metrics.

        Args:
            model_id (str): Model identifier.
            operation (str): Operation performed.
            duration (float): Execution time in seconds.
            input_tokens (int): Number of input tokens.
            output_tokens (int): Number of output tokens.
            success (bool): Whether execution was successful.
        Returns:
            None
        """
        if not self.enabled:
            return
            
        with self.lock:
            if model_id not in self.model_metrics:
                now = datetime.utcnow().timestamp()
                self.model_metrics[model_id] = {
                    "total_uses": 0,
                    "successful_uses": 0,
                    "total_duration": 0.0,
                    "total_input_tokens": 0,
                    "total_output_tokens": 0,
                    "operations": defaultdict(int),
                    "durations": [],
                    "first_use": now,
                    "last_use": now
                }
                
            metrics = self.model_metrics[model_id]
            
            # Update usage counts
            metrics["total_uses"] += 1
            if success:
                metrics["successful_uses"] += 1
                
            # Update timing metrics
            metrics["total_duration"] += duration
            metrics["durations"].append(duration)
            
            # Limit the size of durations list
            max_samples = get_config_value(self.metrics_config, "max_samples", 1000)
            if len(metrics["durations"]) > max_samples:
                metrics["durations"] = metrics["durations"][-max_samples:]
                
            # Update token metrics
            metrics["total_input_tokens"] += input_tokens
            metrics["total_output_tokens"] += output_tokens
            
            # Update operation counts
            metrics["operations"][operation] += 1
            
            # Update timestamp
            metrics["last_use"] = datetime.utcnow().timestamp()
            
            # Record time series data
            self._record_time_series("model", {
                "model_id": model_id,
                "operation": operation,
                "duration": duration,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "success": success
            })
            
    def record_language_operation(
        self,
        source_lang: str,
        target_lang: Optional[str],
        operation: str,
        duration: float,
        input_size: int,
        output_size: int,
        success: bool
    ) -> None:
        """
        Record language operation metrics.

        Args:
            source_lang (str): Source language code.
            target_lang (Optional[str]): Target language code (if applicable).
            operation (str): Operation performed.
            duration (float): Execution time in seconds.
            input_size (int): Size of input in characters/tokens.
            output_size (int): Size of output in characters/tokens.
            success (bool): Whether operation was successful.
        Returns:
            None
        """
        if not self.enabled:
            return
            
        with self.lock:
            # Create language pair key
            lang_key = f"{source_lang}"
            if target_lang:
                lang_key = f"{source_lang}-{target_lang}"
                
            operation_key = f"{lang_key}:{operation}"
            
            if operation_key not in self.language_metrics:
                now = datetime.utcnow().timestamp()
                self.language_metrics[operation_key] = {
                    "total_operations": 0,
                    "successful_operations": 0,
                    "total_duration": 0.0,
                    "durations": [],
                    "input_sizes": [],
                    "output_sizes": [],
                    "first_operation": now,
                    "last_operation": now,
                    "source_language": source_lang,
                    "target_language": target_lang
                }
                
            metrics = self.language_metrics[operation_key]
            
            # Update operation counts
            metrics["total_operations"] += 1
            if success:
                metrics["successful_operations"] += 1
                
            # Update timing metrics
            metrics["total_duration"] += duration
            metrics["durations"].append(duration)
            
            # Limit the size of durations list
            max_samples = get_config_value(self.metrics_config, "max_samples", 1000)
            if len(metrics["durations"]) > max_samples:
                metrics["durations"] = metrics["durations"][-max_samples:]
                
            # Update size metrics
            metrics["input_sizes"].append(input_size)
            metrics["output_sizes"].append(output_size)
            
            # Update timestamp
            metrics["last_operation"] = datetime.utcnow().timestamp()
            
            # Record time series data
            self._record_time_series("language", {
                "source_lang": source_lang,
                "target_lang": target_lang,
                "operation": operation,
                "duration": duration,
                "input_size": input_size,
                "output_size": output_size,
                "success": success
            })
            
    def record_system_metrics(
        self,
        memory_usage: float,
        cpu_usage: float,
        disk_usage: float,
        gpu_usage: Optional[float] = None
    ) -> None:
        """
        Record system resource usage metrics.

        Args:
            memory_usage (float): Memory usage as a fraction (0-1).
            cpu_usage (float): CPU usage as a fraction (0-1).
            disk_usage (float): Disk usage as a fraction (0-1).
            gpu_usage (Optional[float], optional): GPU usage as a fraction (0-1) if available.
        Returns:
            None
        """
        if not self.enabled:
            return
            
        with self.lock:
            # Update system metrics
            self.system_metrics["current_memory_usage"] = memory_usage
            self.system_metrics["peak_memory_usage"] = max(
                self.system_metrics["peak_memory_usage"], memory_usage
            )
            self.system_metrics["uptime_seconds"] = datetime.utcnow().timestamp() - self.system_metrics["start_time"]
            
            # Record time series data
            self._record_time_series("system", {
                "memory_usage": memory_usage,
                "cpu_usage": cpu_usage,
                "disk_usage": disk_usage,
                "gpu_usage": gpu_usage,
                "uptime_seconds": self.system_metrics["uptime_seconds"]
            })
            
            # Log warnings for high resource usage
            if memory_usage > self.thresholds.get("memory_usage_critical", 0.95):
                logger.warning(f"Critical memory usage: {memory_usage:.1%}")
            elif memory_usage > self.thresholds.get("memory_usage_warning", 0.80):
                logger.info(f"High memory usage: {memory_usage:.1%}")
                
    # --- Private and helper methods grouped below ---
    def _record_time_series(self, series_name: str, data: Dict[str, Any]) -> None:
        """
        Record data point in time series, adding an ISO8601 UTC timestamp.

        Args:
            series_name (str): Name of the time series.
            data (Dict[str, Any]): Data point to record.
        Returns:
            None
        """
        # Initialize time series if needed
        if series_name not in self.time_series:
            self.time_series[series_name] = []

        # Add ISO8601 UTC timestamp to data
        data["timestamp"] = datetime.utcnow().isoformat()

        # Append to time series
        self.time_series[series_name].append(data)

        # Limit time series size to prevent memory issues
        max_points = get_config_value(self.metrics_config, "max_time_series_points", 1000)
        if len(self.time_series[series_name]) > max_points:
            self.time_series[series_name] = self.time_series[series_name][-max_points:]
            
    def get_api_metrics(self) -> Dict[str, Any]:
        """
        Get aggregated API metrics.
        
        Returns:
            Dictionary with API metrics
        """
        if not self.enabled:
            return {}
            
        with self.lock:
            metrics = {
                "endpoints": {},
                "overall": {
                    "total_requests": self.system_metrics["total_requests"],
                    "total_errors": self.system_metrics["total_errors"],
                    "error_rate": 0.0,
                    "avg_response_time": 0.0,
                    "p95_response_time": 0.0,
                    "status_code_counts": {}
                }
            }
            
            # Calculate overall error rate
            if metrics["overall"]["total_requests"] > 0:
                metrics["overall"]["error_rate"] = (
                    metrics["overall"]["total_errors"] / metrics["overall"]["total_requests"]
                )
                
            # Calculate overall response times
            all_response_times = []
            for times in self.request_times.values():
                all_response_times.extend(times)
                
            if all_response_times:
                metrics["overall"]["avg_response_time"] = statistics.mean(all_response_times)
                metrics["overall"]["p95_response_time"] = self._percentile(all_response_times, 95)
                
            # Aggregate status codes
            all_status_codes = defaultdict(int)
            for endpoint_codes in self.status_codes.values():
                for code, count in endpoint_codes.items():
                    all_status_codes[code] += count
            metrics["overall"]["status_code_counts"] = dict(all_status_codes)
            
            # Calculate per-endpoint metrics
            for endpoint, times in self.request_times.items():
                if not times:
                    continue
                    
                error_count = self.error_counts.get(endpoint, 0)
                request_count = len(times)
                
                endpoint_metrics = {
                    "request_count": request_count,
                    "error_count": error_count,
                    "error_rate": error_count / request_count if request_count > 0 else 0.0,
                    "avg_response_time": statistics.mean(times),
                    "min_response_time": min(times),
                    "max_response_time": max(times),
                    "p95_response_time": self._percentile(times, 95),
                    "status_codes": dict(self.status_codes.get(endpoint, {}))
                }
                
                metrics["endpoints"][endpoint] = endpoint_metrics
                
            return metrics
            
    def get_pipeline_metrics(self) -> Dict[str, Any]:
        """
        Get aggregated pipeline metrics.
        
        Returns:
            Dictionary with pipeline metrics
        """
        if not self.enabled or not self.pipeline_metrics:
            return {}
            
        with self.lock:
            metrics = {
                "pipelines": {},
                "overall": {
                    "total_executions": 0,
                    "successful_executions": 0,
                    "success_rate": 0.0,
                    "avg_execution_time": 0.0
                }
            }
            
            total_executions = 0
            total_successful = 0
            all_durations = []
            
            # Process each pipeline's metrics
            for pipeline_key, pipeline_data in self.pipeline_metrics.items():
                pipeline_id, operation = pipeline_key.split(":", 1)
                
                # Calculate statistics
                durations = pipeline_data["durations"]
                if not durations:
                    continue
                    
                total_executions += pipeline_data["total_executions"]
                total_successful += pipeline_data["successful_executions"]
                all_durations.extend(durations)
                
                success_rate = (
                    pipeline_data["successful_executions"] / 
                    pipeline_data["total_executions"]
                ) if pipeline_data["total_executions"] > 0 else 0.0
                
                # Calculate size metrics
                input_sizes = pipeline_data["input_sizes"]
                output_sizes = pipeline_data["output_sizes"]
                
                avg_input_size = statistics.mean(input_sizes) if input_sizes else 0
                avg_output_size = statistics.mean(output_sizes) if output_sizes else 0
                
                # Calculate throughput (characters per second)
                avg_throughput = (
                    avg_output_size / statistics.mean(durations)
                ) if durations and statistics.mean(durations) > 0 else 0
                
                # Add to metrics
                if pipeline_id not in metrics["pipelines"]:
                    metrics["pipelines"][pipeline_id] = {
                        "operations": {}
                    }
                    
                metrics["pipelines"][pipeline_id]["operations"][operation] = {
                    "total_executions": pipeline_data["total_executions"],
                    "successful_executions": pipeline_data["successful_executions"],
                    "success_rate": success_rate,
                    "avg_execution_time": statistics.mean(durations),
                    "min_execution_time": min(durations),
                    "max_execution_time": max(durations),
                    "p95_execution_time": self._percentile(durations, 95),
                    "avg_input_size": avg_input_size,
                    "avg_output_size": avg_output_size,
                    "avg_throughput": avg_throughput,
                    "first_execution": datetime.fromtimestamp(
                        pipeline_data["first_execution"]
                    ).isoformat(),
                    "last_execution": datetime.fromtimestamp(
                        pipeline_data["last_execution"]
                    ).isoformat()
                }
                
            # Calculate overall metrics
            metrics["overall"]["total_executions"] = total_executions
            metrics["overall"]["successful_executions"] = total_successful
            
            if total_executions > 0:
                metrics["overall"]["success_rate"] = total_successful / total_executions
                
            if all_durations:
                metrics["overall"]["avg_execution_time"] = statistics.mean(all_durations)
                metrics["overall"]["p95_execution_time"] = self._percentile(all_durations, 95)
                
            return metrics
            
    def get_model_metrics(self) -> Dict[str, Any]:
        """
        Get aggregated model usage metrics.
        
        Returns:
            Dictionary with model metrics
        """
        if not self.enabled or not self.model_metrics:
            return {}
            
        with self.lock:
            metrics = {
                "models": {},
                "overall": {
                    "total_uses": 0,
                    "successful_uses": 0,
                    "success_rate": 0.0,
                    "total_input_tokens": 0,
                    "total_output_tokens": 0,
                    "avg_execution_time": 0.0
                }
            }
            
            total_uses = 0
            total_successful = 0
            total_input_tokens = 0
            total_output_tokens = 0
            all_durations = []
            
            # Process each model's metrics
            for model_id, model_data in self.model_metrics.items():
                # Calculate statistics
                durations = model_data["durations"]
                if not durations:
                    continue
                    
                total_uses += model_data["total_uses"]
                total_successful += model_data["successful_uses"]
                total_input_tokens += model_data["total_input_tokens"]
                total_output_tokens += model_data["total_output_tokens"]
                all_durations.extend(durations)
                
                success_rate = (
                    model_data["successful_uses"] / 
                    model_data["total_uses"]
                ) if model_data["total_uses"] > 0 else 0.0
                
                # Calculate tokens per second
                tokens_per_second = 0
                if model_data["total_duration"] > 0:
                    total_tokens = model_data["total_input_tokens"] + model_data["total_output_tokens"]
                    tokens_per_second = total_tokens / model_data["total_duration"]
                    
                # Add to metrics
                metrics["models"][model_id] = {
                    "total_uses": model_data["total_uses"],
                    "successful_uses": model_data["successful_uses"],
                    "success_rate": success_rate,
                    "total_input_tokens": model_data["total_input_tokens"],
                    "total_output_tokens": model_data["total_output_tokens"],
                    "tokens_per_second": tokens_per_second,
                    "avg_execution_time": statistics.mean(durations),
                    "min_execution_time": min(durations),
                    "max_execution_time": max(durations),
                    "p95_execution_time": self._percentile(durations, 95),
                    "operations": dict(model_data["operations"]),
                    "first_use": datetime.fromtimestamp(
                        model_data["first_use"]
                    ).isoformat(),
                    "last_use": datetime.fromtimestamp(
                        model_data["last_use"]
                    ).isoformat()
                }
                
            # Calculate overall metrics
            metrics["overall"]["total_uses"] = total_uses
            metrics["overall"]["successful_uses"] = total_successful
            metrics["overall"]["total_input_tokens"] = total_input_tokens
            metrics["overall"]["total_output_tokens"] = total_output_tokens
            
            if total_uses > 0:
                metrics["overall"]["success_rate"] = total_successful / total_uses
                
            if all_durations:
                metrics["overall"]["avg_execution_time"] = statistics.mean(all_durations)
                metrics["overall"]["p95_execution_time"] = self._percentile(all_durations, 95)
                
            # Calculate token efficiency
            if total_input_tokens > 0:
                metrics["overall"]["token_efficiency"] = total_output_tokens / total_input_tokens
                
            return metrics
            
    def get_language_metrics(self) -> Dict[str, Any]:
        """
        Get aggregated language metrics.
        
        Returns:
            Dictionary with language metrics
        """
        if not self.enabled or not self.language_metrics:
            return {}
            
        with self.lock:
            metrics = {
                "language_pairs": {},
                "languages": {},
                "operations": {}
            }
            
            # Process language pair metrics
            for op_key, op_data in self.language_metrics.items():
                lang_key, operation = op_key.split(":", 1)
                source_lang = op_data["source_language"]
                target_lang = op_data["target_language"]
                
                # Calculate statistics
                durations = op_data["durations"]
                if not durations:
                    continue
                    
                success_rate = (
                    op_data["successful_operations"] / 
                    op_data["total_operations"]
                ) if op_data["total_operations"] > 0 else 0.0
                
                # Calculate size metrics
                input_sizes = op_data["input_sizes"]
                output_sizes = op_data["output_sizes"]
                
                avg_input_size = statistics.mean(input_sizes) if input_sizes else 0
                avg_output_size = statistics.mean(output_sizes) if output_sizes else 0
                
                # Create metrics entry
                if lang_key not in metrics["language_pairs"]:
                    metrics["language_pairs"][lang_key] = {
                        "source_language": source_lang,
                        "target_language": target_lang,
                        "operations": {},
                        "total_operations": 0,
                        "successful_operations": 0
                    }
                    
                # Add operation metrics
                metrics["language_pairs"][lang_key]["operations"][operation] = {
                    "total_operations": op_data["total_operations"],
                    "successful_operations": op_data["successful_operations"],
                    "success_rate": success_rate,
                    "avg_execution_time": statistics.mean(durations),
                    "min_execution_time": min(durations),
                    "max_execution_time": max(durations),
                    "p95_execution_time": self._percentile(durations, 95),
                    "avg_input_size": avg_input_size,
                    "avg_output_size": avg_output_size
                }
                
                # Update totals
                metrics["language_pairs"][lang_key]["total_operations"] += op_data["total_operations"]
                metrics["language_pairs"][lang_key]["successful_operations"] += op_data["successful_operations"]
                
                # Track per-language metrics
                for lang in [source_lang, target_lang]:
                    if lang and lang not in metrics["languages"]:
                        metrics["languages"][lang] = {
                            "total_operations": 0,
                            "as_source": 0,
                            "as_target": 0
                        }
                        
                    if lang:
                        metrics["languages"][lang]["total_operations"] += op_data["total_operations"]
                        
                        if lang == source_lang:
                            metrics["languages"][lang]["as_source"] += op_data["total_operations"]
                            
                        if lang == target_lang:
                            metrics["languages"][lang]["as_target"] += op_data["total_operations"]
                            
                # Track per-operation metrics
                if operation not in metrics["operations"]:
                    metrics["operations"][operation] = {
                        "total_operations": 0,
                        "successful_operations": 0,
                        "language_pairs": set()
                    }
                    
                metrics["operations"][operation]["total_operations"] += op_data["total_operations"]
                metrics["operations"][operation]["successful_operations"] += op_data["successful_operations"]
                metrics["operations"][operation]["language_pairs"].add(lang_key)
                
            # Convert sets to lists for JSON serialization
            for op in metrics["operations"].values():
                op["language_pairs"] = list(op["language_pairs"])
                
            return metrics
            
    def get_system_metrics(self) -> Dict[str, Any]:
        """
        Get system metrics.
        
        Returns:
            Dictionary with system metrics
        """
        if not self.enabled:
            return {}
            
        with self.lock:
            metrics = {
                "uptime_seconds": self.system_metrics["uptime_seconds"],
                "uptime_formatted": self._format_duration(self.system_metrics["uptime_seconds"]),
                "start_time": datetime.fromtimestamp(self.system_metrics["start_time"]).isoformat(),
                "current_time": datetime.now().isoformat(),
                "memory_usage": {
                    "current": self.system_metrics["current_memory_usage"],
                    "peak": self.system_metrics["peak_memory_usage"]
                },
                "request_metrics": {
                    "total_requests": self.system_metrics["total_requests"],
                    "total_errors": self.system_metrics["total_errors"],
                    "error_rate": (
                        self.system_metrics["total_errors"] / 
                        self.system_metrics["total_requests"]
                    ) if self.system_metrics["total_requests"] > 0 else 0.0,
                    "requests_per_second": (
                        self.system_metrics["total_requests"] / 
                        self.system_metrics["uptime_seconds"]
                    ) if self.system_metrics["uptime_seconds"] > 0 else 0.0
                }
            }
            
            return metrics
            
    def _format_duration(self, seconds: float) -> str:
        """
        Format duration in seconds to a human-readable string.

        Args:
            seconds (float): Duration in seconds.
        Returns:
            str: Human-readable duration string.
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
        
    def _percentile(self, data: List[float], percentile: float) -> float:
        """
        Calculate percentile value from a list of values.

        Args:
            data (List[float]): List of values.
            percentile (float): Percentile to calculate (0-100).
        Returns:
            float: Percentile value.
        """
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = (percentile / 100.0) * (len(sorted_data) - 1)
        # Interpolate between two values if needed
        floor_index = math.floor(index)
        ceil_index = math.ceil(index)
        if floor_index == ceil_index:
            return sorted_data[floor_index]
        floor_value = sorted_data[floor_index]
        ceil_value = sorted_data[ceil_index]
        # Linear interpolation
        fraction = index - floor_index
        return floor_value + (ceil_value - floor_value) * fraction
        
    def get_all_metrics(self) -> Dict[str, Any]:
        """
        Get all metrics.
        
        Returns:
            Dictionary with all metrics
        """
        if not self.enabled:
            return {}
            
        return {
            "api": self.get_api_metrics(),
            "pipeline": self.get_pipeline_metrics(),
            "model": self.get_model_metrics(),
            "language": self.get_language_metrics(),
            "system": self.get_system_metrics(),
            "timestamp": datetime.now().isoformat()
        }
        
    def get_time_series(
        self,
        series_name: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get time series data.
        
        Args:
            series_name: Name of the time series
            start_time: Start time as UNIX timestamp
            end_time: End time as UNIX timestamp
            limit: Maximum number of data points to return
            
        Returns:
            List of time series data points
        """
        if not self.enabled:
            return []
            
        with self.lock:
            if series_name not in self.time_series:
                return []
                
            series = self.time_series[series_name]
            
            # Filter by time range if specified
            if start_time is not None or end_time is not None:
                start_time = start_time or 0
                end_time = end_time or float("inf")
                
                series = [
                    point for point in series
                    if start_time <= point["timestamp"] <= end_time
                ]
                
            # Apply limit if specified
            if limit is not None and limit > 0:
                series = series[-limit:]
                
            return series
            
    def save_metrics(self) -> None:
        """Save metrics to disk."""
        if not self.enabled:
            return
            
        try:
            # Get current timestamp for filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save all metrics to a single file
            all_metrics = self.get_all_metrics()
            all_metrics_file = self.metrics_dir / f"metrics_{timestamp}.json"
            
            with open(all_metrics_file, "w", encoding="utf-8") as f:
                json.dump(all_metrics, f, indent=2)
                
            # Save time series data if enabled
            if get_config_value(self.metrics_config, "save_time_series", True):
                time_series_dir = self.metrics_dir / "time_series"
                os.makedirs(time_series_dir, exist_ok=True)
                
                for series_name, series_data in self.time_series.items():
                    if series_data:
                        series_file = time_series_dir / f"{series_name}_{timestamp}.json"
                        with open(series_file, "w", encoding="utf-8") as f:
                            json.dump(series_data, f, indent=2)
                            
            logger.info(f"Metrics saved to {all_metrics_file}")
            
        except Exception as e:
            logger.error(f"Error saving metrics: {str(e)}", exc_info=True)
            
    async def _cleanup_old_metrics(self) -> None:
        """
        Clean up old metrics files from disk based on retention policy.
        """
        if not self.enabled or self.retention_days <= 0:
            return

        try:
            # Calculate cutoff date
            cutoff_date = datetime.now() - timedelta(days=self.retention_days)
            logger.info(f"Cleaning up metrics files older than {cutoff_date.isoformat()}")

            # Clean up main metrics directory
            deleted_count = 0
            for file_path in self.metrics_dir.glob("*.json"):
                if file_path.is_file():
                    file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                    if file_time < cutoff_date:
                        file_path.unlink()
                        deleted_count += 1

            # Clean up time series directory
            time_series_dir = self.metrics_dir / "time_series"
            if time_series_dir.exists():
                for file_path in time_series_dir.glob("*.json"):
                    if file_path.is_file():
                        file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                        if file_time < cutoff_date:
                            file_path.unlink()
                            deleted_count += 1

            if deleted_count > 0:
                logger.info(f"Deleted {deleted_count} old metrics files")

        except Exception as e:
            logger.error(f"Error cleaning up old metrics: {str(e)}", exc_info=True)