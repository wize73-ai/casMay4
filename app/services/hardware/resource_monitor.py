"""
Resource Utilization Monitoring Module for CasaLingua

This module provides comprehensive resource monitoring capabilities
to track CPU, GPU, memory and I/O usage during model operations.

Author: Exygy Development Team
Version: 1.0.0
License: MIT
"""

import os
import time
import json
import asyncio
import psutil
import threading
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
from functools import wraps

# Optional GPU monitoring with nvidia-smi if available
try:
    import pynvml
    NVIDIA_AVAILABLE = True
except ImportError:
    NVIDIA_AVAILABLE = False

# Try to import torch for GPU memory tracking with PyTorch
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from app.utils.config import load_config, get_config_value
from app.utils.logging import get_logger

logger = get_logger(__name__)

class ResourceMonitor:
    """
    Monitors system resource utilization during model operations
    and other processing-intensive tasks.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the resource monitor.
        
        Args:
            config: Application configuration
        """
        self.config = config or load_config()
        self.monitoring_config = get_config_value(self.config, "monitoring", {})
        
        # Configure monitoring settings
        self.enabled = get_config_value(self.monitoring_config, "enabled", True)
        self.interval = get_config_value(self.monitoring_config, "interval_seconds", 1.0)
        self.log_dir = Path(get_config_value(self.monitoring_config, "log_dir", "logs/metrics"))
        self.log_file = self.log_dir / f"resources_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self.include_gpu = get_config_value(self.monitoring_config, "include_gpu", True)
        self.include_io = get_config_value(self.monitoring_config, "include_io", True)
        self.include_network = get_config_value(self.monitoring_config, "include_network", True)
        self.include_per_process = get_config_value(self.monitoring_config, "include_per_process", False)
        
        # Initialize monitoring state
        self.monitoring_tasks = {}
        self.monitoring_data = {}
        self.lock = threading.RLock()
        
        # Initialize GPU monitoring if available
        self.gpu_count = 0
        if self.include_gpu and NVIDIA_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.gpu_count = pynvml.nvmlDeviceGetCount()
                logger.info(f"GPU monitoring enabled - detected {self.gpu_count} NVIDIA GPUs")
            except Exception as e:
                logger.warning(f"Failed to initialize NVIDIA GPU monitoring: {str(e)}")
                self.gpu_count = 0
        elif self.include_gpu and TORCH_AVAILABLE and torch.cuda.is_available():
            self.gpu_count = torch.cuda.device_count()
            logger.info(f"GPU monitoring enabled via PyTorch - detected {self.gpu_count} CUDA devices")
        
        # Create log directory
        os.makedirs(self.log_dir, exist_ok=True)
        logger.info("Resource monitor initialized")
    
    async def start_monitoring(self, task_id: str, labels: Dict[str, Any] = None) -> str:
        """Start monitoring for a specific task.
        
        Args:
            task_id: Unique identifier for the task
            labels: Additional labels for tracking
            
        Returns:
            Task ID for tracking
        """
        if not self.enabled:
            return task_id
        
        # If no task ID provided, generate one
        if not task_id:
            task_id = f"task_{int(time.time() * 1000)}"
        
        async with asyncio.Lock():
            if task_id in self.monitoring_tasks:
                # If already monitoring, stop previous
                await self.stop_monitoring(task_id)
            
            # Initialize task data
            self.monitoring_data[task_id] = {
                "start_time": time.time(),
                "labels": labels or {},
                "samples": []
            }
            
            # Start monitoring task
            cancel_event = asyncio.Event()
            monitor_task = asyncio.create_task(
                self._collect_metrics(task_id, cancel_event)
            )
            
            self.monitoring_tasks[task_id] = {
                "task": monitor_task,
                "cancel_event": cancel_event
            }
            
            logger.debug(f"Started resource monitoring for task {task_id}")
            return task_id
    
    async def stop_monitoring(self, task_id: str) -> Dict[str, Any]:
        """Stop monitoring for a specific task and return results.
        
        Args:
            task_id: Task identifier to stop monitoring
            
        Returns:
            Monitoring data for the task
        """
        if not self.enabled or task_id not in self.monitoring_tasks:
            return {}
        
        async with asyncio.Lock():
            # Set cancel event
            self.monitoring_tasks[task_id]["cancel_event"].set()
            
            # Wait for task to complete
            try:
                await self.monitoring_tasks[task_id]["task"]
            except asyncio.CancelledError:
                pass
            
            # Record end time
            if task_id in self.monitoring_data:
                self.monitoring_data[task_id]["end_time"] = time.time()
                
                # Save to file
                await self._save_data(task_id)
                
                # Get result copy
                result = self.monitoring_data[task_id].copy()
                
                # Clean up
                del self.monitoring_tasks[task_id]
                
                logger.debug(f"Stopped resource monitoring for task {task_id}")
                return result
            
            return {}
    
    async def get_current_usage(self) -> Dict[str, Any]:
        """Get current resource usage snapshot.
        
        Returns:
            Dictionary with current resource usage
        """
        return self._collect_usage_sample()
    
    async def _collect_metrics(self, task_id: str, cancel_event: asyncio.Event) -> None:
        """
        Collect resource metrics periodically for a specific task.
        
        Args:
            task_id: Task identifier
            cancel_event: Event to signal monitoring cancellation
        """
        try:
            while not cancel_event.is_set():
                # Collect sample
                sample = self._collect_usage_sample()
                sample["timestamp"] = time.time()
                
                # Store sample
                self.monitoring_data[task_id]["samples"].append(sample)
                
                # Wait for next collection
                try:
                    await asyncio.wait_for(
                        cancel_event.wait(),
                        timeout=self.interval
                    )
                except asyncio.TimeoutError:
                    # Normal timeout, continue collecting
                    pass
        except Exception as e:
            logger.error(f"Error in resource monitoring for {task_id}: {str(e)}")
    
    def _collect_usage_sample(self) -> Dict[str, Any]:
        """
        Collect a single resource usage sample.
        
        Returns:
            Dictionary with resource usage metrics
        """
        sample = {
            "timestamp": time.time(),
            "cpu": {},
            "memory": {},
        }
        
        # CPU metrics
        cpu_times = psutil.cpu_times_percent(interval=None, percpu=False)
        cpu_count = psutil.cpu_count()
        cpu_freq = psutil.cpu_freq()
        
        sample["cpu"] = {
            "usage_percent": psutil.cpu_percent(),
            "count": cpu_count,
            "user_percent": cpu_times.user,
            "system_percent": cpu_times.system,
            "idle_percent": cpu_times.idle,
            "freq_current": cpu_freq.current if cpu_freq else None,
            "freq_min": cpu_freq.min if cpu_freq and hasattr(cpu_freq, 'min') else None,
            "freq_max": cpu_freq.max if cpu_freq and hasattr(cpu_freq, 'max') else None,
        }
        
        # Memory metrics
        virtual_mem = psutil.virtual_memory()
        swap_mem = psutil.swap_memory()
        
        sample["memory"] = {
            "total": virtual_mem.total,
            "available": virtual_mem.available,
            "used": virtual_mem.used,
            "free": virtual_mem.free,
            "percent": virtual_mem.percent,
            "swap_total": swap_mem.total,
            "swap_used": swap_mem.used,
            "swap_free": swap_mem.free,
            "swap_percent": swap_mem.percent
        }
        
        # GPU metrics if available
        if self.include_gpu and self.gpu_count > 0:
            sample["gpu"] = []
            
            # Try NVIDIA-SMI first
            if NVIDIA_AVAILABLE:
                try:
                    for i in range(self.gpu_count):
                        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to W
                        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                        
                        gpu_info = {
                            "index": i,
                            "name": pynvml.nvmlDeviceGetName(handle),
                            "util_gpu": util.gpu,
                            "util_memory": util.memory,
                            "memory_total": mem_info.total,
                            "memory_used": mem_info.used,
                            "memory_free": mem_info.free,
                            "power_usage": power,
                            "temperature": temp
                        }
                        sample["gpu"].append(gpu_info)
                except Exception as e:
                    logger.warning(f"Error collecting NVIDIA GPU metrics: {str(e)}")
            
            # If NVIDIA-SMI failed or not available, try PyTorch
            elif TORCH_AVAILABLE and torch.cuda.is_available():
                try:
                    for i in range(self.gpu_count):
                        # Get device properties
                        props = torch.cuda.get_device_properties(i)
                        
                        # Get memory usage
                        memory_allocated = torch.cuda.memory_allocated(i)
                        memory_reserved = torch.cuda.memory_reserved(i)
                        
                        gpu_info = {
                            "index": i,
                            "name": props.name,
                            "memory_total": props.total_memory,
                            "memory_used": memory_allocated,
                            "memory_reserved": memory_reserved,
                            "compute_capability": f"{props.major}.{props.minor}"
                        }
                        sample["gpu"].append(gpu_info)
                except Exception as e:
                    logger.warning(f"Error collecting PyTorch GPU metrics: {str(e)}")
        
        # Disk I/O metrics
        if self.include_io:
            disk_io = psutil.disk_io_counters()
            sample["disk"] = {
                "read_count": disk_io.read_count if disk_io else 0,
                "write_count": disk_io.write_count if disk_io else 0,
                "read_bytes": disk_io.read_bytes if disk_io else 0,
                "write_bytes": disk_io.write_bytes if disk_io else 0,
                "read_time": disk_io.read_time if disk_io else 0,
                "write_time": disk_io.write_time if disk_io else 0
            }
        
        # Network metrics
        if self.include_network:
            net_io = psutil.net_io_counters()
            sample["network"] = {
                "bytes_sent": net_io.bytes_sent,
                "bytes_recv": net_io.bytes_recv,
                "packets_sent": net_io.packets_sent,
                "packets_recv": net_io.packets_recv,
                "errin": net_io.errin,
                "errout": net_io.errout,
                "dropin": net_io.dropin,
                "dropout": net_io.dropout
            }
        
        # Per-process metrics
        if self.include_per_process:
            sample["processes"] = {}
            current_process = psutil.Process()
            sample["processes"]["current"] = self._get_process_metrics(current_process)
            
            # Get parent process
            try:
                parent = current_process.parent()
                if parent:
                    sample["processes"]["parent"] = self._get_process_metrics(parent)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            
            # Get children processes
            sample["processes"]["children"] = []
            try:
                children = current_process.children(recursive=True)
                for child in children:
                    sample["processes"]["children"].append(
                        self._get_process_metrics(child)
                    )
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        return sample
    
    def _get_process_metrics(self, process: 'psutil.Process') -> Dict[str, Any]:
        """
        Get metrics for a specific process.
        
        Args:
            process: Process object to collect metrics for
            
        Returns:
            Dictionary with process metrics
        """
        try:
            metrics = {
                "pid": process.pid,
                "name": process.name(),
                "status": process.status(),
                "cpu_percent": process.cpu_percent(),
                "memory_percent": process.memory_percent(),
                "memory_rss": process.memory_info().rss,
                "memory_vms": process.memory_info().vms,
                "threads": process.num_threads(),
                "create_time": process.create_time()
            }
            
            # Try to get command line
            try:
                metrics["cmdline"] = process.cmdline()
            except (psutil.AccessDenied, psutil.ZombieProcess):
                pass
            
            # Try to get user
            try:
                metrics["username"] = process.username()
            except (psutil.AccessDenied, psutil.ZombieProcess):
                pass
                
            return metrics
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess) as e:
            return {"pid": process.pid, "error": str(e)}
        except Exception as e:
            return {"pid": process.pid, "error": f"Unknown error: {str(e)}"}
    
    async def _save_data(self, task_id: str) -> None:
        """
        Save monitoring data to file.
        
        Args:
            task_id: Task identifier to save data for
        """
        if task_id not in self.monitoring_data:
            return
        
        # Create file path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_task_id = ''.join(c if c.isalnum() else '_' for c in task_id)
        file_path = self.log_dir / f"resources_{safe_task_id}_{timestamp}.json"
        
        # Save data
        try:
            # Get task data
            task_data = self.monitoring_data[task_id]
            
            # Save to file
            with open(file_path, "w") as f:
                json.dump(task_data, f, indent=2)
                
            logger.debug(f"Saved resource monitoring data for task {task_id} to {file_path}")
        except Exception as e:
            logger.error(f"Error saving resource monitoring data for task {task_id}: {str(e)}")
    
    def monitor(self, labels: Dict[str, Any] = None):
        """
        Decorator to monitor resource usage for a function.
        
        Args:
            labels: Additional labels for tracking
            
        Returns:
            Decorated function
        """
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Skip monitoring if disabled
                if not self.enabled:
                    return await func(*args, **kwargs)
                
                # Generate task ID
                task_id = f"{func.__name__}_{int(time.time() * 1000)}"
                
                # Start monitoring
                await self.start_monitoring(task_id, labels)
                
                try:
                    # Call original function
                    result = await func(*args, **kwargs)
                    return result
                finally:
                    # Stop monitoring
                    await self.stop_monitoring(task_id)
            
            return wrapper
        
        return decorator
    
    async def analyze_task(self, task_id: str) -> Dict[str, Any]:
        """
        Analyze resource usage data for a specific task.
        
        Args:
            task_id: Task identifier to analyze
            
        Returns:
            Dictionary with analysis results
        """
        if task_id not in self.monitoring_data:
            return {"error": "Task not found"}
        
        # Get task data
        task_data = self.monitoring_data[task_id]
        samples = task_data["samples"]
        
        if not samples:
            return {"error": "No samples collected"}
        
        # Calculate duration
        duration = task_data.get("end_time", time.time()) - task_data["start_time"]
        
        # Prepare result
        result = {
            "task_id": task_id,
            "start_time": task_data["start_time"],
            "end_time": task_data.get("end_time", time.time()),
            "duration": duration,
            "sample_count": len(samples),
            "labels": task_data["labels"],
            "cpu": {
                "min": 100.0,
                "max": 0.0,
                "avg": 0.0,
                "std_dev": 0.0
            },
            "memory": {
                "min_percent": 100.0,
                "max_percent": 0.0,
                "avg_percent": 0.0,
                "std_dev_percent": 0.0,
                "peak": 0
            }
        }
        
        # Process CPU metrics
        cpu_usages = [s["cpu"]["usage_percent"] for s in samples]
        result["cpu"]["min"] = min(cpu_usages)
        result["cpu"]["max"] = max(cpu_usages)
        result["cpu"]["avg"] = sum(cpu_usages) / len(cpu_usages)
        
        # Calculate CPU standard deviation
        cpu_variance = sum((x - result["cpu"]["avg"]) ** 2 for x in cpu_usages) / len(cpu_usages)
        result["cpu"]["std_dev"] = cpu_variance ** 0.5
        
        # Process memory metrics
        memory_percents = [s["memory"]["percent"] for s in samples]
        memory_used = [s["memory"]["used"] for s in samples]
        
        result["memory"]["min_percent"] = min(memory_percents)
        result["memory"]["max_percent"] = max(memory_percents)
        result["memory"]["avg_percent"] = sum(memory_percents) / len(memory_percents)
        result["memory"]["peak"] = max(memory_used)
        
        # Calculate memory standard deviation
        memory_variance = sum((x - result["memory"]["avg_percent"]) ** 2 for x in memory_percents) / len(memory_percents)
        result["memory"]["std_dev_percent"] = memory_variance ** 0.5
        
        # Process GPU metrics if available
        if self.include_gpu and self.gpu_count > 0 and "gpu" in samples[0]:
            result["gpu"] = []
            
            for gpu_idx in range(len(samples[0]["gpu"])):
                gpu_result = {
                    "index": gpu_idx,
                    "name": samples[0]["gpu"][gpu_idx]["name"],
                    "utilization": {
                        "min": 100.0,
                        "max": 0.0,
                        "avg": 0.0
                    },
                    "memory": {
                        "min_used": float('inf'),
                        "max_used": 0.0,
                        "avg_used": 0.0,
                        "peak_used": 0.0
                    }
                }
                
                # Collect GPU metrics
                gpu_utils = []
                gpu_mem_used = []
                gpu_temps = []
                
                for sample in samples:
                    if "gpu" in sample and gpu_idx < len(sample["gpu"]):
                        gpu_data = sample["gpu"][gpu_idx]
                        
                        # Utilization
                        if "util_gpu" in gpu_data:
                            gpu_utils.append(gpu_data["util_gpu"])
                        
                        # Memory
                        if "memory_used" in gpu_data:
                            gpu_mem_used.append(gpu_data["memory_used"])
                        
                        # Temperature
                        if "temperature" in gpu_data:
                            gpu_temps.append(gpu_data["temperature"])
                
                # Calculate statistics
                if gpu_utils:
                    gpu_result["utilization"]["min"] = min(gpu_utils)
                    gpu_result["utilization"]["max"] = max(gpu_utils)
                    gpu_result["utilization"]["avg"] = sum(gpu_utils) / len(gpu_utils)
                
                if gpu_mem_used:
                    gpu_result["memory"]["min_used"] = min(gpu_mem_used)
                    gpu_result["memory"]["max_used"] = max(gpu_mem_used)
                    gpu_result["memory"]["avg_used"] = sum(gpu_mem_used) / len(gpu_mem_used)
                    gpu_result["memory"]["peak_used"] = max(gpu_mem_used)
                
                if gpu_temps:
                    gpu_result["temperature"] = {
                        "min": min(gpu_temps),
                        "max": max(gpu_temps),
                        "avg": sum(gpu_temps) / len(gpu_temps)
                    }
                
                result["gpu"].append(gpu_result)
        
        # Process disk I/O metrics if available
        if self.include_io and "disk" in samples[0]:
            # Get first and last sample
            first_sample = samples[0]["disk"]
            last_sample = samples[-1]["disk"]
            
            # Calculate I/O rates
            read_bytes = last_sample["read_bytes"] - first_sample["read_bytes"]
            write_bytes = last_sample["write_bytes"] - first_sample["write_bytes"]
            
            result["disk"] = {
                "read_bytes_total": read_bytes,
                "write_bytes_total": write_bytes,
                "read_rate": read_bytes / duration if duration > 0 else 0,
                "write_rate": write_bytes / duration if duration > 0 else 0
            }
        
        # Process network metrics if available
        if self.include_network and "network" in samples[0]:
            # Get first and last sample
            first_sample = samples[0]["network"]
            last_sample = samples[-1]["network"]
            
            # Calculate network rates
            bytes_sent = last_sample["bytes_sent"] - first_sample["bytes_sent"]
            bytes_recv = last_sample["bytes_recv"] - first_sample["bytes_recv"]
            
            result["network"] = {
                "bytes_sent_total": bytes_sent,
                "bytes_recv_total": bytes_recv,
                "send_rate": bytes_sent / duration if duration > 0 else 0,
                "recv_rate": bytes_recv / duration if duration > 0 else 0
            }
        
        return result
    
    async def close(self):
        """Close the resource monitor and clean up."""
        if not self.enabled:
            return
        
        # Stop all monitoring tasks
        task_ids = list(self.monitoring_tasks.keys())
        for task_id in task_ids:
            await self.stop_monitoring(task_id)
        
        # Clean up GPU monitoring
        if self.include_gpu and NVIDIA_AVAILABLE and self.gpu_count > 0:
            try:
                pynvml.nvmlShutdown()
            except Exception as e:
                logger.warning(f"Error shutting down NVIDIA GPU monitoring: {str(e)}")
        
        logger.info("Resource monitor closed")


# Singleton instance
_resource_monitor = None

def get_resource_monitor(config: Optional[Dict[str, Any]] = None) -> ResourceMonitor:
    """Get or create the resource monitor singleton instance.
    
    Args:
        config: Optional configuration
        
    Returns:
        ResourceMonitor instance
    """
    global _resource_monitor
    if _resource_monitor is None:
        _resource_monitor = ResourceMonitor(config)
    return _resource_monitor