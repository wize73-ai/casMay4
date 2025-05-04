# app/services/hardware/detector.py
"""
Hardware Detector Module for CasaLingua
Detects and reports system hardware capabilities to optimize application performance
"""

import os
import platform
import subprocess
import sys
import json
import logging
import re
from typing import Dict, Any, List, Optional, Tuple, Union
import time
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text

# Initialize rich console
console = Console()

# Configure logging
logger = logging.getLogger(__name__)

# Try to import optional dependencies with fallbacks
try:
    import psutil
    HAVE_PSUTIL = True
except ImportError:
    HAVE_PSUTIL = False
    logger.warning("psutil not available - hardware detection will be limited")

try:
    import cpuinfo
    HAVE_CPUINFO = True
except ImportError:
    HAVE_CPUINFO = False
    logger.warning("cpuinfo not available - CPU detection will be limited")

try:
    import torch
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False
    logger.warning("PyTorch not available - GPU detection will be limited")

try:
    import numpy as np
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False
    logger.warning("NumPy not available - some features will be limited")

class HardwareDetector:
    """
    Hardware detection and reporting for optimizing CasaLingua performance
    """
    
    def __init__(self, config=None):
        self.config = config
        self.system_info = {}
        self.memory_info = {}
        self.cpu_info = {}
        self.gpu_info = {}
        self.disk_info = {}
        self.network_info = {}
        self.applied_config = {}

    def apply_configuration(self, config: Dict[str, Any]) -> None:
        """
        Apply the recommended configuration to internal state or runtime hints.

        Args:
            config (Dict[str, Any]): Recommended configuration dictionary
        """
        logger.info("Applying configuration to hardware context")
        self.applied_config = config
        
        console.print(Panel(
            "[bold green]Hardware Configuration Applied[/bold green]",
            border_style="green"
        ))
        
    def detect_all(self) -> Dict[str, Any]:
        """
        Detect all hardware components
        
        Returns:
            Dict[str, Any]: Complete hardware information
        """
        console.print("[bold cyan]Starting Hardware Detection...[/bold cyan]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            # Create tasks for each detection phase
            system_task = progress.add_task("[cyan]Detecting system information...", total=100)
            memory_task = progress.add_task("[cyan]Detecting memory...", total=100, visible=False)
            cpu_task = progress.add_task("[cyan]Detecting CPU...", total=100, visible=False)
            gpu_task = progress.add_task("[cyan]Detecting GPU...", total=100, visible=False)
            disk_task = progress.add_task("[cyan]Detecting storage...", total=100, visible=False)
            network_task = progress.add_task("[cyan]Detecting network...", total=100, visible=False)
            
            # Detect system
            self.detect_system()
            progress.update(system_task, advance=100, description="[green]System detection complete")
            
            # Show next task
            progress.update(memory_task, visible=True)
            self.detect_memory()
            progress.update(memory_task, advance=100, description="[green]Memory detection complete")
            
            # Show next task
            progress.update(cpu_task, visible=True)
            self.detect_cpu()
            progress.update(cpu_task, advance=100, description="[green]CPU detection complete")
            
            # Show next task
            progress.update(gpu_task, visible=True)
            self.detect_gpu()
            progress.update(gpu_task, advance=100, description="[green]GPU detection complete")
            
            # Show next task
            progress.update(disk_task, visible=True)
            self.detect_disk()
            progress.update(disk_task, advance=100, description="[green]Storage detection complete")
            
            # Show next task
            progress.update(network_task, visible=True)
            self.detect_network()
            progress.update(network_task, advance=100, description="[green]Network detection complete")
        
        # Display summary
        self.display_hardware_summary()
        
        return self.get_all_info()
    
    def display_hardware_summary(self):
        """Display a formatted summary of detected hardware"""
        # Create a summary table
        table = Table(title="[bold cyan]Hardware Detection Results[/bold cyan]")
        table.add_column("Component", style="cyan", no_wrap=True)
        table.add_column("Details", style="white")
        
        # System information
        system_name = self.system_info.get("os_name", self.system_info.get("platform", "Unknown"))
        system_version = self.system_info.get("os_version", self.system_info.get("platform_version", ""))
        table.add_row("System", f"{system_name} {system_version}")
        
        # CPU information
        cpu_brand = self.cpu_info.get("brand", "Unknown CPU")
        cpu_cores = self.cpu_info.get("count_logical", 0)
        cpu_physical = self.cpu_info.get("count_physical", 0)
        table.add_row("CPU", f"{cpu_brand} ({cpu_physical} cores / {cpu_cores} threads)")
        
        # Memory information
        total_gb = self.memory_info.get("total_gb", 0)
        available_gb = self.memory_info.get("available_gb", 0)
        table.add_row("Memory", f"{total_gb:.1f} GB total / {available_gb:.1f} GB available")
        
        # GPU information
        if self.gpu_info.get("has_gpu", False):
            gpu_devices = self.gpu_info.get("devices", [])
            if gpu_devices:
                gpu_names = [device.get("name", "Unknown GPU") for device in gpu_devices]
                gpu_info = ", ".join(gpu_names)
            else:
                gpu_info = "GPU detected"
                
            if self.gpu_info.get("cuda_available", False):
                gpu_info += " (CUDA enabled)"
            elif self.gpu_info.get("mps_available", False):
                gpu_info += " (MPS enabled)"
                
            table.add_row("GPU", gpu_info)
        else:
            table.add_row("GPU", "[dim]None detected[/dim]")
        
        # Disk information
        disk_total = self.disk_info.get("total_size_gb", 0)
        disk_free = self.disk_info.get("total_free_gb", 0)
        table.add_row("Storage", f"{disk_total:.1f} GB total / {disk_free:.1f} GB free")
        
        # Network information
        active_interfaces = [i["name"] for i in self.network_info.get("interfaces", []) if i.get("is_up")]
        if active_interfaces:
            table.add_row("Network", f"{len(active_interfaces)} active interfaces")
        else:
            table.add_row("Network", "[dim]No active interfaces[/dim]")
        
        # Display the table
        console.print(table)
        
        # Show recommendations panel
        recommendations = self.recommend_config()
        
        console.print(Panel(
            f"[bold cyan]Hardware Recommendations[/bold cyan]\n"
            f"Device: [yellow]{recommendations.get('device', 'cpu')}[/yellow]\n"
            f"Model Size: [green]{recommendations.get('memory', {}).get('model_size', 'unknown')}[/green]\n"
            f"Batch Size: [magenta]{recommendations.get('memory', {}).get('batch_size', 'unknown')}[/magenta]",
            border_style="blue"
        ))
    
    def detect_system(self) -> Dict[str, Any]:
        """
        Detect basic system information
        
        Returns:
            Dict[str, Any]: System information
        """
        try:
            self.system_info = {
                "platform": platform.system(),
                "platform_release": platform.release(),
                "platform_version": platform.version(),
                "architecture": platform.machine(),
                "hostname": platform.node(),
                "processor": platform.processor(),
                "python_version": sys.version,
                "python_build": platform.python_build(),
                "python_compiler": platform.python_compiler()
            }

            # Try to get more detailed OS information
            if platform.system() == "Linux":
                try:
                    # Get Linux distribution details
                    with open("/etc/os-release") as f:
                        os_info = {}
                        for line in f:
                            if "=" in line:
                                key, value = line.strip().split("=", 1)
                                os_info[key] = value.strip('"')
                    self.system_info["os_name"] = os_info.get("NAME", "Linux")
                    self.system_info["os_version"] = os_info.get("VERSION", "")
                    self.system_info["os_id"] = os_info.get("ID", "")
                except Exception as e:
                    logger.warning(f"Failed to get detailed Linux information: {e}")

            elif platform.system() == "Darwin":  # macOS
                try:
                    macos_version = subprocess.check_output(["sw_vers", "-productVersion"]).decode().strip()
                    self.system_info["os_name"] = "macOS"
                    self.system_info["os_version"] = macos_version
                except Exception as e:
                    logger.warning(f"Failed to get detailed macOS information: {e}")
                # Add mac model key for macOS
                try:
                    model = subprocess.check_output(["sysctl", "-n", "hw.model"]).decode().strip()
                    self.system_info["mac_model"] = model
                except Exception as e:
                    logger.warning(f"Failed to get mac model info: {e}")

            elif platform.system() == "Windows":
                try:
                    import winreg
                    key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows NT\CurrentVersion")
                    self.system_info["os_name"] = "Windows"
                    self.system_info["os_version"] = f"{platform.version()} (Build {platform.release()})"
                    self.system_info["product_name"] = winreg.QueryValueEx(key, "ProductName")[0]
                except Exception as e:
                    logger.warning(f"Failed to get detailed Windows information: {e}")

            logger.info(f"Detected system: {self.system_info.get('platform')} {self.system_info.get('platform_version')}")
            return self.system_info

        except Exception as e:
            logger.error(f"Error detecting system information: {e}")
            return {}
    
    def detect_cpu(self) -> Dict[str, Any]:
        """
        Detect CPU information
        
        Returns:
            Dict[str, Any]: CPU information
        """
        try:
            # Initialize with basic info
            self.cpu_info = {
                "brand": "Unknown",
                "count_logical": 1,
                "count_physical": 1
            }
            
            # Get CPU info from cpuinfo package if available
            if HAVE_CPUINFO:
                try:
                    cpu_data = cpuinfo.get_cpu_info()
                    self.cpu_info.update({
                        "brand": cpu_data.get("brand_raw", "Unknown"),
                        "arch": cpu_data.get("arch", "Unknown"),
                        "bits": cpu_data.get("bits", 0),
                        "features": cpu_data.get("flags", []),
                        "l2_cache_size": cpu_data.get("l2_cache_size", 0),
                        "l3_cache_size": cpu_data.get("l3_cache_size", 0),
                        "vendor_id": cpu_data.get("vendor_id_raw", "Unknown"),
                        "supports_avx": "avx" in cpu_data.get("flags", []),
                        "supports_avx2": "avx2" in cpu_data.get("flags", []),
                        "supports_avx512": any("avx512" in flag for flag in cpu_data.get("flags", [])),
                        "supports_sse2": "sse2" in cpu_data.get("flags", []),
                        "supports_ssse3": "ssse3" in cpu_data.get("flags", []),
                        "supports_sse4_1": "sse4_1" in cpu_data.get("flags", []),
                        "supports_sse4_2": "sse4_2" in cpu_data.get("flags", [])
                    })
                except Exception as e:
                    logger.warning(f"Failed to get detailed CPU information from cpuinfo: {e}")
            
            # Get CPU counts from psutil if available
            if HAVE_PSUTIL:
                try:
                    self.cpu_info.update({
                        "count_logical": psutil.cpu_count(logical=True),
                        "count_physical": psutil.cpu_count(logical=False)
                    })
                    
                    # Get CPU frequency if available
                    cpu_freq = psutil.cpu_freq()
                    if cpu_freq:
                        self.cpu_info.update({
                            "current_freq_mhz": cpu_freq.current,
                            "min_freq_mhz": cpu_freq.min,
                            "max_freq_mhz": cpu_freq.max
                        })
                    
                    # Get CPU usage percentages
                    self.cpu_info["usage_percent"] = psutil.cpu_percent(interval=0.1, percpu=True)
                    self.cpu_info["average_usage_percent"] = psutil.cpu_percent(interval=0.1)
                except Exception as e:
                    logger.warning(f"Failed to get CPU counts from psutil: {e}")
            
            # Try alternative methods for CPU brand if still unknown
            if self.cpu_info.get("brand") == "Unknown":
                self.cpu_info["brand"] = self._get_cpu_name_alternative()
                
            # If count_physical is None, use logical count
            if self.cpu_info["count_physical"] is None:
                self.cpu_info["count_physical"] = self.cpu_info["count_logical"]
            
            logger.info(f"Detected CPU: {self.cpu_info.get('brand')} with {self.cpu_info.get('count_logical')} logical cores")
            return self.cpu_info
            
        except Exception as e:
            logger.error(f"Error detecting CPU information: {e}")
            self.cpu_info = {
                "brand": "Unknown",
                "count_logical": 1,
                "count_physical": 1
            }
            return self.cpu_info
    
    def detect_memory(self) -> Dict[str, Any]:
        """
        Detect memory information
        
        Returns:
            Dict[str, Any]: Memory information
        """
        try:
            # Initialize with default values
            self.memory_info = {
                "total_gb": 4.0,         # Conservative default
                "available_gb": 2.0,      # Conservative default
                "used_gb": 2.0,
                "percent_used": 50.0
            }
            
            # Get virtual memory stats if psutil is available
            if HAVE_PSUTIL:
                virtual_memory = psutil.virtual_memory()
                
                self.memory_info.update({
                    "total_gb": round(virtual_memory.total / (1024**3), 2),
                    "available_gb": round(virtual_memory.available / (1024**3), 2),
                    "used_gb": round(virtual_memory.used / (1024**3), 2),
                    "percent_used": virtual_memory.percent
                })
                
                # Get swap memory if available
                try:
                    swap_memory = psutil.swap_memory()
                    self.memory_info.update({
                        "swap_total_gb": round(swap_memory.total / (1024**3), 2),
                        "swap_used_gb": round(swap_memory.used / (1024**3), 2),
                        "swap_percent_used": swap_memory.percent
                    })
                except Exception as e:
                    logger.warning(f"Failed to get swap memory information: {e}")
            
            logger.info(f"Detected Memory: {self.memory_info.get('total_gb')} GB total, {self.memory_info.get('available_gb')} GB available")
            return self.memory_info
            
        except Exception as e:
            logger.error(f"Error detecting memory information: {e}")
            return self.memory_info
    
    def detect_gpu(self) -> Dict[str, Any]:
        """
        Detect GPU information
        
        Returns:
            Dict[str, Any]: GPU information
        """
        try:
            # Initialize with default values
            self.gpu_info = {
                "has_gpu": False,
                "cuda_available": False,
                "device_count": 0,
                "devices": []
            }

            # Check for CUDA via PyTorch if available
            if HAVE_TORCH:
                try:
                    self.gpu_info["cuda_available"] = torch.cuda.is_available()

                    if torch.cuda.is_available():
                        self.gpu_info["has_gpu"] = True
                        self.gpu_info["device_count"] = torch.cuda.device_count()
                        self.gpu_info["cuda_version"] = torch.version.cuda

                        # Get information about each device
                        for i in range(torch.cuda.device_count()):
                            device_props = torch.cuda.get_device_properties(i)
                            device_properties = {
                                "index": i,
                                "name": torch.cuda.get_device_name(i),
                                "total_memory_gb": round(device_props.total_memory / (1024**3), 2),
                                "major": device_props.major,
                                "minor": device_props.minor,
                                "multi_processor_count": device_props.multi_processor_count
                            }

                            # Add memory usage if available
                            try:
                                torch.cuda.set_device(i)
                                allocated = torch.cuda.memory_allocated(i)
                                reserved = torch.cuda.memory_reserved(i)
                                device_properties["allocated_memory_gb"] = round(allocated / (1024**3), 2)
                                device_properties["reserved_memory_gb"] = round(reserved / (1024**3), 2)
                            except Exception as e:
                                logger.warning(f"Failed to get memory usage for GPU {i}: {e}")

                            self.gpu_info["devices"].append(device_properties)

                    # Always check for MPS (Metal Performance Shaders) on macOS if present
                    if platform.system() == "Darwin":
                        try:
                            # Always check for MPS
                            mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
                            self.gpu_info["mps_available"] = mps_available
                            if mps_available:
                                self.gpu_info["has_gpu"] = True
                                # Add MPS device if not already present
                                mps_device = {
                                    "index": 0,
                                    "name": "Apple Silicon GPU",
                                    "type": "MPS"
                                }
                                # Optionally, try to detect specific M4 (Pro/Max) for newer Apple Silicon
                                cpu_brand = ""
                                if hasattr(self, "cpu_info") and self.cpu_info and "brand" in self.cpu_info:
                                    cpu_brand = self.cpu_info["brand"]
                                else:
                                    try:
                                        cpu_brand = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
                                    except Exception:
                                        pass
                                if cpu_brand:
                                    if any(m in cpu_brand for m in ["M4 Pro", "M4 Max"]):
                                        mps_device["name"] = "Apple M4 GPU"
                                        mps_device["apple_silicon"] = True
                                        mps_device["m4_family"] = True
                                    elif "M4" in cpu_brand:
                                        mps_device["name"] = "Apple M4 GPU"
                                        mps_device["apple_silicon"] = True
                                        mps_device["m4_family"] = True
                                    elif "M3" in cpu_brand:
                                        mps_device["name"] = "Apple M3 GPU"
                                        mps_device["apple_silicon"] = True
                                        mps_device["m3_family"] = True
                                    elif "M2" in cpu_brand:
                                        mps_device["name"] = "Apple M2 GPU"
                                        mps_device["apple_silicon"] = True
                                    elif "M1" in cpu_brand:
                                        mps_device["name"] = "Apple M1 GPU"
                                        mps_device["apple_silicon"] = True
                                # Only add if not already present
                                if not any(d.get("type") == "MPS" for d in self.gpu_info["devices"]):
                                    self.gpu_info["devices"].append(mps_device)
                                if not self.gpu_info.get("device_count"):
                                    self.gpu_info["device_count"] = 1
                        except (ImportError, AttributeError, Exception):
                            self.gpu_info["mps_available"] = False
                except Exception as e:
                    logger.warning(f"Failed to detect GPU using PyTorch: {e}")

            # Try to get additional information on Linux with nvidia-smi
            if platform.system() == "Linux" and self.gpu_info["has_gpu"]:
                try:
                    nvidia_smi_output = subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used,driver_version", "--format=csv,noheader,nounits"]).decode()

                    for i, line in enumerate(nvidia_smi_output.strip().split("\n")):
                        if i < len(self.gpu_info["devices"]):
                            values = [x.strip() for x in line.split(",")]
                            if len(values) >= 9:
                                self.gpu_info["devices"][i]["temperature_c"] = float(values[2])
                                self.gpu_info["devices"][i]["utilization_gpu_percent"] = float(values[3])
                                self.gpu_info["devices"][i]["utilization_memory_percent"] = float(values[4])
                                self.gpu_info["devices"][i]["memory_total_mib"] = float(values[5])
                                self.gpu_info["devices"][i]["memory_free_mib"] = float(values[6])
                                self.gpu_info["devices"][i]["memory_used_mib"] = float(values[7])
                                self.gpu_info["driver_version"] = values[8]
                except Exception as e:
                    logger.warning(f"Failed to get additional GPU information using nvidia-smi: {e}")

            if self.gpu_info["has_gpu"]:
                if self.gpu_info.get("cuda_available"):
                    logger.info(f"Detected {self.gpu_info.get('device_count')} CUDA GPU(s): {', '.join([d.get('name', 'Unknown') for d in self.gpu_info.get('devices', [])])}")
                elif self.gpu_info.get("mps_available"):
                    logger.info("Detected Apple GPU with MPS support")
                else:
                    logger.info("GPU detected but not accessible through CUDA or MPS")
            else:
                logger.info("No GPU detected")

            return self.gpu_info

        except Exception as e:
            logger.error(f"Error detecting GPU information: {e}")
            return {"has_gpu": False}
    
    def detect_disk(self) -> Dict[str, Any]:
        """
        Detect disk information
        
        Returns:
            Dict[str, Any]: Disk information
        """
        try:
            # Initialize with default values
            disk_info = {
                "total_size_gb": 100.0,  # Conservative default
                "total_used_gb": 50.0,
                "total_free_gb": 50.0,
                "total_percent_used": 50.0,
                "partitions": []
            }
            
            if HAVE_PSUTIL:
                # Get partitions
                partitions = psutil.disk_partitions()
                
                total_size = 0
                total_used = 0
                total_free = 0
                
                # Get information for each partition
                for partition in partitions:
                    try:
                        if platform.system() == "Windows" and "cdrom" in partition.opts or partition.fstype == "":
                            # Skip CD-ROM drives or non-mounted partitions on Windows
                            continue
                            
                        usage = psutil.disk_usage(partition.mountpoint)
                        
                        partition_info = {
                            "device": partition.device,
                            "mountpoint": partition.mountpoint,
                            "fstype": partition.fstype,
                            "opts": partition.opts,
                            "total_gb": round(usage.total / (1024**3), 2),
                            "used_gb": round(usage.used / (1024**3), 2),
                            "free_gb": round(usage.free / (1024**3), 2),
                            "percent_used": usage.percent
                        }
                        
                        disk_info["partitions"].append(partition_info)
                        
                        # Only add to totals if it's a real device (skip network mounts, etc.)
                        if (platform.system() == "Windows" and partition.device.startswith("\\")) or \
                           (platform.system() in ["Linux", "Darwin"] and partition.device.startswith("/")):
                            total_size += usage.total
                            total_used += usage.used
                            total_free += usage.free
                            
                    except (PermissionError, FileNotFoundError) as e:
                        # Skip partitions that can't be accessed
                        continue
                
                # Calculate totals
                if total_size > 0:
                    disk_info["total_size_gb"] = round(total_size / (1024**3), 2)
                    disk_info["total_used_gb"] = round(total_used / (1024**3), 2)
                    disk_info["total_free_gb"] = round(total_free / (1024**3), 2)
                    disk_info["total_percent_used"] = round(total_used / total_size * 100, 2)
                
                # Get disk I/O stats if available
                try:
                    disk_io = psutil.disk_io_counters(perdisk=False)
                    disk_info["io_read_bytes"] = disk_io.read_bytes
                    disk_info["io_write_bytes"] = disk_io.write_bytes
                    disk_info["io_read_count"] = disk_io.read_count
                    disk_info["io_write_count"] = disk_io.write_count
                    disk_info["io_read_time"] = disk_io.read_time
                    disk_info["io_write_time"] = disk_io.write_time
                except (AttributeError, OSError) as e:
                    # Disk I/O stats not available
                    pass
            
            self.disk_info = disk_info
            
            logger.info(f"Detected Disk: {disk_info.get('total_size_gb')} GB total, {disk_info.get('total_free_gb')} GB free")
            return self.disk_info
            
        except Exception as e:
            logger.error(f"Error detecting disk information: {e}")
            return {"total_size_gb": 100.0, "total_free_gb": 50.0}
    
    def detect_network(self) -> Dict[str, Any]:
        """
        Detect network information
        
        Returns:
            Dict[str, Any]: Network information
        """
        try:
            # Initialize with default values
            network_info = {
                "interfaces": []
            }
            
            if HAVE_PSUTIL:
                # Get network interfaces
                net_if_addrs = psutil.net_if_addrs()
                net_if_stats = psutil.net_if_stats()
                
                # Get information for each interface
                for interface, addrs in net_if_addrs.items():
                    interface_info = {
                        "name": interface,
                        "addresses": [],
                        "is_up": False,
                        "speed_mb": None,
                        "duplex": None,
                        "mtu": None
                    }
                    
                    # Get addresses
                    for addr in addrs:
                        addr_info = {
                            "family": str(addr.family),
                            "address": addr.address,
                            "netmask": addr.netmask,
                            "broadcast": addr.broadcast,
                            "ptp": addr.ptp
                        }
                        interface_info["addresses"].append(addr_info)
                    
                    # Get interface stats if available
                    if interface in net_if_stats:
                        stats = net_if_stats[interface]
                        interface_info["is_up"] = stats.isup
                        interface_info["speed_mb"] = stats.speed
                        interface_info["duplex"] = stats.duplex
                        interface_info["mtu"] = stats.mtu
                    
                    network_info["interfaces"].append(interface_info)
                
                # Get network I/O stats if available
                try:
                    net_io = psutil.net_io_counters(pernic=False)
                    network_info["io_bytes_sent"] = net_io.bytes_sent
                    network_info["io_bytes_recv"] = net_io.bytes_recv
                    network_info["io_packets_sent"] = net_io.packets_sent
                    network_info["io_packets_recv"] = net_io.packets_recv
                    network_info["io_errin"] = net_io.errin
                    network_info["io_errout"] = net_io.errout
                    network_info["io_dropin"] = net_io.dropin
                    network_info["io_dropout"] = net_io.dropout
                except (AttributeError, OSError) as e:
                    # Network I/O stats not available
                    pass
            
            self.network_info = network_info
            
            # Log active interfaces
            active_interfaces = [i["name"] for i in network_info["interfaces"] if i.get("is_up")]
            logger.info(f"Detected Network: {len(active_interfaces)} active interfaces: {', '.join(active_interfaces)}")
            
            return self.network_info
            
        except Exception as e:
            logger.error(f"Error detecting network information: {e}")
            return {"interfaces": []}
    
    def get_all_info(self) -> Dict[str, Any]:
        """
        Get all hardware information
        
        Returns:
            Dict[str, Any]: All hardware information
        """
        return {
            "system": self.system_info,
            "cpu": self.cpu_info,
            "memory": self.memory_info,
            "gpu": self.gpu_info,
            "disk": self.disk_info,
            "network": self.network_info,
            "timestamp": time.time()
        }
    
    def get_hardware_summary(self) -> Dict[str, Any]:
        """
        Get a summary of hardware information
        
        Returns:
            Dict[str, Any]: Hardware summary
        """
        summary = {}
        
        # System summary
        if self.system_info:
            summary["system"] = {
                "platform": self.system_info.get("platform"),
                "os_name": self.system_info.get("os_name", self.system_info.get("platform")),
                "os_version": self.system_info.get("os_version", self.system_info.get("platform_version")),
                "architecture": self.system_info.get("architecture")
            }
        
        # CPU summary
        if self.cpu_info:
            summary["cpu"] = {
                "brand": self.cpu_info.get("brand"),
                "cores_logical": self.cpu_info.get("count_logical"),
                "cores_physical": self.cpu_info.get("count_physical"),
                "supports_avx": self.cpu_info.get("supports_avx"),
                "supports_avx2": self.cpu_info.get("supports_avx2")
            }
        
        # Memory summary
        if self.memory_info:
            summary["memory"] = {
                "total_gb": self.memory_info.get("total_gb"),
                "available_gb": self.memory_info.get("available_gb")
            }
        
        # GPU summary
        if self.gpu_info:
            summary["gpu"] = {
                "has_gpu": self.gpu_info.get("has_gpu", False),
                "cuda_available": self.gpu_info.get("cuda_available", False),
                "mps_available": self.gpu_info.get("mps_available", False),
                "device_count": self.gpu_info.get("device_count", 0)
            }
            
            # Add GPU device summaries
            if self.gpu_info.get("devices"):
                summary["gpu"]["devices"] = []
                for device in self.gpu_info.get("devices", []):
                    device_summary = {
                        "name": device.get("name"),
                        "total_memory_gb": device.get("total_memory_gb")
                    }
                    summary["gpu"]["devices"].append(device_summary)
        
        # Disk summary
        if self.disk_info:
            summary["disk"] = {
                "total_size_gb": self.disk_info.get("total_size_gb"),
                "total_free_gb": self.disk_info.get("total_free_gb")
            }
        
        return summary
    
    def recommend_config(self) -> Dict[str, Any]:
        """
        Recommend optimal configuration based on detected hardware
        
        Returns:
            Dict[str, Any]: Configuration recommendations
        """
        # Make sure all detections have run
        if not hasattr(self, 'gpu_info') or not self.gpu_info:
            self.detect_gpu()
        if not hasattr(self, 'cpu_info') or not self.cpu_info:
            self.detect_cpu()
        if not hasattr(self, 'memory_info') or not self.memory_info:
            self.detect_memory()
        if not hasattr(self, 'disk_info') or not self.disk_info:
            self.detect_disk()

        recommendations = {}

        # Get references to hardware info
        gpu_info = self.gpu_info if hasattr(self, 'gpu_info') else {}
        memory_info = self.memory_info if hasattr(self, 'memory_info') else {}
        cpu_info = self.cpu_info if hasattr(self, 'cpu_info') else {}
        
        # Device recommendations
        recommendations["use_gpu"] = gpu_info.get("has_gpu", False) if isinstance(gpu_info, dict) else False

        if gpu_info.get("cuda_available", False):
            recommendations["device"] = "cuda"
        elif gpu_info.get("mps_available", False):
            recommendations["device"] = "mps"
        else:
            recommendations["device"] = "cpu"

        # Memory recommendations
        available_memory = memory_info.get("available_gb", 0)
        recommendations["memory"] = {}

        if available_memory > 0:
            # Conservative RAM usage estimate (leave headroom for OS and other processes)
            usable_memory = min(available_memory * 0.8, available_memory - 2)

            # Recommendations for batch sizes, etc.
            if usable_memory >= 16:
                recommendations["memory"]["batch_size"] = 32
                recommendations["memory"]["max_sequence_length"] = 2048
                recommendations["memory"]["model_size"] = "large"
            elif usable_memory >= 8:
                recommendations["memory"]["batch_size"] = 16
                recommendations["memory"]["max_sequence_length"] = 1024
                recommendations["memory"]["model_size"] = "medium"
            elif usable_memory >= 4:
                recommendations["memory"]["batch_size"] = 8
                recommendations["memory"]["max_sequence_length"] = 512
                recommendations["memory"]["model_size"] = "small"
            else:
                recommendations["memory"]["batch_size"] = 4
                recommendations["memory"]["max_sequence_length"] = 256
                recommendations["memory"]["model_size"] = "tiny"
        else:
            # Default conservative settings
            recommendations["memory"]["batch_size"] = 4
            recommendations["memory"]["max_sequence_length"] = 256
            recommendations["memory"]["model_size"] = "tiny"

        # CPU recommendations
        cpu_cores = cpu_info.get("count_logical", 1)
        recommendations["cpu"] = {}

        recommendations["cpu"]["num_workers"] = max(1, min(cpu_cores - 1, 8))  # Leave one core for OS
        recommendations["cpu"]["pin_memory"] = recommendations["device"] == "cuda"

        # Threading recommendations
        if cpu_cores >= 16:
            recommendations["cpu"]["use_threads"] = True
            recommendations["cpu"]["num_threads"] = max(1, min(cpu_cores - 2, 16))
        elif cpu_cores >= 4:
            recommendations["cpu"]["use_threads"] = True
            recommendations["cpu"]["num_threads"] = max(1, cpu_cores - 1)
        else:
            recommendations["cpu"]["use_threads"] = False
            recommendations["cpu"]["num_threads"] = 1

        # Model precision recommendations
        gpu_devices = gpu_info.get("devices", [])
        gpu_memory = sum(device.get("total_memory_gb", 0) for device in gpu_devices)
        
        if recommendations["device"] == "cuda":
            if gpu_memory >= 16:
                recommendations["precision"] = "float16"
            elif gpu_memory >= 8:
                recommendations["precision"] = "float16"
            else:
                recommendations["precision"] = "int8"
        elif recommendations["device"] == "mps":
            # Apple MPS device - enhanced logic for Apple Silicon (M1/M2/M3/M4)
            recommendations["precision"] = "float16"
            # Check for high-end Apple Silicon
            for device in gpu_devices:
                if device.get("type") == "MPS" and device.get("apple_silicon"):
                    if device.get("m3_family") or device.get("m4_family"):
                        # For M3/M4 we can use larger models
                        recommendations["memory"]["model_size"] = "large" 
                        recommendations["memory"]["batch_size"] = 24
                        recommendations["memory"]["max_sequence_length"] = 1536
        else:
            # CPU precision
            if cpu_info.get("supports_avx2", False):
                recommendations["precision"] = "float16"
            else:
                recommendations["precision"] = "int8"

        # Disk recommendations
        disk_info = self.disk_info if hasattr(self, 'disk_info') else {}
        disk_free = disk_info.get("total_free_gb", 0)
        recommendations["storage"] = {}

        if disk_free >= 100:
            recommendations["storage"]["cache_models"] = True
            recommendations["storage"]["cache_embeddings"] = True
            recommendations["storage"]["vector_db"] = "faiss"
        elif disk_free >= 20:
            recommendations["storage"]["cache_models"] = True
            recommendations["storage"]["cache_embeddings"] = False
            recommendations["storage"]["vector_db"] = "faiss"
        else:
            recommendations["storage"]["cache_models"] = False
            recommendations["storage"]["cache_embeddings"] = False
            recommendations["storage"]["vector_db"] = "hnswlib"

        # RAG specific recommendations
        recommendations["rag"] = {}

        # Embedding model size based on available memory and GPU
        if recommendations["device"] == "cuda" and gpu_memory >= 8:
            recommendations["rag"]["embedding_model"] = "large"
        elif (recommendations["device"] == "cuda" and gpu_memory >= 4) or available_memory >= 16:
            recommendations["rag"]["embedding_model"] = "medium"
        else:
            recommendations["rag"]["embedding_model"] = "small"

        # Chunk size and overlap
        recommendations["rag"]["chunk_size"] = 512
        recommendations["rag"]["chunk_overlap"] = 50

        # Retrieval strategy
        if recommendations["device"] == "cuda" or available_memory >= 16:
            recommendations["rag"]["retriever"] = "hybrid"
            recommendations["rag"]["hybrid_alpha"] = 0.7
        elif cpu_cores >= 8 and available_memory >= 8:
            recommendations["rag"]["retriever"] = "dense"
        else:
            recommendations["rag"]["retriever"] = "sparse"

        return recommendations
        
    def _get_cpu_name_alternative(self) -> str:
        """Get CPU model name using platform-specific methods."""
        system = platform.system()
        
        if system == "Darwin":  # macOS
            try:
                output = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
                
                # Check if Apple Silicon
                if not output or "Apple" not in output:
                    # Try alternative for Apple Silicon
                    model = subprocess.check_output(["sysctl", "-n", "hw.model"]).decode().strip()
                    if "Mac" in model:
                        return f"Apple {model}"
                        
                return output or "Unknown macOS CPU"
            except Exception:
                if platform.machine() == "arm64":
                    return "Apple Silicon"
                return "Unknown macOS CPU"
                
        elif system == "Linux":
            try:
                with open("/proc/cpuinfo", "r") as f:
                    for line in f:
                        if "model name" in line:
                            return line.split(":")[1].strip()
                return "Unknown Linux CPU"
            except Exception:
                return "Unknown Linux CPU"
                
        elif system == "Windows":
            try:
                from subprocess import PIPE, Popen
                process = Popen(["wmic", "cpu", "get", "name"], stdout=PIPE, stderr=PIPE)
                output = process.communicate()[0].decode().strip().split("\n")[1]
                return output
            except Exception:
                return "Unknown Windows CPU"
                
        return f"Unknown {system} CPU"
    
    def export_to_json(self, filepath: str, include_all: bool = False) -> bool:
        """
        Export hardware information to a JSON file
        
        Args:
            filepath (str): Path to save the JSON file
            include_all (bool): Whether to include all details or just summary
            
        Returns:
            bool: Success status
        """
        try:
            # Ensure we have the necessary information
            if not hasattr(self, 'system_info') or not self.system_info:
                self.detect_all()
                
            # Get data to export
            data = self.get_all_info() if include_all else self.get_hardware_summary()
            
            # Add recommendations
            data["recommendations"] = self.recommend_config()
            
            # Export to JSON file
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
                
            logger.info(f"Exported hardware information to {filepath}")
            console.print(f"[green]✓ Hardware information exported to:[/green] [cyan]{filepath}[/cyan]")
            return True
                
        except Exception as e:
            logger.error(f"Error exporting hardware information: {e}")
            console.print(f"[red]⚠ Error exporting hardware information: {str(e)}[/red]")
            return False