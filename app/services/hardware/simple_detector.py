# app/services/hardware/simple_detector.py
"""
Simplified Hardware Detector that avoids colorama issues
for system startup and graceful degradation
"""

import os
import platform
import sys
import logging
import time
from typing import Dict, Any

# Configure logging
logger = logging.getLogger(__name__)

class SimpleHardwareDetector:
    """
    Simplified hardware detection without fancy terminal UI
    to ensure consistent startup across environments
    """
    
    def __init__(self, config=None):
        self.config = config
        self.system_info = {}
        self.memory_info = {}
        self.cpu_info = {}
        self.gpu_info = {}
        self.disk_info = {}
        self.applied_config = {}

    def detect_all(self) -> Dict[str, Any]:
        """
        Detect all hardware components - simplified version
        
        Returns:
            Dict[str, Any]: Complete hardware information
        """
        logger.info("Starting simplified hardware detection...")
        
        # Detect each component sequentially
        self.detect_system()
        logger.info("System detection complete")
        
        self.detect_memory()
        logger.info("Memory detection complete")
        
        self.detect_cpu()
        logger.info("CPU detection complete")
        
        self.detect_gpu()
        logger.info("GPU detection complete")
        
        # Log summary
        self._log_hardware_summary()
        
        return self.get_all_info()
    
    def _log_hardware_summary(self):
        """Log a simple hardware summary"""
        system = self.system_info.get("platform", "Unknown")
        system_version = self.system_info.get("platform_version", "")
        cpu = self.cpu_info.get("brand", "Unknown CPU")
        cpu_cores = self.cpu_info.get("count_logical", 0)
        memory = self.memory_info.get("total_gb", 0)
        gpu_available = self.gpu_info.get("cuda_available", False) or self.gpu_info.get("mps_available", False)
        
        logger.info(f"Hardware Summary: {system} {system_version}, {cpu} ({cpu_cores} cores), {memory:.1f}GB RAM, GPU: {'Available' if gpu_available else 'Not available'}")
        
        # Log device recommendation
        recommendations = self.recommend_config()
        logger.info(f"Recommended device: {recommendations.get('device', 'cpu')}")
    
    def detect_system(self) -> Dict[str, Any]:
        """Detect basic system information with enhanced Apple Silicon detection"""
        try:
            # Basic system info
            self.system_info = {
                "platform": platform.system(),
                "platform_release": platform.release(),
                "platform_version": platform.version(),
                "architecture": platform.machine(),
                "hostname": platform.node(),
                "processor": platform.processor(),
                "python_version": sys.version
            }
            
            # Enhanced Mac detection for Apple Silicon
            if platform.system() == "Darwin" and platform.machine() == "arm64":
                # Check for Apple Silicon model (M1/M2/M3/M4)
                try:
                    # Get Mac model
                    mac_model = os.popen('sysctl -n hw.model').read().strip()
                    self.system_info["mac_model"] = mac_model
                    
                    # Try to get CPU brand string
                    cpu_brand = os.popen('sysctl -n machdep.cpu.brand_string').read().strip()
                    self.system_info["cpu_brand"] = cpu_brand
                    
                    # Additional Apple Silicon detection via memory size for M4 Pro/Max with 48GB
                    # This is useful because the hw.model sometimes doesn't specify M4 directly
                    hw_memsize = os.popen('sysctl -n hw.memsize').read().strip()
                    if hw_memsize and hw_memsize.isdigit():
                        # Convert to GB
                        total_gb = int(hw_memsize) / (1024**3)
                        # Check if 48GB RAM, which is unique to M4 Pro/Max
                        if abs(total_gb - 48) < 1:  # Within 1GB of 48GB
                            self.system_info["apple_silicon"] = "M4"
                            self.system_info["processor_type"] = "apple_silicon_m4_pro_max"
                            self.system_info["apple_silicon_variant"] = "M4 Pro/Max"
                            self.system_info["memory_config"] = "48GB"
                            self.system_info["is_m4_pro_max"] = True
                            logger.info("Detected M4 Pro/Max with 48GB RAM configuration")
                    
                    # If not already detected via memory, try to determine from CPU/model info
                    if "apple_silicon" not in self.system_info:
                        if "M4" in cpu_brand or "M4" in mac_model:
                            self.system_info["apple_silicon"] = "M4"
                            self.system_info["processor_type"] = "apple_silicon_m4"
                            # Get CPU cores to check if it's likely a Pro/Max variant
                            cpu_cores = os.popen('sysctl -n hw.ncpu').read().strip()
                            if cpu_cores and cpu_cores.isdigit() and int(cpu_cores) >= 14:
                                self.system_info["apple_silicon_variant"] = "M4 Pro/Max"
                                self.system_info["is_m4_pro_max"] = True
                            else:
                                self.system_info["apple_silicon_variant"] = "M4"
                        elif "M3" in cpu_brand or "M3" in mac_model:
                            self.system_info["apple_silicon"] = "M3"
                            self.system_info["processor_type"] = "apple_silicon_m3" 
                        elif "M2" in cpu_brand or "M2" in mac_model:
                            self.system_info["apple_silicon"] = "M2"
                            self.system_info["processor_type"] = "apple_silicon_m2"
                        elif "M1" in cpu_brand or "M1" in mac_model:
                            self.system_info["apple_silicon"] = "M1"
                            self.system_info["processor_type"] = "apple_silicon_m1"
                        else:
                            self.system_info["apple_silicon"] = "Unknown"
                            self.system_info["processor_type"] = "apple_silicon"
                        
                    # Try to identify the specific variant
                    if "apple_silicon_variant" not in self.system_info:
                        # Check memory size for more precise detection
                        if hw_memsize and hw_memsize.isdigit():
                            total_gb = int(hw_memsize) / (1024**3)
                            if self.system_info.get("apple_silicon") == "M4":
                                if abs(total_gb - 48) < 1:  # 48GB
                                    self.system_info["apple_silicon_variant"] = "M4 Pro/Max"
                                    self.system_info["is_m4_pro_max"] = True
                                elif abs(total_gb - 32) < 1:  # 32GB
                                    self.system_info["apple_silicon_variant"] = "M4 Pro/Max"
                                    self.system_info["is_m4_pro_max"] = True
                                elif abs(total_gb - 24) < 1:  # 24GB
                                    self.system_info["apple_silicon_variant"] = "M4 Pro"
                                    self.system_info["is_m4_pro_max"] = True
                                elif abs(total_gb - 16) < 1:  # 16GB
                                    self.system_info["apple_silicon_variant"] = "M4"
                                else:
                                    self.system_info["apple_silicon_variant"] = "M4"
                            # Similar logic for other M-series if needed
                                    
                    logger.info(f"Detected Apple Silicon: {self.system_info.get('apple_silicon', 'Unknown')} "
                              f"{self.system_info.get('apple_silicon_variant', '')}")
                except Exception as e:
                    logger.warning(f"Error detecting Apple Silicon model: {e}")
                    self.system_info["processor_type"] = "apple_silicon"
            
            logger.info(f"Detected system: {self.system_info.get('platform')} {self.system_info.get('platform_version')}")
            return self.system_info
        except Exception as e:
            logger.error(f"Error detecting system information: {e}")
            return {}
    
    def detect_memory(self) -> Dict[str, Any]:
        """Detect memory information with enhanced detection for Apple Silicon"""
        # Default conservative values
        self.memory_info = {
            "total_gb": 8.0,
            "available_gb": 4.0
        }
        
        try:
            # Try to import psutil if available
            import psutil
            virtual_memory = psutil.virtual_memory()
            
            self.memory_info = {
                "total_gb": round(virtual_memory.total / (1024**3), 2),
                "available_gb": round(virtual_memory.available / (1024**3), 2)
            }
            
            # Enhanced detection for Apple Silicon Mac - more accurate on some macOS versions
            if platform.system() == "Darwin":
                try:
                    # Try using sysctl to get physical memory size
                    hw_memsize = os.popen('sysctl -n hw.memsize').read().strip()
                    if hw_memsize and hw_memsize.isdigit():
                        # Convert to GB
                        total_gb = int(hw_memsize) / (1024**3)
                        self.memory_info["total_gb"] = round(total_gb, 2)
                        
                        # If total memory is detected as 48GB or close, we know it's an M4 Pro/Max
                        if abs(total_gb - 48) < 1:  # Within 1GB of 48GB
                            logger.info("Detected 48GB memory configuration (M4 Pro/Max)")
                            # Add metadata about the memory configuration
                            self.memory_info["apple_silicon_memory"] = "48GB"
                            self.memory_info["memory_type"] = "unified"
                            self.memory_info["memory_configuration"] = "m4_pro_max_48gb"
                            
                            # Calculate available memory as a percentage of total if psutil is available
                            if "available_gb" in self.memory_info:
                                avail_ratio = self.memory_info["available_gb"] / self.memory_info["total_gb"]
                                self.memory_info["available_gb"] = round(48 * avail_ratio, 2)
                            else:
                                # Estimate available memory as 50% of total
                                self.memory_info["available_gb"] = 24.0
                            
                            # Get more memory details for M4 Pro/Max
                            try:
                                # Try to get page size and vm stats for more accurate memory assessment
                                page_size = int(os.popen('sysctl -n hw.pagesize').read().strip())
                                vm_stat = os.popen('vm_stat').read().strip()
                                
                                # Parse vm_stat output for better memory insights
                                if vm_stat:
                                    lines = vm_stat.split('\n')
                                    stats = {}
                                    for line in lines[1:]:  # Skip the first line
                                        if ':' in line:
                                            key, val = line.split(':')
                                            stats[key.strip()] = int(val.strip().replace('.', ''))
                                    
                                    # Free pages are available for allocation
                                    if "Pages free" in stats:
                                        free_pages = stats["Pages free"]
                                        free_memory = (free_pages * page_size) / (1024**3)
                                        self.memory_info["free_gb"] = round(free_memory, 2)
                                    
                                    # Calculate a more accurate available memory figure for M4
                                    if all(k in stats for k in ["Pages free", "Pages inactive", "Pages speculative"]):
                                        available_pages = stats["Pages free"] + stats["Pages inactive"] + stats["Pages speculative"]
                                        self.memory_info["available_gb"] = round((available_pages * page_size) / (1024**3), 2)
                                        logger.info(f"Enhanced memory detection: {self.memory_info['available_gb']} GB available")
                            except Exception as e:
                                logger.warning(f"Error getting detailed memory stats: {e}")
                        
                        # Also detect other M-series memory configurations
                        elif abs(total_gb - 32) < 1:  # 32GB
                            self.memory_info["apple_silicon_memory"] = "32GB"
                            self.memory_info["memory_type"] = "unified"
                            self.memory_info["memory_configuration"] = "apple_silicon_32gb"
                        elif abs(total_gb - 24) < 1:  # 24GB
                            self.memory_info["apple_silicon_memory"] = "24GB"
                            self.memory_info["memory_type"] = "unified"
                            self.memory_info["memory_configuration"] = "apple_silicon_24gb"
                        elif abs(total_gb - 16) < 1:  # 16GB
                            self.memory_info["apple_silicon_memory"] = "16GB"
                            self.memory_info["memory_type"] = "unified"
                            self.memory_info["memory_configuration"] = "apple_silicon_16gb"
                        elif abs(total_gb - 8) < 1:  # 8GB
                            self.memory_info["apple_silicon_memory"] = "8GB"
                            self.memory_info["memory_type"] = "unified"
                            self.memory_info["memory_configuration"] = "apple_silicon_8gb"
                except Exception as e:
                    logger.warning(f"Error detecting memory with sysctl: {e}")
        except ImportError:
            logger.warning("psutil not available - using enhanced detection")
            # Try using platform-specific commands
            if platform.system() == "Darwin":
                try:
                    # Try using sysctl to get physical memory size
                    hw_memsize = os.popen('sysctl -n hw.memsize').read().strip()
                    if hw_memsize and hw_memsize.isdigit():
                        # Convert to GB
                        total_gb = int(hw_memsize) / (1024**3)
                        self.memory_info["total_gb"] = round(total_gb, 2)
                        # Estimate available memory as 50% of total (conservative)
                        self.memory_info["available_gb"] = round(total_gb * 0.5, 2)
                        
                        # If total memory is detected as 48GB or close, we know it's an M4 Pro/Max
                        if abs(total_gb - 48) < 1:  # Within 1GB of 48GB
                            logger.info("Detected 48GB memory configuration (M4 Pro/Max)")
                            # Add metadata about the memory configuration
                            self.memory_info["apple_silicon_memory"] = "48GB"
                            self.memory_info["memory_type"] = "unified"
                            self.memory_info["memory_configuration"] = "m4_pro_max_48gb"
                            # For M4 Pro/Max with 48GB, we can use a higher available estimate
                            self.memory_info["available_gb"] = round(total_gb * 0.6, 2)  # 60% available is reasonable
                except Exception as e:
                    logger.warning(f"Error detecting memory with sysctl: {e}")
        except Exception as e:
            logger.error(f"Error detecting memory: {e}")
        
        # Add memory model tagging
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            if self.memory_info.get("total_gb", 0) >= 48:
                self.memory_info["memory_model"] = "high_capacity"
                self.memory_info["suitable_for_large_models"] = True
            elif self.memory_info.get("total_gb", 0) >= 32:
                self.memory_info["memory_model"] = "high_capacity"
                self.memory_info["suitable_for_large_models"] = True
            elif self.memory_info.get("total_gb", 0) >= 16:
                self.memory_info["memory_model"] = "medium_capacity"
                self.memory_info["suitable_for_medium_models"] = True
            else:
                self.memory_info["memory_model"] = "standard_capacity"
        
        logger.info(f"Detected Memory: {self.memory_info.get('total_gb')} GB total, {self.memory_info.get('available_gb')} GB available")
        if "memory_model" in self.memory_info:
            logger.info(f"Memory Model: {self.memory_info.get('memory_model')}")
        return self.memory_info
    
    def detect_cpu(self) -> Dict[str, Any]:
        """Detect CPU information with enhanced Apple Silicon detection"""
        # Default values
        self.cpu_info = {
            "brand": "Unknown CPU",
            "count_logical": 1,
            "count_physical": 1
        }
        
        try:
            # Try to get CPU count via os module
            import os
            self.cpu_info["count_logical"] = os.cpu_count() or 1
            
            # Try to get more detailed info if psutil is available
            try:
                import psutil
                self.cpu_info["count_physical"] = psutil.cpu_count(logical=False) or 1
            except ImportError:
                pass
                
            # Try to get CPU name
            system = platform.system()
            if system == "Darwin":  # macOS
                try:
                    # First try the standard CPU brand string
                    output = os.popen('sysctl -n machdep.cpu.brand_string').read().strip()
                    if output:
                        self.cpu_info["brand"] = output
                    
                    # For Apple Silicon, this may not show up properly, so check architecture
                    if platform.machine() == "arm64":
                        # Get Mac model for Apple Silicon identification
                        mac_model = os.popen('sysctl -n hw.model').read().strip()
                        self.cpu_info["mac_model"] = mac_model
                        
                        # Check memory size - this is the most reliable way to identify M4 Pro/Max with 48GB
                        hw_memsize = os.popen('sysctl -n hw.memsize').read().strip()
                        if hw_memsize and hw_memsize.isdigit():
                            total_gb = int(hw_memsize) / (1024**3)
                            
                            # This is most likely an M4 Pro/Max based on 48GB memory
                            if abs(total_gb - 48) < 1:  # Within 1GB of 48GB
                                # Get CPU cores to distinguish between Pro and Max variants
                                cpu_cores = os.popen('sysctl -n hw.ncpu').read().strip()
                                if cpu_cores and cpu_cores.isdigit():
                                    cores = int(cpu_cores)
                                    # Get active/performance/efficiency cores
                                    try:
                                        # This might not work on all macOS versions, but worth trying
                                        perf_cores = int(os.popen('sysctl -n hw.perflevel1.physicalcpu').read().strip() or '0')
                                        eff_cores = int(os.popen('sysctl -n hw.perflevel0.physicalcpu').read().strip() or '0')
                                        
                                        # Physical cores is the sum of performance and efficiency cores
                                        physical_cores = perf_cores + eff_cores
                                        
                                        # Detect specific M4 model based on core count
                                        if physical_cores >= 14:
                                            # Likely M4 Max
                                            self.cpu_info["brand"] = "Apple M4 Max"
                                            self.cpu_info["chip_variant"] = "M4 Max"
                                            self.cpu_info["performance_cores"] = perf_cores
                                            self.cpu_info["efficiency_cores"] = eff_cores
                                        elif physical_cores >= 11:
                                            # Likely M4 Pro
                                            self.cpu_info["brand"] = "Apple M4 Pro"
                                            self.cpu_info["chip_variant"] = "M4 Pro"
                                            self.cpu_info["performance_cores"] = perf_cores
                                            self.cpu_info["efficiency_cores"] = eff_cores
                                        else:
                                            # Default to M4 Pro/Max
                                            self.cpu_info["brand"] = "Apple M4 Pro/Max"
                                            self.cpu_info["chip_variant"] = "M4 Pro/Max"
                                    except:
                                        # Fallback detection based on logical cores
                                        if cores >= 24:
                                            # Likely M4 Max
                                            self.cpu_info["brand"] = "Apple M4 Max"
                                            self.cpu_info["chip_variant"] = "M4 Max"
                                            self.cpu_info["count_physical"] = cores // 2  # Approximate
                                        elif cores >= 14:
                                            # Likely M4 Pro
                                            self.cpu_info["brand"] = "Apple M4 Pro"
                                            self.cpu_info["chip_variant"] = "M4 Pro"
                                            self.cpu_info["count_physical"] = cores // 2  # Approximate
                                        else:
                                            # Default to M4 Pro/Max
                                            self.cpu_info["brand"] = "Apple M4 Pro/Max"
                                            self.cpu_info["chip_variant"] = "M4 Pro/Max"
                                            self.cpu_info["count_physical"] = cores // 2  # Approximate
                                else:
                                    # Can't get CPU cores, default to M4 Pro/Max
                                    self.cpu_info["brand"] = "Apple M4 Pro/Max"
                                    self.cpu_info["chip_variant"] = "M4 Pro/Max"
                                    self.cpu_info["count_logical"] = 14  # Default for M4 Pro
                                    self.cpu_info["count_physical"] = 7   # Default for M4 Pro
                                
                                # Common attributes for all M4 Pro/Max with 48GB RAM
                                self.cpu_info["apple_silicon"] = True
                                self.cpu_info["chip_type"] = "M4_Pro_Max"
                                self.cpu_info["architecture"] = "arm64"
                                self.cpu_info["core_architecture"] = "ARMv8.5-A"
                                self.cpu_info["neural_engine"] = True
                                self.cpu_info["ram_unified"] = True
                                self.cpu_info["ram_size_gb"] = 48
                                # Get CPU frequency if available
                                try:
                                    freq_data = os.popen('sysctl -n hw.cpufrequency').read().strip()
                                    if freq_data and freq_data.isdigit():
                                        freq_mhz = int(freq_data) / 1000000
                                        self.cpu_info["frequency_mhz"] = round(freq_mhz, 2)
                                except:
                                    pass
                                    
                                # Log detailed detection
                                logger.info(f"Detected {self.cpu_info['brand']} with {self.cpu_info.get('count_logical', 0)} logical cores")
                                
                            # Handle other memory configurations for Apple Silicon
                            elif abs(total_gb - 32) < 1:  # 32GB
                                # Could be M4/M3/M2 Pro/Max
                                if "M4" in mac_model or (hasattr(self, 'system_info') and self.system_info.get('apple_silicon') == 'M4'):
                                    self.cpu_info["brand"] = "Apple M4 Pro"
                                    self.cpu_info["chip_type"] = "M4_Pro"
                                elif "M3" in mac_model or (hasattr(self, 'system_info') and self.system_info.get('apple_silicon') == 'M3'):
                                    self.cpu_info["brand"] = "Apple M3 Pro/Max"
                                    self.cpu_info["chip_type"] = "M3_Pro_Max"
                                else:
                                    self.cpu_info["brand"] = "Apple M-series Pro/Max"
                                    self.cpu_info["chip_type"] = "M_Series_Pro_Max"
                                self.cpu_info["apple_silicon"] = True
                                self.cpu_info["ram_unified"] = True
                                self.cpu_info["ram_size_gb"] = 32
                            elif abs(total_gb - 24) < 1:  # 24GB
                                # Could be M4/M3/M2 Pro
                                if "M4" in mac_model or (hasattr(self, 'system_info') and self.system_info.get('apple_silicon') == 'M4'):
                                    self.cpu_info["brand"] = "Apple M4 Pro"
                                    self.cpu_info["chip_type"] = "M4_Pro"
                                elif "M3" in mac_model or (hasattr(self, 'system_info') and self.system_info.get('apple_silicon') == 'M3'):
                                    self.cpu_info["brand"] = "Apple M3 Pro"
                                    self.cpu_info["chip_type"] = "M3_Pro"
                                else:
                                    self.cpu_info["brand"] = "Apple M-series Pro"
                                    self.cpu_info["chip_type"] = "M_Series_Pro"
                                self.cpu_info["apple_silicon"] = True
                                self.cpu_info["ram_unified"] = True
                                self.cpu_info["ram_size_gb"] = 24
                            elif abs(total_gb - 16) < 1:  # 16GB
                                # Standard M4/M3/M2/M1
                                self.cpu_info["brand"] = "Apple M-series"
                                self.cpu_info["apple_silicon"] = True
                                self.cpu_info["chip_type"] = "M_series"
                                self.cpu_info["ram_unified"] = True
                                self.cpu_info["ram_size_gb"] = 16
                            else:
                                # Other Apple Silicon
                                self.cpu_info["brand"] = "Apple M-series"
                                self.cpu_info["apple_silicon"] = True
                                self.cpu_info["chip_type"] = "M_series"
                                self.cpu_info["ram_unified"] = True
                                
                        # If we have system_info with apple_silicon already detected, use that
                        elif hasattr(self, 'system_info') and 'apple_silicon' in self.system_info:
                            silicon_gen = self.system_info['apple_silicon']
                            if silicon_gen == "M4":
                                self.cpu_info["brand"] = "Apple M4"
                                self.cpu_info["chip_type"] = "M4"
                            elif silicon_gen == "M3":
                                self.cpu_info["brand"] = "Apple M3"
                                self.cpu_info["chip_type"] = "M3"
                            elif silicon_gen == "M2":
                                self.cpu_info["brand"] = "Apple M2"
                                self.cpu_info["chip_type"] = "M2"
                            elif silicon_gen == "M1":
                                self.cpu_info["brand"] = "Apple M1"
                                self.cpu_info["chip_type"] = "M1"
                            else:
                                self.cpu_info["brand"] = "Apple Silicon"
                                self.cpu_info["chip_type"] = "Apple_Silicon"
                            self.cpu_info["apple_silicon"] = True
                        else:
                            # Fallback for generic Apple Silicon
                            self.cpu_info["brand"] = "Apple Silicon"
                            self.cpu_info["apple_silicon"] = True
                            self.cpu_info["chip_type"] = "Apple_Silicon"
                    else:
                        # For Intel Mac
                        if not self.cpu_info.get("brand") or "Unknown" in self.cpu_info.get("brand", ""):
                            self.cpu_info["brand"] = "Intel Mac CPU"
                            self.cpu_info["architecture"] = "x86_64"
                except Exception as e:
                    logger.warning(f"Error detecting Apple Silicon CPU details: {e}")
                    # Fallback detection for Apple Silicon
                    if platform.machine() == "arm64":
                        self.cpu_info["brand"] = "Apple Silicon"
                        self.cpu_info["apple_silicon"] = True
            elif system == "Linux":
                try:
                    with open("/proc/cpuinfo", "r") as f:
                        for line in f:
                            if "model name" in line:
                                self.cpu_info["brand"] = line.split(":")[1].strip()
                                break
                except:
                    pass
        except Exception as e:
            logger.error(f"Error detecting CPU: {e}")
        
        logger.info(f"Detected CPU: {self.cpu_info.get('brand')} with {self.cpu_info.get('count_logical')} logical cores")
        if self.cpu_info.get("apple_silicon") and self.cpu_info.get("chip_variant"):
            logger.info(f"Apple Silicon variant: {self.cpu_info.get('chip_variant')}")
        return self.cpu_info
    
    def detect_gpu(self) -> Dict[str, Any]:
        """Detect GPU information with enhanced Apple Silicon detection"""
        # Default values
        self.gpu_info = {
            "has_gpu": False,
            "cuda_available": False,
            "mps_available": False,
        }
        
        try:
            # Check for CUDA via PyTorch if available
            try:
                import torch
                self.gpu_info["cuda_available"] = torch.cuda.is_available()
                
                if torch.cuda.is_available():
                    self.gpu_info["has_gpu"] = True
                    self.gpu_info["device_count"] = torch.cuda.device_count()
                
                # Check for MPS on Apple Silicon
                if platform.system() == "Darwin" and hasattr(torch.backends, "mps"):
                    self.gpu_info["mps_available"] = torch.backends.mps.is_available()
                    if torch.backends.mps.is_available():
                        self.gpu_info["has_gpu"] = True
                        self.gpu_info["mps_device"] = "mps"
                
            except ImportError:
                logger.warning("PyTorch not available - using fallback GPU detection")
                # Even without PyTorch, detect Apple Silicon GPU
                if platform.system() == "Darwin" and platform.machine() == "arm64":
                    self.gpu_info["has_gpu"] = True
                    self.gpu_info["mps_available"] = True  # Assume MPS is available on Apple Silicon
                    self.gpu_info["mps_device"] = "mps"
                    
            except Exception as e:
                logger.warning(f"Error detecting GPU via PyTorch: {e}")
                # Fallback detection for Apple Silicon
                if platform.system() == "Darwin" and platform.machine() == "arm64":
                    self.gpu_info["has_gpu"] = True
                    self.gpu_info["mps_available"] = True  # Assume MPS is available on Apple Silicon
                    self.gpu_info["mps_device"] = "mps"
            
            # Enhanced Apple Silicon GPU detection
            if platform.system() == "Darwin" and platform.machine() == "arm64":
                # Mark this as an Apple Silicon GPU
                self.gpu_info["has_gpu"] = True
                self.gpu_info["apple_silicon_gpu"] = True
                
                # Get the Metal device info if possible
                try:
                    # Check for IORegistry information about the GPU
                    ioreg_output = os.popen('ioreg -rc AppleM1Device -l').read().strip()
                    if "M4" in ioreg_output:
                        self.gpu_info["gpu_model"] = "M4"
                    elif "M3" in ioreg_output:
                        self.gpu_info["gpu_model"] = "M3"
                    elif "M2" in ioreg_output:
                        self.gpu_info["gpu_model"] = "M2"
                    elif "M1" in ioreg_output:
                        self.gpu_info["gpu_model"] = "M1"
                    
                    # Try to get GPU core count from ioreg
                    import re
                    core_match = re.search(r'gpu-core-count"\s*=\s*(\d+)', ioreg_output)
                    if core_match:
                        self.gpu_info["gpu_core_count"] = int(core_match.group(1))
                except Exception as e:
                    logger.warning(f"Error getting detailed GPU info: {e}")
                
                # Check CPU info for specific M4 Pro/Max detection with 48GB RAM
                m4_detected = False
                
                # See if we've already detected M4 Pro/Max via memory size
                if hasattr(self, 'memory_info') and self.memory_info.get('memory_configuration') == "m4_pro_max_48gb":
                    m4_detected = True
                    gpu_name = "Apple M4 Pro/Max Integrated GPU"
                    gpu_core_count = self.gpu_info.get("gpu_core_count", 0)
                    
                    # Determine if Pro or Max based on GPU core count
                    if gpu_core_count >= 40:
                        gpu_name = "Apple M4 Max Integrated GPU"
                        gpu_variant = "M4 Max"
                        gpu_compute_units = gpu_core_count
                    elif gpu_core_count >= 28:
                        gpu_name = "Apple M4 Pro Integrated GPU"
                        gpu_variant = "M4 Pro"
                        gpu_compute_units = gpu_core_count
                    else:
                        gpu_name = "Apple M4 Pro Integrated GPU"
                        gpu_variant = "M4 Pro"
                        gpu_compute_units = 28  # Default assumption
                    
                    # Enhanced M4 Pro/Max GPU info
                    self.gpu_info["gpu_name"] = gpu_name
                    self.gpu_info["gpu_variant"] = gpu_variant
                    self.gpu_info["gpu_compute_units"] = gpu_compute_units
                    self.gpu_info["gpu_memory_gb"] = 48  # Unified memory
                    self.gpu_info["gpu_memory_unified"] = True
                    self.gpu_info["gpu_family"] = "Apple M4"
                    self.gpu_info["neural_engine_cores"] = 16  # M4 has 16-core Neural Engine
                    self.gpu_info["mps_optimized"] = True
                    self.gpu_info["gpu_tensorflow_compatible"] = True
                    self.gpu_info["gpu_pytorch_compatible"] = True
                    self.gpu_info["gpu_performance_level"] = "high"
                    
                    # Add device details
                    self.gpu_info["devices"] = [{
                        "name": gpu_name,
                        "type": "MPS",
                        "apple_silicon": True,
                        "m4_family": True,
                        "m4_variant": gpu_variant,
                        "compute_units": gpu_compute_units,
                        "memory_gb": 48,
                        "memory_unified": True,
                        "neural_engine_cores": 16,
                        "index": 0,
                    }]
                
                # Check if we have identified an M4 Pro/Max system in CPU detection
                elif hasattr(self, 'cpu_info') and 'chip_type' in self.cpu_info:
                    if self.cpu_info.get('chip_type') == "M4_Pro_Max":
                        # For M4 Pro/Max with 48GB RAM, identify the integrated GPU
                        memory_gb = self.memory_info.get('total_gb', 0) if hasattr(self, 'memory_info') else 0
                        
                        if abs(memory_gb - 48) < 1:  # 48GB memory configuration
                            m4_detected = True
                            
                            # Determine exact variant based on CPU info
                            if self.cpu_info.get('chip_variant', '') == "M4 Max":
                                # M4 Max GPU with more GPU cores
                                self.gpu_info["gpu_name"] = "Apple M4 Max Integrated GPU"
                                self.gpu_info["gpu_variant"] = "M4 Max"
                                self.gpu_info["gpu_compute_units"] = 40  # Default for M4 Max
                            else:
                                # M4 Pro GPU
                                self.gpu_info["gpu_name"] = "Apple M4 Pro Integrated GPU"
                                self.gpu_info["gpu_variant"] = "M4 Pro"
                                self.gpu_info["gpu_compute_units"] = 28  # Default for M4 Pro
                            
                            # Common M4 GPU attributes
                            self.gpu_info["gpu_memory_gb"] = 48  # Unified memory
                            self.gpu_info["gpu_memory_unified"] = True
                            self.gpu_info["gpu_family"] = "Apple M4"
                            self.gpu_info["neural_engine_cores"] = 16  # M4 has 16-core Neural Engine
                            self.gpu_info["mps_optimized"] = True
                            
                            # Add device details
                            self.gpu_info["devices"] = [{
                                "name": self.gpu_info["gpu_name"],
                                "type": "MPS",
                                "apple_silicon": True,
                                "m4_family": True,
                                "m4_variant": self.gpu_info["gpu_variant"],
                                "compute_units": self.gpu_info["gpu_compute_units"],
                                "memory_gb": 48,
                                "memory_unified": True,
                                "neural_engine_cores": 16,
                                "index": 0,
                            }]
                    
                # If M4 wasn't specifically detected, check for other Apple Silicon GPUs
                if not m4_detected:
                    # Try to determine M-series from system_info
                    if hasattr(self, 'system_info') and 'apple_silicon' in self.system_info:
                        silicon_gen = self.system_info['apple_silicon']
                        memory_gb = self.memory_info.get('total_gb', 0) if hasattr(self, 'memory_info') else 0
                        
                        if silicon_gen == "M4":
                            # Generic M4
                            self.gpu_info["gpu_name"] = "Apple M4 Integrated GPU"
                            self.gpu_info["gpu_family"] = "Apple M4"
                            self.gpu_info["neural_engine_cores"] = 16
                        elif silicon_gen == "M3":
                            # Generic M3
                            self.gpu_info["gpu_name"] = "Apple M3 Integrated GPU"
                            self.gpu_info["gpu_family"] = "Apple M3"
                            self.gpu_info["neural_engine_cores"] = 16
                        elif silicon_gen == "M2":
                            # Generic M2
                            self.gpu_info["gpu_name"] = "Apple M2 Integrated GPU"
                            self.gpu_info["gpu_family"] = "Apple M2"
                            self.gpu_info["neural_engine_cores"] = 16
                        elif silicon_gen == "M1":
                            # Generic M1
                            self.gpu_info["gpu_name"] = "Apple M1 Integrated GPU"
                            self.gpu_info["gpu_family"] = "Apple M1"
                            self.gpu_info["neural_engine_cores"] = 16
                        else:
                            # Generic Apple Silicon
                            self.gpu_info["gpu_name"] = "Apple Silicon Integrated GPU"
                            self.gpu_info["gpu_family"] = "Apple Silicon"
                            self.gpu_info["neural_engine_cores"] = 16
                        
                        # Add memory info
                        self.gpu_info["gpu_memory_gb"] = memory_gb
                        self.gpu_info["gpu_memory_unified"] = True
                        
                        # Add device details
                        self.gpu_info["devices"] = [{
                            "name": self.gpu_info["gpu_name"],
                            "type": "MPS",
                            "apple_silicon": True,
                            "memory_gb": memory_gb,
                            "memory_unified": True,
                            "neural_engine_cores": self.gpu_info["neural_engine_cores"],
                            "index": 0,
                        }]
                    else:
                        # Generic Apple Silicon GPU info
                        self.gpu_info["gpu_name"] = "Apple Integrated GPU"
                        self.gpu_info["gpu_family"] = "Apple Silicon"
                        self.gpu_info["devices"] = [{
                            "name": "Apple Integrated GPU",
                            "type": "MPS",
                            "apple_silicon": True,
                            "index": 0,
                        }]
                
                # Add ML performance metrics for models - this is useful for the model configuration
                if "gpu_variant" in self.gpu_info and self.gpu_info["gpu_variant"] in ["M4 Pro", "M4 Max"]:
                    if self.gpu_info["gpu_variant"] == "M4 Max":
                        # Higher performance profile for M4 Max
                        self.gpu_info["ml_performance"] = {
                            "batch_size_recommendation": 128,
                            "max_sequence_length": 4096,
                            "performance_class": "ultra_high",
                            "suitable_models": ["large", "medium", "small"],
                            "transformer_tflops": 800,  # Approximate TFLOPs for transformers
                            "fp16_support": True,
                            "int8_support": True,
                            "recommended_precision": "float16"
                        }
                    else:  # M4 Pro
                        # High performance profile for M4 Pro
                        self.gpu_info["ml_performance"] = {
                            "batch_size_recommendation": 64,
                            "max_sequence_length": 4096,
                            "performance_class": "high",
                            "suitable_models": ["large", "medium", "small"],
                            "transformer_tflops": 600,  # Approximate TFLOPs for transformers
                            "fp16_support": True,
                            "int8_support": True,
                            "recommended_precision": "float16"
                        }
                
        except Exception as e:
            logger.error(f"Error detecting GPU: {e}")
        
        if self.gpu_info["has_gpu"]:
            if self.gpu_info["cuda_available"]:
                logger.info(f"Detected GPU with CUDA support")
            elif self.gpu_info["mps_available"]:
                if "gpu_name" in self.gpu_info:
                    logger.info(f"Detected {self.gpu_info['gpu_name']} with MPS support")
                    if "gpu_compute_units" in self.gpu_info:
                        logger.info(f"GPU has {self.gpu_info['gpu_compute_units']} compute units")
                    if "neural_engine_cores" in self.gpu_info:
                        logger.info(f"Neural Engine: {self.gpu_info['neural_engine_cores']} cores")
                else:
                    logger.info("Detected Apple GPU with MPS support")
            else:
                logger.info("GPU detected but not accessible through CUDA or MPS")
        else:
            logger.info("No GPU detected")
        
        return self.gpu_info
    
    def get_all_info(self) -> Dict[str, Any]:
        """Get all hardware information with enhanced Apple Silicon support"""
        # Create a comprehensive hardware info dictionary
        hardware_info = {
            "system": self.system_info,
            "cpu": self.cpu_info,
            "memory": self.memory_info,
            "gpu": self.gpu_info,
            "timestamp": time.time()
        }
        
        # Add Apple Silicon specific fields
        is_apple_silicon = platform.system() == "Darwin" and platform.machine() == "arm64"
        if is_apple_silicon:
            hardware_info["is_apple_silicon"] = True
            
            # Check for M4 Pro/Max with 48GB RAM through multiple detection methods
            is_m4_pro_max = False
            
            # Method 1: Memory size
            memory_gb = self.memory_info.get('total_gb', 0)
            if abs(memory_gb - 48) < 1:
                is_m4_pro_max = True
            
            # Method 2: CPU chip type
            if self.cpu_info.get('chip_type') == "M4_Pro_Max":
                is_m4_pro_max = True
            
            # Method 3: Memory configuration
            if self.memory_info.get('memory_configuration') == "m4_pro_max_48gb":
                is_m4_pro_max = True
            
            # Method 4: GPU variant
            if self.gpu_info.get('gpu_variant') in ["M4 Pro", "M4 Max"]:
                is_m4_pro_max = True
                # Check if we have the exact variant for even better recommendations
                m4_exact_variant = self.gpu_info.get('gpu_variant')
            else:
                m4_exact_variant = None
            
            # Method 5: Check for explicit M4 flag in system_info
            if self.system_info.get('is_m4_pro_max'):
                is_m4_pro_max = True
            
            # Enhanced info for M4 Pro/Max with 48GB RAM
            if is_m4_pro_max:
                # Determine if Pro or Max for more specific details
                is_m4_max = (m4_exact_variant == "M4 Max" or 
                            self.cpu_info.get('chip_variant') == "M4 Max" or
                            self.gpu_info.get('gpu_compute_units', 0) >= 40)
                
                # Set enhanced hardware info for M4 Pro/Max
                hardware_info["apple_silicon_model"] = "M4_Max" if is_m4_max else "M4_Pro"
                hardware_info["apple_silicon_memory"] = "48GB"
                hardware_info["apple_silicon_integrated_gpu"] = True
                hardware_info["apple_silicon_neural_engine"] = True
                hardware_info["apple_silicon_unified_memory"] = True
                
                # Add ML capabilities
                hardware_info["ml_capabilities"] = {
                    "model_size_support": "large",
                    "batch_size_recommended": 128 if is_m4_max else 64,
                    "max_sequence_length": 4096,
                    "precision": "float16",
                    "neural_engine_compatible": True,
                    "mps_optimized": True
                }
                
                # Add specific configuration details
                variant_name = "M4 Max" if is_m4_max else "M4 Pro"
                hardware_info["config_notes"] = f"High-performance {variant_name} with 48GB RAM - optimized for large models and concurrent processing"
                
                # Add RAM info specific to 48GB configuration
                hardware_info["ram_specs"] = {
                    "size_gb": 48,
                    "type": "unified",
                    "bandwidth": "400 GB/s" if is_m4_max else "300 GB/s",  # Approximate memory bandwidth
                    "suitable_for_large_models": True
                }
                
                # Add performance profile
                hardware_info["performance_profile"] = "ultra_high" if is_m4_max else "high"
                
                # Recommended device mode
                hardware_info["recommended_device"] = "mps"
            elif self.system_info.get('apple_silicon') in ["M3", "M2", "M1"]:
                # Handle other Apple Silicon models
                hardware_info["apple_silicon_model"] = f"{self.system_info.get('apple_silicon')}"
                if self.memory_info.get('apple_silicon_memory'):
                    hardware_info["apple_silicon_memory"] = self.memory_info.get('apple_silicon_memory')
                else:
                    hardware_info["apple_silicon_memory"] = f"{int(memory_gb)}GB"
                    
                # Add variant if known
                if self.cpu_info.get('chip_variant'):
                    hardware_info["apple_silicon_model"] += f" {self.cpu_info.get('chip_variant').split(' ')[-1]}"
            else:
                hardware_info["apple_silicon_model"] = "Unknown"
                hardware_info["apple_silicon_memory"] = f"{int(memory_gb)}GB"
        
        # Add detected ML performance profile if available
        if hasattr(self, 'gpu_info') and "ml_performance" in self.gpu_info:
            hardware_info["ml_performance_profile"] = self.gpu_info["ml_performance"]
        
        # Add recommended configuration
        hardware_info["recommended_config"] = self.recommend_config()
        
        return hardware_info
    
    def recommend_config(self) -> Dict[str, Any]:
        """Recommend optimal configuration based on detected hardware with enhanced Apple Silicon support"""
        recommendations = {}
        
        # Device recommendations
        if self.gpu_info.get("cuda_available", False):
            recommendations["device"] = "cuda"
        elif self.gpu_info.get("mps_available", False):
            recommendations["device"] = "mps"
        else:
            recommendations["device"] = "cpu"
        
        # Memory recommendations
        total_memory = self.memory_info.get("total_gb", 0)
        available_memory = self.memory_info.get("available_gb", 0)
        recommendations["memory"] = {}
        
        # Check for Apple Silicon with 48GB RAM (M4 Pro/Max)
        is_apple_silicon = platform.system() == "Darwin" and platform.machine() == "arm64"
        is_m4_pro_max = False
        
        # Check for M4 Pro/Max via different methods for better detection
        # Method 1: Check memory size - most reliable way
        if is_apple_silicon and abs(total_memory - 48) < 1:
            is_m4_pro_max = True
            
        # Method 2: Check CPU chip type
        if hasattr(self, 'cpu_info') and self.cpu_info.get('chip_type') == "M4_Pro_Max":
            is_m4_pro_max = True
            
        # Method 3: Check memory configuration
        if hasattr(self, 'memory_info') and self.memory_info.get('memory_configuration') == "m4_pro_max_48gb":
            is_m4_pro_max = True
        
        # Method 4: Check GPU variant
        if hasattr(self, 'gpu_info') and self.gpu_info.get('gpu_variant') in ["M4 Pro", "M4 Max"]:
            is_m4_pro_max = True
            # Check if we have the exact variant for even better recommendations
            m4_exact_variant = self.gpu_info.get('gpu_variant')
        else:
            m4_exact_variant = None
            
        # Special optimization for M4 Pro/Max with 48GB RAM
        if is_m4_pro_max:
            # For M4 Pro/Max with 48GB, we can use the largest models with optimized settings
            logger.info("Recommending enhanced configuration for M4 Pro/Max with 48GB RAM")
            
            # Check if we're dealing with M4 Max (higher performance) or M4 Pro
            is_m4_max = (m4_exact_variant == "M4 Max" or 
                         (hasattr(self, 'cpu_info') and self.cpu_info.get('chip_variant') == "M4 Max") or
                         (hasattr(self, 'gpu_info') and self.gpu_info.get('gpu_compute_units', 0) >= 40))
            
            # Common recommendations for both Pro and Max
            recommendations["memory"]["model_size"] = "large"
            recommendations["memory"]["max_sequence_length"] = 4096
            recommendations["apple_silicon_optimized"] = True
            recommendations["high_performance"] = True
            recommendations["m4_specific"] = True
            recommendations["prioritize_gpu"] = True
            recommendations["unified_memory"] = True
            recommendations["neural_engine_compatible"] = True
            
            # Set the device settings specifically for MPS
            if recommendations["device"] == "mps":
                recommendations["mps_optimized"] = True
                recommendations["metal_performance_shaders"] = True
                
                # Add ML framework recommendations
                recommendations["frameworks"] = {
                    "pytorch": {
                        "device": "mps",
                        "precision": "float16", 
                        "compile_mode": "reduce-overhead"
                    },
                    "tensorflow": {
                        "device": "metal",
                        "xla": True
                    }
                }
                
                # MBART models work better with the 48GB model
                recommendations["priority_models"] = ["mbart_translation", "mbart_large_50", "rag_generator"]
                recommendations["model_loading"] = "unified"  # Keep models loaded in memory
            
            # Different batch sizes based on M4 Pro vs M4 Max
            if is_m4_max:
                # M4 Max can handle larger batches
                recommendations["memory"]["batch_size"] = 128
                recommendations["threads"] = 16
                recommendations["performance_profile"] = "maximum"
                
                # M4 Max specific optimizations
                if hasattr(self, 'gpu_info') and "ml_performance" in self.gpu_info:
                    # Copy the ML performance recommendations directly
                    recommendations["ml_performance"] = self.gpu_info["ml_performance"]
                else:
                    # Default high-performance settings for M4 Max
                    recommendations["ml_performance"] = {
                        "batch_size": 128,
                        "max_sequence_length": 4096,
                        "performance_class": "ultra_high",
                        "suitable_models": ["large", "medium", "small"],
                        "precision": "float16"
                    }
            else:
                # M4 Pro - still high performance but with more modest batch sizes
                recommendations["memory"]["batch_size"] = 64
                recommendations["threads"] = 12
                recommendations["performance_profile"] = "high"
                
                # M4 Pro specific optimizations
                if hasattr(self, 'gpu_info') and "ml_performance" in self.gpu_info:
                    # Copy the ML performance recommendations directly
                    recommendations["ml_performance"] = self.gpu_info["ml_performance"]
                else:
                    # Default high-performance settings for M4 Pro
                    recommendations["ml_performance"] = {
                        "batch_size": 64,
                        "max_sequence_length": 4096,
                        "performance_class": "high",
                        "suitable_models": ["large", "medium", "small"],
                        "precision": "float16"
                    }
            
            # Set model capacities for different ML tasks with 48GB RAM
            recommendations["model_capacities"] = {
                "translation": {
                    "size": "large",
                    "recommended_model": "mbart-large-50-many-to-many-mmt",
                    "max_batch": 96,
                    "concurrent_requests": 12
                },
                "rag": {
                    "size": "large",
                    "context_length": 8192,
                    "embedding_dimensions": 1536,
                    "max_documents": 10000
                },
                "summarization": {
                    "size": "large",
                    "max_document_length": 16384
                },
                "language_detection": {
                    "size": "large",
                    "concurrent_requests": 32
                }
            }
            
            # Log the enhanced configuration for M4 Pro/Max
            logger.info(f"Optimized for Apple M4 {'Max' if is_m4_max else 'Pro'} with 48GB RAM")
            logger.info(f"Recommended batch size: {recommendations['memory']['batch_size']}")
            logger.info(f"Recommended sequence length: {recommendations['memory']['max_sequence_length']}")
            
        # Other Apple Silicon recommendations based on memory
        elif is_apple_silicon and total_memory >= 32:
            # High-end Apple Silicon (M3 Pro/Max or M2 Ultra)
            recommendations["memory"]["model_size"] = "large"
            recommendations["memory"]["batch_size"] = 32
            recommendations["memory"]["max_sequence_length"] = 2048
            recommendations["apple_silicon_optimized"] = True
            recommendations["high_performance"] = True
            
            if recommendations["device"] == "mps":
                recommendations["mps_optimized"] = True
                recommendations["priority_models"] = ["mbart_translation", "rag_generator"]
                
        elif is_apple_silicon and total_memory >= 16:
            # Standard Apple Silicon (M1/M2/M3)
            recommendations["memory"]["model_size"] = "medium"
            recommendations["memory"]["batch_size"] = 16
            recommendations["memory"]["max_sequence_length"] = 1024
            recommendations["apple_silicon_optimized"] = True
            
            if recommendations["device"] == "mps":
                recommendations["mps_optimized"] = True
                
        # Non-Apple Silicon or lower memory recommendations
        elif total_memory >= 24:
            recommendations["memory"]["model_size"] = "large"
            recommendations["memory"]["batch_size"] = 24
            recommendations["memory"]["max_sequence_length"] = 2048
        elif total_memory >= 16:
            recommendations["memory"]["model_size"] = "medium"
            recommendations["memory"]["batch_size"] = 16
            recommendations["memory"]["max_sequence_length"] = 1024
        elif total_memory >= 8:
            recommendations["memory"]["model_size"] = "medium"
            recommendations["memory"]["batch_size"] = 8
            recommendations["memory"]["max_sequence_length"] = 512
        else:
            recommendations["memory"]["model_size"] = "small"
            recommendations["memory"]["batch_size"] = 4
            recommendations["memory"]["max_sequence_length"] = 256
        
        # Set precision recommendation
        if recommendations["device"] == "cuda":
            recommendations["precision"] = "float16"
        elif recommendations["device"] == "mps":
            # For Apple Silicon, we can use float16
            recommendations["precision"] = "float16"
            
            # For M4 Pro/Max with 48GB, we have already set all the optimizations above
            if not is_m4_pro_max:
                # For other Apple Silicon, set basic MPS optimizations
                recommendations["mps_optimized"] = True
        else:
            # For CPU, use int8 quantization for better performance
            recommendations["precision"] = "int8"
            
            # Special case: high-memory systems on CPU can still use better precision
            if total_memory >= 64:
                recommendations["precision"] = "float16"  # Use fp16 on high-memory CPU systems
            elif total_memory >= 32:
                recommendations["precision"] = "bfloat16"  # Use bf16 on medium-high memory CPU systems
        
        return recommendations