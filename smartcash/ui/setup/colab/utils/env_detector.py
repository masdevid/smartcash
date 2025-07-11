"""
File: smartcash/ui/setup/colab/utils/env_detector.py
Deskripsi: Utility untuk deteksi informasi environment

Fungsi-fungsi ini digunakan untuk mendeteksi informasi lingkungan seperti:
- Versi Python
- Informasi sistem operasi
- Ketersediaan GPU
- Informasi penyimpanan
- Tipe runtime (Colab/local)
"""

import os
import sys
import platform
import traceback
from typing import Dict, Any, Tuple, Callable, Union

def safe_get(func: Callable, default: Any = None) -> Any:
    """Safely execute a function and return default if error occurs.
    
    Args:
        func: Function to execute
        default: Default value to return if function fails
        
    Returns:
        Result of function or default value
    """
    try:
        return func()
    except Exception:
        return default

def detect_environment_info(check_drive: bool = False) -> Dict[str, Any]:
    """Detect and return comprehensive environment information."""
    # Initialize with default values
    result = {
        'python_version': 'Unknown',
        'os': {},
        'status': 'success',
        'runtime': {'type': 'unknown', 'gpu': 'unknown', 'display': 'Unknown'},
        'gpu': 'No GPU available',
        'storage_info': {},
        'is_colab': False,
        'cpu_cores': 0,
        'total_ram': 0,
        'memory_info': {},
        'drive_checked': check_drive,
        'drive_mounted': None,
        'drive_status': 'not_checked'
    }
    
    # Helper function to safely update result dictionary
    def safe_update(updates: Dict[str, Any]) -> None:
        for key, value in updates.items():
            if value is not None:
                result[key] = value
    
    try:
        # Get basic info
        safe_update({
            'python_version': safe_get(_get_python_version, 'Unknown'),
            'os': safe_get(_get_os_info, {})
        })
        
        # Get runtime information
        safe_update(safe_get(get_runtime_type, {}))
        
        # Get system information
        safe_update({
            'cpu_cores': safe_get(_get_cpu_cores, 0),
            'total_ram': safe_get(_get_total_ram, 0),
            'memory_info': safe_get(_get_memory_info, {})
        })
        
        # Get GPU information
        gpu_info = safe_get(_get_gpu_info, 'No GPU available')
        if gpu_info != 'No GPU available':
            safe_update({
                'gpu': {
                    'available': True,
                    'name': gpu_info,
                    'details': safe_get(_get_gpu_details, {})
                }
            })
        
        # Get storage information
        safe_update({'storage_info': safe_get(_get_storage_info, {})})
        
        # Get network information
        network_info = safe_get(_get_network_info, {})
        if network_info:
            safe_update({'network_info': network_info})
        
        # Get environment variables
        env_vars = safe_get(_get_environment_variables, {})
        if env_vars:
            safe_update({'environment_variables': env_vars})
        
        # Check if running in Colab and get drive status if requested
        if safe_get(_is_google_colab, False):
            result['is_colab'] = True
            
            if check_drive:
                is_drive_mounted, mount_path = safe_get(_is_drive_mounted, (False, ''))
                result['drive_mounted'] = is_drive_mounted
                if is_drive_mounted and mount_path:
                    result['drive_mount_path'] = mount_path
                    try:
                        import os
                        test_file = os.path.join(mount_path, '.smartcash_write_test')
                        with open(test_file, 'w') as f:
                            f.write('test')
                        os.remove(test_file)
                        result['drive_write_access'] = True
                    except (IOError, OSError):
                        result['drive_write_access'] = False
            else:
                result['drive_status'] = 'not_checked'
    
    except Exception as e:
        result['status'] = f'error: {str(e)}'
        print(f"Error in detect_environment_info: {str(e)}")
    
    return result

def _get_python_version() -> str:
    """🐍 Get Python version"""
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

def _get_platform_info() -> str:
    """🖥️ Get platform information"""
    return f"{platform.system()} {platform.release()} ({platform.machine()})"

def _get_os_info() -> Dict[str, str]:
    """Get detailed OS information.
    
    Returns:
        Dictionary containing OS information including:
        - system: OS name (e.g., 'Linux', 'Windows', 'Darwin')
        - release: OS release version
        - version: OS version
        - machine: Machine type (e.g., 'x86_64')
        - processor: Processor information
    """
    return {
        'system': platform.system(),
        'release': platform.release(),
        'version': platform.version(),
        'machine': platform.machine(),
        'processor': platform.processor() or 'Unknown',
        'platform': platform.platform(),
        'node': platform.node()
    }

def _is_google_colab() -> bool:
    """🔍 Check if running in Google Colab"""
    try:
        import google.colab
        return True
    except ImportError:
        return False

def _get_gpu_info() -> str:
    """🎮 Get GPU information"""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
        return 'No GPU available'
    except ImportError:
        return 'PyTorch not available'

def _is_drive_mounted() -> Tuple[bool, str]:
    """💾 Check if Google Drive is mounted and return the mount path if available
    
    Returns:
        tuple: (is_mounted: bool, mount_path: str)
    """
    try:
        # Check if running in Colab
        if not _is_google_colab():
            return False, ''
            
        from google.colab import drive
        import os
        
        # Default mount path in Colab
        mount_path = '/content/drive'
        
        # Check if already mounted
        if os.path.ismount(mount_path):
            return True, mount_path
            
        # Try to mount if not already mounted
        try:
            drive.mount(mount_path)
            return True, mount_path
        except Exception as e:
            print(f"Failed to mount Google Drive: {e}")
            return False, ''
            
    except Exception as e:
        print(f"Error checking Google Drive mount: {e}")
        return False, ''

def _get_cpu_cores() -> int:
    """🖥️ Get number of CPU cores"""
    import multiprocessing
    return multiprocessing.cpu_count()

def _get_total_ram() -> int:
    """💾 Get total RAM in bytes"""
    try:
        import psutil
        return psutil.virtual_memory().total
    except ImportError:
        return 0

def _get_memory_info() -> Dict[str, Any]:
    """💾 Get detailed memory information"""
    try:
        import psutil
        memory = psutil.virtual_memory()
        return {
            'total': memory.total,
            'available': memory.available,
            'percent_used': memory.percent,
            'used': memory.used,
            'free': memory.free,
            'total_gb': round(memory.total / (1024**3), 2),
            'available_gb': round(memory.available / (1024**3), 2),
            'used_gb': round(memory.used / (1024**3), 2)
        }
    except ImportError:
        return {}

def _get_storage_info() -> Dict[str, Any]:
    """💽 Get detailed storage information"""
    try:
        import psutil
        disk = psutil.disk_usage('/')
        return {
            'total': disk.total,
            'used': disk.used,
            'free': disk.free,
            'percent_used': round((disk.used / disk.total) * 100, 2),
            'total_gb': round(disk.total / (1024**3), 2),
            'used_gb': round(disk.used / (1024**3), 2),
            'free_gb': round(disk.free / (1024**3), 2)
        }
    except ImportError:
        return {}

def _get_gpu_details() -> Dict[str, Any]:
    """🎮 Get detailed GPU information"""
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            devices = []
            for i in range(device_count):
                props = torch.cuda.get_device_properties(i)
                devices.append({
                    'id': i,
                    'name': torch.cuda.get_device_name(i),
                    'memory_total': props.total_memory,
                    'memory_total_gb': round(props.total_memory / (1024**3), 2),
                    'major': props.major,
                    'minor': props.minor,
                    'multi_processor_count': props.multi_processor_count
                })
            
            return {
                'available': True,
                'device_count': device_count,
                'current_device': torch.cuda.current_device(),
                'devices': devices,
                'cuda_version': torch.version.cuda
            }
        else:
            return {
                'available': False,
                'reason': 'CUDA not available',
                'device_count': 0,
                'devices': []
            }
    except ImportError:
        return {
            'available': False,
            'reason': 'PyTorch not installed',
            'device_count': 0,
            'devices': []
        }

def _get_network_info() -> Dict[str, Any]:
    """🌐 Get network information"""
    try:
        import psutil
        import socket
        
        # Get hostname
        hostname = socket.gethostname()
        
        # Get network interfaces
        interfaces = []
        for interface, addrs in psutil.net_if_addrs().items():
            for addr in addrs:
                if addr.family == socket.AF_INET:  # IPv4
                    interfaces.append({
                        'interface': interface,
                        'ip': addr.address,
                        'netmask': addr.netmask,
                        'broadcast': addr.broadcast
                    })
        
        return {
            'hostname': hostname,
            'interfaces': interfaces
        }
    except ImportError:
        return {}

def _get_environment_variables() -> Dict[str, str]:
    """🌍 Get relevant environment variables"""
    import os
    
    relevant_vars = [
        'PATH', 'PYTHONPATH', 'HOME', 'USER', 'SHELL',
        'LANG', 'LC_ALL', 'TZ', 'DISPLAY',
        'SMARTCASH_ROOT', 'SMARTCASH_ENV', 'SMARTCASH_DATA_ROOT',
        'CUDA_VISIBLE_DEVICES', 'NVIDIA_VISIBLE_DEVICES'
    ]
    
    env_vars = {}
    for var in relevant_vars:
        value = os.environ.get(var)
        if value is not None:
            env_vars[var] = value
    
    return env_vars

def get_runtime_type() -> Dict[str, str]:
    """Get detailed runtime type information.
    
    Returns:
        Dictionary containing:
        - type: Runtime type (e.g., 'colab', 'local')
        - gpu: GPU availability ('available' or 'not available')
        - display: Formatted display string
    """
    try:
        is_colab = _is_google_colab()
        has_gpu = _get_gpu_info() != 'No GPU available'
        
        runtime_type = 'colab' if is_colab else 'local'
        gpu_status = 'available' if has_gpu else 'not available'
        
        return {
            'type': runtime_type,
            'gpu': gpu_status,
            'display': f"{runtime_type.capitalize()} ({gpu_status} GPU)"
        }
    except Exception as e:
        return {
            'type': 'unknown',
            'gpu': 'unknown',
            'display': f'Unknown ({str(e)})'
        }
