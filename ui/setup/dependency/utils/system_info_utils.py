"""
File: smartcash/ui/setup/dependency/utils/system_info_utils.py
Deskripsi: System information utilities dengan consolidated approach
"""

from typing import Dict, Any
import platform
import sys

def get_comprehensive_system_info() -> Dict[str, Any]:
    """Get comprehensive system information dengan one-liner approach"""
    
    base_info = {
        'platform': platform.system(),
        'platform_release': platform.release(),
        'python_version': platform.python_version(),
        'architecture': platform.machine(),
        'processor': platform.processor()
    }
    
    # Memory info dengan safe fallback
    memory_info = _get_memory_info_safe()
    base_info.update(memory_info)
    
    # Environment detection
    environment_info = _detect_environment()
    base_info.update(environment_info)
    
    # GPU info
    gpu_info = _get_gpu_info_safe()
    base_info.update(gpu_info)
    
    return base_info

def _get_memory_info_safe() -> Dict[str, Any]:
    """Get memory info dengan safe error handling - one-liner fallback"""
    try:
        import psutil
        return {
            'memory_total_gb': round(psutil.virtual_memory().total / (1024**3), 2),
            'memory_available_gb': round(psutil.virtual_memory().available / (1024**3), 2),
            'disk_free_gb': round(psutil.disk_usage('/').free / (1024**3), 2)
        }
    except Exception:
        return {'memory_total_gb': 0, 'memory_available_gb': 0, 'disk_free_gb': 0}

def _detect_environment() -> Dict[str, Any]:
    """Detect environment (Colab, local, etc.) - one-liner detection"""
    try:
        import google.colab
        return {'environment': 'Google Colab', 'colab': True}
    except ImportError:
        return {'environment': 'Local', 'colab': False}

def _get_gpu_info_safe() -> Dict[str, Any]:
    """Get GPU info dengan safe error handling - one-liner approach"""
    try:
        import torch
        if torch.cuda.is_available():
            return {
                'cuda_available': True,
                'cuda_device_count': torch.cuda.device_count(),
                'cuda_device_name': torch.cuda.get_device_name(0)
            }
    except ImportError:
        pass
    
    return {'cuda_available': False, 'cuda_device_count': 0, 'cuda_device_name': 'N/A'}

def format_system_info_html(system_info: Dict[str, Any]) -> str:
    """Format system info sebagai HTML table - one-liner template"""
    
    info_rows = [
        ('Environment', system_info.get('environment', 'Unknown')),
        ('Platform', f"{system_info.get('platform', 'Unknown')} {system_info.get('platform_release', '')}"),
        ('Python', system_info.get('python_version', 'Unknown')),
        ('Architecture', system_info.get('architecture', 'Unknown')),
        ('Memory', f"{system_info.get('memory_available_gb', 0):.1f}GB / {system_info.get('memory_total_gb', 0):.1f}GB"),
        ('CUDA', '✅ Available' if system_info.get('cuda_available') else '❌ Not Available')
    ]
    
    table_rows = ''.join(f"<tr><td><strong>{label}:</strong></td><td>{value}</td></tr>" 
                        for label, value in info_rows)
    
    return f"""
    <div style="background:#f0f8ff;padding:12px;border-radius:6px;margin:10px 0;">
        <h4 style="margin:0 0 8px 0;color:#2c3e50;">💻 System Information</h4>
        <table style="width:100%;font-size:12px;">{table_rows}</table>
    </div>
    """

def get_system_summary_text(system_info: Dict[str, Any]) -> str:
    """Get system summary sebagai plain text - one-liner"""
    return (f"{system_info.get('environment', 'Unknown')} | "
            f"Python {system_info.get('python_version', 'Unknown')} | "
            f"{system_info.get('memory_available_gb', 0):.1f}GB RAM | "
            f"CUDA: {'Yes' if system_info.get('cuda_available') else 'No'}")

def check_system_requirements() -> Dict[str, Any]:
    """Check basic system requirements untuk SmartCash"""
    
    system_info = get_comprehensive_system_info()
    
    requirements_check = {
        'python_version_ok': _check_python_version(system_info.get('python_version', '')),
        'memory_sufficient': system_info.get('memory_total_gb', 0) >= 2.0,  # Minimum 2GB
        'disk_space_ok': system_info.get('disk_free_gb', 0) >= 1.0,  # Minimum 1GB
        'platform_supported': system_info.get('platform', '') in ['Linux', 'Windows', 'Darwin']
    }
    
    # Overall status
    requirements_check['all_requirements_met'] = all(requirements_check.values())
    
    return requirements_check

def _check_python_version(python_version: str) -> bool:
    """Check apakah Python version memenuhi minimum requirement - one-liner"""
    try:
        major, minor = map(int, python_version.split('.')[:2])
        return major >= 3 and minor >= 7  # Python 3.7+
    except Exception:
        return False

def get_environment_specific_info() -> Dict[str, Any]:
    """Get environment-specific information"""
    
    env_info = {}
    
    # Colab-specific info
    if _detect_environment().get('colab'):
        env_info.update(_get_colab_specific_info())
    
    # Local environment info
    else:
        env_info.update(_get_local_environment_info())
    
    return env_info

def _get_colab_specific_info() -> Dict[str, Any]:
    """Get Colab-specific information - one-liner"""
    import os
    return {
        'colab_pro': os.environ.get('COLAB_GPU') is not None,
        'mounted_drive': '/content/drive' in os.listdir('/content') if os.path.exists('/content') else False,
        'runtime_type': 'GPU' if os.environ.get('COLAB_GPU') else 'CPU'
    }

def _get_local_environment_info() -> Dict[str, Any]:
    """Get local environment information - one-liner"""
    import os
    return {
        'virtual_env': os.environ.get('VIRTUAL_ENV') is not None,
        'conda_env': os.environ.get('CONDA_DEFAULT_ENV') is not None,
        'user_home': os.path.expanduser('~')
    }

def format_requirements_check_html(requirements: Dict[str, Any]) -> str:
    """Format requirements check sebagai HTML - one-liner approach"""
    
    check_items = [
        ('Python Version (3.7+)', requirements.get('python_version_ok', False)),
        ('Memory (2GB+)', requirements.get('memory_sufficient', False)),
        ('Disk Space (1GB+)', requirements.get('disk_space_ok', False)),
        ('Platform Support', requirements.get('platform_supported', False))
    ]
    
    status_rows = ''.join(
        f"<tr><td>{item}</td><td>{'✅ OK' if status else '❌ Fail'}</td></tr>"
        for item, status in check_items
    )
    
    overall_status = "✅ All requirements met" if requirements.get('all_requirements_met') else "⚠️ Some requirements not met"
    bg_color = "#e8f5e9" if requirements.get('all_requirements_met') else "#fff3cd"
    
    return f"""
    <div style="background:{bg_color};padding:12px;border-radius:6px;margin:10px 0;">
        <h4 style="margin:0 0 8px 0;color:#2c3e50;">🔍 System Requirements Check</h4>
        <table style="width:100%;font-size:12px;">{status_rows}</table>
        <div style="margin-top:8px;font-weight:bold;">{overall_status}</div>
    </div>
    """