"""
file_path: smartcash/ui/setup/colab/components/env_info_panel.py

Environment Information Panel for Colab.

Copied from env_config version but imports updated to reference local utils.
"""

import ipywidgets as widgets
from typing import Dict, Any
from smartcash.ui.setup.colab.utils.env_detector import detect_environment_info

def create_env_info_panel(env_info: Dict[str, Any] = None) -> widgets.HTML:
    """Create an environment information panel.
    
    Args:
        env_info: Optional pre-fetched environment info. If not provided,
                 it will be detected automatically.
                 
    Returns:
        Environment info panel HTML widget
    """
    if env_info is None:
        env_info = detect_environment_info()
    
    return widgets.HTML(
        value=_format_env_info_content(env_info),
        layout=widgets.Layout(
            width='100%',
            padding='15px',
            border='1px solid #e0e0e0',
            border_radius='6px',
            margin='10px 0',
            background='#f9f9f9'
        )
    )

def _format_env_info_content(env_info: Dict[str, Any]) -> str:
    """Format environment information as HTML.
    
    Args:
        env_info: Dictionary containing environment information
        
    Returns:
        Formatted HTML string
    """
    # Get runtime information
    runtime_info = env_info.get('runtime', {})
    runtime_display = runtime_info.get('display', 'Unknown Environment')
    
    # Get system information
    python_version = env_info.get('python_version', 'N/A')
    os_info = env_info.get('os', {})
    os_name = os_info.get('system', 'N/A')
    os_release = os_info.get('release', 'N/A')
    
    # Get enhanced memory information
    memory_info = env_info.get('memory_info', {})
    total_ram = memory_info.get('total_gb', env_info.get('total_ram', 0) / (1024**3) if env_info.get('total_ram') else 0)
    available_ram = memory_info.get('available_gb', 0)
    ram_usage = memory_info.get('percent_used', 0)
    
    # Get resource information
    cpu_cores = env_info.get('cpu_cores', 'N/A')
    
    # Get storage information
    storage_info = env_info.get('storage_info', {})
    storage_status = _format_enhanced_storage_info(storage_info)
    
    # Get enhanced GPU information
    gpu_info = _format_enhanced_gpu_info(env_info)
    
    # Get drive information with write access
    drive_status = _get_enhanced_drive_status(env_info)
    
    # Get network information
    network_info = env_info.get('network_info', {})
    hostname = network_info.get('hostname', 'N/A')
    
    # Get environment variables summary
    env_vars = env_info.get('environment_variables', {})
    smartcash_status = _get_smartcash_env_status(env_vars)
    
    return f"""
    <div style="font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6;">
        <div style="margin-bottom: 15px; padding-bottom: 10px; border-bottom: 1px solid #e0e0e0;">
            <h3 style="margin: 0; color: #333;">🌐 {runtime_display}</h3>
        </div>
        
        <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; margin-top: 15px;">
            <div>
                <h4 style="color: #333; margin: 0 0 10px 0; padding-bottom: 5px; border-bottom: 2px solid #2196f3;">
                    🖥️ System Information
                </h4>
                <p><strong>OS:</strong> {os_name} {os_release}</p>
                <p><strong>Python:</strong> {python_version}</p>
                <p><strong>Environment:</strong> {_get_env_type(env_info)}</p>
                <p><strong>Hostname:</strong> {hostname}</p>
                <p><strong>SmartCash:</strong> {smartcash_status}</p>
            </div>
            
            <div>
                <h4 style="color: #333; margin: 0 0 10px 0; padding-bottom: 5px; border-bottom: 2px solid #4caf50;">
                    ⚡ Resources
                </h4>
                <p><strong>CPU Cores:</strong> {cpu_cores}</p>
                <p><strong>Total RAM:</strong> {total_ram:.1f}GB</p>
                <p><strong>Available RAM:</strong> {available_ram:.1f}GB ({100-ram_usage:.1f}% free)</p>
                <p><strong>Storage:</strong> {storage_status}</p>
            </div>
            
            <div>
                <h4 style="color: #333; margin: 0 0 10px 0; padding-bottom: 5px; border-bottom: 2px solid #ff9800;">
                    🎮 Hardware & Drive
                </h4>
                <p><strong>GPU:</strong> {gpu_info}</p>
                <p><strong>Drive:</strong> {drive_status}</p>
                {_format_additional_info(env_info)}
            </div>
        </div>
    </div>
    """

def _format_gpu_info(env_info: Dict[str, Any]) -> str:
    """Format GPU information for display.
    
    Args:
        env_info: Dictionary containing GPU information
        
    Returns:
        Formatted GPU information string
    """
    if not env_info.get('gpu_available', False):
        return '❌ Not available'
        
    gpu_info = env_info.get('gpu', {})
    if isinstance(gpu_info, str):
        return f'✅ {gpu_info}'
    
    gpu_name = gpu_info.get('device_name', 'GPU')
    gpu_memory = _format_bytes(gpu_info.get('total_memory')) if 'total_memory' in gpu_info else ''
    
    return f'✅ {gpu_name} {gpu_memory}'.strip()

def _format_storage_info(storage_info: Any) -> str:
    """Format storage information for display.
    
    Args:
        storage_info: Dictionary containing storage information or error string
        
    Returns:
        Formatted storage information string
    """
    try:
        # Handle case where storage_info is a string (error message)
        if isinstance(storage_info, str):
            return f"Error: {storage_info}"
            
        # Handle case where storage_info is None or empty
        if not storage_info:
            return 'N/A'
            
        # Ensure we have numeric values
        total = float(storage_info.get('total', 0)) if storage_info.get('total') is not None else 0
        used = float(storage_info.get('used', 0)) if storage_info.get('used') is not None else 0
        
        # Calculate percentage if we have valid total
        if total > 0:
            used_percent = (used / total) * 100
            return f"{_format_bytes(used)} / {_format_bytes(total)} ({used_percent:.1f}% used)"
        
        # Fallback to just showing used space if total is 0 or not available
        return f"{_format_bytes(used)} used"
        
    except Exception as e:
        # Log the error and return a user-friendly message
        import logging
        logging.error(f"Error formatting storage info: {e}")
        return "Storage info unavailable"

def _get_drive_status(env_info: Dict[str, Any]) -> str:
    """Get formatted drive status.
    
    Args:
        env_info: Dictionary containing drive information
        
    Returns:
        Formatted drive status string
    """
    if env_info.get('drive_mounted'):
        mount_path = env_info.get('drive_mount_path', '')
        return f"✅ Mounted at {mount_path}" if mount_path else "✅ Mounted"
    return "❌ Not mounted"

def _get_env_type(env_info: Dict[str, Any]) -> str:
    """Determine environment type.
    
    Args:
        env_info: Dictionary containing environment information
        
    Returns:
        Environment type as string
    """
    if env_info.get('is_colab'):
        return "Google Colab"
    return "Local"

def _format_enhanced_storage_info(storage_info: Dict[str, Any]) -> str:
    """Format enhanced storage information for display.
    
    Args:
        storage_info: Dictionary containing enhanced storage information
        
    Returns:
        Formatted storage information string
    """
    if not storage_info:
        return 'N/A'
    
    try:
        total_gb = storage_info.get('total_gb', 0)
        used_gb = storage_info.get('used_gb', 0)
        free_gb = storage_info.get('free_gb', 0)
        percent_used = storage_info.get('percent_used', 0)
        
        if total_gb > 0:
            return f"{used_gb:.1f}GB / {total_gb:.1f}GB ({percent_used:.1f}% used, {free_gb:.1f}GB free)"
        
        return f"{used_gb:.1f}GB used"
        
    except Exception:
        return "Storage info unavailable"

def _format_enhanced_gpu_info(env_info: Dict[str, Any]) -> str:
    """Format enhanced GPU information for display.
    
    Args:
        env_info: Dictionary containing enhanced GPU information
        
    Returns:
        Formatted GPU information string
    """
    gpu_details = env_info.get('gpu_details', {})
    
    if not gpu_details.get('available', False):
        reason = gpu_details.get('reason', 'Not available')
        return f'❌ {reason}'
    
    devices = gpu_details.get('devices', [])
    if not devices:
        return '❌ No GPU devices found'
    
    # Show primary GPU
    primary_gpu = devices[0]
    gpu_name = primary_gpu.get('name', 'Unknown GPU')
    gpu_memory = primary_gpu.get('memory_total_gb', 0)
    device_count = gpu_details.get('device_count', 1)
    
    gpu_info = f'✅ {gpu_name}'
    if gpu_memory > 0:
        gpu_info += f' ({gpu_memory:.1f}GB)'
    if device_count > 1:
        gpu_info += f' +{device_count-1} more'
    
    return gpu_info

def _get_enhanced_drive_status(env_info: Dict[str, Any]) -> str:
    """Get enhanced drive status with write access information.
    
    Args:
        env_info: Dictionary containing drive information
        
    Returns:
        Enhanced drive status string
    """
    if not env_info.get('drive_mounted', False):
        return "❌ Not mounted"
    
    mount_path = env_info.get('drive_mount_path', '/content/drive')
    
    # Check if we have write access info from verification
    # This would come from verification operations
    return f"✅ Mounted at {mount_path}"

def _get_smartcash_env_status(env_vars: Dict[str, str]) -> str:
    """Get SmartCash environment status.
    
    Args:
        env_vars: Dictionary of environment variables
        
    Returns:
        SmartCash environment status string
    """
    required_vars = ['SMARTCASH_ROOT', 'SMARTCASH_ENV', 'SMARTCASH_DATA_ROOT']
    set_vars = [var for var in required_vars if var in env_vars]
    
    if len(set_vars) == len(required_vars):
        env_type = env_vars.get('SMARTCASH_ENV', 'unknown')
        return f"✅ Configured ({env_type})"
    elif len(set_vars) > 0:
        return f"⚠️ Partial ({len(set_vars)}/{len(required_vars)} vars)"
    else:
        return "❌ Not configured"

def _format_additional_info(env_info: Dict[str, Any]) -> str:
    """Format additional information section.
    
    Args:
        env_info: Dictionary containing environment information
        
    Returns:
        Additional information HTML
    """
    additional_info = []
    
    # Network interfaces count
    network_info = env_info.get('network_info', {})
    interfaces = network_info.get('interfaces', [])
    if interfaces:
        additional_info.append(f"<p><strong>Network:</strong> {len(interfaces)} interface(s)</p>")
    
    # CUDA version if available
    gpu_details = env_info.get('gpu_details', {})
    if gpu_details.get('available') and 'cuda_version' in gpu_details:
        cuda_version = gpu_details['cuda_version']
        additional_info.append(f"<p><strong>CUDA:</strong> {cuda_version}</p>")
    
    return '\n'.join(additional_info)

def _format_bytes(bytes_value: int) -> str:
    """Format bytes into a human-readable string.
    
    Args:
        bytes_value: Size in bytes
        
    Returns:
        Formatted string with appropriate unit (B, KB, MB, GB, TB)
    """
    if not isinstance(bytes_value, (int, float)) or bytes_value < 0:
        return 'N/A'
        
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            if unit == 'B':
                return f"{int(bytes_value)} {unit}"
            return f"{bytes_value:.1f} {unit}"
        bytes_value /= 1024.0
    
    return f"{bytes_value:.1f} PB"
