"""
File: smartcash/ui/dataset/downloader/handlers/defaults.py
Deskripsi: Simplified defaults dengan fallback handling
"""

from typing import Dict, Any
from smartcash.ui.utils.fallback_utils import try_operation_safe

# Simple default configuration
DEFAULT_CONFIG = {
    'roboflow': {
        'workspace': 'smartcash-wo2us',
        'project': 'rupiah-emisi-2022',
        'version': '3',
        'api_key': '',
        'format': 'yolov5pytorch'
    },
    'local': {
        'output_dir': '/content/data',
        'backup_dir': '/content/data/backup',
        'organize_dataset': True,
        'backup_enabled': False
    },
    'advanced': {
        'retry_attempts': 3,
        'timeout_seconds': 30,
        'chunk_size_kb': 8
    }
}

def get_default_api_key() -> str:
    """Get API key dari environment dengan fallback."""
    import os
    
    # Check environment variables
    api_key = os.environ.get('ROBOFLOW_API_KEY', '')
    if api_key:
        return api_key
    
    # Try Colab userdata
    try:
        from google.colab import userdata
        return userdata.get('ROBOFLOW_API_KEY', '')
    except Exception:
        return ''

def get_default_downloader_config() -> Dict[str, Any]:
    """Get default configuration dengan environment detection."""
    config = DEFAULT_CONFIG.copy()
    
    # Try to get environment-specific paths
    try:
        from smartcash.common.environment import get_environment_manager
        env_manager = get_environment_manager()
        
        if env_manager.is_colab and env_manager.is_drive_mounted:
            config['local']['output_dir'] = '/content/drive/MyDrive/SmartCash/data'
            config['local']['backup_dir'] = '/content/drive/MyDrive/SmartCash/data/backup'
        elif env_manager.is_colab:
            config['local']['output_dir'] = '/content/data'
            config['local']['backup_dir'] = '/content/data/backup'
        else:
            config['local']['output_dir'] = 'data'
            config['local']['backup_dir'] = 'data/backup'
            
    except Exception:
        # Fallback to simple paths
        config['local']['output_dir'] = '/content/data'
        config['local']['backup_dir'] = '/content/data/backup'
    
    # Try to get API key
    api_key = try_operation_safe(get_default_api_key, fallback_value='')
    if api_key:
        config['roboflow']['api_key'] = api_key
    
    return config

def get_simple_config() -> Dict[str, Any]:
    """Get simple config tanpa environment detection."""
    return DEFAULT_CONFIG.copy()