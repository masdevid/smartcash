"""
File: smartcash/ui/dataset/downloader/handlers/defaults.py
Deskripsi: Default configuration values untuk downloader module
"""

from smartcash.common.environment import get_environment_manager

def get_default_api_key() -> str:
    """Detect API key dari environment sources."""
    import os
    
    # Environment variables
    for env_key in ['ROBOFLOW_API_KEY', 'ROBOFLOW_KEY', 'RF_API_KEY']:
        if api_key := os.environ.get(env_key, '').strip():
            return api_key if len(api_key) > 10 else ''
    
    # Google Colab userdata
    try:
        from google.colab import userdata
        for key_name in ['ROBOFLOW_API_KEY', 'roboflow_api_key', 'API_KEY']:
            try:
                if api_key := userdata.get(key_name, '').strip():
                    return api_key if len(api_key) > 10 else ''
            except Exception:
                continue
    except ImportError:
        pass
    
    return ''

def get_default_paths() -> dict:
    """Get default paths berdasarkan environment."""
    env_manager = get_environment_manager()
    
    if env_manager.is_colab and env_manager.is_drive_mounted:
        base_path = str(env_manager.drive_path / 'dataset')
    else:
        base_path = 'data'
    
    return {
        'output_dir': f"{base_path}/downloads",
        'data_dir': f"{base_path}/processed",
        'backup_dir': f"{base_path}/backup"
    }

# Default configuration dengan inheritance support
DEFAULT_CONFIG = {
    '_base_': ['base_config'],  # Inherit dari base_config.yaml
    
    # Dataset identification
    'workspace': 'smartcash-wo2us',
    'project': 'rupiah-emisi-2022',
    'version': '3',
    'api_key': get_default_api_key(),
    
    # Download options
    'output_format': 'yolov5pytorch',
    'validate_download': True,
    'organize_dataset': True,
    'backup_existing': False,
    
    # Progress options
    'progress_enabled': True,
    'show_detailed_progress': False,
    'log_level': 'INFO',
    
    # Performance options
    'retry_attempts': 3,
    'timeout_seconds': 30,
    'chunk_size_kb': 8,
    'parallel_downloads': 1,
    
    # Paths (dynamic berdasarkan environment)
    **get_default_paths(),
    
    # Metadata
    'module_name': 'downloader',
    'version': '1.0.0',
    'created_by': 'SmartCash Dataset Downloader',
    'description': 'Configuration untuk dataset downloader dengan Roboflow integration'
}

# Validation rules untuk config values
VALIDATION_RULES = {
    'retry_attempts': {'min': 1, 'max': 10, 'default': 3},
    'timeout_seconds': {'min': 10, 'max': 300, 'default': 30},
    'chunk_size_kb': {'min': 1, 'max': 64, 'default': 8},
    'parallel_downloads': {'min': 1, 'max': 4, 'default': 1}
}

REQUIRED_FIELDS = ['workspace', 'project', 'version']

OPTIONAL_FIELDS_WITH_DEFAULTS = {
    'api_key': '',
    'output_format': 'yolov5pytorch',
    'validate_download': True,
    'organize_dataset': True
}