"""
File: smartcash/ui/dataset/download/handlers/defaults.py
Deskripsi: Default configuration untuk download module dengan environment-aware settings
"""

from typing import Dict, Any
import os

# Default configuration untuk download module
DEFAULT_CONFIG = {
    # Dataset identification
    'workspace': 'smartcash-wo2us',
    'project': 'rupiah-emisi-2022', 
    'version': '3',
    
    # Paths (akan di-override dengan environment-aware paths)
    'output_dir': 'data/downloads',
    'backup_dir': 'data/backup',
    
    # Processing options
    'organize_dataset': True,
    'backup_before_download': False,
    
    # Metadata
    'module_name': 'download',
    'parent_module': 'dataset',
    'version': '1.0.0',
    'created_by': 'DownloadConfigHandler',
    'description': 'SmartCash dataset download configuration',
    
    # UI preferences
    'show_advanced': False,
    'auto_validate': True,
    'progress_tracking': True,
    
    # Security (tidak menyimpan API key ke config)
    'api_key_env_vars': ['ROBOFLOW_API_KEY', 'ROBOFLOW_KEY', 'RF_API_KEY'],
    'api_key_colab_keys': ['ROBOFLOW_API_KEY', 'roboflow_api_key', 'ROBOFLOW_KEY', 'API_KEY']
}

def get_environment_aware_config() -> Dict[str, Any]:
    """Get default config dengan environment-aware paths."""
    config = DEFAULT_CONFIG.copy()
    
    try:
        from smartcash.common.environment import get_environment_manager
        from smartcash.common.constants.paths import get_paths_for_environment
        
        env_manager = get_environment_manager()
        paths = get_paths_for_environment(env_manager.is_colab, env_manager.is_drive_mounted)
        
        # Update paths dengan environment-appropriate values
        config.update({
            'output_dir': paths.get('downloads', 'data/downloads'),
            'backup_dir': paths.get('backup', 'data/backup'),
            'environment_type': 'Google Drive' if env_manager.is_drive_mounted else 'Local Storage',
            'storage_persistent': env_manager.is_drive_mounted,
            'colab_environment': env_manager.is_colab
        })
        
    except Exception:
        # Fallback ke paths default jika error
        pass
    
    return config

def get_production_config() -> Dict[str, Any]:
    """Get production-optimized config."""
    config = get_environment_aware_config()
    config.update({
        'auto_validate': True,
        'backup_before_download': True,  # Safer untuk production
        'progress_tracking': True,
        'show_advanced': False
    })
    return config

def get_development_config() -> Dict[str, Any]:
    """Get development-optimized config."""
    config = get_environment_aware_config()
    config.update({
        'auto_validate': False,  # Manual validation untuk debugging
        'backup_before_download': False,  # Faster iteration
        'progress_tracking': True,
        'show_advanced': True
    })
    return config

def get_minimal_config() -> Dict[str, Any]:
    """Get minimal config untuk testing."""
    return {
        'workspace': 'test-workspace',
        'project': 'test-project',
        'version': '1',
        'output_dir': 'test/downloads',
        'backup_dir': 'test/backup',
        'organize_dataset': True,
        'backup_before_download': False
    }

def get_smartcash_defaults() -> Dict[str, Any]:
    """Get SmartCash-specific defaults."""
    config = get_environment_aware_config()
    config.update({
        'workspace': 'smartcash-wo2us',
        'project': 'rupiah-emisi-2022',
        'version': '3',
        'description': 'SmartCash Rupiah Detection Dataset - 2022 Emission Series'
    })
    return config

def validate_default_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate default config dan return validation result."""
    validation_result = {'valid': True, 'errors': [], 'warnings': []}
    
    # Required fields validation
    required_fields = ['workspace', 'project', 'version', 'output_dir']
    missing_fields = [field for field in required_fields if not config.get(field)]
    validation_result['errors'].extend([f"Missing required field: {field}" for field in missing_fields])
    
    # Path validation
    path_fields = ['output_dir', 'backup_dir']
    for field in path_fields:
        if config.get(field) and not isinstance(config[field], str):
            validation_result['errors'].append(f"Field {field} harus berupa string")
    
    # Boolean fields validation
    boolean_fields = ['organize_dataset', 'backup_before_download', 'auto_validate', 'progress_tracking']
    for field in boolean_fields:
        if field in config and not isinstance(config[field], bool):
            validation_result['warnings'].append(f"Field {field} should be boolean, got {type(config[field])}")
    
    validation_result['valid'] = len(validation_result['errors']) == 0
    return validation_result

def merge_with_user_config(user_config: Dict[str, Any], use_environment_aware: bool = True) -> Dict[str, Any]:
    """Merge user config dengan defaults."""
    base_config = get_environment_aware_config() if use_environment_aware else DEFAULT_CONFIG.copy()
    
    # Deep merge user config
    merged_config = base_config.copy()
    merged_config.update(user_config)
    
    # Ensure essential fields tidak di-override dengan empty values
    essential_fields = ['workspace', 'project', 'version']
    for field in essential_fields:
        if not merged_config.get(field) and base_config.get(field):
            merged_config[field] = base_config[field]
    
    return merged_config

def get_config_for_environment(env_type: str) -> Dict[str, Any]:
    """Get config berdasarkan environment type."""
    env_configs = {
        'colab_drive': get_environment_aware_config(),
        'colab_local': get_development_config(),
        'local': get_development_config(),
        'production': get_production_config(),
        'testing': get_minimal_config()
    }
    
    return env_configs.get(env_type, get_environment_aware_config())

# Export default untuk backward compatibility
CONFIG_DEFAULTS = DEFAULT_CONFIG
SMARTCASH_DEFAULTS = get_smartcash_defaults()

# One-liner utilities
get_default = lambda key, fallback=None: DEFAULT_CONFIG.get(key, fallback)
is_default_value = lambda key, value: DEFAULT_CONFIG.get(key) == value
get_field_default = lambda field: DEFAULT_CONFIG.get(field, '')
has_default_field = lambda field: field in DEFAULT_CONFIG

# Environment detection utilities
detect_api_key = lambda: next((os.environ.get(var, '') for var in DEFAULT_CONFIG['api_key_env_vars'] if os.environ.get(var)), '')
get_colab_api_key = lambda: _get_colab_userdata_key()

def _get_colab_userdata_key() -> str:
    """Get API key dari Google Colab userdata."""
    try:
        from google.colab import userdata
        return next((userdata.get(key, '') for key in DEFAULT_CONFIG['api_key_colab_keys'] 
                    if _safe_get_userdata(userdata, key)), '')
    except ImportError:
        return ''

def _safe_get_userdata(userdata, key: str) -> str:
    """Safe get userdata dengan error handling."""
    try:
        return userdata.get(key, '').strip()
    except Exception:
        return ''