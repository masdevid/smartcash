"""
File: smartcash/ui/dataset/downloader/configs/downloader_config_constants.py
Description: Centralized configuration constants and mappings for downloader module.
"""

from typing import Dict, Any, List, Tuple

# Default configuration values
DEFAULT_ROBOFLOW_CONFIG = {
    'workspace': 'smartcash-wo2us',
    'project': 'rupiah-emisi-2022',
    'version': '3',
    'api_key': '',
    'output_format': 'yolov5pytorch'
}

DEFAULT_DOWNLOAD_CONFIG = {
    'enabled': True,
    'target_dir': 'data',
    'temp_dir': 'data/downloads',
    'backup_existing': False,
    'validate_download': True,
    'organize_dataset': True,
    'rename_files': True,
    'retry_count': 3,
    'timeout': 30,
    'chunk_size': 65536,  # 64KB for optimal performance (standardized)
    'parallel_downloads': True,
    'max_workers': 4
}

DEFAULT_UUID_CONFIG = {
    'enabled': True,
    'backup_before_rename': False,
    'batch_size': 1000,
    'parallel_workers': 4,
    'validate_consistency': True,
    'target_splits': ['train', 'valid', 'test'],
    'file_patterns': ['.jpg', '.jpeg', '.png', '.bmp'],
    'label_patterns': ['.txt'],
    'progress_reporting': True
}

DEFAULT_VALIDATION_CONFIG = {
    'enabled': True,
    'check_file_integrity': True,
    'verify_image_format': True,
    'validate_labels': True,
    'check_dataset_structure': True,
    'minimum_images_per_split': {
        'train': 100,
        'valid': 50,
        'test': 25
    },
    'allowed_extensions': ['.jpg', '.jpeg', '.png', '.bmp'],
    'max_image_size_mb': 50,
    'generate_report': True,
    'parallel_workers': 4
}

DEFAULT_CLEANUP_CONFIG = {
    'auto_cleanup_downloads': False,
    'preserve_original_structure': True,
    'backup_dir': 'data/backup/downloads',
    'temp_cleanup_patterns': [
        '*.tmp',
        '*.temp',
        '*_download_*',
        '*.zip'
    ],
    'keep_download_logs': True,
    'cleanup_on_error': True,
    'parallel_workers': 4
}

DEFAULT_FILE_NAMING_CONFIG = {
    'uuid_format': True,
    'naming_strategy': 'research_uuid',
    'preserve_original': False
}

# UI Component to Config mapping
UI_FIELD_MAPPINGS: List[Tuple[str, str, str, Any]] = [
    # Format: (ui_component_key, config_path, config_key, default_value)
    
    # Roboflow dataset settings
    ('workspace_input', 'data.roboflow', 'workspace', DEFAULT_ROBOFLOW_CONFIG['workspace']),
    ('project_input', 'data.roboflow', 'project', DEFAULT_ROBOFLOW_CONFIG['project']),
    ('version_input', 'data.roboflow', 'version', DEFAULT_ROBOFLOW_CONFIG['version']),
    ('api_key_input', 'data.roboflow', 'api_key', DEFAULT_ROBOFLOW_CONFIG['api_key']),
    
    # Download options
    ('validate_checkbox', 'download', 'validate_download', DEFAULT_DOWNLOAD_CONFIG['validate_download']),
    ('backup_checkbox', 'download', 'backup_existing', DEFAULT_DOWNLOAD_CONFIG['backup_existing']),
    ('target_dir', 'download', 'target_dir', DEFAULT_DOWNLOAD_CONFIG['target_dir']),
    ('temp_dir', 'download', 'temp_dir', DEFAULT_DOWNLOAD_CONFIG['temp_dir']),
    ('organize_dataset', 'download', 'organize_dataset', DEFAULT_DOWNLOAD_CONFIG['organize_dataset']),
    ('rename_files', 'download', 'rename_files', DEFAULT_DOWNLOAD_CONFIG['rename_files']),
    ('retry_count', 'download', 'retry_count', DEFAULT_DOWNLOAD_CONFIG['retry_count']),
    ('timeout', 'download', 'timeout', DEFAULT_DOWNLOAD_CONFIG['timeout']),
    ('chunk_size', 'download', 'chunk_size', DEFAULT_DOWNLOAD_CONFIG['chunk_size']),
    ('parallel_downloads', 'download', 'parallel_downloads', DEFAULT_DOWNLOAD_CONFIG['parallel_downloads']),
    ('max_workers', 'download', 'max_workers', DEFAULT_DOWNLOAD_CONFIG['max_workers']),
    
    # UUID renaming settings
    ('uuid_enabled', 'uuid_renaming', 'enabled', DEFAULT_UUID_CONFIG['enabled']),
    ('uuid_backup_before_rename', 'uuid_renaming', 'backup_before_rename', DEFAULT_UUID_CONFIG['backup_before_rename']),
    ('uuid_batch_size', 'uuid_renaming', 'batch_size', DEFAULT_UUID_CONFIG['batch_size']),
    ('uuid_parallel_workers', 'uuid_renaming', 'parallel_workers', DEFAULT_UUID_CONFIG['parallel_workers']),
    ('uuid_validate_consistency', 'uuid_renaming', 'validate_consistency', DEFAULT_UUID_CONFIG['validate_consistency']),
    ('uuid_progress_reporting', 'uuid_renaming', 'progress_reporting', DEFAULT_UUID_CONFIG['progress_reporting']),
    
    # File naming settings
    ('uuid_format', 'data.file_naming', 'uuid_format', DEFAULT_FILE_NAMING_CONFIG['uuid_format']),
    ('naming_strategy', 'data.file_naming', 'naming_strategy', DEFAULT_FILE_NAMING_CONFIG['naming_strategy']),
    ('preserve_original', 'data.file_naming', 'preserve_original', DEFAULT_FILE_NAMING_CONFIG['preserve_original']),
    
    # Validation settings
    ('validation_enabled', 'validation', 'enabled', DEFAULT_VALIDATION_CONFIG['enabled']),
    ('check_file_integrity', 'validation', 'check_file_integrity', DEFAULT_VALIDATION_CONFIG['check_file_integrity']),
    ('verify_image_format', 'validation', 'verify_image_format', DEFAULT_VALIDATION_CONFIG['verify_image_format']),
    ('validate_labels', 'validation', 'validate_labels', DEFAULT_VALIDATION_CONFIG['validate_labels']),
    ('check_dataset_structure', 'validation', 'check_dataset_structure', DEFAULT_VALIDATION_CONFIG['check_dataset_structure']),
    ('max_image_size_mb', 'validation', 'max_image_size_mb', DEFAULT_VALIDATION_CONFIG['max_image_size_mb']),
    ('generate_report', 'validation', 'generate_report', DEFAULT_VALIDATION_CONFIG['generate_report']),
    
    # Cleanup settings
    ('auto_cleanup_downloads', 'cleanup', 'auto_cleanup_downloads', DEFAULT_CLEANUP_CONFIG['auto_cleanup_downloads']),
    ('preserve_original_structure', 'cleanup', 'preserve_original_structure', DEFAULT_CLEANUP_CONFIG['preserve_original_structure']),
    ('backup_dir', 'cleanup', 'backup_dir', DEFAULT_CLEANUP_CONFIG['backup_dir']),
    ('keep_download_logs', 'cleanup', 'keep_download_logs', DEFAULT_CLEANUP_CONFIG['keep_download_logs']),
    ('cleanup_on_error', 'cleanup', 'cleanup_on_error', DEFAULT_CLEANUP_CONFIG['cleanup_on_error'])
]

# Required fields for validation
REQUIRED_FIELDS = {
    'data.roboflow.workspace': 'Workspace name is required',
    'data.roboflow.project': 'Project name is required',
    'data.roboflow.version': 'Version is required'
}

# Optional fields with warnings
OPTIONAL_FIELDS_WITH_WARNINGS = {
    'data.roboflow.api_key': 'API key is empty - will try to use Colab secrets'
}

def get_nested_value(config: Dict[str, Any], path: str, default: Any = None) -> Any:
    """
    Get nested value from config using dot notation.
    
    Args:
        config: Configuration dictionary
        path: Dot notation path (e.g., 'data.roboflow.workspace')
        default: Default value if path not found
        
    Returns:
        Value at path or default
    """
    keys = path.split('.')
    current = config
    
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    
    return current

def set_nested_value(config: Dict[str, Any], path: str, value: Any) -> None:
    """
    Set nested value in config using dot notation.
    
    Args:
        config: Configuration dictionary
        path: Dot notation path (e.g., 'data.roboflow.workspace')
        value: Value to set
    """
    keys = path.split('.')
    current = config
    
    # Navigate to parent
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    
    # Set final value
    current[keys[-1]] = value

def get_default_config_structure() -> Dict[str, Any]:
    """
    Get the complete default configuration structure.
    
    Returns:
        Default configuration dictionary
    """
    return {
        'config_version': '1.0',
        'module_name': 'downloader',
        'version': '1.0',
        'created_by': 'smartcash',
        'description': 'Dataset downloader configuration',
        
        'data': {
            'source': 'roboflow',
            'dir': 'data',
            'roboflow': DEFAULT_ROBOFLOW_CONFIG.copy(),
            'file_naming': DEFAULT_FILE_NAMING_CONFIG.copy(),
            'local': {
                'train': 'data/train',
                'valid': 'data/valid',
                'test': 'data/test'
            }
        },
        
        'download': DEFAULT_DOWNLOAD_CONFIG.copy(),
        'uuid_renaming': DEFAULT_UUID_CONFIG.copy(),
        'validation': DEFAULT_VALIDATION_CONFIG.copy(),
        'cleanup': DEFAULT_CLEANUP_CONFIG.copy(),
        
        'ui_settings': {
            'auto_validate': True,
            'show_advanced': False,
            'theme': 'light'
        },
        
        'history': {
            'created_at': None,
            'updated_at': None,
            'version': '1.0'
        }
    }