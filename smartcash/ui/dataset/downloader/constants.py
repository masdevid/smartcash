"""
Downloader module-specific constants.

This module contains constants used throughout the downloader module.
"""

# Button configuration
DOWNLOAD_BUTTON_ID = 'download_button'
CHECK_BUTTON_ID = 'check_button'
CLEANUP_BUTTON_ID = 'cleanup_button'

# Operation types
OPERATION_DOWNLOAD = 'download'
OPERATION_CHECK = 'check'
OPERATION_CLEANUP = 'cleanup'

# UI component keys
UI_COMPONENT_DOWNLOAD_BUTTON = 'download_button'
UI_COMPONENT_CHECK_BUTTON = 'check_button'
UI_COMPONENT_CLEANUP_BUTTON = 'cleanup_button'
UI_COMPONENT_ACTION_CONTAINER = 'action_container'
UI_COMPONENT_OPERATION_MANAGER = 'operation_manager'
UI_COMPONENT_OPERATION_SUMMARY = 'operation_summary'

# Default configuration keys
CONFIG_DATA_DIR = 'data_dir'
CONFIG_TARGET_DIR = 'target_dir'
CONFIG_ROBOFLOW_WORKSPACE = 'workspace'
CONFIG_ROBOFLOW_PROJECT = 'project'
CONFIG_ROBOFLOW_VERSION = 'version'

# File and directory defaults
DEFAULT_DATA_DIR = 'data'
DEFAULT_TARGET_DIR = 'data'

# Roboflow API
ROBOFLOW_API_KEY_ENV = 'ROBOFLOW_API_KEY'
ROBOFLOW_FORMAT_YOLOV5 = 'yolov5'

# Module metadata
MODULE_NAME = 'downloader'
MODULE_GROUP = 'dataset'
MODULE_TITLE = '📥 Dataset Downloader'

# UI Configuration
UI_CONFIG = {
    'title': MODULE_TITLE,
    'subtitle': "Download dataset Roboflow untuk SmartCash dengan UUID renaming dan validasi otomatis",
    'icon': "📥",
    'module_name': MODULE_NAME,
    'parent_module': MODULE_GROUP,
    'version': "1.0.0"
}

# Button configuration
BUTTON_CONFIG = {
    'download': {
        'text': '📥 Download',
        'style': 'primary',
        'tooltip': 'Download dataset from Roboflow',
        'order': 1
    },
    'check': {
        'text': '🔍 Check',
        'style': 'info',
        'tooltip': 'Check dataset status and integrity',
        'order': 2
    },
    'cleanup': {
        'text': '🗑️ Cleanup',
        'style': 'danger',
        'tooltip': 'Remove dataset files from local storage',
        'order': 3
    }
}

# Validation rules for form fields
VALIDATION_RULES = {
    'workspace': {'required': True, 'min_length': 1},
    'project': {'required': True, 'min_length': 1},
    'version': {'required': True, 'min_length': 1},
    'api_key': {'required': True, 'min_length': 10}
}