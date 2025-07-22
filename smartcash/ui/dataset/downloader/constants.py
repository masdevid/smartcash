"""
Downloader module-specific constants.

This module contains constants used throughout the downloader module.
"""
from enum import Enum, auto
from typing import Dict, Any, TypedDict, Literal

# Type definitions
class ButtonConfig(TypedDict):
    """Type definition for button configuration."""
    text: str
    style: str
    tooltip: str
    order: int

class ValidationRule(TypedDict, total=False):
    """Type definition for validation rules."""
    required: bool
    min_length: int
    max_length: int
    pattern: str

# Type aliases
ButtonStyle = Literal['primary', 'secondary', 'success', 'danger', 'warning', 'info', 'light', 'dark']

class OperationType(Enum):
    """Enumeration of operation types in the downloader module."""
    DOWNLOAD = 'download'
    CHECK = 'check'
    CLEANUP = 'cleanup'

class UIComponent(Enum):
    """Enumeration of UI component identifiers."""
    ACTION_CONTAINER = 'action_container'
    OPERATION_MANAGER = 'operation_manager'
    OPERATION_SUMMARY = 'operation_summary'
    
    @classmethod
    def button_id(cls, operation: OperationType) -> str:
        """Get the button ID for an operation type."""
        return operation.value

# Default configuration keys
CONFIG_DATA_DIR = 'data_dir'
CONFIG_TARGET_DIR = 'target_dir'
CONFIG_ROBOFLOW_WORKSPACE = 'workspace'
CONFIG_ROBOFLOW_PROJECT = 'project'
CONFIG_ROBOFLOW_VERSION = 'version'


# Module metadata
MODULE_GROUP: str = "dataset"
MODULE_NAME: str = "downloader"
MODULE_TITLE: str = "Dataset Downloader"
MODULE_DESCRIPTION: str = "Download datasets from Roboflow"
MODULE_ICON: str = "📥"
MODULE_VERSION: str = "1.0.0"

# UI configuration
UI_CONFIG: Dict[str, Any] = {
    'title': MODULE_TITLE,  # Required by UI components
    'subtitle': MODULE_DESCRIPTION,  # Required by UI components
    'module_title': MODULE_TITLE,
    'module_description': MODULE_DESCRIPTION,
    'module_icon': MODULE_ICON,
    'module_name': MODULE_NAME,
    'parent_module': MODULE_GROUP,
    'version': MODULE_VERSION
}

# Button configurations
BUTTON_CONFIG: Dict[str, ButtonConfig] = {
    OperationType.DOWNLOAD.value: {
        'text': '📥 Download',
        'style': 'success',
        'tooltip': 'Download dataset from Roboflow',
        'order': 1
    },
    OperationType.CHECK.value: {
        'text': '🔍 Check',
        'style': 'info',
        'tooltip': 'Check dataset status and integrity',
        'order': 2
    },
    OperationType.CLEANUP.value: {
        'text': '🗑️ Cleanup',
        'style': 'danger',
        'tooltip': 'Remove dataset files from local storage',
        'order': 3
    }
}

# Field validation rules
VALIDATION_RULES: Dict[str, ValidationRule] = {
    'workspace': {'required': True, 'min_length': 1},
    'project': {'required': True, 'min_length': 1},
    'version': {'required': True, 'min_length': 1},
    'api_key': {'required': True, 'min_length': 10}
}

# Default directories
DEFAULT_DATA_DIR: str = 'data'
DEFAULT_TARGET_DIR: str = 'data'

# Roboflow API
ROBOFLOW_API_KEY_ENV: str = 'ROBOFLOW_API_KEY'
ROBOFLOW_FORMAT_YOLOV5: str = 'yolov5'