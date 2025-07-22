"""
File: smartcash/ui/setup/colab/configs/colab_defaults.py
Description: Default configuration for colab module following dependency pattern
"""

from typing import Dict, Any

def get_default_colab_config() -> Dict[str, Any]:
    """
    Get default colab configuration.
    
    Returns:
        Dict containing default colab configuration
    """
    return {
        "environment": {
            "type": "colab",
            "auto_mount_drive": True,
            "mount_path": "/content/drive",
            "base_path": "/content",
            "project_name": "SmartCash",
            "gpu_enabled": False,
            "tpu_enabled": False
        },
        "setup": {
            "auto_start": False,
            "stop_on_error": True,
            "verify_setup": True,
            "max_retries": 3,
            "retry_delay": 5,
            "backup_existing": True,
            "create_symlinks": True,
            "stages": [
                "environment_detection",
                "drive_mount",
                "gpu_setup",
                "folder_setup",
                "config_sync",
                "verify"
            ]
        },
        "paths": {
            "drive_base": "/content/drive/MyDrive/SmartCash",
            "colab_base": "/content",
            "repo_base": "/content/smartcash/smartcash",
            "config_dir": "configs",
            "data_dir": "data",
            "models_dir": "models",
            "outputs_dir": "outputs",
            "logs_dir": "logs"
        },
        "logging": {
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "file_logging": True,
            "console_logging": True
        },
        "ui": {
            "theme": "dark",
            "show_progress": True,
            "show_summary": True,
            "auto_scroll_logs": True,
            "expand_logs": True,
            "show_advanced_options": False
        }
    }

def get_available_environments() -> Dict[str, Dict[str, Any]]:
    """
    Get available environment configurations.
    
    Returns:
        Dict containing available environment configurations
    """
    return {
        'colab': {
            'display_name': 'Google Colab',
            'description': 'Google Colaboratory environment',
            'supports_gpu': True,
            'supports_tpu': True,
            'mount_required': True,
            'base_path': '/content'
        },
        'kaggle': {
            'display_name': 'Kaggle Notebooks',
            'description': 'Kaggle notebook environment',
            'supports_gpu': True,
            'supports_tpu': False,
            'mount_required': False,
            'base_path': '/kaggle/working'
        },
        'local': {
            'display_name': 'Local Environment',
            'description': 'Local development environment',
            'supports_gpu': True,
            'supports_tpu': False,
            'mount_required': False,
            'base_path': '.'
        }
    }

def get_setup_stages_config() -> Dict[str, Dict[str, Any]]:
    """
    Get setup stages configuration.
    
    Returns:
        Dict containing setup stages configuration
    """
    return {
        'environment_detection': {
            'display_name': 'Environment Detection',
            'description': 'Detect current runtime environment',
            'required': True,
            'timeout': 30
        },
        'drive_mount': {
            'display_name': 'Drive Mount',
            'description': 'Mount Google Drive (Colab only)',
            'required': False,
            'timeout': 120
        },
        'gpu_setup': {
            'display_name': 'GPU Setup',
            'description': 'Configure GPU acceleration',
            'required': False,
            'timeout': 60
        },
        'folder_setup': {
            'display_name': 'Folder Setup',
            'description': 'Create project directory structure',
            'required': True,
            'timeout': 30
        },
        'config_sync': {
            'display_name': 'Config Sync',
            'description': 'Synchronize configuration files',
            'required': True,
            'timeout': 60
        },
        'verify': {
            'display_name': 'Verification',
            'description': 'Verify setup completion',
            'required': True,
            'timeout': 30
        }
    }

def get_gpu_configurations() -> Dict[str, Dict[str, Any]]:
    """
    Get GPU configuration options.
    
    Returns:
        Dict containing GPU configurations
    """
    return {
        'none': {
            'display_name': 'No GPU',
            'description': 'CPU-only execution',
            'memory_gb': 0,
            'compute_capability': None
        },
        'k80': {
            'display_name': 'Tesla K80',
            'description': 'NVIDIA Tesla K80 (12GB)',
            'memory_gb': 12,
            'compute_capability': '3.7'
        },
        't4': {
            'display_name': 'Tesla T4',
            'description': 'NVIDIA Tesla T4 (16GB)',
            'memory_gb': 16,
            'compute_capability': '7.5'
        },
        'p100': {
            'display_name': 'Tesla P100',
            'description': 'NVIDIA Tesla P100 (16GB)',
            'memory_gb': 16,
            'compute_capability': '6.0'
        },
        'v100': {
            'display_name': 'Tesla V100',
            'description': 'NVIDIA Tesla V100 (16GB)',
            'memory_gb': 16,
            'compute_capability': '7.0'
        }
    }

# Legacy support
DEFAULT_CONFIG = get_default_colab_config()
