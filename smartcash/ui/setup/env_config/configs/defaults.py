"""
File: smartcash/ui/setup/env_config/configs/defaults.py

Default configuration untuk environment setup.
"""

from typing import Dict, Any

DEFAULT_CONFIG: Dict[str, Any] = {
    'version': '1.0.0',
    'environment': {
        'type': 'colab',
        'auto_mount_drive': True,
        'mount_path': '/content/drive',
        'base_path': '/content',
        'project_name': 'SmartCash'
    },
    'setup': {
        'auto_start': False,
        'stop_on_error': True,
        'verify_setup': True,
        'max_retries': 3,
        'retry_delay': 5,
        'backup_existing': True,
        'create_symlinks': True,
        'stages': [
            'drive_mount',
            'config_sync', 
            'folder_setup',
            'verify'
        ]
    },
    'paths': {
        'drive_base': '/content/drive/MyDrive/SmartCash',
        'colab_base': '/content/SmartCash',
        'config_dir': 'configs',
        'data_dir': 'data',
        'models_dir': 'models',
        'outputs_dir': 'outputs',
        'logs_dir': 'logs'
    },
    'logging': {
        'level': 'INFO',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'file_logging': True,
        'console_logging': True
    },
    'ui': {
        'theme': 'dark',
        'show_progress': True,
        'show_summary': True,
        'auto_scroll_logs': True,
        'expand_logs': True
    }
}