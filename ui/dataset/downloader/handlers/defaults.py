"""
File: smartcash/ui/dataset/downloader/handlers/defaults.py
Deskripsi: Default configuration values untuk downloader module
"""

from pathlib import Path
from typing import Dict, Any, Optional
from smartcash.common.environment import get_environment_manager
from smartcash.common.config import ConfigManager, get_config_manager

# Default configuration untuk downloader
DEFAULT_CONFIG = {
    'roboflow': {
        'workspace': '',
        'project': '',
        'version': '1',
        'api_key': '',
        'format': 'yolov5pytorch',
        'location': '',
        'download_options': {
            'include_augmented': False,
            'include_test': True,
            'include_train': True,
            'include_valid': True,
            'include_raw': False
        }
    },
    'local': {
        'output_dir': '',
        'backup_dir': '',
        'organize_dataset': True,
        'backup_enabled': False,
        'progress_enabled': True,
        'show_detailed_progress': True
    },
    'advanced': {
        'retry_attempts': 3,
        'timeout_seconds': 30,
        'chunk_size_kb': 8
    }
}

# Validation rules untuk config values
VALIDATION_RULES = {
    'roboflow': {
        'workspace': {'min_length': 2, 'required': True, 'type': str},
        'project': {'min_length': 2, 'required': True, 'type': str},
        'version': {'min_length': 1, 'required': True, 'type': (str, int)},
        'api_key': {'min_length': 20, 'required': True, 'type': str},
        'format': {
            'required': True,
            'type': str,
            'choices': ['yolov5', 'yolov8', 'coco', 'pascal-voc', 'tfrecord']
        },
        'location': {'type': str, 'default': ''},
        'download_options': {
            'type': dict,
            'schema': {
                'include_augmented': {'type': bool, 'default': False},
                'include_test': {'type': bool, 'default': False},
                'include_train': {'type': bool, 'default': True},
                'include_valid': {'type': bool, 'default': True},
                'include_raw': {'type': bool, 'default': False}
            }
        }
    },
    'local': {
        'output_dir': {'type': str, 'required': True},
        'backup_dir': {'type': str, 'required': False},
        'organize_dataset': {'type': bool, 'default': True},
        'backup_enabled': {'type': bool, 'default': False},
        'progress_enabled': {'type': bool, 'default': True},
        'show_detailed_progress': {'type': bool, 'default': True}
    },
    'advanced': {
        'retry_attempts': {'type': int, 'min': 1, 'max': 10, 'default': 3},
        'timeout_seconds': {'type': int, 'min': 10, 'max': 300, 'default': 30},
        'chunk_size_kb': {'type': int, 'min': 1, 'max': 1024, 'default': 8}
    }
}

def get_default_api_key() -> str:
    """Detect API key dari environment sources."""
    import os
    
    # Cari dari environment variables
    for env_key in ['ROBOFLOW_API_KEY', 'ROBOFLOW_KEY', 'RF_API_KEY']:
        if api_key := os.environ.get(env_key, '').strip():
            return api_key if len(api_key) > 10 else ''
    
    # Coba dari Google Colab userdata
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
    env = get_environment_manager()
    base_path = str(env.drive_path / 'dataset') if env.is_colab and env.is_drive_mounted else 'data'
    return {
        'data': {
            'dir': f"{base_path}",
            'preprocessed_dir': f"{base_path}/preprocessed"
        },
        'cleanup': {
            'backup_dir': f"{base_path}/backup",
            'backup_enabled': False
        },
        'dataset': {
            'backup': {
                'enabled': False,
                'dir': f"{base_path}/backup/dataset"
            },
            'export': {
                'format': 'yolov5pytorch',
                'include_visualization': True
            }
        }
    }

def get_default_downloader_config() -> Dict[str, Any]:
    """Get default configuration untuk downloader."""
    config_manager = get_config_manager()
    
    # Load konfigurasi
    base_config = config_manager.load('base_config.yaml') or {}
    dataset_config = config_manager.load('dataset_config.yaml') or {}
    
    # Dapatkan default paths
    default_paths = get_default_paths()
    
    # Fungsi untuk merge dictionaries
    def merge_dicts(a: Dict, b: Dict) -> Dict:
        for key in b:
            if key in a:
                if isinstance(a[key], dict) and isinstance(b[key], dict):
                    merge_dicts(a[key], b[key])
                else:
                    a[key] = b[key]
            else:
                a[key] = b[key]
        return a
    
    # Gabungkan konfigurasi
    config = merge_dicts(base_config, dataset_config)
    
    # Inisialisasi bagian konfigurasi jika tidak ada
    config.setdefault('roboflow', {})
    config.setdefault('local', {})
    config.setdefault('advanced', {})
    
    # Update konfigurasi roboflow dengan nilai default
    roboflow = config['roboflow']
    roboflow.update({
        'workspace': roboflow.get('workspace', ''),
        'project': roboflow.get('project', ''),
        'version': str(roboflow.get('version', '1')),
        'api_key': roboflow.get('api_key', get_default_api_key()),
        'format': roboflow.get('format', 'yolov5pytorch'),
        'location': roboflow.get('location', ''),
        'download_options': {
            'include_augmented': roboflow.get('download_options', {}).get('include_augmented', False),
            'include_test': roboflow.get('download_options', {}).get('include_test', True),
            'include_train': roboflow.get('download_options', {}).get('include_train', True),
            'include_valid': roboflow.get('download_options', {}).get('include_valid', True),
            'include_raw': roboflow.get('download_options', {}).get('include_raw', False)
        }
    })
    
    # Update konfigurasi local dengan nilai default
    local = config['local']
    local.update({
        'output_dir': local.get('output_dir', str(default_paths.get('data_dir', ''))),
        'backup_dir': local.get('backup_dir', str(default_paths.get('backup_dir', ''))),
        'organize_dataset': local.get('organize_dataset', True),
        'backup_enabled': local.get('backup_enabled', False),
        'progress_enabled': local.get('progress_enabled', True),
        'show_detailed_progress': local.get('show_detailed_progress', True)
    })
    
    # Update konfigurasi advanced dengan nilai default
    advanced = config['advanced']
    advanced.update({
        'retry_attempts': advanced.get('retry_attempts', 3),
        'timeout_seconds': advanced.get('timeout_seconds', 30),
        'chunk_size_kb': advanced.get('chunk_size_kb', 8)
    })
    
    return config

# Validation rules untuk config values
VALIDATION_RULES = {
    'roboflow': {
        'workspace': {'min_length': 2, 'required': True, 'type': str},
        'project': {'min_length': 2, 'required': True, 'type': str},
        'version': {'min': 1, 'max': 1000, 'default': 1, 'type': (str, int)},
        'api_key': {'min_length': 10, 'required': True, 'type': str},
        'format': {
            'required': True,
            'type': str,
            'choices': ['yolov5pytorch', 'yolov5', 'coco', 'pascal-voc', 'tfrecord']
        },
        'location': {'type': str, 'default': ''},
        'download_options': {
            'type': dict,
            'schema': {
                'include_augmented': {'type': bool, 'default': False},
                'include_test': {'type': bool, 'default': False},
                'include_train': {'type': bool, 'default': True},
                'include_valid': {'type': bool, 'default': True},
                'include_raw': {'type': bool, 'default': False}
            }
        }
    },
    'local': {
        'output_dir': {'type': str, 'required': True},
        'backup_dir': {'type': str, 'required': False},
        'organize_dataset': {'type': bool, 'default': True},
        'backup_enabled': {'type': bool, 'default': False},
        'progress_enabled': {'type': bool, 'default': True},
        'show_detailed_progress': {'type': bool, 'default': True}
    },
    'advanced': {
        'retry_attempts': {'type': int, 'min': 1, 'max': 10, 'default': 3},
        'timeout_seconds': {'type': int, 'min': 10, 'max': 300, 'default': 30},
        'chunk_size_kb': {'type': int, 'min': 1, 'max': 1024, 'default': 8}
    }
}