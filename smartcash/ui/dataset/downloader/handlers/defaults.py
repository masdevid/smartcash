"""
File: smartcash/ui/dataset/downloader/handlers/defaults.py
Deskripsi: Updated default configuration tanpa format options (hardcoded yolov5pytorch)
"""

from typing import Dict, Any

def get_default_download_config() -> Dict[str, Any]:
    """Get default configuration untuk download module (format hardcoded ke yolov5pytorch)"""
    
    return {
        # Dataset identification
        'workspace': 'smartcash-wo2us',
        'project': 'rupiah-emisi-2022', 
        'version': '3',
        'api_key': '',  # Will be injected from Colab Secret
        
        # Download settings (format hardcoded)
        'output_format': 'yolov5pytorch',  # Hardcoded - tidak bisa diubah
        'validate_download': True,
        'organize_dataset': True,
        'backup_existing': False,
        
        # UI settings
        'auto_check_on_load': False,
        'show_progress_details': True,
        'confirm_destructive_operations': True,
        
        # Performance settings
        'download_timeout': 300,  # 5 minutes
        'retry_count': 3,
        'chunk_size': 8192,
        
        # Validation settings
        'validate_credentials': True,
        'validate_disk_space': True,
        'min_free_space_mb': 500,
        
        # Cleanup settings
        'auto_cleanup_temp': True,
        'preserve_backups': 1,
        'cleanup_on_exit': False,
        
        # Progress tracking
        'progress_update_interval': 0.5,  # seconds
        'detailed_progress': True,
        'show_file_progress': True,
        
        # Error handling
        'continue_on_errors': False,
        'max_retry_attempts': 3,
        'retry_delay_seconds': 2,
        
        # Advanced options
        'use_symlinks': False,
        'preserve_metadata': True,
        'compress_backup': True,
        
        # Metadata
        'module_name': 'downloader',
        'version': '3',
        'created_by': 'SmartCash Download Module',
        'description': 'Configuration for dataset download operations (YOLOv5 format)',
        
        # Internal flags (not exposed in UI)
        '_api_key_source': 'unknown',
        '_api_key_valid': False,
        '_last_check_timestamp': None,
        '_last_download_timestamp': None,
        '_format_locked': True  # Indicate format is hardcoded
    }

def get_download_validation_rules() -> Dict[str, Any]:
    """Get validation rules untuk download config (tanpa format validation)"""
    
    return {
        'required_fields': [
            'workspace', 'project', 'version', 'api_key'
        ],
        'optional_fields': [
            'validate_download', 'organize_dataset', 'backup_existing'
        ],
        'field_constraints': {
            'workspace': {
                'min_length': 2,
                'max_length': 50,
                'pattern': r'^[a-zA-Z0-9_-]+$',
                'description': 'Alphanumeric, underscore, dash only'
            },
            'project': {
                'min_length': 2,
                'max_length': 50,
                'pattern': r'^[a-zA-Z0-9_-]+$',
                'description': 'Alphanumeric, underscore, dash only'
            },
            'version': {
                'min_length': 1,
                'max_length': 10,
                'pattern': r'^[a-zA-Z0-9.]+$',
                'description': 'Alphanumeric and dots only'
            },
            'api_key': {
                'min_length': 10,
                'max_length': 200,
                'pattern': r'^[a-zA-Z0-9_-]+$',
                'description': 'Alphanumeric, underscore, dash only'
            }
        },
        'hardcoded_format': 'yolov5pytorch',  # Format is hardcoded
        'boolean_fields': [
            'validate_download', 'organize_dataset', 'backup_existing'
        ]
    }

def get_download_form_layout() -> Dict[str, Any]:
    """Get form layout configuration untuk UI (tanpa format field)"""
    
    return {
        'sections': [
            {
                'title': 'Dataset Configuration',
                'icon': '🔧',
                'fields': ['workspace', 'project', 'version', 'api_key'],
                'collapsible': False
            },
            {
                'title': 'Download Options', 
                'icon': '⚙️',
                'fields': ['validate_download', 'organize_dataset', 'backup_existing'],
                'collapsible': True,
                'collapsed': False
            }
        ],
        'field_widths': {
            'workspace': '100%',
            'project': '100%', 
            'version': '100%',
            'api_key': '100%'
        },
        'field_help_texts': {
            'workspace': 'Nama workspace di Roboflow (contoh: smartcash-wo2us)',
            'project': 'Nama project di workspace (contoh: rupiah-emisi-2022)',
            'version': 'Versi dataset (angka, contoh: 3)',
            'api_key': 'API key dari Roboflow dashboard atau Colab Secret',
            'validate_download': 'Validasi integritas dataset setelah download',
            'organize_dataset': 'Organisir struktur folder train/valid/test',
            'backup_existing': 'Backup dataset existing sebelum download baru'
        },
        'format_info': {
            'hardcoded': True,
            'format': 'yolov5pytorch',
            'description': 'Format YOLOv5 PyTorch (hardcoded untuk konsistensi)'
        }
    }

def get_operation_button_config() -> Dict[str, Any]:
    """Get configuration untuk operation buttons"""
    
    return {
        'check_button': {
            'label': '📊 Check Dataset',
            'tooltip': 'Periksa ketersediaan dataset tanpa download',
            'style': '',  # Default grey
            'width': '140px',
            'requires_confirmation': False,
            'validation_required': True,
            'progress_type': 'check'
        },
        'download_button': {
            'label': '📥 Download',
            'tooltip': 'Download dataset ke sistem (format YOLOv5)',
            'style': '',  # Default grey
            'width': '120px', 
            'requires_confirmation': 'conditional',  # Only if data exists
            'validation_required': True,
            'progress_type': 'download'
        },
        'cleanup_button': {
            'label': '🧹 Cleanup',
            'tooltip': 'Bersihkan file dataset dan cache',
            'style': '',  # Default grey
            'width': '120px',
            'requires_confirmation': True,  # Always confirm destructive operation
            'validation_required': False,
            'progress_type': 'cleanup'
        }
    }

def get_progress_step_config() -> Dict[str, Any]:
    """Get configuration untuk progress steps"""
    
    return {
        'check': {
            'steps': ['validate', 'connect', 'metadata', 'local_check', 'report'],
            'step_weights': {'validate': 10, 'connect': 20, 'metadata': 30, 'local_check': 30, 'report': 10},
            'progress_bars': ['overall']
        },
        'download': {
            'steps': ['validate', 'connect', 'metadata', 'download', 'extract', 'organize'],
            'step_weights': {'validate': 5, 'connect': 10, 'metadata': 10, 'download': 50, 'extract': 15, 'organize': 10},
            'progress_bars': ['overall', 'step', 'current']
        },
        'cleanup': {
            'steps': ['scan', 'confirm', 'cleanup', 'verify'],
            'step_weights': {'scan': 20, 'confirm': 5, 'cleanup': 70, 'verify': 5},
            'progress_bars': ['overall']
        }
    }

def get_api_key_config() -> Dict[str, Any]:
    """Get configuration untuk API key management"""
    
    return {
        'secret_names': [
            'ROBOFLOW_API_KEY',
            'roboflow_api_key',
            'API_KEY', 
            'api_key',
            'SMARTCASH_API_KEY',
            'smartcash_api_key'
        ],
        'validation': {
            'min_length': 10,
            'max_length': 200,
            'allowed_chars': r'^[a-zA-Z0-9_-]+$'
        },
        'sources': {
            'colab_secret': {
                'priority': 1,
                'description': 'Colab Secret (Recommended)',
                'icon': '🔑'
            },
            'manual': {
                'priority': 2, 
                'description': 'Manual Input',
                'icon': '📝'
            },
            'config': {
                'priority': 3,
                'description': 'Config File',
                'icon': '📄'
            }
        }
    }

# One-liner utilities untuk config access (updated tanpa format)
get_required_fields = lambda: get_download_validation_rules()['required_fields']
get_hardcoded_format = lambda: get_download_validation_rules()['hardcoded_format']
get_boolean_fields = lambda: get_download_validation_rules()['boolean_fields']
get_api_secret_names = lambda: get_api_key_config()['secret_names']
get_button_config = lambda button_name: get_operation_button_config().get(button_name, {})
get_progress_config = lambda operation: get_progress_step_config().get(operation, {})
is_format_locked = lambda: True  # Always True karena format hardcoded