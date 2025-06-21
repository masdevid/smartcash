"""
File: smartcash/ui/dataset/downloader/handlers/config_extractor.py
Deskripsi: Ekstraksi konfigurasi downloader dari UI components sesuai dengan dataset_config.yaml
"""

from typing import Dict, Any
from datetime import datetime

def extract_downloader_config(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Ekstraksi konfigurasi downloader yang konsisten dengan dataset_config.yaml"""
    # One-liner value extraction dengan fallback
    get_value = lambda key, default: getattr(ui_components.get(key, type('', (), {'value': default})()), 'value', default)
    
    # Metadata untuk config yang diperbarui
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Struktur konfigurasi sesuai dengan dataset_config.yaml
    return {
        'config_version': '1.0',
        'updated_at': current_time,
        '_base_': 'base_config.yaml',
        
        'data': {
            'source': 'roboflow',
            'dir': get_value('data_dir', 'data'),
            
            'roboflow': {
                'workspace': get_value('workspace_input', '').strip(),
                'project': get_value('project_input', '').strip(),
                'version': get_value('version_input', '').strip(),
                'api_key': get_value('api_key_input', '').strip(),
                'output_format': 'yolov5pytorch'
            },
            
            'file_naming': {
                'uuid_format': True,
                'naming_strategy': 'research_uuid',
                'preserve_original': False,
                'backup_before_rename': get_value('backup_checkbox', False)
            },
            
            'local': {
                'train': 'data/train',
                'valid': 'data/valid', 
                'test': 'data/test'
            }
        },
        
        'download': {
            'enabled': True,
            'target_dir': get_value('target_dir', 'data'),
            'temp_dir': 'data/downloads',
            'backup_existing': get_value('backup_checkbox', False),
            'validate_download': get_value('validate_checkbox', True),
            'organize_dataset': True,
            'rename_files': True,
            'retry_count': 3,
            'timeout': 30,
            'chunk_size': 262144,
            'parallel_downloads': True,
            'max_workers': get_value('max_workers', _get_optimal_download_workers())
        },
        
        'uuid_renaming': {
            'enabled': True,
            'backup_before_rename': get_value('backup_checkbox', False),
            'batch_size': 1000,
            'parallel_workers': _get_optimal_rename_workers(),
            'validate_consistency': True,
            'target_splits': ['train', 'valid', 'test'],
            'file_patterns': ['*.jpg', '*.jpeg', '*.png', '*.bmp'],
            'label_patterns': ['*.txt'],
            'progress_reporting': True
        },
        
        'validation': {
            'enabled': get_value('validate_checkbox', True),
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
            'parallel_workers': _get_optimal_validation_workers()
        },
        
        'cleanup': {
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
            'parallel_workers': _get_optimal_io_workers()
        }
    }

def _get_optimal_download_workers() -> int:
    """Get optimal workers untuk download operations"""
    from smartcash.common.threadpools import get_download_workers
    return get_download_workers()

def _get_optimal_rename_workers() -> int:
    """Get optimal workers untuk file renaming operations"""
    from smartcash.common.threadpools import get_rename_workers
    return get_rename_workers(5000)  # Estimate 5k files

def _get_optimal_validation_workers() -> int:
    """Get optimal workers untuk validation operations"""
    from smartcash.common.threadpools import get_optimal_thread_count
    return get_optimal_thread_count('io')

def _get_optimal_io_workers() -> int:
    """Get optimal workers untuk I/O operations"""
    from smartcash.common.threadpools import optimal_io_workers
    return optimal_io_workers()