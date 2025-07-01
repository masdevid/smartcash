"""
File: smartcash/ui/dataset/downloader/handlers/config_extractor.py
Deskripsi: Ekstraksi konfigurasi downloader dari UI components sesuai dengan dataset_config.yaml
"""

from typing import Dict, Any, Optional, List, Tuple, TypeVar, Callable, Union
from datetime import datetime

from smartcash.ui.utils.ui_logger import get_module_logger
from smartcash.ui.handlers.error_handler import handle_ui_errors, create_error_response
from smartcash.ui.utils.error_utils import ErrorHandler, ErrorContext
from smartcash.common.worker_utils import get_optimal_worker_count, get_worker_counts_for_operations

# Initialize module logger
logger = get_module_logger('smartcash.ui.dataset.downloader.config_extractor')
T = TypeVar('T')

@handle_ui_errors(error_component_title="Config Extraction Error")
def extract_downloader_config(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Extract configuration dari UI components untuk downloader
    
    Args:
        ui_components: Dictionary of UI components
        
    Returns:
        Dictionary berisi konfigurasi downloader sesuai dengan dataset_config.yaml
    """
    # Create error context for better tracing
    ctx = ErrorContext(
        component="extract_downloader_config",
        operation="extract_config"
    )
    
    # Use ErrorHandler for consistent error handling
    handler = ErrorHandler(
        context=ctx,
        logger=logger
    )
    
    # One-liner value extraction dengan fallback
    get_value = lambda key, default: getattr(ui_components.get(key, type('', (), {'value': default})()), 'value', default)
    
    logger.debug("Extracting configuration from UI components")
    
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
            'max_workers': get_optimal_worker_count('io_bound')
        },
        
        'uuid_renaming': {
            'enabled': True,
            'backup_before_rename': get_value('backup_checkbox', False),
            'batch_size': 1000,
            'parallel_workers': get_optimal_worker_count('mixed'),
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
            'parallel_workers': get_optimal_worker_count('cpu_bound')
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
            'parallel_workers': get_optimal_worker_count('io_bound')
        }
    }



@handle_ui_errors(error_component_title="Worker Counts Error")
def get_optimal_worker_counts() -> Dict[str, int]:
    """Get all optimal worker counts for downloader operations with fail-fast error handling"""
    # Create error context for better tracing
    ctx = ErrorContext(
        component="get_optimal_worker_counts",
        operation="get_all_workers"
    )
    
    # Use ErrorHandler for consistent error handling
    handler = ErrorHandler(
        context=ctx,
        logger=logger
    )
    
    try:
        # Use centralized worker counts utility
        worker_counts = get_worker_counts_for_operations()
        logger.debug(f"Retrieved optimal worker counts: {worker_counts}")
        return worker_counts
    except Exception as e:
        error_msg = f"Failed to get optimal worker counts: {str(e)}"
        handler.handle_error(error=e, message=error_msg)
        # Return safe defaults
        return {
            'download': 4,
            'validation': 2,
            'uuid_renaming': 2,
            'preprocessing': 2
        }