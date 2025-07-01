"""
File: smartcash/ui/dataset/downloader/handlers/defaults.py
Deskripsi: Hardcoded default configuration untuk reset operations tanpa dependency ke yaml files
"""

from typing import Dict, Any, Optional, TypeVar, Callable, Union

from smartcash.ui.utils.ui_logger import get_module_logger
from smartcash.ui.handlers.error_handler import handle_ui_errors, create_error_response
from smartcash.ui.utils.error_utils import ErrorHandler, ErrorContext
from smartcash.ui.utils.fallback_utils import safe_execute
from smartcash.common.worker_utils import get_optimal_worker_count

# Initialize module logger
logger = get_module_logger('smartcash.ui.dataset.downloader.defaults')
T = TypeVar('T')

@handle_ui_errors(error_component_title="Default Downloader Config Error")
def get_default_downloader_config() -> Dict[str, Any]:
    """
    Get hardcoded default configuration untuk downloader reset operations.
    Tidak bergantung pada yaml files untuk menghindari circular dependency.
    Uses fail-fast approach with centralized error handling.
    
    Returns:
        Dictionary berisi default configuration
    """
    logger.debug("Getting default downloader configuration")
    return {
        'data': {
            'source': 'roboflow',
            'roboflow': {
                'workspace': 'smartcash-wo2us',
                'project': 'rupiah-emisi-2022',
                'version': '3',
                'api_key': '',
                'output_format': 'yolov5pytorch'
            },
            'file_naming': {
                'uuid_format': True,
                'naming_strategy': 'research_uuid',
                'preserve_original': False
            }
        },
        'download': {
            'rename_files': True,
            'organize_dataset': True,
            'validate_download': True,
            'backup_existing': False,
            'retry_count': 3,
            'timeout': 30,
            'chunk_size': 8192
        },
        'uuid_renaming': {
            'enabled': True,
            'backup_before_rename': False,
            'batch_size': 1000,
            'parallel_workers': get_optimal_worker_count('mixed'),
            'validate_consistency': True
        }
    }

@handle_ui_errors(error_component_title="Roboflow Config Error")
def get_roboflow_defaults() -> Dict[str, str]:
    """Get default Roboflow configuration untuk UI reset dengan fail-fast approach"""
    # Create error context for better tracing
    ctx = ErrorContext(
        component="get_roboflow_defaults",
        operation="retrieve_defaults"
    )
    
    # Use ErrorHandler for consistent error handling
    handler = ErrorHandler(
        context=ctx,
        logger=logger
    )
    
    try:
        config = get_default_downloader_config()
        roboflow = config['data']['roboflow']
        logger.debug("Retrieved default Roboflow configuration")
        return {
            'workspace': roboflow['workspace'],
            'project': roboflow['project'], 
            'version': roboflow['version'],
            'api_key': roboflow['api_key']
        }
    except KeyError as e:
        # Specific handling for key errors with detailed message
        error_msg = f"Missing required key in config: {str(e)}"
        handler.handle_error(error=e, message=error_msg)
        # Fallback to safe defaults
        return {
            'workspace': get_default_workspace(),
            'project': get_default_project(),
            'version': get_default_version(),
            'api_key': ''
        }

@handle_ui_errors(error_component_title="Download Config Error")
def get_download_defaults() -> Dict[str, Any]:
    """Get default download options untuk UI reset dengan fail-fast approach"""
    # Create error context for better tracing
    ctx = ErrorContext(
        component="get_download_defaults",
        operation="retrieve_defaults"
    )
    
    # Use ErrorHandler for consistent error handling
    handler = ErrorHandler(
        context=ctx,
        logger=logger
    )
    
    try:
        download = get_default_downloader_config()['download']
        logger.debug("Retrieved default download options")
        return {
            'validate_download': download['validate_download'],
            'backup_existing': download['backup_existing'],
            'rename_files': download['rename_files'],
            'organize_dataset': download['organize_dataset']
        }
    except KeyError as e:
        # Specific handling for key errors with detailed message
        error_msg = f"Missing required key in download config: {str(e)}"
        handler.handle_error(error=e, message=error_msg)
        # Fallback to safe defaults
        return {
            'validate_download': is_validation_enabled_by_default(),
            'backup_existing': False,
            'rename_files': True,
            'organize_dataset': True
        }

@handle_ui_errors(error_component_title="UUID Config Error")
def get_uuid_defaults() -> Dict[str, Any]:
    """Get default UUID renaming settings untuk reset dengan fail-fast approach"""
    # Create error context for better tracing
    ctx = ErrorContext(
        component="get_uuid_defaults",
        operation="retrieve_defaults"
    )
    
    # Use ErrorHandler for consistent error handling
    handler = ErrorHandler(
        context=ctx,
        logger=logger
    )
    
    try:
        uuid_config = get_default_downloader_config()['uuid_renaming']
        logger.debug("Retrieved default UUID renaming settings")
        return {
            'enabled': uuid_config['enabled'],
            'backup_before_rename': uuid_config['backup_before_rename'],
            'validate_consistency': uuid_config['validate_consistency']
        }
    except KeyError as e:
        # Specific handling for key errors with detailed message
        error_msg = f"Missing required key in UUID config: {str(e)}"
        handler.handle_error(error=e, message=error_msg)
        # Fallback to safe defaults
        return {
            'enabled': is_uuid_enabled_by_default(),
            'backup_before_rename': False,
            'validate_consistency': True
        }

# One-liner utilities untuk quick access with proper error handling
@handle_ui_errors(error_component_title="Default Config Error", return_type=str)
def get_default_workspace() -> str:
    """Get default workspace name with fail-fast error handling"""
    return 'smartcash-wo2us'

@handle_ui_errors(error_component_title="Default Config Error", return_type=str)
def get_default_project() -> str:
    """Get default project name with fail-fast error handling"""
    return 'rupiah-emisi-2022'

@handle_ui_errors(error_component_title="Default Config Error", return_type=str)
def get_default_version() -> str:
    """Get default version with fail-fast error handling"""
    return '3'

@handle_ui_errors(error_component_title="Default Config Error", return_type=bool)
def is_uuid_enabled_by_default() -> bool:
    """Check if UUID is enabled by default with fail-fast error handling"""
    return True

@handle_ui_errors(error_component_title="Default Config Error", return_type=bool)
def is_validation_enabled_by_default() -> bool:
    """Check if validation is enabled by default with fail-fast error handling"""
    return True