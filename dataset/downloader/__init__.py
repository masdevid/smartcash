"""
File: smartcash/dataset/downloader/__init__.py
Deskripsi: Optimized factory dan exports dengan one-liner style dan enhanced integration
"""

from typing import Dict, Any, Optional
from smartcash.common.logger import get_logger

def get_downloader_instance(config: Dict[str, Any], logger=None):
    """
    Optimized factory untuk downloader instance dengan enhanced error handling.
    
    Args:
        config: Configuration dictionary
        logger: Optional logger instance
        
    Returns:
        Downloader service instance
    """
    try:
        from smartcash.dataset.downloader.download_service import create_download_service
        return create_download_service(config, logger)
    except ImportError as e:
        logger = logger or get_logger('downloader.factory')
        logger.error(f"❌ Import error: {str(e)}")
        raise ImportError(f"Cannot import download service: {str(e)}")

def create_roboflow_downloader(api_key: str, config: Dict[str, Any] = None, logger=None):
    """
    One-liner Roboflow downloader creation dengan optimized defaults.
    
    Args:
        api_key: Roboflow API key
        config: Optional config override
        logger: Optional logger
        
    Returns:
        Configured downloader instance
    """
    # One-liner config merging
    merged_config = {
        'api_key': api_key, 'output_format': 'yolov5pytorch',
        'validate_download': True, 'organize_dataset': True, 'backup_existing': False,
        **(config or {})
    }
    
    return get_downloader_instance(merged_config, logger)

def create_optimized_downloader(api_key: str, workspace: str, project: str, version: str, 
                               logger=None, **kwargs) -> Any:
    """
    One-liner optimized downloader dengan pre-configured parameters.
    
    Args:
        api_key: Roboflow API key
        workspace: Roboflow workspace
        project: Roboflow project
        version: Dataset version
        logger: Optional logger
        **kwargs: Additional config parameters
        
    Returns:
        Pre-configured downloader instance
    """
    # One-liner optimized config creation
    optimized_config = {
        'api_key': api_key, 'workspace': workspace, 'project': project, 'version': version,
        'output_format': 'yolov5pytorch', 'validate_download': True, 'organize_dataset': True,
        'backup_existing': False, 'retry_count': 3, 'timeout': 30, 'chunk_size': 8192,
        **kwargs
    }
    
    return get_downloader_instance(optimized_config, logger)

# One-liner component factories
def create_roboflow_client(api_key: str, logger=None):
    """One-liner Roboflow client factory"""
    from smartcash.dataset.downloader.roboflow_client import create_roboflow_client
    return create_roboflow_client(api_key, logger)

def create_file_processor(logger=None, max_workers: int = None):
    """One-liner file processor factory"""
    from smartcash.dataset.downloader.file_processor import create_file_processor
    return create_file_processor(logger, max_workers)

def create_progress_tracker():
    """One-liner progress tracker factory"""
    from smartcash.dataset.downloader.progress_tracker import create_download_tracker
    return create_download_tracker()

def create_dataset_validator(logger=None, max_workers: int = None):
    """One-liner dataset validator factory"""
    from smartcash.dataset.downloader.validators import create_dataset_validator
    return create_dataset_validator(logger, max_workers)

# One-liner utility functions
def validate_config_quick(config: Dict[str, Any]) -> bool:
    """Quick one-liner config validation"""
    required_fields = ['api_key', 'workspace', 'project', 'version']
    return all(config.get(field, '').strip() for field in required_fields)

def get_default_download_config(api_key: str = "", workspace: str = "", project: str = "", version: str = "") -> Dict[str, Any]:
    """One-liner default config creation"""
    return {
        'api_key': api_key, 'workspace': workspace, 'project': project, 'version': version,
        'output_format': 'yolov5pytorch', 'validate_download': True, 'organize_dataset': True,
        'backup_existing': False, 'retry_count': 3, 'timeout': 30, 'chunk_size': 8192
    }

def create_download_session(api_key: str, workspace: str, project: str, version: str, logger=None) -> Dict[str, Any]:
    """
    One-liner download session creation dengan all components.
    
    Returns:
        Dictionary containing all downloader components
    """
    config = get_default_download_config(api_key, workspace, project, version)
    
    return {
        'service': get_downloader_instance(config, logger),
        'client': create_roboflow_client(api_key, logger),
        'processor': create_file_processor(logger),
        'validator': create_dataset_validator(logger),
        'tracker': create_progress_tracker(),
        'config': config
    }

# Enhanced error handling factories
def create_safe_downloader(config: Dict[str, Any], logger=None, fallback_config: Dict[str, Any] = None):
    """One-liner safe downloader creation dengan fallback"""
    try:
        return get_downloader_instance(config, logger)
    except Exception as e:
        logger = logger or get_logger('downloader.safe_factory')
        logger.warning(f"⚠️ Primary config failed: {str(e)}, using fallback")
        return get_downloader_instance(fallback_config or get_default_download_config(), logger) if fallback_config else None

def get_downloader_info(downloader_instance) -> Dict[str, Any]:
    """One-liner downloader instance information"""
    return (downloader_instance.get_service_info() if hasattr(downloader_instance, 'get_service_info') 
            else {'type': type(downloader_instance).__name__, 'available': True})

# Compatibility aliases
create_downloader = get_downloader_instance  # Backward compatibility
get_roboflow_downloader = create_roboflow_downloader  # Alternative name

# Export everything untuk comprehensive access
__all__ = [
    # Main factories
    'get_downloader_instance', 'create_roboflow_downloader', 'create_optimized_downloader',
    
    # Component factories  
    'create_roboflow_client', 'create_file_processor', 'create_progress_tracker', 'create_dataset_validator',
    
    # Utility functions
    'validate_config_quick', 'get_default_download_config', 'create_download_session',
    
    # Enhanced factories
    'create_safe_downloader', 'get_downloader_info',
    
    # Compatibility aliases
    'create_downloader', 'get_roboflow_downloader'
]

# One-liner version info
__version__ = '2.0.0'
__description__ = 'Optimized SmartCash dataset downloader dengan enhanced performance dan one-liner style'