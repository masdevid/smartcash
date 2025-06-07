"""
File: smartcash/dataset/downloader/__init__.py
Deskripsi: UPDATED factory exports dengan backend services baru
"""

from typing import Dict, Any, Optional
from smartcash.common.logger import get_logger


def get_downloader_instance(config: Dict[str, Any], logger=None) -> Optional['DownloadService']:
    """
    Main factory untuk downloader instance - matching UI expectation
    
    Args:
        config: Configuration dengan format yang diharapkan UI
        logger: Logger instance
        
    Returns:
        DownloadService instance atau None jika error
    """
    try:
        from smartcash.dataset.downloader.download_service import create_download_service
        return create_download_service(config, logger)
    except Exception as e:
        logger = logger or get_logger('downloader.factory')
        logger.error(f"❌ Error creating downloader: {str(e)}")
        return None


# Backend services exports - NEW
def get_dataset_scanner(logger=None):
    """Get dataset scanner service"""
    from smartcash.dataset.downloader.dataset_scanner import create_dataset_scanner
    return create_dataset_scanner(logger)


def get_cleanup_service(logger=None):
    """Get cleanup service"""
    from smartcash.dataset.downloader.cleanup_service import create_cleanup_service
    return create_cleanup_service(logger)


def get_progress_tracker(callback=None):
    """Get progress tracker"""
    from smartcash.dataset.downloader.progress_tracker import create_progress_tracker
    return create_progress_tracker(callback)


def create_download_session(api_key: str, workspace: str = None, project: str = None, 
                           version: str = None, logger=None, **kwargs) -> Dict[str, Any]:
    """Create complete download session dengan UI compatibility"""
    config = get_default_config(api_key)
    
    # Override defaults dengan parameter yang diberikan
    config.update({k: v for k, v in {
        'workspace': workspace, 'project': project, 'version': version
    }.items() if v is not None})
    
    # Apply additional kwargs
    config.update(kwargs)
    
    service = get_downloader_instance(config, logger)
    
    return {
        'service': service,
        'config': config,
        'ready': service is not None,
        'has_progress_callback': hasattr(service, 'set_progress_callback') if service else False
    }


def get_default_config(api_key: str = '') -> Dict[str, Any]:
    """Get default configuration dengan format yang konsisten dengan UI"""
    return {
        # Core Roboflow parameters
        'api_key': api_key,
        'workspace': 'smartcash-wo2us',
        'project': 'rupiah-emisi-2022', 
        'version': '3',
        'output_format': 'yolov5pytorch',
        
        # Processing options (matching UI expectations)
        'rename_files': True,
        'validate_download': True,
        'organize_dataset': True,
        'backup_existing': False,
        
        # Performance settings
        'retry_count': 3,
        'timeout': 30,
        'chunk_size': 8192
    }


def validate_service_compatibility(service) -> Dict[str, Any]:
    """Validate service compatibility dengan UI expectations"""
    if not service:
        return {'compatible': False, 'missing': ['service_instance']}
    
    required_methods = ['download_dataset', 'set_progress_callback']
    missing_methods = [method for method in required_methods 
                      if not hasattr(service, method)]
    
    return {
        'compatible': len(missing_methods) == 0,
        'missing': missing_methods,
        'service_type': type(service).__name__,
        'has_progress_support': hasattr(service, 'set_progress_callback')
    }


def create_ui_compatible_config(ui_config: Dict[str, Any]) -> Dict[str, Any]:
    """Convert UI config format ke format yang diharapkan core service"""
    roboflow = ui_config.get('data', {}).get('roboflow', {})
    download = ui_config.get('download', {})
    
    return {
        # Extract dari nested structure UI
        'api_key': roboflow.get('api_key', ''),
        'workspace': roboflow.get('workspace', ''),
        'project': roboflow.get('project', ''),
        'version': roboflow.get('version', ''),
        'output_format': roboflow.get('output_format', 'yolov5pytorch'),
        
        # Extract download options
        'rename_files': download.get('rename_files', True),
        'validate_download': download.get('validate_download', True),
        'organize_dataset': download.get('organize_dataset', True), 
        'backup_existing': download.get('backup_existing', False),
        'retry_count': download.get('retry_count', 3),
        'timeout': download.get('timeout', 30)
    }


# Convenience functions
create_roboflow_downloader = lambda api_key, **kwargs: get_downloader_instance(
    {**get_default_config(api_key), **kwargs}
)

validate_config_quick = lambda config: all(
    config.get(f, '').strip() for f in ['api_key', 'workspace', 'project', 'version']
)

# UPDATED exports dengan backend services
__all__ = [
    # Core factory (UI dependency)
    'get_downloader_instance',
    'create_download_session', 
    'get_default_config',
    'create_roboflow_downloader',
    'validate_config_quick',
    'validate_service_compatibility',
    'create_ui_compatible_config',
    
    # Backend services (NEW)
    'get_dataset_scanner',
    'get_cleanup_service', 
    'get_progress_tracker'
]