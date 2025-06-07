"""
File: smartcash/dataset/downloader/__init__.py
Deskripsi: Clean factory exports dengan reduced duplication
"""

from typing import Dict, Any, Optional
from smartcash.common.logger import get_logger


def get_downloader_instance(config: Dict[str, Any], logger=None) -> Optional['DownloadService']:
    """Main factory untuk downloader instance"""
    try:
        from smartcash.dataset.downloader.download_service import create_download_service
        return create_download_service(config, logger)
    except Exception as e:
        logger = logger or get_logger('downloader.factory')
        logger.error(f"❌ Error creating downloader: {str(e)}")
        return None


def create_download_session(api_key: str, workspace: str = None, project: str = None, 
                           version: str = None, logger=None, **kwargs) -> Dict[str, Any]:
    """Create complete download session"""
    config = get_default_config(api_key)
    
    # Override defaults
    if workspace:
        config['workspace'] = workspace
    if project:
        config['project'] = project
    if version:
        config['version'] = version
    
    # Apply additional kwargs
    config.update(kwargs)
    
    service = get_downloader_instance(config, logger)
    
    return {
        'service': service,
        'config': config,
        'ready': service is not None
    }


def get_default_config(api_key: str = '') -> Dict[str, Any]:
    """Get default configuration"""
    return {
        'api_key': api_key,
        'workspace': 'smartcash-wo2us',
        'project': 'rupiah-emisi-2022',
        'version': '3',
        'output_format': 'yolov5pytorch',
        'rename_files': True,
        'validate_download': True,
        'organize_dataset': True,
        'backup_existing': False
    }


# Convenience functions
create_roboflow_downloader = lambda api_key, **kwargs: get_downloader_instance(
    {**get_default_config(api_key), **kwargs}
)

validate_config_quick = lambda config: all(
    config.get(f, '').strip() for f in ['api_key', 'workspace', 'project', 'version']
)

# Export minimal essentials
__all__ = [
    'get_downloader_instance',
    'create_download_session', 
    'get_default_config',
    'create_roboflow_downloader',
    'validate_config_quick'
]