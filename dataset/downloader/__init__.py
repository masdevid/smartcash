"""
File: smartcash/dataset/downloader/__init__.py
Deskripsi: Factory untuk proper integration antara UI dan service layer
"""

from typing import Dict, Any, Optional
from smartcash.common.logger import get_logger

def get_downloader_instance(config: Dict[str, Any], logger=None):
    """
    Factory untuk create downloader instance dengan proper integration.
    
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
    Create Roboflow downloader dengan default config.
    
    Args:
        api_key: Roboflow API key
        config: Optional config override
        logger: Optional logger
        
    Returns:
        Configured downloader instance
    """
    default_config = {
        'api_key': api_key,
        'output_format': 'yolov5pytorch',
        'validate_download': True,
        'organize_dataset': True,
        'backup_existing': False
    }
    
    if config:
        default_config.update(config)
    
    return get_downloader_instance(default_config, logger)

# Export untuk backward compatibility
__all__ = ['get_downloader_instance', 'create_roboflow_downloader']