"""
File: smartcash/dataset/downloader/__init__.py
Deskripsi: Entry point untuk dataset downloader dengan factory pattern dan lazy loading
"""

from typing import Dict, Any, Optional
from .download_service import DownloadService, create_download_service
from .roboflow_client import RoboflowClient, create_roboflow_client  
from .file_processor import FileProcessor, create_file_processor
from .validators import DatasetValidator, FileValidator, create_dataset_validator, create_file_validator
from .progress_tracker import DownloadProgressTracker, create_download_tracker

__all__ = [
    'DownloadService', 'create_download_service',
    'RoboflowClient', 'create_roboflow_client',
    'FileProcessor', 'create_file_processor', 
    'DatasetValidator', 'FileValidator', 'create_dataset_validator', 'create_file_validator',
    'DownloadProgressTracker', 'create_download_tracker',
    'get_downloader_instance', 'create_downloader_config'
]

# Lazy loading pattern untuk heavy dependencies
_downloader_instances = {}

def get_downloader_instance(config: Dict[str, Any], logger=None) -> DownloadService:
    """Get atau create downloader instance dengan caching untuk performance."""
    cache_key = f"{config.get('workspace', '')}_{config.get('project', '')}_{id(logger) if logger else 'default'}"
    
    if cache_key not in _downloader_instances:
        _downloader_instances[cache_key] = create_download_service(config, logger)
    
    return _downloader_instances[cache_key]

def create_downloader_config(workspace: str, project: str, version: str, api_key: str, **kwargs) -> Dict[str, Any]:
    """Factory untuk create downloader config dengan sensible defaults."""
    from smartcash.ui.dataset.downloader.handlers.defaults import DEFAULT_CONFIG
    
    config = DEFAULT_CONFIG.copy()
    config.update({
        'workspace': workspace,
        'project': project, 
        'version': version,
        'api_key': api_key,
        **kwargs
    })
    
    return config

def clear_downloader_cache():
    """Clear downloader instance cache untuk cleanup."""
    global _downloader_instances
    _downloader_instances.clear()