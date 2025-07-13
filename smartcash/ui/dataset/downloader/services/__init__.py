"""
Downloader services module.

This module provides a clean, organized set of services for the downloader functionality,
including dataset scanning, configuration validation, and secret management.

Services are organized into submodules:
- core: Base service classes and interfaces
- backends: Services that interact with external systems
- validators: Services for data validation
- utils: Utility services for common operations
"""

import logging
from typing import Dict, Any, Optional, Type, TypeVar

# Import the new services
from .core.base_service import BaseService
from .backends.dataset_scanner import DatasetScannerService
from .validators.config_validator import ConfigValidatorService
from .utils.secret_manager import SecretManagerService
from .downloader_service import DownloaderService

# Type variable for service classes
T = TypeVar('T', bound=BaseService)

# Service registry
_service_instances: Dict[Type[BaseService], BaseService] = {}

# Default logger
_logger = logging.getLogger(__name__)

def get_service(service_class: Type[T], logger: Optional[logging.Logger] = None) -> T:
    """Get or create a service instance.
    
    Args:
        service_class: The service class to get an instance of
        logger: Optional logger instance to use
        
    Returns:
        An instance of the requested service class
    """
    if service_class not in _service_instances:
        _service_instances[service_class] = service_class(logger or _logger)
    return _service_instances[service_class]

# Public API
def get_dataset_scanner(logger: Optional[logging.Logger] = None) -> DatasetScannerService:
    """Get the dataset scanner service."""
    return get_service(DatasetScannerService, logger)

def get_config_validator(logger: Optional[logging.Logger] = None) -> ConfigValidatorService:
    """Get the config validator service."""
    return get_service(ConfigValidatorService, logger)

def get_secret_manager(logger: Optional[logging.Logger] = None) -> SecretManagerService:
    """Get the secret manager service."""
    return get_service(SecretManagerService, logger)

__all__ = [
    'DownloaderService',
    'get_dataset_scanner',
    'get_config_validator',
    'get_secret_manager'
]
