"""
Downloader API Module

Provides standardized API functions for downloader operations that match
the interface expected by integration tests and UI modules.
"""

from typing import Dict, Any, Optional

# Re-export all functions from parent downloader module  
from smartcash.dataset.downloader import (
    get_downloader_instance,
    get_cleanup_service,
    get_dataset_scanner,
    get_progress_tracker,
    create_download_session,
    get_default_config,
    validate_service_compatibility,
    create_ui_compatible_config,
    validate_config_quick,
    create_roboflow_downloader
)

# Additional API functions for specific integration test requirements

def create_downloader_service(config: Dict[str, Any], logger=None) -> Optional[Any]:
    """
    Create downloader service instance (alias for get_downloader_instance).
    
    Args:
        config: Configuration dictionary
        logger: Optional logger instance
        
    Returns:
        Downloader service instance or None
    """
    return get_downloader_instance(config, logger)


def validate_downloader_config(config: Dict[str, Any]) -> bool:
    """
    Validate downloader configuration (alias for validate_config_quick).
    
    Args:
        config: Configuration to validate
        
    Returns:
        True if config is valid, False otherwise
    """
    return validate_config_quick(config)


def get_scanner_service(config: Optional[Dict[str, Any]] = None, logger=None) -> Optional[Any]:
    """
    Get dataset scanner service instance.
    
    Args:
        config: Optional configuration (unused but kept for compatibility)
        logger: Optional logger instance
        
    Returns:
        Scanner service instance or None
    """
    # config parameter is unused but kept for API compatibility
    _ = config  # Explicitly mark as unused
    return get_dataset_scanner(logger)


# Export all API functions
__all__ = [
    # Core factory functions
    'get_downloader_instance',
    'create_downloader_service',
    'get_cleanup_service',
    'get_dataset_scanner', 
    'get_scanner_service',
    'get_progress_tracker',
    
    # Configuration functions
    'create_download_session',
    'get_default_config',
    'validate_service_compatibility',
    'create_ui_compatible_config',
    'validate_config_quick',
    'validate_downloader_config',
    'create_roboflow_downloader'
]