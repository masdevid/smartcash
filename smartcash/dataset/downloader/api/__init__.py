"""
Dataset Downloader API Module

This module provides API functions for downloader integration tests and external access.
It re-exports the main downloader factory functions with aliases expected by integration tests.
"""

# Module exports for downloader API

# Import from parent downloader module
from smartcash.dataset.downloader import (
    get_downloader_instance,
    get_cleanup_service,
    get_dataset_scanner,
    get_progress_tracker,
    create_download_session,
    get_default_config,
    validate_service_compatibility,
    create_ui_compatible_config,
    validate_config_quick
)

# API exports for integration tests
__all__ = [
    'get_downloader_instance',
    'get_cleanup_service', 
    'get_dataset_scanner',
    'get_progress_tracker',
    'create_download_session',
    'get_default_config',
    'validate_service_compatibility',
    'create_ui_compatible_config',
    'validate_config_quick'
]