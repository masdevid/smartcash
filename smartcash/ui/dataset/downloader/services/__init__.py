"""
Downloader backend bridge services.

This module provides services that act as a bridge to backend functionality
like progress callbacks, API calls, and data transformations.
"""

from .downloader_service import DownloaderService
from .backend_utils import get_existing_dataset_count
from .validation_utils import validate_config
from .progress_utils import create_progress_callback

__all__ = [
    'DownloaderService',
    'get_existing_dataset_count',
    'validate_config', 
    'create_progress_callback'
]
