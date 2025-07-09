"""
Downloader backend bridge services.

This module provides services that act as a bridge to backend functionality
like progress callbacks, API calls, and data transformations.
"""

from .downloader_service import DownloaderService
from .backend_utils import get_existing_dataset_count
from .validation_utils import validate_config

__all__ = [
    'DownloaderService',
    'get_existing_dataset_count',
    'validate_config'
]
