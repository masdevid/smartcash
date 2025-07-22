"""
Downloader operation handlers and factories.

This module contains operation-specific handlers and factory functions for download operations.
"""

from .downloader_base_operation import DownloaderBaseOperation, DownloaderOperationPhase
from .download_operation import DownloadOperation
from .download_check_operation import DownloadCheckOperation
from .download_cleanup_operation import DownloadCleanupOperation
from .download_factory import (
    DownloaderOperationFactory,
    create_download_operation,
    create_check_operation,
    create_cleanup_operation,
    execute_download_operation,
    execute_check_operation,
    execute_cleanup_operation
)

__all__ = [
    'DownloaderBaseOperation',
    'DownloaderOperationPhase',
    'DownloadOperation',
    'DownloadCheckOperation',
    'DownloadCleanupOperation',
    'DownloaderOperationFactory',
    'create_download_operation',
    'create_check_operation',
    'create_cleanup_operation',
    'execute_download_operation',
    'execute_check_operation',
    'execute_cleanup_operation'
]