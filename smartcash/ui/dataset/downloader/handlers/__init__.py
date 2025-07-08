"""
Downloader module-specific handlers.

This module contains handler implementations for the downloader module.
"""

from .downloader_ui_handler import DownloaderUIHandler
from .download_handler import setup_download_handlers

__all__ = [
    'DownloaderUIHandler',
    'setup_download_handlers'
]
