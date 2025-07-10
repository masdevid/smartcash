"""
Downloader UI components.

This module provides UI component definitions for the downloader module.
"""

from .downloader_ui import create_downloader_ui
from .operation_summary import create_operation_summary

# Maintain backward compatibility
create_downloader_ui_components = create_downloader_ui

__all__ = [
    'create_downloader_ui',
    'create_operation_summary'
]
