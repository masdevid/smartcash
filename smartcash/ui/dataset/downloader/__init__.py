"""
File: smartcash/ui/dataset/downloader/__init__.py
Description: Dataset Downloader module entry point with UIModule pattern.

This module provides the UIModule pattern API for the dataset downloader,
including UI initialization, display, and component access.
"""

from typing import Dict, Any, Optional

# Conditional import to avoid issues in non-Jupyter environments
try:
    from IPython.display import display
except ImportError:
    # Fallback for non-Jupyter environments
    def display(obj):
        print(obj)

# Import UIModule components
from smartcash.ui.dataset.downloader.downloader_uimodule import (
    DownloaderUIModule,
    create_downloader_uimodule,
    get_downloader_uimodule,
    reset_downloader_uimodule,
    initialize_downloader_ui
)


def get_downloader_components(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Get downloader UI components.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Dictionary of UI components
    """
    module = create_downloader_uimodule(config=config, auto_initialize=True)
    return getattr(module, 'get_ui_components', lambda: {})()

# Export public API
__all__ = [
    # Core UIModule components
    'DownloaderUIModule',
    'create_downloader_uimodule',
    'get_downloader_uimodule',
    'reset_downloader_uimodule',
    'initialize_downloader_ui',
    
    # Helper functions
    'get_downloader_components'
]