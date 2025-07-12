"""
File: smartcash/ui/dataset/downloader/__init__.py
Description: Dataset Downloader module entry point with new UIModule pattern support.

This module provides both the new UIModule pattern API and maintains backward
compatibility with the existing initializer-based API.
"""

# New UIModule pattern API (recommended)
from smartcash.ui.dataset.downloader.downloader_uimodule import (
    DownloaderUIModule,
    create_downloader_uimodule,
    get_downloader_uimodule,
    reset_downloader_uimodule,
    initialize_downloader_ui as initialize_downloader_ui_new,
    get_downloader_components as get_downloader_components_new,
    display_downloader_ui as display_downloader_ui_new
)

# Legacy initializer-based API (backward compatibility)
from smartcash.ui.dataset.downloader.downloader_initializer import (
    initialize_downloader_ui as initialize_downloader_ui_legacy,
    display_downloader_ui as display_downloader_ui_legacy,
    get_downloader_components as get_downloader_components_legacy
)

# Default to new UIModule pattern, but provide legacy fallback
def initialize_downloader_ui(use_legacy: bool = False, **kwargs):
    """Initialize downloader UI.
    
    Args:
        use_legacy: If True, use legacy initializer pattern
        **kwargs: Arguments passed to the initializer
    """
    if use_legacy:
        return initialize_downloader_ui_legacy(**kwargs)
    else:
        return initialize_downloader_ui_new(**kwargs)

def display_downloader_ui(use_legacy: bool = False, **kwargs):
    """Display downloader UI.
    
    Args:
        use_legacy: If True, use legacy initializer pattern
        **kwargs: Arguments passed to the initializer
    """
    if use_legacy:
        return display_downloader_ui_legacy(**kwargs)
    else:
        return display_downloader_ui_new(**kwargs)

def get_downloader_components(use_legacy: bool = False, **kwargs):
    """Get downloader components.
    
    Args:
        use_legacy: If True, use legacy initializer pattern
        **kwargs: Arguments passed to the initializer
    """
    if use_legacy:
        return get_downloader_components_legacy(**kwargs)
    else:
        return get_downloader_components_new(**kwargs)

# Export both APIs
__all__ = [
    # New UIModule pattern API
    'DownloaderUIModule',
    'create_downloader_uimodule',
    'get_downloader_uimodule', 
    'reset_downloader_uimodule',
    
    # Main API functions (default to new pattern)
    'initialize_downloader_ui',
    'display_downloader_ui',
    'get_downloader_components',
    
    # Legacy API (explicit)
    'initialize_downloader_ui_legacy',
    'display_downloader_ui_legacy',
    'get_downloader_components_legacy'
]