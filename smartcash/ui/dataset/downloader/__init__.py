"""
Downloader Module - Data handling and processing for downloader

file_path: /Users/masdevid/Projects/smartcash/smartcash/ui/dataset/downloader/__init__.py
"""

from .downloader_uimodule import DownloaderUIModule
from .downloader_ui_factory import DownloaderUIFactory, create_downloader_display

def initialize_downloader_ui(config=None, **kwargs):
    """
    Initialize and display the downloader UI.
    
    Args:
        config: Optional configuration dict
        **kwargs: Additional arguments for UI initialization
        
    Returns:
        None (displays the UI using IPython.display)
    """
    DownloaderUIFactory.create_and_display_downloader(config=config, **kwargs)

# Export main classes and functions
__all__ = [
    'DownloaderUIModule',
    'DownloaderUIFactory',
    'initialize_downloader_ui',
    'create_downloader_display'
]
