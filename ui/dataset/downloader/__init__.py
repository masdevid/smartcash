"""
File: smartcash/ui/dataset/downloader/__init__.py
Deskripsi: Simple entry point yang langsung return UI widget
"""

from .downloader_init import initialize_downloader_ui, get_downloader_status, setup_downloader

__all__ = ['initialize_downloader_ui', 'get_downloader_status', 'setup_downloader']

# Main public API
def setup_downloader(env=None, config=None, **kwargs):
    """Main entry point - return UI widget langsung."""
    return initialize_downloader_ui(env=env, config=config, **kwargs)