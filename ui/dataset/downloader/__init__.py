"""
File: smartcash/ui/dataset/downloader/__init__.py
Deskripsi: Simple entry point yang langsung return UI widget
"""

from smartcash.ui.dataset.downloader.downloader_init import initialize_downloader_ui, get_downloader_status

__all__ = ['initialize_downloader_ui', 'get_downloader_status']

