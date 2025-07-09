"""
File: smartcash/ui/dataset/downloader/__init__.py
Deskripsi: Simple entry point yang langsung return UI widget
"""

from smartcash.ui.dataset.downloader.downloader_initializer import initialize_downloader_ui, display_downloader_ui, get_downloader_components


__all__ = ['initialize_downloader_ui', 'display_downloader_ui', 'get_downloader_components']