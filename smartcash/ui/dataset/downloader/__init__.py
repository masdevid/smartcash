"""
File: smartcash/ui/dataset/downloader/__init__.py
Deskripsi: Simple entry point yang langsung return UI widget
"""

from smartcash.ui.dataset.downloader.downloader_initializer import initialize_downloader, DownloaderInitializer


__all__ = ['DownloaderInitializer', 'initialize_downloader']