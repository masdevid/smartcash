"""
Downloader module untuk SmartCash dengan CommonInitializer pattern.

Modul ini menyediakan antarmuka untuk mengunduh dataset dengan dukungan
konfigurasi yang fleksibel dan manajemen state yang kuat.
"""

# Ekspor utama
from smartcash.ui.dataset.downloader.downloader_initializer import (
    DownloaderInitializer,
    initialize_downloader_ui
)

# Aliases for backward compatibility
init_downloader = initialize_downloader_ui

__all__ = [
    # Classes
    'DownloaderInitializer',
    
    # Factory functions
    'initialize_downloader_ui',
    
    # Aliases for backward compatibility
    'init_downloader'
]