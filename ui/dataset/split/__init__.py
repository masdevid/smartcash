"""
File: smartcash/ui/dataset/split/__init__.py
Deskripsi: Ekspor utilitas dan fungsi split dataset
"""

from smartcash.ui.dataset.split.split_init import (
    display_split_config,
    create_split_config_ui,
    SplitConfigInitializer
)
from smartcash.ui.dataset.split.handlers.config_handler import SplitConfigHandler

# Public API dengan nama yang konsisten
def initialize_split_ui(config=None, **kwargs):
    """🎯 Inisialisasi UI split dataset dengan konfigurasi opsional.
    
    Ini adalah entry point utama untuk fungsionalitas split dataset.
    Menampilkan UI di notebook tanpa mengembalikan komponen.
    
    Args:
        config: Dictionary konfigurasi opsional
        **kwargs: Argumen tambahan untuk initializer
    """
    display_split_config(config, **kwargs)


def get_split_ui_components(config=None, **kwargs):
    """📦 Mendapatkan komponen UI split dataset untuk akses programmatik.
    
    Fungsi ini mengembalikan dictionary komponen UI untuk manipulasi programmatik
    tanpa menampilkan UI di notebook.
    
    Args:
        config: Dictionary konfigurasi opsional
        **kwargs: Argumen tambahan untuk initializer
        
    Returns:
        Dictionary komponen UI
    """
    return create_split_config_ui(config, **kwargs)


# Alias untuk backward compatibility dan convenience
create_split_config_cell = display_split_config
get_split_config_components = create_split_config_ui
create_split_init = display_split_config

__all__ = [
    # Main API
    'initialize_split_ui',
    'get_split_ui_components',
    
    # Direct exports
    'display_split_config',
    'create_split_config_ui',
    'SplitConfigInitializer',
    'SplitConfigHandler',
    
    # Backward compatibility aliases
    'create_split_config_cell',
    'get_split_config_components',
    'create_split_init'
]