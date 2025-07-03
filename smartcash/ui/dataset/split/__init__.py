"""
File: smartcash/ui/dataset/split/__init__.py
Deskripsi: Ekspor utilitas dan fungsi split dataset
"""

from smartcash.ui.dataset.split.split_initializer import (
    create_split_config_cell,
    get_split_config_components,
    SplitInitializer
)
from smartcash.ui.dataset.split.handlers.config_handler import SplitConfigHandler

# Untuk backward compatibility
create_split_config_ui = create_split_config_cell
display_split_config = create_split_config_cell  # Alias untuk backward compatibility

# Public API
def initialize_split_ui(config=None):
    """🎯 Inisialisasi UI split dataset dengan konfigurasi opsional.
    
    Ini adalah entry point utama untuk fungsionalitas split dataset.
    Menampilkan UI di notebook tanpa mengembalikan komponen.
    
    Args:
        config: Dictionary konfigurasi opsional
    """
    create_split_config_cell(config)


def get_split_ui_components(config=None):
    """📦 Mendapatkan komponen UI split dataset untuk akses programmatik.
    
    Fungsi ini mengembalikan dictionary komponen UI untuk manipulasi programmatik
    tanpa menampilkan UI di notebook.
    
    Args:
        config: Dictionary konfigurasi opsional
        
    Returns:
        Dictionary komponen UI
    """
    return get_split_config_components(config)


# Alias untuk backward compatibility
create_split_init = create_split_config_cell

__all__ = [
    'initialize_split_ui',
    'get_split_ui_components',
    'create_split_init',
    'create_split_config_ui',  # Backward compatibility
    'create_split_config_cell',
    'get_split_config_components',
    'display_split_config',  # Backward compatibility
    'SplitInitializer',  # Updated from SplitConfigInitializer
    'SplitConfigHandler'
]
