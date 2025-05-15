"""
File: smartcash/ui/dataset/augmentation/handlers/visualization_handler.py
Deskripsi: Handler visualisasi untuk augmentasi dataset (wrapper ke shared handler)
"""

from typing import Dict, Any
# display, clear_output tidak digunakan langsung dalam file ini (digunakan dalam shared_handler)
# ICONS dan create_status_indicator tidak digunakan langsung dalam file ini

def setup_visualization_handlers(ui_components: Dict[str, Any], env=None, config=None) -> Dict[str, Any]:
    """
    Setup handler untuk visualisasi dataset augmentasi.
    
    Args:
        ui_components: Dictionary komponen UI
        env: Environment manager
        config: Konfigurasi aplikasi
        
    Returns:
        Dictionary UI components yang telah diupdate
    """
    # Gunakan shared visualization handler
    from smartcash.ui.handlers.visualization_handler import setup_visualization_handlers as setup_shared_visualization_handlers
    
    # Delegasikan ke shared handler
    ui_components = setup_shared_visualization_handlers(ui_components, 'augmentation', env, config)
    
    return ui_components