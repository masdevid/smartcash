"""
File: smartcash/ui/dataset/preprocessing/handlers/visualization_handler.py
Deskripsi: Handler visualisasi untuk preprocessing dataset (wrapper ke shared handler)
"""

from typing import Dict, Any
from IPython.display import display, clear_output
from pathlib import Path
from smartcash.ui.utils.constants import ICONS
from smartcash.ui.utils.alert_utils import create_status_indicator

def setup_visualization_handlers(ui_components: Dict[str, Any], env=None, config=None) -> Dict[str, Any]:
    """
    Setup handler untuk visualisasi dataset preprocessing.
    
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
    ui_components = setup_shared_visualization_handlers(ui_components, 'preprocessing', env, config)
    
    return ui_components