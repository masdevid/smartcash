"""
File: smartcash/ui/setup/env_config_initializer.py
Deskripsi: Initializer untuk modul konfigurasi environment dengan implementasi DRY
"""

from typing import Dict, Any
from IPython.display import display

def initialize_env_config_ui():
    """
    Inisialisasi UI dan handler untuk konfigurasi environment menggunakan cell template.
    
    Returns:
        Dictionary komponen UI yang terinisialisasi
    """
    try:
        # Gunakan cell template standar untuk konsistensi
        from smartcash.ui.cell_template import setup_cell
        from smartcash.ui.setup.env_config_component import create_env_config_ui
        from smartcash.ui.setup.env_config_handlers import setup_env_config_handlers
        
        # Gunakan UI logger yang ada
        from smartcash.ui.utils.ui_logger import log_to_ui
        
        # Inisialisasi dengan cell template
        ui_components = setup_cell(
            cell_name="env_config",
            create_ui_func=create_env_config_ui,
            setup_handlers_func=setup_env_config_handlers
        )
        
        # Tampilkan UI
        if 'ui' in ui_components:
            display(ui_components['ui'])
            
        return ui_components
        
    except Exception as e:
        # Fallback minimal menggunakan alert_utils
        from smartcash.ui.utils.alert_utils import create_info_alert
        from smartcash.ui.utils.fallback_utils import create_fallback_ui
        
        # Gunakan factory fallback yang ada
        import ipywidgets as widgets
        error_ui = create_fallback_ui(
            {'module_name': 'env_config'},
            f"Error inisialisasi environment config: {str(e)}",
            "error"
        )
        
        display(error_ui['ui'])
        return error_ui