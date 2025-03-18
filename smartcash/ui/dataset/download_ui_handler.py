"""
File: smartcash/ui/dataset/download_ui_handler.py
Deskripsi: Handler untuk UI events pada download dataset
"""

from typing import Dict, Any
import ipywidgets as widgets

def setup_ui_handlers(ui_components: Dict[str, Any], env=None, config=None) -> Dict[str, Any]:
    """Setup handler untuk events UI download dataset."""
    try:
        # Validasi input
        if not isinstance(ui_components.get('download_options'), widgets.RadioButtons):
            from smartcash.ui.dataset.download_initialization import update_status_panel
            update_status_panel(ui_components, "error", "Download options widget tidak valid")
            return ui_components
        
        def on_download_option_change(change):
            """Handler untuk perubahan opsi download."""
            if change['name'] != 'value':
                return
                
            if change['new'] == 'Roboflow (Online)':
                # Gunakan fungsi get_api_key_info untuk mendapatkan info API
                from smartcash.ui.dataset.download_initialization import get_api_key_info
                api_key_info = get_api_key_info(ui_components)
                
                # Update container dengan settings dan info API
                if 'download_settings_container' in ui_components and 'roboflow_settings' in ui_components:
                    ui_components['download_settings_container'].children = [
                        ui_components['roboflow_settings'], 
                        api_key_info
                    ]
                
                from smartcash.ui.dataset.download_initialization import update_status_panel
                update_status_panel(ui_components, "info", "Mempersiapkan download dari Roboflow")
            
            elif change['new'] == 'Local Data (Upload)':
                # Ganti ke komponen upload lokal
                if 'download_settings_container' in ui_components and 'local_upload' in ui_components:
                    ui_components['download_settings_container'].children = [
                        ui_components['local_upload']
                    ]
                
                from smartcash.ui.dataset.download_initialization import update_status_panel
                update_status_panel(ui_components, "info", "Siap untuk upload dataset lokal")
        
        # Register event handler
        if 'download_options' in ui_components:
            ui_components['download_options'].observe(on_download_option_change, names='value')
        
    except Exception as e:
        from smartcash.ui.dataset.download_initialization import update_status_panel
        update_status_panel(ui_components, "error", f"Error setup UI handlers: {str(e)}")
    
    return ui_components