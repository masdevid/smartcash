"""
File: smartcash/ui/dataset/download_ui_handler.py
Deskripsi: Handler untuk UI events pada download dataset
"""

from typing import Dict, Any
import ipywidgets as widgets

try:
    from smartcash.ui.utils.fallback import create_status_message, handle_download_status
    from smartcash.ui.dataset.download_initialization import get_api_key_info
except ImportError:
    def create_status_message(message, status_type='info', **kwargs):
        return f"📢 {message}"
    
    def handle_download_status(ui_components, message, status_type='info', **kwargs):
        print(f"{status_type.upper()}: {message}")
    
    def get_api_key_info(ui_components):
        return widgets.HTML(value="API Key Info")

def setup_ui_handlers(
    ui_components: Dict[str, Any], 
    env=None, 
    config=None
) -> Dict[str, Any]:
    """
    Setup handler untuk events UI download dataset.
    
    Args:
        ui_components: Dictionary komponen UI
        env: Environment manager
        config: Konfigurasi aplikasi
        
    Returns:
        Dictionary UI components yang telah diupdate
    """
    try:
        # Validasi input
        if not isinstance(ui_components.get('download_options'), widgets.RadioButtons):
            handle_download_status(
                ui_components, 
                "Download options widget tidak valid", 
                'error'
            )
            return ui_components
        
        def on_download_option_change(change):
            """Handler untuk perubahan opsi download."""
            if change['name'] == 'value':
                if change['new'] == 'Roboflow (Online)':
                    # Gunakan fungsi get_api_key_info untuk mendapatkan info API
                    api_key_info = get_api_key_info(ui_components)
                    
                    # Update container dengan settings dan info API
                    ui_components['download_settings_container'].children = [
                        ui_components['roboflow_settings'], 
                        api_key_info
                    ]
                    
                    handle_download_status(
                        ui_components, 
                        "Mempersiapkan download dari Roboflow", 
                        'info'
                    )
                
                elif change['new'] == 'Local Data (Upload)':
                    # Ganti ke komponen upload lokal
                    ui_components['download_settings_container'].children = [
                        ui_components['local_upload']
                    ]
                    
                    handle_download_status(
                        ui_components, 
                        "Siap untuk upload dataset lokal", 
                        'info'
                    )
        
        # Register event handler
        ui_components['download_options'].observe(
            on_download_option_change, 
            names='value'
        )
        
    except Exception as e:
        handle_download_status(
            ui_components, 
            f"Error setup UI handlers: {str(e)}", 
            'error'
        )
    
    return ui_components