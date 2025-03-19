"""
File: smartcash/ui/dataset/download_ui_handler.py
Deskripsi: Handler UI untuk proses download dataset dengan pemisahan progress handler
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional
from IPython.display import display, HTML

from smartcash.common.logger import get_logger
from smartcash.ui.utils.constants import ICONS


def setup_ui_handlers(ui_components: Dict[str, Any], env=None, config=None) -> Dict[str, Any]:
    """Setup handler untuk events UI download dataset."""
    try:
        # Validasi input
        if not isinstance(ui_components.get('download_options'), widgets.RadioButtons):
            from smartcash.ui.dataset.download_initialization import update_status_panel
            update_status_panel(ui_components, "error", "Download options widget tidak valid")
            return ui_components
        
        # Cek apakah sudah ada API key dari Google Secret
        has_secret_key = False
        try:
            from google.colab import userdata
            secret_key = userdata.get('ROBOFLOW_API_KEY')
            has_secret_key = bool(secret_key)
        except ImportError:
            pass
        
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
                    
                    # Jika ada secret key, pastikan field API tetap disembunyikan
                    if has_secret_key and hasattr(ui_components['roboflow_settings'], 'children') and len(ui_components['roboflow_settings'].children) > 0:
                        ui_components['roboflow_settings'].children[0].layout.display = 'none'
                
                from smartcash.ui.dataset.download_initialization import update_status_panel
                if has_secret_key:
                    update_status_panel(ui_components, "info", "Mempersiapkan download dari Roboflow dengan API key dari Google Secret")
                else:
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
        
        # Trigger initial event handler based on selected option
        if ui_components.get('download_options') and ui_components['download_options'].value == 'Roboflow (Online)':
            on_download_option_change({'name': 'value', 'new': 'Roboflow (Online)'})
        
        # Setup progress tracking with observer
        # Periksa apakah komponen progress tersedia
        has_progress_components = (
            'progress_bar' in ui_components and 
            'progress_label' in ui_components and
            'status_output' in ui_components
        )
        
        if has_progress_components:
            # Inisialisasi progress handler yang sudah dipindahkan ke handler terpisah
            try:
                from smartcash.ui.handlers.download_progress_handler import DownloadProgressHandler
                progress_handler = DownloadProgressHandler(ui_components)
                ui_components['progress_handler'] = progress_handler
                
                logger = ui_components.get('logger') or get_logger("download_ui_handler")
                logger.info(f"{ICONS['success']} Handler progres download berhasil diinisialisasi")
            except ImportError as e:
                logger = ui_components.get('logger') or get_logger("download_ui_handler")
                logger.warning(f"{ICONS['warning']} Tidak dapat memuat DownloadProgressHandler: {str(e)}")
            
    except Exception as e:
        from smartcash.ui.dataset.download_initialization import update_status_panel
        update_status_panel(ui_components, "error", f"Error setup UI handlers: {str(e)}")
    
    return ui_components