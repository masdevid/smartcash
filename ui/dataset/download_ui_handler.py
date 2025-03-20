"""
File: smartcash/ui/dataset/download_ui_handler.py
Deskripsi: Handler UI untuk proses download dataset dengan integrasi utils standar untuk observer dan progress
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional
from IPython.display import display

from smartcash.ui.utils.constants import ICONS

def setup_ui_handlers(ui_components: Dict[str, Any], env=None, config=None) -> Dict[str, Any]:
    """Setup handler untuk events UI download dataset dengan utils standar."""
    # Dapatkan logger dari UI components
    logger = ui_components.get('logger')
    
    try:
        # Validasi input dengan utils standar
        from smartcash.ui.handlers.error_handler import handle_ui_error
        
        if not isinstance(ui_components.get('download_options'), widgets.RadioButtons):
            from smartcash.ui.dataset.download_initialization import update_status_panel
            update_status_panel(ui_components, "error", "Download options widget tidak valid")
            return ui_components
        
        # Definisikan handler perubahan opsi download yang lebih terstruktur
        def on_download_option_change(change):
            """Handler untuk perubahan opsi download dengan utils standar."""
            if change['name'] != 'value':
                return
                
            # Import fungsi API key info yang sudah diperbaiki
            from smartcash.ui.dataset.download_initialization import get_api_key_info, update_status_panel
                
            if change['new'] == 'Roboflow (Online)':
                # Dapatkan info API key dengan fungsi yang sudah diperbaiki
                api_key_info = get_api_key_info(ui_components)
                
                # Update container dengan settings Roboflow
                if 'download_settings_container' in ui_components and 'roboflow_settings' in ui_components:
                    ui_components['download_settings_container'].children = [
                        ui_components['roboflow_settings'], 
                        api_key_info
                    ]
                    
                    # Jika ada secret key, pastikan field API tetap disembunyikan
                    if ui_components.get('api_key_available', False) and hasattr(ui_components['roboflow_settings'], 'children') and len(ui_components['roboflow_settings'].children) > 0:
                        ui_components['roboflow_settings'].children[0].layout.display = 'none'
                
                update_status_panel(ui_components, "info", f"{ICONS['info']} Mempersiapkan download dari Roboflow")
            
            elif change['new'] == 'Local Data (Upload)':
                # Ganti ke komponen upload lokal
                if 'download_settings_container' in ui_components and 'local_upload' in ui_components:
                    ui_components['download_settings_container'].children = [ui_components['local_upload']]
                
                update_status_panel(ui_components, "info", f"{ICONS['upload']} Siap untuk upload dataset lokal")
        
        # Register event handler dengan validasi
        if 'download_options' in ui_components:
            ui_components['download_options'].observe(on_download_option_change, names='value')
        
        # Trigger inisialisasi awal berdasarkan opsi yang dipilih
        if ui_components.get('download_options') and ui_components['download_options'].value == 'Roboflow (Online)':
            on_download_option_change({'name': 'value', 'new': 'Roboflow (Online)'})
        
        # Setup event observers untuk tracking progress dengan utils standar
        try:
            # Setup observer standard di component handlers
            from smartcash.ui.handlers.observer_handler import register_ui_observer
            
            # Register untuk berbagai event download
            events = [
                "DOWNLOAD_START", "DOWNLOAD_PROGRESS", "DOWNLOAD_COMPLETE", "DOWNLOAD_ERROR",
                "EXPORT_START", "EXPORT_PROGRESS", "EXPORT_COMPLETE", "EXPORT_ERROR",
                "UPLOAD_START", "UPLOAD_PROGRESS", "UPLOAD_COMPLETE", "UPLOAD_ERROR"
            ]
            
            # Register observer untuk semua event
            for event in events:
                register_ui_observer(ui_components, event, 'status')
                
            if logger: logger.info(f"{ICONS['success']} Observer untuk {len(events)} events berhasil didaftarkan")
        except ImportError as e:
            if logger: logger.warning(f"{ICONS['warning']} Observer tidak tersedia: {str(e)}")
        
        # Setup progress handler dengan komponen standar jika belum ada
        if 'progress_handler' not in ui_components:
            try:
                from smartcash.ui.handlers.download_progress_handler import DownloadProgressHandler
                ui_components['progress_handler'] = DownloadProgressHandler(ui_components)
                if logger: logger.info(f"{ICONS['success']} Progress handler berhasil diinisialisasi")
            except ImportError as e:
                if logger: logger.warning(f"{ICONS['warning']} Progress handler tidak tersedia: {str(e)}")
            
    except Exception as e:
        # Handle error dengan komponen standar
        from smartcash.ui.handlers.error_handler import handle_ui_error
        handle_ui_error(e, ui_components.get('status'), True, "Error setup UI handlers")
    
    return ui_components