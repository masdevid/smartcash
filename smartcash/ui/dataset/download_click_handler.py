"""
File: smartcash/ui/dataset/download_click_handler.py
Deskripsi: Handler untuk tombol download dataset dengan dukungan dialog konfirmasi
"""

import os
from pathlib import Path
from typing import Dict, Any
from IPython.display import display, clear_output, HTML

def setup_click_handlers(ui_components: Dict[str, Any], env=None, config=None) -> Dict[str, Any]:
    """
    Setup handler untuk klik tombol download.
    
    Args:
        ui_components: Dictionary komponen UI
        env: Environment manager
        config: Konfigurasi aplikasi
        
    Returns:
        Dictionary UI components yang telah diupdate
    """
    try:
        from smartcash.ui.components.alerts import create_status_indicator
        from smartcash.ui.utils.constants import ICONS
        from smartcash.ui.dataset.download_initialization import update_status_panel
        from smartcash.ui.dataset.roboflow_download_handler import download_from_roboflow
        from smartcash.ui.dataset.local_upload_handler import process_local_upload
        # Import confirmation handler
        from smartcash.ui.dataset.download_confirmation_handler import setup_confirmation_handlers
    except ImportError:
        # Fallback jika module tidak tersedia
        def create_status_indicator(status, message):
            icons = {'success': '✅', 'warning': '⚠️', 'error': '❌', 'info': 'ℹ️'}
            icon = icons.get(status, 'ℹ️')
            return HTML(f"<div style='padding:8px'>{icon} {message}</div>")
        
        ICONS = {
            'processing': '🔄',
            'success': '✅',
            'warning': '⚠️',
            'error': '❌'
        }
    
    # Handler download button click
    def on_download_click(b):
        with ui_components['status']:
            clear_output(wait=True)
            display(create_status_indicator("info", f"{ICONS['processing']} Memulai download dataset..."))
        
        if 'progress_bar' in ui_components:
            ui_components['progress_bar'].value = 0
            
        try:
            download_option = ui_components['download_options'].value
            
            # Notify event if observer manager available
            if 'observer_manager' in ui_components:
                try:
                    from smartcash.components.observer.event_dispatcher_observer import EventDispatcher
                    EventDispatcher.notify(
                        event_type="DOWNLOAD_START",
                        sender="download_handler",
                        message=f"Memulai download dataset dari {download_option}"
                    )
                except ImportError:
                    pass
            
            if download_option == 'Roboflow (Online)':
                download_from_roboflow(ui_components, env, config)
            elif download_option == 'Local Data (Upload)':
                process_local_upload(ui_components, env, config)
                
        except Exception as e:
            with ui_components['status']:
                display(create_status_indicator("error", f"{ICONS['error']} Error: {str(e)}"))
            
            # Log error
            if 'logger' in ui_components:
                ui_components['logger'].error(f"{ICONS['error']} Error saat download dataset: {str(e)}")
    
    # Periksa struktur dataset
    def validate_dataset_structure(data_dir):
        splits = ['train', 'valid', 'test']
        valid_structure = True
        missing_parts = []
        
        for split in splits:
            split_dir = Path(data_dir) / split
            images_dir = split_dir / 'images'
            labels_dir = split_dir / 'labels'
            
            if not split_dir.exists():
                missing_parts.append(f"{split}/")
                valid_structure = False
            elif not images_dir.exists():
                missing_parts.append(f"{split}/images/")
                valid_structure = False
            elif not labels_dir.exists():
                missing_parts.append(f"{split}/labels/")
                valid_structure = False
        
        if valid_structure:
            with ui_components['status']:
                display(create_status_indicator("success", f"{ICONS['success']} Struktur dataset valid dan siap digunakan"))
        else:
            with ui_components['status']:
                display(create_status_indicator("warning", 
                    f"{ICONS['warning']} Struktur dataset belum lengkap: {', '.join(missing_parts)} tidak ditemukan"))
    
    # Tambahkan handler download asli ke UI components
    ui_components['on_download_click'] = on_download_click
    
    # Tambahkan fungsi validasi ke UI components
    ui_components['validate_dataset_structure'] = validate_dataset_structure
    
    # Setup handler konfirmasi
    try:
        ui_components = setup_confirmation_handlers(ui_components, env, config)
    except Exception as e:
        # Jika gagal setup confirmation handler, gunakan handler langsung
        ui_components['download_button'].on_click(on_download_click)
        if 'logger' in ui_components:
            ui_components['logger'].warning(f"⚠️ Gagal setup dialog konfirmasi: {str(e)}")
    
    return ui_components