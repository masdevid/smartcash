"""
File: smartcash/ui/dataset/download_click_handler.py
Deskripsi: Handler untuk tombol download dataset dengan perbaikan validasi indeks array
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, Callable
from IPython.display import display, clear_output
from smartcash.ui.utils.constants import ICONS
from smartcash.ui.utils.alerts import create_status_indicator

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
    # Setup handler untuk tombol download dengan validasi callback
    def on_download_click(b):
        with ui_components['status']:
            clear_output(wait=True)
            display(create_status_indicator("info", f"{ICONS['processing']} Memulai download dataset..."))
        
        if 'progress_bar' in ui_components:
            ui_components['progress_bar'].value = 0
            ui_components['progress_bar'].description = 'Download: 0%'
            
        try:
            # Periksa dahulu apakah komponen UI ada sebelum mengaksesnya
            download_option = ui_components.get('download_options', None)
            if download_option is None:
                raise ValueError("Komponen download_options tidak ditemukan")
                
            download_option_value = download_option.value
            
            # Notify event if observer manager available
            if 'observer_manager' in ui_components:
                try:
                    from smartcash.components.observer.event_dispatcher_observer import EventDispatcher
                    EventDispatcher.notify(
                        event_type="DOWNLOAD_START",
                        sender="download_handler",
                        message=f"Memulai download dataset dari {download_option_value}"
                    )
                except ImportError:
                    pass
            
            # Execute download based on option
            if download_option_value == 'Roboflow (Online)':
                # Import handler on-demand untuk mengurangi dependencies
                from smartcash.ui.dataset.roboflow_download_handler import download_from_roboflow
                download_from_roboflow(ui_components, env, config)
            elif download_option_value == 'Local Data (Upload)':
                from smartcash.ui.dataset.local_upload_handler import process_local_upload
                process_local_upload(ui_components, env, config)
                
        except Exception as e:
            with ui_components['status']:
                display(create_status_indicator("error", f"{ICONS['error']} Error: {str(e)}"))
            
            # Log error
            if 'logger' in ui_components:
                ui_components['logger'].error(f"{ICONS['error']} Error saat download dataset: {str(e)}")
    
    # Validasi dataset
    def validate_dataset_structure(data_dir):
        """Validasi struktur dataset dasar tanpa fallback berlebihan."""
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
        
        with ui_components['status']:
            if valid_structure:
                display(create_status_indicator("success", 
                    f"{ICONS['success']} Struktur dataset valid dan siap digunakan"))
            else:
                display(create_status_indicator("warning", 
                    f"{ICONS['warning']} Struktur dataset belum lengkap: {', '.join(missing_parts)} tidak ditemukan"))
    
    # Setup dialog konfirmasi dengan validasi handler
    try:
        from smartcash.ui.dataset.download_confirmation_handler import setup_confirmation_handlers
        ui_components = setup_confirmation_handlers(ui_components, env, config)
    except ImportError as e:
        # Fallback ke handler langsung jika tidak ada confirmation handler
        if 'download_button' in ui_components:
            # Reset handlers terlebih dahulu untuk menghindari duplikasi
            if hasattr(ui_components['download_button'], '_click_handlers'):
                try:
                    ui_components['download_button']._click_handlers.callbacks.clear()
                except (AttributeError, IndexError):
                    pass
            # Register fresh handler
            ui_components['download_button'].on_click(on_download_click)
        
        if 'logger' in ui_components:
            ui_components['logger'].warning(f"{ICONS['warning']} Dialog konfirmasi tidak tersedia: {str(e)}")
    
    # Tambahkan fungsi ke ui_components
    ui_components['on_download_click'] = on_download_click
    ui_components['validate_dataset_structure'] = validate_dataset_structure
    
    return ui_components