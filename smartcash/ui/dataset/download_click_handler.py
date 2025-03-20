"""
File: smartcash/ui/dataset/download_click_handler.py
Deskripsi: Handler untuk tombol download dataset dengan integrasi utils dan error handling standar
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, Callable
from IPython.display import display, clear_output, HTML
from smartcash.ui.utils.constants import ICONS
from smartcash.ui.utils.alert_utils import create_status_indicator

def setup_click_handlers(ui_components: Dict[str, Any], env=None, config=None) -> Dict[str, Any]:
    """
    Setup handler untuk klik tombol download dengan integrasi utils standar.
    
    Args:
        ui_components: Dictionary komponen UI
        env: Environment manager
        config: Konfigurasi aplikasi
        
    Returns:
        Dictionary UI components yang telah diupdate
    """
    # Setup handler untuk tombol download dengan penanganan error standar
    from smartcash.ui.handlers.error_handler import try_except_decorator
    
    @try_except_decorator(ui_components.get('status'))
    def on_download_click(b):
        """Handler tombol download dengan error handling standar."""
        with ui_components['status']:
            clear_output(wait=True)
            display(create_status_indicator("info", f"{ICONS['processing']} Memulai download dataset..."))
        
        # Reset progress bar dengan utils standar
        if 'progress_bar' in ui_components:
            ui_components['progress_bar'].value = 0
            ui_components['progress_bar'].description = 'Download: 0%'
            
        # Notify observer jika tersedia dengan utils standar
        try:
            from smartcash.components.observer.event_dispatcher_observer import EventDispatcher
            download_option = ui_components.get('download_options').value if ui_components.get('download_options') else None
            EventDispatcher.notify(event_type="DOWNLOAD_START", sender="download_handler", 
                                 message=f"Memulai download dataset dari {download_option}")
        except ImportError:
            pass
        
        # Execute download based on option dengan utils standar
        download_option = ui_components.get('download_options').value if ui_components.get('download_options') else None
        
        if download_option == 'Roboflow (Online)':
            # Import handler on-demand untuk mengurangi dependencies
            from smartcash.ui.dataset.roboflow_download_handler import download_from_roboflow
            download_from_roboflow(ui_components, env, config)
        elif download_option == 'Local Data (Upload)':
            from smartcash.ui.dataset.local_upload_handler import process_local_upload
            process_local_upload(ui_components, env, config)
    
    # Validasi dataset dengan utils standar
    def validate_dataset_structure(data_dir):
        """Validasi struktur dataset dasar dengan utils standar."""
        # Import utils untuk validasi file
        from smartcash.ui.utils.file_utils import directory_tree
        
        splits = ['train', 'valid', 'test']
        valid_structure = True
        missing_parts = []
        
        # Validasi struktur dengan utils standar
        for split in splits:
            split_dir = Path(data_dir) / split
            images_dir = split_dir / 'images'
            labels_dir = split_dir / 'labels'
            
            if not split_dir.exists(): missing_parts.append(f"{split}/"); valid_structure = False
            elif not images_dir.exists(): missing_parts.append(f"{split}/images/"); valid_structure = False
            elif not labels_dir.exists(): missing_parts.append(f"{split}/labels/"); valid_structure = False
        
        # Tampilkan status dengan utils standar
        with ui_components['status']:
            if valid_structure:
                display(create_status_indicator("success", 
                    f"{ICONS['success']} Struktur dataset valid dan siap digunakan"))
                
                # Tampilkan struktur direktori dengan tree
                tree_html = directory_tree(data_dir, max_depth=2)
                display(HTML(f"<div style='margin-top:15px'><h4>Struktur Dataset</h4>{tree_html}</div>"))
            else:
                display(create_status_indicator("warning", 
                    f"{ICONS['warning']} Struktur dataset belum lengkap: {', '.join(missing_parts)} tidak ditemukan"))
    
    # Setup dialog konfirmasi dengan komponen standar
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