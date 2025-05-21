"""
File: smartcash/ui/dataset/download/handlers/confirmation_handler.py
Deskripsi: Handler konfirmasi untuk download dataset
"""

from typing import Dict, Any, Callable, Optional, List
from IPython.display import display
from smartcash.ui.dataset.download.utils.logger_helper import log_message, setup_ui_logger

def confirm_download(ui_components: Dict[str, Any], endpoint: str = 'Roboflow') -> bool:
    """
    Tampilkan dialog konfirmasi untuk download dataset dari Roboflow.
    
    Args:
        ui_components: Dictionary komponen UI
        endpoint: Endpoint download (default: 'Roboflow')
        
    Returns:
        bool: True jika pengguna mengkonfirmasi, False jika dibatalkan
    """
    # Setup logger jika belum
    ui_components = setup_ui_logger(ui_components)
    
    # Import modul yang diperlukan
    from smartcash.ui.components.confirmation_dialog import create_confirmation_dialog
    from smartcash.ui.components.status_panel import update_status_panel
    
    # Log message sebelum konfirmasi
    log_message(ui_components, "Menunggu konfirmasi download dataset", "info", "⏳")
    
    output_dir = ui_components['output_dir'].value
    
    # Buat pesan konfirmasi berdasarkan endpoint
    message = f"Anda akan mengunduh dataset dalam format YOLO v5 ke direktori {output_dir}. "
    message += _get_endpoint_details(ui_components, endpoint)
    
    # Pastikan confirmation_area ada dan valid
    if 'confirmation_area' not in ui_components or not hasattr(ui_components['confirmation_area'], 'clear_output'):
        log_message(ui_components, "Area konfirmasi tidak tersedia", "error", "❌")
        return False
    
    # Tampilkan dialog konfirmasi dengan Promise/Future untuk mengembalikan hasil
    try:
        from ipywidgets import Button, HBox, VBox, HTML
        import threading
        
        result_event = threading.Event()
        result_value = [False]  # Gunakan list untuk bisa diubah dari dalam fungsi
        
        def on_confirm(b):
            result_value[0] = True
            result_event.set()
            if 'confirmation_area' in ui_components:
                ui_components['confirmation_area'].clear_output()
            log_message(ui_components, "Konfirmasi download diterima", "info", "✅")
            
        def on_cancel(b):
            result_value[0] = False
            result_event.set()
            if 'confirmation_area' in ui_components:
                ui_components['confirmation_area'].clear_output()
            log_message(ui_components, "Download dibatalkan oleh pengguna", "info", "❌")
            update_status_panel(
                ui_components['status_panel'],
                "Download dibatalkan",
                "info"
            )
        
        # Buat tombol konfirmasi dan pembatalan
        confirm_button = Button(description="Ya, Lanjutkan", button_style="success")
        confirm_button.on_click(on_confirm)
        
        cancel_button = Button(description="Batal", button_style="danger")
        cancel_button.on_click(on_cancel)
        
        # Buat dialog konfirmasi
        title = HTML(f"<h3>Konfirmasi Download Dataset</h3>")
        message_html = HTML(f"<p>{message}</p>")
        buttons = HBox([confirm_button, cancel_button])
        dialog = VBox([title, message_html, buttons])
        
        # Tampilkan dialog
        ui_components['confirmation_area'].clear_output()
        with ui_components['confirmation_area']:
            display(dialog)
        
        # Update status panel
        update_status_panel(
            ui_components['status_panel'],
            "Silakan konfirmasi untuk melanjutkan download dataset",
            "warning"
        )
        
        # Tunggu hasil dari pengguna (ini akan mem-block)
        result_event.wait(timeout=300)  # Timeout 5 menit
        return result_value[0]
        
    except Exception as e:
        log_message(ui_components, f"Error saat menampilkan konfirmasi: {str(e)}", "error", "❌")
        return False

def _get_endpoint_details(ui_components: Dict[str, Any], endpoint: str) -> str:
    """Dapatkan detail spesifik untuk Roboflow."""
    workspace = ui_components['workspace'].value
    project = ui_components['project'].value
    version = ui_components['version'].value
    return f"Dataset akan diunduh dari Roboflow (workspace: {workspace}, project: {project}, version: {version})."