"""
File: smartcash/ui/dataset/download/handlers/confirmation_handler.py
Deskripsi: Handler konfirmasi untuk download dataset
"""

from typing import Dict, Any, Callable, Optional, List
from IPython.display import display
from smartcash.ui.dataset.download.utils.logger_helper import log_message, setup_ui_logger
from smartcash.ui.dataset.download.utils.ui_state_manager import update_status_panel, enable_download_button, disable_buttons
from smartcash.ui.dataset.download.utils.progress_manager import reset_progress_bar
from smartcash.ui.dataset.download.handlers.download_executor import download_from_roboflow, process_download_result

def confirm_download(ui_components: Dict[str, Any], endpoint: str = 'Roboflow') -> None:
    """
    Tampilkan dialog konfirmasi untuk download dataset dari Roboflow.
    Menggunakan callback pattern yang kompatibel dengan Colab.
    
    Args:
        ui_components: Dictionary komponen UI
        endpoint: Endpoint download (default: 'Roboflow')
    """
    # Setup logger jika belum
    ui_components = setup_ui_logger(ui_components)
    
    # Import modul yang diperlukan
    from smartcash.ui.components.confirmation_dialog import create_confirmation_dialog
    
    # Log message sebelum konfirmasi
    log_message(ui_components, "Menunggu konfirmasi download dataset", "info", "⏳")
    
    output_dir = ui_components['output_dir'].value
    
    # Buat pesan konfirmasi berdasarkan endpoint
    message = f"Anda akan mengunduh dataset dalam format YOLO v5 ke direktori {output_dir}. "
    message += _get_endpoint_details(ui_components, endpoint)
    
    # Pastikan confirmation_area ada dan valid
    if 'confirmation_area' not in ui_components or not hasattr(ui_components['confirmation_area'], 'clear_output'):
        log_message(ui_components, "Area konfirmasi tidak tersedia", "error", "❌")
        return
    
    # Reset status konfirmasi
    ui_components['confirmation_result'] = False
    
    # Callback untuk tombol konfirmasi    
    def on_confirm(b):
        # Bersihkan area konfirmasi
        ui_components['confirmation_area'].clear_output()
        
        # Update status panel
        update_status_panel(
            ui_components,
            "Memulai proses download dataset",
            "info"
        )
        
        # Set hasil konfirmasi
        ui_components['confirmation_result'] = True
        
        # Log konfirmasi
        log_message(ui_components, "Konfirmasi download diterima", "info", "✅")
        
        # Lanjutkan dengan proses download
        _execute_download_after_confirm(ui_components)
    
    # Callback untuk tombol batal
    def on_cancel(b):
        # Bersihkan area konfirmasi
        ui_components['confirmation_area'].clear_output()
        
        # Update status panel
        update_status_panel(
            ui_components,
            "Download dibatalkan oleh pengguna",
            "info"
        )
        
        # Set hasil konfirmasi
        ui_components['confirmation_result'] = False
        
        # Log pembatalan
        log_message(ui_components, "Download dibatalkan oleh pengguna", "info", "❌")
        
        # Enable tombol download
        enable_download_button(ui_components)
    
    # Gunakan component dialog konfirmasi
    dialog = create_confirmation_dialog(
        title="Konfirmasi Download Dataset",
        message=message,
        on_confirm=on_confirm,
        on_cancel=on_cancel
    )
    
    # Tampilkan dialog
    ui_components['confirmation_area'].clear_output()
    with ui_components['confirmation_area']:
        display(dialog)
    
    # Update status panel
    update_status_panel(
        ui_components,
        "Silakan konfirmasi untuk melanjutkan download dataset",
        "warning"
    )

def _execute_download_after_confirm(ui_components: Dict[str, Any]) -> None:
    """
    Eksekusi download setelah konfirmasi diterima.
    
    Args:
        ui_components: Dictionary komponen UI
    """
    # Reset progress bar setelah konfirmasi
    reset_progress_bar(ui_components)
    
    # Kumpulkan parameter dari UI untuk logging
    params = {
        'workspace': ui_components['workspace'].value,
        'project': ui_components['project'].value,
        'version': ui_components['version'].value,
        'api_key': ui_components['api_key'].value,
        'output_dir': ui_components['output_dir'].value,
        'backup_before_download': ui_components.get('backup_checkbox', {}).value if 'backup_checkbox' in ui_components else False,
        'backup_dir': ui_components.get('backup_dir', {}).value if 'backup_dir' in ui_components else ''
    }
    
    # Log parameter yang akan digunakan
    log_message(ui_components, "Parameter download:", "info", "ℹ️")
    for key, value in params.items():
        if key == 'api_key':
            masked_key = value[:4] + "****" if value and len(value) > 4 else "****"
            log_message(ui_components, f"- {key}: {masked_key}", "debug", "🔑")
        else:
            log_message(ui_components, f"- {key}: {value}", "debug", "🔹")
    
    # Nonaktifkan tombol lain selama download
    disable_buttons(ui_components, True)
    
    # Jalankan download dari Roboflow
    result = download_from_roboflow(ui_components)
    
    # Proses hasil download
    process_download_result(ui_components, result)

def _get_endpoint_details(ui_components: Dict[str, Any], endpoint: str) -> str:
    """Dapatkan detail spesifik untuk Roboflow."""
    workspace = ui_components['workspace'].value
    project = ui_components['project'].value
    version = ui_components['version'].value
    return f"Dataset akan diunduh dari Roboflow (workspace: {workspace}, project: {project}, version: {version})."