"""
File: smartcash/ui/dataset/download/handlers/download_handler.py
Deskripsi: Handler untuk proses download dataset dengan dukungan observer dan delegasi ke service yang sesuai
"""

from typing import Dict, Any, Optional
from smartcash.ui.dataset.download.utils.logger_helper import log_message, setup_ui_logger
from smartcash.ui.dataset.download.utils.ui_state_manager import (
    enable_download_button,
    ensure_confirmation_area,
    update_status_panel
)
from smartcash.ui.dataset.download.handlers.confirmation_handler import confirm_download

__all__ = [
    'handle_download_button_click',
    'execute_download'
]

def handle_download_button_click(ui_components: Dict[str, Any], button: Any = None) -> None:
    """
    Handler untuk tombol download pada UI download.
    
    Args:
        ui_components: Dictionary komponen UI
        button: Button widget (opsional)
    """
    try:
        # Setup logger jika belum
        ui_components = setup_ui_logger(ui_components)
        
        # Disable tombol download jika button adalah widget
        if button and hasattr(button, 'disabled'):
            button.disabled = True
        else:
            # Coba nonaktifkan tombol dari ui_components
            if 'download_button' in ui_components and hasattr(ui_components['download_button'], 'disabled'):
                ui_components['download_button'].disabled = True
        
        # Reset log output saat tombol diklik
        if 'log_output' in ui_components and hasattr(ui_components['log_output'], 'clear_output'):
            ui_components['log_output'].clear_output(wait=True)
        
        # Log pesan persiapan
        log_message(ui_components, "Memulai persiapan download dataset...", "info", "🚀")
        
        # Pastikan kita memiliki UI area untuk konfirmasi
        ui_components = ensure_confirmation_area(ui_components)
            
        # Ekstrak parameter dari UI untuk validasi
        workspace = ui_components['workspace'].value
        project = ui_components['project'].value
        version = ui_components['version'].value
        api_key = ui_components['api_key'].value
        
        # Validasi parameter dasar
        if not workspace or not project or not version or not api_key:
            log_message(ui_components, "Parameter download tidak lengkap. Mohon isi workspace, project, version, dan API key.", "error", "❌")
            
            # Update status panel
            update_status_panel(
                ui_components,
                "Parameter download tidak lengkap",
                "error"
            )
            
            # Aktifkan kembali tombol download
            enable_download_button(ui_components, button)
            return
        
        # Tampilkan dialog konfirmasi
        # Semua proses akan dilanjutkan melalui callback pada confirmation_handler.py
        confirm_download(ui_components)
            
    except Exception as e:
        # Log error
        log_message(ui_components, f"Error saat persiapan download: {str(e)}", "error", "❌")
        
        # Aktifkan kembali tombol download
        enable_download_button(ui_components, button)

def execute_download(ui_components: Dict[str, Any], endpoint: str = 'Roboflow') -> None:
    """
    Eksekusi proses download dataset dari Roboflow.
    
    Args:
        ui_components: Dictionary komponen UI
        endpoint: Parameter dipertahankan untuk kompatibilitas, selalu 'Roboflow'
    """
    try:
        # Setup logger jika belum
        ui_components = setup_ui_logger(ui_components)
        
        # Jalankan download berdasarkan endpoint yang dipilih
        if endpoint.lower() == 'roboflow':
            # Import modul dibutuhkan untuk menjalankan proses download
            from smartcash.ui.dataset.download.utils.progress_manager import show_progress
            from smartcash.ui.dataset.download.handlers.download_executor import download_from_roboflow, process_download_result
            
            # Tampilkan progress
            show_progress(ui_components, "Memulai download dari Roboflow...")
            
            # Log message dengan logger helper
            log_message(ui_components, "Memulai proses download dataset dari Roboflow", "info", "🚀")
            
            # Jalankan download
            result = download_from_roboflow(ui_components)
            
            # Proses hasil jika ada
            process_download_result(ui_components, result)
        else:
            # Endpoint tidak didukung
            log_message(ui_components, f"Endpoint '{endpoint}' tidak didukung", "error", "❌")
            
            # Reset UI
            from smartcash.ui.dataset.download.utils.ui_state_manager import reset_ui_after_download
            reset_ui_after_download(ui_components)
    except Exception as e:
        # Tampilkan error dengan logger helper
        log_message(ui_components, f"Error saat eksekusi download: {str(e)}", "error", "❌")
        
        # Reset UI
        from smartcash.ui.dataset.download.utils.ui_state_manager import reset_ui_after_download
        reset_ui_after_download(ui_components)