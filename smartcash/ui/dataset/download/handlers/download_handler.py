"""
File: smartcash/ui/dataset/download/handlers/download_handler.py
Deskripsi: Handler untuk proses download dataset dengan dukungan observer dan delegasi ke service yang sesuai
"""

import os
import time
import datetime
from typing import Dict, Any, Optional, List, Tuple, Callable
from pathlib import Path
from IPython.display import display

from smartcash.dataset.manager import DatasetManager
from smartcash.dataset.services.downloader.download_service import DownloadService
from smartcash.common.config import get_config_manager
from smartcash.ui.dataset.download.utils.notification_manager import notify_log, notify_progress
from smartcash.ui.dataset.download.utils.ui_observers import register_ui_observers
from smartcash.ui.dataset.download.handlers.confirmation_handler import confirm_download
from smartcash.ui.dataset.download.utils.logger_helper import log_message, setup_ui_logger

__all__ = [
    'handle_download_button_click',
    'execute_download',
    '_reset_progress_bar',
    '_show_progress',
    '_update_progress',
    '_reset_ui_after_download',
    '_disable_buttons'
]

class DownloadHandler:
    """Handler untuk proses download dataset."""
    
    def __init__(self, ui_components: Dict[str, Any]):
        """
        Inisialisasi DownloadHandler.
        
        Args:
            ui_components: Dictionary komponen UI
        """
        self.ui_components = ui_components
        self.dataset_manager = DatasetManager()
    
    def download(self) -> None:
        """Eksekusi proses download dataset."""
        try:
            self.ui_components['download_running'] = True
            
            # Ambil parameter dari UI
            params = {
                'workspace': self.ui_components['workspace'].value,
                'project': self.ui_components['project'].value,
                'version': self.ui_components['version'].value,
                'api_key': self.ui_components['api_key'].value,
                'output_dir': self.ui_components['output_dir'].value,
                'validate_dataset': self.ui_components['validate_dataset'].value,
                'backup_before_download': self.ui_components['backup_checkbox'].value,
                'backup_dir': self.ui_components['backup_dir'].value
            }
            
            # Notifikasi parameter yang diterima
            log_message(
                self.ui_components,
                f"Menerima parameter download: workspace={params['workspace']}, project={params['project']}, version={params['version']}",
                "info",
                "ℹ️"
            )
            
            # Jalankan download dengan parameter
            self.dataset_manager.download_from_roboflow(**params)
            
            self.ui_components['download_running'] = False
        except Exception as e:
            self.ui_components['download_running'] = False
            raise

def _reset_progress_bar(ui_components: Dict[str, Any]) -> None:
    """
    Reset progress bar ke kondisi awal.
    
    Args:
        ui_components: Dictionary komponen UI
    """
    try:
        # Reset progress bar
        if 'progress_bar' in ui_components and ui_components['progress_bar'] is not None:
            ui_components['progress_bar'].value = 0
            ui_components['progress_bar'].description = "Progress: 0%"
            ui_components['progress_bar'].layout.visibility = 'hidden'
        
        # Reset labels
        if 'overall_label' in ui_components and ui_components['overall_label'] is not None:
            ui_components['overall_label'].value = ""
            ui_components['overall_label'].layout.visibility = 'hidden'
            
        if 'step_label' in ui_components and ui_components['step_label'] is not None:
            ui_components['step_label'].value = ""
            ui_components['step_label'].layout.visibility = 'hidden'
            
        if 'current_progress' in ui_components and ui_components['current_progress'] is not None:
            ui_components['current_progress'].value = 0
            ui_components['current_progress'].description = "Step 0/0"
            ui_components['current_progress'].layout.visibility = 'hidden'
    except Exception as e:
        log_message(ui_components, f"Gagal mereset progress bar: {str(e)}", "warning", "⚠️")

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
        if hasattr(button, 'disabled'):
            button.disabled = True
        
        # Reset log output saat tombol diklik
        if 'log_output' in ui_components and hasattr(ui_components['log_output'], 'clear_output'):
            ui_components['log_output'].clear_output(wait=True)
        
        # Log pesan persiapan
        log_message(ui_components, "Memulai persiapan download dataset...", "info", "🚀")
        
        # Pastikan kita memiliki UI area untuk konfirmasi
        if 'confirmation_area' not in ui_components:
            from ipywidgets import Output
            ui_components['confirmation_area'] = Output()
            log_message(ui_components, "Area konfirmasi dibuat otomatis", "info", "ℹ️")
            
            # Tambahkan ke UI jika ada area untuk itu
            if 'ui' in ui_components and hasattr(ui_components['ui'], 'children'):
                try:
                    # Coba tambahkan ke UI container (bukan UI ideal, tapi berfungsi sebagai fallback)
                    children = list(ui_components['ui'].children)
                    children.append(ui_components['confirmation_area'])
                    ui_components['ui'].children = tuple(children)
                except Exception as e:
                    log_message(ui_components, f"Tidak bisa menambahkan area konfirmasi ke UI: {str(e)}", "warning", "⚠️")
        
        # Import modul yang diperlukan
        from smartcash.ui.dataset.download.handlers.confirmation_handler import confirm_download
        
        # Tampilkan konfirmasi download (menunggu konfirmasi dengan callback)
        confirm_result = confirm_download(ui_components)
        
        # Jalankan proses hanya jika pengguna mengkonfirmasi
        if confirm_result:
            # Reset progress bar setelah konfirmasi
            _reset_progress_bar(ui_components)
            
            # Ekstrak parameter dari UI
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
            _disable_buttons(ui_components, True)
            
            # Jalankan download dengan endpoint Roboflow
            execute_download(ui_components, 'Roboflow')
        else:
            # Log pembatalan download
            log_message(ui_components, "Download dibatalkan oleh pengguna", "info", "❌")
            
            # Aktifkan kembali tombol download
            if hasattr(button, 'disabled'):
                button.disabled = False
            
    except Exception as e:
        # Log error
        log_message(ui_components, f"Error saat persiapan download: {str(e)}", "error", "❌")
    finally:
        # Enable kembali tombol download jika button adalah widget dan belum diaktifkan
        if hasattr(button, 'disabled') and button.disabled:
            button.disabled = False

def _show_progress(ui_components: Dict[str, Any], message: str = "") -> None:
    """
    Tampilkan dan reset progress bar dengan layout yang konsisten.
    
    Args:
        ui_components: Dictionary komponen UI
        message: Pesan progress awal
    """
    # Set flag download_running ke True
    ui_components['download_running'] = True

    # Reset progress bar terlebih dahulu
    _reset_progress_bar(ui_components)

    # Pastikan progress container terlihat
    if 'progress_container' in ui_components and hasattr(ui_components['progress_container'], 'layout'):
        ui_components['progress_container'].layout.display = 'block'
        ui_components['progress_container'].layout.visibility = 'visible'

    # Kirim notifikasi progress dimulai
    notify_progress(
        sender=ui_components,
        event_type="start",
        progress=0,
        total=100,
        message=message or "Memulai proses...",
        step=1,
        total_steps=5
    )
    
    # Log message dengan logger helper
    log_message(ui_components, message or "Memulai proses download...", "info", "🚀")

    # Pastikan log accordion terbuka
    if 'log_accordion' in ui_components and hasattr(ui_components['log_accordion'], 'selected_index'):
        ui_components['log_accordion'].selected_index = 0  # Buka accordion pertama

def _update_progress(ui_components: Dict[str, Any], value: int, message: Optional[str] = None) -> None:
    """
    Update progress bar dengan layout yang konsisten.
    
    Args:
        ui_components: Dictionary komponen UI
        value: Nilai progress (0-100)
        message: Pesan progress opsional
    """
    # Pastikan progress container terlihat
    if 'progress_container' in ui_components and hasattr(ui_components['progress_container'], 'layout'):
        ui_components['progress_container'].layout.display = 'block'
        ui_components['progress_container'].layout.visibility = 'visible'
    
    # Pastikan value adalah integer
    try:
        value = int(float(value))
    except (ValueError, TypeError):
        value = 0
    
    # Kirim notifikasi progress update
    notify_progress(
        sender=ui_components,
        event_type="update",
        progress=value,
        total=100,
        message=message or "",
        step=min(5, max(1, int(value / 20))),  # Estimasi langkah berdasarkan persentase
        total_steps=5
    )
    
    # Log progress message jika ada pesan
    if message:
        log_message(ui_components, f"Progress {value}%: {message}", "debug", "⏳")

def _disable_buttons(ui_components: Dict[str, Any], disabled: bool) -> None:
    """
    Nonaktifkan/aktifkan tombol-tombol UI.
    
    Args:
        ui_components: Dictionary komponen UI
        disabled: True untuk nonaktifkan, False untuk aktifkan
    """
    # Daftar tombol yang perlu dinonaktifkan
    button_keys = ['download_button', 'check_button', 'reset_button', 'cleanup_button']
    
    # Set status disabled untuk semua tombol
    for key in button_keys:
        if key in ui_components and hasattr(ui_components[key], 'disabled'):
            ui_components[key].disabled = disabled
            
            # Atur visibilitas tombol jika disabled
            if hasattr(ui_components[key], 'layout'):
                if disabled:
                    # Sembunyikan tombol reset dan cleanup saat proses berjalan
                    if key in ['reset_button', 'cleanup_button']:
                        ui_components[key].layout.display = 'none'
                else:
                    # Tampilkan kembali semua tombol dengan konsisten
                    ui_components[key].layout.display = 'inline-block'

def _reset_ui_after_download(ui_components: Dict[str, Any]) -> None:
    """Reset UI setelah proses download selesai."""
    # Aktifkan kembali tombol (fungsi ini juga akan mengatur display='inline-block')
    _disable_buttons(ui_components, False)
    
    # Reset progress bar
    _reset_progress_bar(ui_components)
    
    # Sembunyikan progress container setelah beberapa detik
    if 'progress_container' in ui_components and hasattr(ui_components['progress_container'], 'layout'):
        # Setelah proses selesai, biarkan progress bar terlihat sebentar
        # kemudian sembunyikan setelah beberapa detik
        time.sleep(3)
        ui_components['progress_container'].layout.display = 'none'
    
    # Bersihkan area konfirmasi jika ada
    if 'confirmation_area' in ui_components and hasattr(ui_components['confirmation_area'], 'clear_output'):
        ui_components['confirmation_area'].clear_output()
    
    # Update status panel jika tersedia
    if 'status_panel' in ui_components:
        from smartcash.ui.utils.alert_utils import update_status_panel
        update_status_panel(ui_components['status_panel'], 'Download selesai', 'success')
    elif 'update_status_panel' in ui_components and callable(ui_components['update_status_panel']):
        ui_components['update_status_panel'](ui_components, 'success', '✅ Download selesai')
    
    # Log message dengan logger helper
    log_message(ui_components, "Proses download telah selesai", "info", "✅")
    
    # Pastikan log accordion tetap terbuka
    if 'log_accordion' in ui_components and hasattr(ui_components['log_accordion'], 'selected_index'):
        ui_components['log_accordion'].selected_index = 0  # Buka accordion pertama
    
    # Cleanup UI jika tersedia
    if 'cleanup_ui' in ui_components and callable(ui_components['cleanup_ui']):
        ui_components['cleanup_ui'](ui_components)
    elif 'cleanup' in ui_components and callable(ui_components['cleanup']):
        ui_components['cleanup']()
        
    # Set flag download_running ke False jika ada
    ui_components['download_running'] = False

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
            # Tampilkan progress
            _show_progress(ui_components, "Memulai download dari Roboflow...")
            
            # Log message dengan logger helper
            log_message(ui_components, "Memulai proses download dataset dari Roboflow", "info", "🚀")
            
            # Pastikan observer terdaftar
            observer_manager = register_ui_observers(ui_components)
            
            # Jalankan download
            result = _download_from_roboflow(ui_components)
            
            # Proses hasil jika ada
            if result:
                _process_download_result(ui_components, result)
        else:
            # Endpoint tidak didukung
            log_message(ui_components, f"Endpoint '{endpoint}' tidak didukung", "error", "❌")
            
            # Reset UI
            _reset_ui_after_download(ui_components)
    except Exception as e:
        # Tampilkan error dengan logger helper
        log_message(ui_components, f"Error saat eksekusi download: {str(e)}", "error", "❌")
        
        # Reset UI
        _reset_ui_after_download(ui_components)

def _download_from_roboflow(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Download dataset dari Roboflow menggunakan dataset manager.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dict[str, Any]: Hasil download
    """
    # Tampilkan progress container
    if 'progress_container' in ui_components:
        ui_components['progress_container'].layout.display = 'block'
    
    # Kirim notifikasi progress dimulai dan log message
    _update_progress(ui_components, 0, "Mempersiapkan download dataset...")
    log_message(ui_components, "Mempersiapkan download dataset...", "info", "🔄")
    
    # Ambil parameter dari UI
    params = {
        'workspace': ui_components['workspace'].value,
        'project': ui_components['project'].value,
        'version': ui_components['version'].value,
        'api_key': ui_components['api_key'].value,
        'output_dir': ui_components['output_dir'].value
    }
    
    # Parameter tambahan jika tersedia
    if 'backup_checkbox' in ui_components and hasattr(ui_components['backup_checkbox'], 'value'):
        backup_existing = ui_components['backup_checkbox'].value
    else:
        backup_existing = False
    
    if 'backup_dir' in ui_components and hasattr(ui_components['backup_dir'], 'value'):
        backup_dir = ui_components['backup_dir'].value
    else:
        backup_dir = None
    
    # Validasi parameter
    if not params['api_key']:
        log_message(ui_components, "API Key tidak ditemukan. Mohon masukkan API Key Roboflow.", "error", "❌")
        _reset_ui_after_download(ui_components)
        return {"status": "error", "message": "API Key tidak ditemukan"}
    
    # Pastikan output_dir valid dan ada
    output_dir = params['output_dir']
    try:
        # Buat direktori jika belum ada
        os.makedirs(output_dir, exist_ok=True)
        
        # Periksa apakah direktori dapat ditulis
        test_file = os.path.join(output_dir, '.test_write')
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)
        
        log_message(ui_components, f"Direktori output dibuat/ditemukan: {output_dir}", "info", "📁")
    except Exception as e:
        error_msg = f"Gagal membuat direktori output: {str(e)}"
        log_message(ui_components, error_msg, "error", "❌")
        
        # Coba gunakan direktori alternatif
        alt_output_dir = os.path.join(os.path.expanduser('~'), 'smartcash_downloads')
        try:
            os.makedirs(alt_output_dir, exist_ok=True)
            params['output_dir'] = alt_output_dir
            log_message(ui_components, f"Menggunakan direktori output alternatif: {alt_output_dir}", "warning", "⚠️")
        except Exception as e2:
            log_message(ui_components, f"Gagal membuat direktori alternatif: {str(e2)}", "error", "❌")
            _reset_ui_after_download(ui_components)
            return {"status": "error", "message": f"Gagal membuat direktori output: {str(e2)}"}
    
    # Jalankan download
    try:
        # Update progress
        _update_progress(ui_components, 10, "Memulai download dari Roboflow...")
        
        dataset_manager = DatasetManager()
        
        # Register observer jika ada
        observer_manager = register_ui_observers(ui_components)
        
        # Coba dapatkan service downloader dan set observer
        try:
            downloader_service = dataset_manager.get_service('downloader')
            downloader_service.set_observer_manager(observer_manager)
        except Exception as e:
            log_message(ui_components, f"Gagal mengatur observer untuk downloader: {str(e)}", "warning", "⚠️")
        
        # Update progress lagi
        _update_progress(ui_components, 20, "Mendownload dataset dari Roboflow...")
        
        # Jalankan download dengan parameter yang sesuai
        # Periksa signature method
        import inspect
        try:
            signature = inspect.signature(dataset_manager.download_from_roboflow)
            valid_params = {}
            
            # Tambahkan parameter yang valid
            for param_name, param in signature.parameters.items():
                if param_name in params:
                    valid_params[param_name] = params[param_name]
            
            # Tambahkan parameter opsional jika didukung
            if 'backup_existing' in signature.parameters:
                valid_params['backup_existing'] = backup_existing
            
            if 'backup_dir' in signature.parameters and backup_dir:
                valid_params['backup_dir'] = backup_dir
                
            # Tambahkan parameter show_progress jika didukung
            if 'show_progress' in signature.parameters:
                valid_params['show_progress'] = True
                
            # Tambahkan parameter verify_integrity jika didukung
            if 'verify_integrity' in signature.parameters:
                valid_params['verify_integrity'] = True
                
            log_message(ui_components, f"Mendownload dataset dengan parameter: {', '.join(valid_params.keys())}", "info", "📥")
            
            # Jalankan download
            result = dataset_manager.download_from_roboflow(**valid_params)
        except Exception as e:
            # Fallback ke parameter minimal
            log_message(ui_components, f"Mencoba download dengan parameter minimal: {str(e)}", "warning", "⚠️")
            result = dataset_manager.download_from_roboflow(
                api_key=params['api_key'],
                workspace=params['workspace'],
                project=params['project'],
                version=params['version'],
                output_dir=params['output_dir']
            )
        
        # Update progress
        _update_progress(ui_components, 90, "Download selesai, memproses hasil...")
        
        return result
        
    except Exception as e:
        # Tangani error
        error_msg = f"Error saat proses download dataset: {str(e)}"
        log_message(ui_components, error_msg, "error", "❌")
        
        # Reset UI
        _reset_ui_after_download(ui_components)
        return {"status": "error", "message": error_msg}

def _process_download_result(ui_components: Dict[str, Any], result: Dict[str, Any]) -> None:
    """
    Proses hasil download dan update UI.
    
    Args:
        ui_components: Dictionary komponen UI
        result: Hasil download dari dataset service
    """
    # Cek apakah result adalah None atau tidak valid
    if result is None:
        error_msg = "Hasil download tidak valid (None)"
        log_message(ui_components, error_msg, "error", "❌")
        
        # Reset UI
        _reset_ui_after_download(ui_components)
        return
    
    # Ekstrak informasi dari result
    success = result.get('success', False) or result.get('status') == 'success'
    message = result.get('message', '')
    
    if success:
        # Download berhasil
        log_message(ui_components, f"Download dataset berhasil: {message}", "success", "✅")
        
        # Gunakan update_progress sebagai notifikasi
        _update_progress(ui_components, 100, "Download selesai")
        
        # Simpan konfigurasi setelah download berhasil
        try:
            from smartcash.ui.dataset.download.handlers.save_handler import handle_save_config
            handle_save_config(ui_components)
        except Exception as e:
            log_message(ui_components, f"Gagal menyimpan konfigurasi: {str(e)}", "warning", "⚠️")
    else:
        # Download gagal
        log_message(ui_components, f"Download dataset gagal: {message}", "error", "❌")
    
    # Reset UI setelah proses selesai
    _reset_ui_after_download(ui_components)