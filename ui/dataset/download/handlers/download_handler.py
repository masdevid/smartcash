"""
File: smartcash/ui/dataset/download/handlers/download_handler.py
Deskripsi: Handler untuk proses download dataset dengan dukungan observer dan delegasi ke service yang sesuai
"""

from typing import Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
import os
import time
from pathlib import Path
from smartcash.dataset.manager import DatasetManager
from smartcash.dataset.services.downloader.download_service import DownloadService
from smartcash.common.config import get_config_manager
from smartcash.ui.dataset.download.utils.notification_manager import notify_log, notify_progress

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
            notify_log(
                sender=self.ui_components,
                message=f"Menerima parameter download: workspace={params['workspace']}, project={params['project']}, version={params['version']}",
                level="info"
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
        print(f"Warning: Failed to reset progress bar: {str(e)}")

def handle_download_button_click(ui_components: Dict[str, Any], button: Any) -> None:
    """
    Handler untuk tombol download pada UI download.
    
    Args:
        ui_components: Dictionary komponen UI
        button: Button widget
    """
    try:
        # Disable tombol download jika button adalah widget
        if hasattr(button, 'disabled'):
            button.disabled = True
        
        # Log pesan persiapan jika ui_components adalah dict
        if isinstance(ui_components, dict) and 'log_output' in ui_components:
            ui_components['log_output'].append_stdout("Memulai persiapan download dataset...")
        
        # Tampilkan konfirmasi download
        from smartcash.ui.dataset.download.handlers.confirmation_handler import confirm_download
        if confirm_download(ui_components):
            # Reset progress bar setelah konfirmasi
            _reset_progress_bar(ui_components)
            
            # Ekstrak parameter dari UI jika ui_components adalah dict
            if isinstance(ui_components, dict):
                params = {
                    'workspace': ui_components['workspace'].value,
                    'project': ui_components['project'].value,
                    'version': ui_components['version'].value,
                    'api_key': ui_components['api_key'].value,
                    'output_dir': ui_components['output_dir'].value,
                    'backup_before_download': ui_components['backup_checkbox'].value if 'backup_checkbox' in ui_components else False,
                    'backup_dir': ui_components['backup_dir'].value if 'backup_dir' in ui_components else ''
                }
                
                # Log parameter yang akan digunakan
                if 'log_output' in ui_components:
                    ui_components['log_output'].append_stdout("Parameter download:")
                    for key, value in params.items():
                        if key == 'api_key':
                            masked_key = value[:4] + "****" if value and len(value) > 4 else "****"
                            ui_components['log_output'].append_stdout(f"- {key}: {masked_key}")
                        else:
                            ui_components['log_output'].append_stdout(f"- {key}: {value}")
            
            # Nonaktifkan tombol lain selama download
            _disable_buttons(ui_components, True)
            
            # Jalankan download dengan endpoint Roboflow
            execute_download(ui_components, 'Roboflow')
        else:
            if isinstance(ui_components, dict) and 'log_output' in ui_components:
                ui_components['log_output'].append_stdout("Download dibatalkan")
                
    except Exception as e:
        if isinstance(ui_components, dict) and 'log_output' in ui_components:
            ui_components['log_output'].append_stderr(f"Error saat persiapan download: {str(e)}")
    finally:
        # Enable kembali tombol download jika button adalah widget
        if hasattr(button, 'disabled'):
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
        import threading
        def hide_progress_container():
            import time
            # Tunggu 3 detik sebelum menyembunyikan progress bar
            time.sleep(3)
            ui_components['progress_container'].layout.display = 'none'
        
        # Jalankan di thread terpisah agar tidak memblokir UI
        threading.Thread(target=hide_progress_container).start()
    
    # Bersihkan area konfirmasi jika ada
    if 'confirmation_area' in ui_components and hasattr(ui_components['confirmation_area'], 'clear_output'):
        ui_components['confirmation_area'].clear_output()
    
    # Update status panel jika tersedia
    if 'status_panel' in ui_components:
        from smartcash.ui.utils.alert_utils import update_status_panel
        update_status_panel(ui_components['status_panel'], 'Download selesai', 'success')
    elif 'update_status_panel' in ui_components and callable(ui_components['update_status_panel']):
        ui_components['update_status_panel'](ui_components, 'info', 'Download selesai')
    
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
        # Jalankan download berdasarkan endpoint yang dipilih
        if endpoint.lower() == 'roboflow':
            # Tampilkan progress
            _show_progress(ui_components, "Memulai download dari Roboflow...")
            
            # Kirim notifikasi log
            notify_log(
                sender=ui_components,
                message=f"Memulai proses download dataset dari Roboflow",
                level="info"
            )
            
            # Pastikan observer terdaftar
            from smartcash.ui.dataset.download.utils.ui_observers import register_ui_observers
            observer_manager = register_ui_observers(ui_components)
            
            # Jalankan download di thread terpisah agar UI tetap responsif
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_download_from_roboflow, ui_components)
                # Tunggu hasil agar thread tidak hilang
                try:
                    result = future.result()
                    # Proses hasil jika ada
                    if result:
                        _process_download_result(ui_components, result)
                except Exception as e:
                    # Tangani error
                    notify_log(
                        sender=ui_components,
                        message=f"Error saat proses download dataset: {str(e)}",
                        level="error"
                    )
                    
                    notify_progress(
                        sender=ui_components,
                        event_type="error",
                        message=f"Error: {str(e)}"
                    )
                    
                    # Reset UI
                    _reset_ui_after_download(ui_components)
        else:
            # Endpoint tidak didukung
            notify_log(
                sender=ui_components,
                message=f"Endpoint '{endpoint}' tidak didukung",
                level="error"
            )
            
            notify_progress(
                sender=ui_components,
                event_type="error",
                message=f"Endpoint tidak didukung: {endpoint}"
            )
            
            # Reset UI
            _reset_ui_after_download(ui_components)
    except Exception as e:
        # Tampilkan error
        notify_log(
            sender=ui_components,
            message=f"Error saat eksekusi download: {str(e)}",
            level="error"
        )
        
        notify_progress(
            sender=ui_components,
            event_type="error",
            message=f"Error: {str(e)}"
        )
        
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
    
    # Kirim notifikasi progress dimulai
    notify_progress(
        sender=ui_components,
        event_type="start",
        progress=0,
        total=100,
        message="Mempersiapkan download dataset...",
        step=1,
        total_steps=5
    )
    
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
        notify_log(
            sender=ui_components,
            message="API Key tidak ditemukan. Mohon masukkan API Key Roboflow.",
            level="error"
        )
        notify_progress(
            sender=ui_components,
            event_type="error",
            message="API Key tidak ditemukan"
        )
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
        
        notify_log(
            sender=ui_components,
            message=f"Direktori output dibuat/ditemukan: {output_dir}",
            level="info"
        )
    except Exception as e:
        error_msg = f"Gagal membuat direktori output: {str(e)}"
        notify_log(
            sender=ui_components,
            message=error_msg,
            level="error"
        )
        # Coba gunakan direktori alternatif
        alt_output_dir = os.path.join(os.path.expanduser('~'), 'smartcash_downloads')
        try:
            os.makedirs(alt_output_dir, exist_ok=True)
            params['output_dir'] = alt_output_dir
            notify_log(
                sender=ui_components,
                message=f"Menggunakan direktori output alternatif: {alt_output_dir}",
                level="warning"
            )
        except Exception as e2:
            notify_log(
                sender=ui_components,
                message=f"Gagal membuat direktori alternatif: {str(e2)}",
                level="error"
            )
            notify_progress(
                sender=ui_components,
                event_type="error",
                message="Error direktori output"
            )
            _reset_ui_after_download(ui_components)
            return {"status": "error", "message": f"Gagal membuat direktori output: {str(e2)}"}
    
    # Jalankan download
    try:
        # Update progress
        notify_progress(
            sender=ui_components,
            event_type="update",
            progress=10,
            total=100,
            message="Memulai download dari Roboflow...",
            step=2,
            total_steps=5
        )
        
        dataset_manager = DatasetManager()
        
        # Register observer jika ada
        from smartcash.ui.dataset.download.utils.ui_observers import register_ui_observers
        observer_manager = register_ui_observers(ui_components)
        
        # Coba dapatkan service downloader dan set observer
        try:
            downloader_service = dataset_manager.get_service('downloader')
            downloader_service.set_observer_manager(observer_manager)
        except Exception as e:
            notify_log(
                sender=ui_components,
                message=f"Warning: Gagal mengatur observer untuk downloader: {str(e)}",
                level="warning"
            )
        
        # Update progress lagi
        notify_progress(
            sender=ui_components,
            event_type="update",
            progress=20,
            total=100,
            message="Mendownload dataset dari Roboflow...",
            step=3,
            total_steps=5
        )
        
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
                
            notify_log(
                sender=ui_components,
                message=f"Mendownload dataset dengan parameter: {', '.join(valid_params.keys())}",
                level="info"
            )
            
            # Jalankan download
            result = dataset_manager.download_from_roboflow(**valid_params)
        except Exception as e:
            # Fallback ke parameter minimal
            notify_log(
                sender=ui_components,
                message=f"Mencoba download dengan parameter minimal: {str(e)}",
                level="warning"
            )
            result = dataset_manager.download_from_roboflow(
                api_key=params['api_key'],
                workspace=params['workspace'],
                project=params['project'],
                version=params['version'],
                output_dir=params['output_dir']
            )
        
        # Update progress
        notify_progress(
            sender=ui_components,
            event_type="update",
            progress=90,
            total=100,
            message="Download selesai, memproses hasil...",
            step=5,
            total_steps=5
        )
        
        return result
        
    except Exception as e:
        # Tangani error
        error_msg = f"Error saat proses download dataset: {str(e)}"
        notify_log(
            sender=ui_components,
            message=error_msg,
            level="error"
        )
        
        notify_progress(
            sender=ui_components,
            event_type="error",
            message=f"Error: {str(e)}"
        )
        
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
        notify_log(
            sender=ui_components,
            message=error_msg,
            level="error"
        )
        
        notify_progress(
            sender=ui_components,
            event_type="error",
            message=error_msg
        )
        
        # Reset UI
        _reset_ui_after_download(ui_components)
        return
    
    # Ekstrak informasi dari result
    success = result.get('success', False) or result.get('status') == 'success'
    message = result.get('message', '')
    
    if success:
        # Download berhasil
        notify_log(
            sender=ui_components,
            message=f"Download dataset berhasil: {message}",
            level="success"
        )
        
        notify_progress(
            sender=ui_components,
            event_type="complete",
            message="Download selesai"
        )
        
        # Simpan konfigurasi setelah download berhasil
        from smartcash.ui.dataset.download.handlers.config_handler import update_config_from_ui
        update_config_from_ui(ui_components)
    else:
        # Download gagal
        notify_log(
            sender=ui_components,
            message=f"Download dataset gagal: {message}",
            level="error"
        )
        
        notify_progress(
            sender=ui_components,
            event_type="error",
            message=f"Error: {message}"
        )
    
    # Reset UI setelah proses selesai
    _reset_ui_after_download(ui_components)