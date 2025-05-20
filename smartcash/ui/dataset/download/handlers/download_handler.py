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
        # Call download_from_roboflow directly for test compatibility
        try:
            self.ui_components['download_running'] = True
            self.dataset_manager.download_from_roboflow(
                workspace=self.ui_components['workspace'].value,
                project=self.ui_components['project'].value,
                version=self.ui_components['version'].value,
                api_key=self.ui_components['api_key'].value,
                output_dir=self.ui_components['output_dir'].value,
                validate_dataset=self.ui_components['validate_dataset'].value,
                backup_before_download=self.ui_components['backup_checkbox'].value,
                backup_dir=self.ui_components['backup_dir'].value
            )
            self.ui_components['download_running'] = False
        except Exception as e:
            self.ui_components['download_running'] = False
            raise
    
    def handle_button_click(self, button: Any) -> None:
        """
        Handler untuk tombol download.
        
        Args:
            button: Button widget
        """
        handle_download_button_click(button, self.ui_components)

def handle_download_button_click(b: Any, ui_components: Dict[str, Any]) -> None:
    """
    Handler untuk tombol download pada UI download.
    
    Args:
        b: Button widget
        ui_components: Dictionary komponen UI
    """
    # Import sistem notifikasi baru
    from smartcash.ui.dataset.download.utils.notification_manager import notify_log
    from smartcash.ui.dataset.download.utils.ui_observers import register_ui_observers
    
    # Daftarkan observer untuk notifikasi UI
    observer_manager = register_ui_observers(ui_components)
    
    # Reset log output saat tombol diklik
    if 'log_output' in ui_components and hasattr(ui_components['log_output'], 'clear_output'):
        ui_components['log_output'].clear_output(wait=True)
    
    # Simpan konfigurasi UI sebelum download
    from smartcash.ui.dataset.download.handlers.config_handler import update_config_from_ui
    update_config_from_ui(ui_components)
    
    # Dapatkan nilai dari komponen UI
    workspace = ui_components.get('workspace', {}).value if 'workspace' in ui_components else ''
    project = ui_components.get('project', {}).value if 'project' in ui_components else ''
    version = ui_components.get('version', {}).value if 'version' in ui_components else ''
    api_key = ui_components.get('api_key', {}).value if 'api_key' in ui_components else ''
    output_dir = ui_components.get('output_dir', {}).value if 'output_dir' in ui_components else 'data'
    validate = ui_components.get('validate_dataset', {}).value if 'validate_dataset' in ui_components else True
    
    # Nonaktifkan tombol selama proses
    _disable_buttons(ui_components, True)
    
    try:
        # Reset progress bar terlebih dahulu
        _reset_progress_bar(ui_components)
        
        # Kirim notifikasi log bahwa proses download akan dimulai
        notify_log(
            sender=ui_components,
            message="Mempersiapkan proses download dataset...",
            level="info"
        )
        
        # Konfirmasi download
        from smartcash.ui.dataset.download.handlers.confirmation_handler import confirm_download, cancel_download
        # Sebelum menampilkan konfirmasi, persiapkan cancel_callback
        def cancel_callback():
            # Pastikan tombol diaktifkan kembali saat cancel
            _disable_buttons(ui_components, False)
            cancel_download(ui_components)
            
            # Kirim notifikasi log bahwa proses download dibatalkan
            notify_log(
                sender=ui_components,
                message="Proses download dibatalkan oleh pengguna",
                level="warning"
            )
            
        # Tetapkan callback ke ui_components agar bisa diakses di confirmation_handler
        ui_components['cancel_download_callback'] = cancel_callback
        
        # Tampilkan konfirmasi
        confirm_download(ui_components, 'Roboflow', b)
        
    except Exception as e:
        # Tampilkan error
        error_msg = f"Error saat persiapan download: {str(e)}"
        notify_log(
            sender=ui_components,
            message=error_msg,
            level="error"
        )
        
        # Aktifkan kembali tombol
        _disable_buttons(ui_components, False)

def execute_download(ui_components: Dict[str, Any], endpoint: str = 'Roboflow') -> None:
    """
    Eksekusi proses download dataset dari Roboflow.
    
    Args:
        ui_components: Dictionary komponen UI
        endpoint: Parameter dipertahankan untuk kompatibilitas, selalu 'Roboflow'
    """
    # Import sistem notifikasi baru
    from smartcash.ui.dataset.download.utils.notification_manager import notify_log, notify_progress
    
    # Pastikan observer sudah terdaftar
    from smartcash.ui.dataset.download.utils.ui_observers import register_ui_observers
    observer_manager = register_ui_observers(ui_components)
    
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
            
            # Jalankan download di thread terpisah agar UI tetap responsif
            with ThreadPoolExecutor(max_workers=1) as executor:
                executor.submit(_download_from_roboflow, ui_components)
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

def _download_from_roboflow(ui_components: Dict[str, Any]) -> None:
    """
    Download dataset dari Roboflow menggunakan dataset manager.
    
    Args:
        ui_components: Dictionary komponen UI
    """
    # Import sistem notifikasi baru
    from smartcash.ui.dataset.download.utils.notification_manager import notify_log, notify_progress
    from smartcash.ui.dataset.download.utils.ui_observers import register_ui_observers
    
    # Ambil logger dari ui_components
    logger = ui_components.get('download_logger') or ui_components.get('logger')
    
    # Daftarkan observer untuk notifikasi UI
    observer_manager = register_ui_observers(ui_components)
    
    # Tampilkan progress container
    if 'progress_container' in ui_components:
        ui_components['progress_container'].layout.display = 'block'
    
    # Kirim notifikasi progress dimulai
    notify_progress(
        sender=ui_components,
        event_type="start",
        message="Mempersiapkan download dataset...",
        step=1,
        total_steps=5
    )
    
    # Ambil konfigurasi dari endpoint handler
    from smartcash.ui.dataset.download.handlers.endpoint_handler import get_endpoint_config
    endpoint_config = get_endpoint_config(ui_components)
    
    # Validasi konfigurasi
    if not endpoint_config.get('api_key'):
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
        return
    
    # Pastikan output_dir valid dan ada
    output_dir = endpoint_config.get('output_dir', 'data')
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
            output_dir = alt_output_dir
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
            return
            
        # Pastikan backup_dir juga ada jika backup diaktifkan
        if config.get('backup_before_download', False):
            if not config.get('backup_dir'):
                config['backup_dir'] = os.path.join(os.path.dirname(config['output_dir']), 'downloads_backup')
            try:
                os.makedirs(config['backup_dir'], exist_ok=True)
                if logger:
                    logger.debug(f"📁 Direktori backup dibuat/ditemukan: {config['backup_dir']}")
            except Exception as e:
                if logger:
                    logger.warning(f"⚠️ Gagal membuat direktori backup, menonaktifkan backup: {str(e)}")
                config['backup_before_download'] = False
        
        # Pastikan parameter verify_integrity diset dengan benar
        if 'validate' in config and 'verify_integrity' not in config:
            config['verify_integrity'] = config.pop('validate')
        
        # Log parameter yang akan dikirim ke dataset_manager
        if logger:
            logger.debug(f"📤 Parameter download_from_roboflow: workspace={config.get('workspace')}, project={config.get('project')}, version={config.get('version')}, output_dir={config.get('output_dir')}, verify_integrity={config.get('verify_integrity', True)}")
            
        # Eksekusi download dengan penanganan error yang lebih baik
        try:
            result = self.dataset_manager.download_from_roboflow(
                api_key=config.get('api_key'),
                workspace=config.get('workspace'),
                project=config.get('project'),
                version=config.get('version'),
                format=config.get('format', 'yolov5pytorch'),
                output_dir=config.get('output_dir'),
                verify_integrity=config.get('verify_integrity', True)
            )
            
            # Proses hasil download
            _process_download_result(ui_components, result)
            
        except Exception as e:
            # Tangani error spesifik dari download_from_roboflow
            error_msg = f"Error saat proses download dataset: {str(e)}"
            if logger: logger.error(f"❌ {error_msg}")
            
            # Update UI dengan error
            from smartcash.ui.utils.ui_logger import log_to_ui
            log_to_ui(ui_components, error_msg, "error", "❌")
            
            # Reset UI
            _reset_ui_after_download(ui_components)
        
    except Exception as e:
        # Tangani error umum
        error_msg = f"Error saat download dataset: {str(e)}"
        if logger: logger.error(f"❌ {error_msg}")
        
        # Update UI dengan error
        from smartcash.ui.utils.ui_logger import log_to_ui
        log_to_ui(ui_components, error_msg, "error", "❌")
        
        # Reset UI
        _reset_ui_after_download(ui_components)

def _process_download_result(ui_components: Dict[str, Any], result: Dict[str, Any]) -> None:
    """
    Proses hasil download dan update UI.
    
    Args:
        ui_components: Dictionary komponen UI
        result: Hasil download dari dataset service
    """
    logger = ui_components.get('download_logger') or ui_components.get('logger')
    
    # Cek apakah result adalah None atau tidak valid
    if result is None:
        error_msg = "Hasil download tidak valid (None)"
        if logger: logger.error(f"❌ {error_msg}")
        
        # Update UI dengan error
        from smartcash.ui.utils.ui_logger import log_to_ui
        log_to_ui(ui_components, error_msg, "error", "❌")
        
        # Reset UI
        _reset_ui_after_download(ui_components)
        return
    
    # Ekstrak informasi dari result dengan penanganan yang lebih baik
    success = result.get('success', False) or result.get('status') == 'success'
    message = result.get('message', '')
    
    # Coba dapatkan dataset_info dari berbagai kemungkinan lokasi
    dataset_info = result.get('dataset_info', {})
    if not dataset_info and 'stats' in result:
                    from IPython.display import display, HTML
                    import ipywidgets as widgets
                    
                    # Header summary
                    display(HTML(f"<h3 style='margin-top:0'>Ringkasan Dataset</h3>"))
                    
                    # Info dataset
                    info_items = [
                        ("Sumber", result.get('source', 'Roboflow')),
                        ("Workspace", result.get('workspace', '-')),
                        ("Project", result.get('project', '-')),
                        ("Version", result.get('version', '-')),
                        ("Format", result.get('format', 'YOLOv5')),
                        ("Lokasi", result.get('output_dir', '-'))
                    ]
                    
                    # Buat grid layout untuk info
                    grid = widgets.GridspecLayout(len(info_items), 2, width='100%')
                    
                    for i, (label, value) in enumerate(info_items):
                        grid[i, 0] = widgets.HTML(f"<b>{label}:</b>")
                        grid[i, 1] = widgets.HTML(f"{value}")
                    
                    display(grid)
                    
                    # Statistik dataset jika tersedia
                    if 'stats' in result:
                        stats = result['stats']
                        display(HTML(f"<h4 style='margin-top:15px'>Statistik Dataset</h4>"))
                        
                        # Buat grid untuk statistik
                        stats_items = [(k, v) for k, v in stats.items()]
                        stats_grid = widgets.GridspecLayout(len(stats_items), 2, width='100%')
                        
                        for i, (label, value) in enumerate(stats_items):
                            stats_grid[i, 0] = widgets.HTML(f"<b>{label}:</b>")
                            stats_grid[i, 1] = widgets.HTML(f"{value}")
                        
                        display(stats_grid)
    else:
        # Download gagal
        notify_progress(
            sender=ui_components,
            event_type="error",
            message=f"Error: {message}"
        )
        
        # Log error
        notify_log(
            sender=ui_components,
            message=f"Download dataset gagal: {message}",
            level="error"
        )
    
    # Reset UI setelah proses selesai
    _reset_ui_after_download(ui_components)

def _reset_progress_bar(ui_components: Dict[str, Any]) -> None:
    """
    Reset progress bar ke nilai awal dengan layout yang konsisten.
    
    Args:
        ui_components: Dictionary komponen UI
    """
    # Pastikan progress bar ada dan dapat diakses
    if 'progress_bar' in ui_components and hasattr(ui_components['progress_bar'], 'value'):
        # Reset nilai progress bar ke 0
        ui_components['progress_bar'].value = 0
    
    # Reset label progress jika ada
    for label_key in ['overall_label', 'step_label', 'progress_message', 'current_progress']:
        if label_key in ui_components and hasattr(ui_components[label_key], 'value'):
            ui_components[label_key].value = ""
    
    # Jangan sembunyikan progress container di sini, biarkan _reset_ui_after_download yang menangani
    # Ini untuk mencegah flicker saat progress bar ditampilkan dan disembunyikan terlalu cepat
    
    # Reset progress tracker jika tersedia
    tracker_key = 'download_tracker'
    if tracker_key in ui_components:
        tracker = ui_components[tracker_key]
        if hasattr(tracker, 'reset'):
            tracker.reset()
        elif hasattr(tracker, 'close'):
            tracker.close()
            
    # Reset step tracker jika tersedia
    step_tracker_key = 'download_step_tracker'
    if step_tracker_key in ui_components:
        tracker = ui_components[step_tracker_key]
        if hasattr(tracker, 'reset'):
            tracker.reset()
        elif hasattr(tracker, 'close'):
            tracker.close()
            
    # Bersihkan area konfirmasi jika ada
    if 'confirmation_area' in ui_components and hasattr(ui_components['confirmation_area'], 'clear_output'):
        ui_components['confirmation_area'].clear_output()

def _show_progress(ui_components: Dict[str, Any], message: str = "") -> None:
    """
    Tampilkan dan reset progress bar dengan layout yang konsisten.
    
    Args:
        ui_components: Dictionary komponen UI
        message: Pesan progress awal
    """
    # Set flag download_running ke True
    ui_components['download_running'] = True
    # Import sistem notifikasi baru
    from smartcash.ui.dataset.download.utils.notification_manager import notify_progress

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

    # Tambahkan flag untuk menandai proses sedang berjalan
    ui_components['download_running'] = True

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
    # Import sistem notifikasi baru
    from smartcash.ui.dataset.download.utils.notification_manager import notify_progress
    
    # Pastikan progress container terlihat
    if 'progress_container' in ui_components and hasattr(ui_components['progress_container'], 'layout'):
        ui_components['progress_container'].layout.display = 'block'
        ui_components['progress_container'].layout.visibility = 'visible'
    
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
    # Import sistem notifikasi baru jika diperlukan
    try:
        from smartcash.ui.dataset.download.utils.notification_manager import notify_log
    except ImportError:
        pass
    
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