"""
File: smartcash/ui/model/handlers/download_handlers.py
Deskripsi: Handler untuk tombol download dan sinkronisasi model pretrained
"""

from typing import Dict, Any, Callable, Optional
from pathlib import Path
import time
from IPython.display import display, HTML

from smartcash.ui.utils.constants import ICONS, COLORS
from smartcash.common.environment import EnvironmentManager
from smartcash.common.logger import get_logger

logger = get_logger(__name__)

def handle_download_sync_button(b, ui_components: Dict[str, Any]) -> None:
    """
    Handler untuk tombol download dan sinkronisasi model pretrained.
    
    Args:
        b: Button widget yang dipicu
        ui_components: Dictionary berisi komponen UI
    """
    # Pastikan status_panel tersedia
    status_panel = ui_components.get('status')
    if not status_panel:
        logger.warning(f"{ICONS.get('warning', '⚠️')} Status panel tidak tersedia")
        return
    
    # Tampilkan status processing
    status_panel.clear_output(wait=True)
    with status_panel:
        display(HTML(f"""<div style="padding:10px; background-color:{COLORS['alert_info_bg']}; 
                 color:{COLORS['alert_info_text']}; border-radius:4px; margin:5px 0;
                 border-left:4px solid {COLORS['alert_info_text']}">
            <p style="margin:5px 0">{ICONS.get('processing', '⏳')} Memeriksa dan memproses model pretrained...</p>
        </div>"""))
    
    # Nonaktifkan tombol selama proses berjalan
    b.disabled = True
    b.description = "Sedang Memproses..."
    
    # Jalankan proses download dan sync secara langsung
    try:
        process_download_sync(ui_components)
    finally:
        # Aktifkan kembali tombol setelah proses selesai
        b.disabled = False
        b.description = "Download & Sync Model"

# Fungsi enable_button tidak diperlukan lagi karena kita tidak menggunakan threading

def process_download_sync(ui_components: Dict[str, Any]) -> None:
    """
    Memproses download dan sinkronisasi model pretrained.
    
    Args:
        ui_components: Dictionary berisi komponen UI
        callback: Fungsi callback yang akan dipanggil setelah proses selesai
    """
    # Dapatkan komponen yang diperlukan
    status_panel = ui_components.get('status')
    log_output = ui_components.get('log')
    models_dir = ui_components.get('models_dir', '/content/models')
    drive_models_dir = ui_components.get('drive_models_dir', '/content/drive/MyDrive/SmartCash/models')
    
    # Bersihkan log output
    if log_output:
        log_output.clear_output(wait=True)
    
    # Fungsi untuk logging
    def log_message(message: str):
        timestamp = time.strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"
        if log_output:
            with log_output:
                from IPython.display import display, HTML
                display(HTML(f"<p>{formatted_message}</p>"))
        logger.info(message)
    
    try:
        # Cek environment
        env_manager = EnvironmentManager()
        in_colab = env_manager.is_colab()
        is_drive_mounted = env_manager.is_drive_mounted()
        
        # Jika di Colab tapi Drive belum ter-mount, coba mount
        if in_colab and not is_drive_mounted:
            log_message(f"{ICONS.get('sync', '🔄')} Mounting Google Drive...")
            success, message = env_manager.mount_drive()
            log_message(message)
            is_drive_mounted = env_manager.is_drive_mounted()
        
        # Cek apakah model sudah ada di Drive
        drive_models_exist = False
        if is_drive_mounted:
            drive_path = Path(drive_models_dir)
            if drive_path.exists():
                # Cek apakah file model ada di Drive
                yolo_path = drive_path / "yolov5s.pt"
                efficientnet_path = drive_path / "efficientnet-b4_notop.h5"
                
                if yolo_path.exists() and efficientnet_path.exists():
                    log_message(f"{ICONS.get('success', '✅')} Model ditemukan di Google Drive")
                    drive_models_exist = True
                    
                    # Sinkronkan dari Drive ke lokal
                    log_message(f"{ICONS.get('download', '📥')} Menyinkronkan model dari Drive ke lokal...")
                    from smartcash.ui.model.handlers.sync_handlers import sync_drive_to_local
                    sync_drive_to_local(models_dir, drive_models_dir, log_message)
                    
                    # Update status
                    if status_panel:
                        status_panel.clear_output(wait=True)
                        with status_panel:
                            from IPython.display import display, HTML
                            display(HTML(f"""<div style="padding:10px; background-color:{COLORS['alert_success_bg']}; 
                                    color:{COLORS['alert_success_text']}; border-radius:4px; margin:5px 0;
                                    border-left:4px solid {COLORS['alert_success_text']}">
                                <p style="margin:5px 0">{ICONS.get('success', '✅')} Model berhasil disinkronkan dari Drive</p>
                            </div>"""))
        
        # Jika model tidak ada di Drive, download dan sinkronkan
        if not drive_models_exist:
            log_message(f"{ICONS.get('download', '📥')} Model tidak ditemukan di Drive, memulai download...")
            
            # Download model
            from smartcash.ui.model.trained.setup import setup_pretrained_models
            model_info = setup_pretrained_models(models_dir, log_message)
            
            # Sinkronkan ke Drive jika tersedia
            if is_drive_mounted:
                log_message(f"{ICONS.get('sync', '🔄')} Menyinkronkan model ke Google Drive...")
                from smartcash.ui.model.handlers.sync_handlers import sync_local_to_drive
                sync_local_to_drive(models_dir, drive_models_dir, model_info, log_message)
            
            # Update status
            if status_panel:
                status_panel.clear_output(wait=True)
                with status_panel:
                    from IPython.display import display, HTML
                    display(HTML(f"""<div style="padding:10px; background-color:{COLORS['alert_success_bg']}; 
                            color:{COLORS['alert_success_text']}; border-radius:4px; margin:5px 0;
                            border-left:4px solid {COLORS['alert_success_text']}">
                        <p style="margin:5px 0">{ICONS.get('success', '✅')} Model berhasil didownload dan disinkronkan</p>
                    </div>"""))
    
    except Exception as e:
        log_message(f"{ICONS.get('error', '❌')} Error: {str(e)}")
        
        # Update status
        if status_panel:
            status_panel.clear_output(wait=True)
            with status_panel:
                from IPython.display import display, HTML
                display(HTML(f"""<div style="padding:10px; background-color:{COLORS['alert_danger_bg']}; 
                        color:{COLORS['alert_danger_text']}; border-radius:4px; margin:5px 0;
                        border-left:4px solid {COLORS['alert_danger_text']}">
                    <p style="margin:5px 0">{ICONS.get('error', '❌')} Error: {str(e)}</p>
                </div>"""))
    
    finally:
        # Tidak perlu callback karena tidak menggunakan threading
        pass
