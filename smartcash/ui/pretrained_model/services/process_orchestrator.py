"""
File: smartcash/ui/pretrained_model/services/process_orchestrator.py
Deskripsi: Orchestrator untuk proses download dan sinkronisasi model pretrained
"""

import time
from typing import Dict, Any, Callable
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from IPython.display import display, HTML

from smartcash.ui.utils.constants import ICONS, COLORS
from smartcash.common.logger import get_logger
from smartcash.ui.pretrained_model.pretrained_initializer import is_drive_mounted, mount_drive
from smartcash.ui.pretrained_model.services.download_service import download_with_progress
from smartcash.ui.pretrained_model.services.sync_service import sync_drive_to_local, sync_local_to_drive
from smartcash.ui.pretrained_model.utils.progress import update_progress_ui

logger = get_logger(__name__)

def process_download_sync(ui_components: Dict[str, Any]) -> None:
    """
    Memproses download dan sinkronisasi model pretrained.
    
    Args:
        ui_components: Dictionary berisi komponen UI
    """
    # Dapatkan komponen yang diperlukan dengan safe checking - one-liner style
    status_panel = ui_components.get('status')
    log_output = ui_components.get('log')
    models_dir = ui_components.get('models_dir', '/content/models')
    drive_models_dir = ui_components.get('drive_models_dir', '/content/drive/MyDrive/SmartCash/models')
    
    # Bersihkan log output jika tersedia
    if log_output: log_output.clear_output(wait=True)
    
    # Fungsi untuk logging dengan timestamp dan emoji kontekstual
    def log_message(message: str, message_type='info'):
        # Tambahkan emoji berdasarkan tipe pesan
        emoji_map = {'info': '📝', 'success': '✅', 'warning': '⚠️', 'error': '❌', 'download': '📥', 'sync': '🔄'}
        emoji = emoji_map.get(message_type, emoji_map['info'])
        
        # Format timestamp dan pesan
        timestamp = time.strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {emoji} {message}"
        
        # Tampilkan di log output jika tersedia
        if log_output:
            with log_output:
                display(HTML(f"<p>{formatted_message}</p>"))
        
        # Log ke logger sesuai tipe pesan
        log_func = getattr(logger, message_type if message_type in ['info', 'warning', 'error'] else 'info')
        log_func(message)
    
    try:
        # Cek apakah di Colab dengan mencoba mengimpor google.colab
        in_colab = False
        try:
            import google.colab
            in_colab = True
        except ImportError:
            in_colab = False
        
        # Cek apakah Drive terpasang
        is_drive_mounted_val = is_drive_mounted()
        
        # Jika di Colab tapi Drive belum ter-mount, coba mount
        if in_colab and not is_drive_mounted_val:
            log_message(f"{ICONS.get('sync', '🔄')} Mounting Google Drive...")
            success, message = mount_drive()
            log_message(message)
            is_drive_mounted_val = is_drive_mounted()
        
        # Cek apakah model sudah ada di Drive
        drive_models_exist = False
        if is_drive_mounted_val:
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
                    sync_drive_to_local(models_dir, drive_models_dir, log_message, ui_components)
                    
                    # Update status
                    if status_panel:
                        status_panel.clear_output(wait=True)
                        with status_panel:
                            display(HTML(f"""<div style="padding:10px; background-color:{COLORS['alert_success_bg']}; 
                                    color:{COLORS['alert_success_text']}; border-radius:4px; margin:5px 0;
                                    border-left:4px solid {COLORS['alert_success_text']}">
                                <p style="margin:5px 0">{ICONS.get('success', '✅')} Model berhasil disinkronkan dari Drive</p>
                            </div>"""))
        
        # Jika model tidak ada di Drive, download dan sinkronkan
        if not drive_models_exist:
            log_message(f"{ICONS.get('download', '📥')} Model tidak ditemukan di Drive, memulai download...")
            
            # Download model dengan progress bar - one-liner style
            model_info = [
                {
                    "name": "yolov5s.pt",
                    "url": "https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5s.pt",
                    "min_size": 10 * 1024 * 1024,  # 10MB
                    "idx": 0
                },
                {
                    "name": "efficientnet-b4_notop.h5",
                    "url": "https://github.com/qubvel/efficientnet/releases/download/v0.6.0/efficientnet-b4_notop.h5",
                    "min_size": 50 * 1024 * 1024,  # 50MB
                    "idx": 1
                }
            ]
            
            # Buat direktori model jika belum ada
            Path(models_dir).mkdir(parents=True, exist_ok=True)
            
            # Download semua model dengan progress tracking secara non-blocking
            model_data = {}
            download_futures = []
            
            # Fungsi untuk menangani model yang sudah ada
            def handle_existing_model(model_path, model_name):
                size_mb = model_path.stat().st_size / (1024 * 1024)
                log_message(f"{model_name} sudah tersedia (<span style='color:{COLORS.get('alert_success_text', 'green')}'>{size_mb:.1f} MB</span>)")
                return {
                    "name": model_name,
                    "path": str(model_path),
                    "size_mb": size_mb
                }
            
            # Mulai proses download untuk semua model secara paralel
            with ThreadPoolExecutor(max_workers=2) as executor:
                for i, model in enumerate(model_info):
                    model_path = Path(models_dir) / model["name"]
                    
                    # Update progress untuk model ini
                    update_progress_ui(ui_components, i, len(model_info), f"Memeriksa {model['name']}...")
                    
                    # Periksa apakah model sudah ada dan ukurannya sesuai
                    if not model_path.exists() or model_path.stat().st_size < model["min_size"]:
                        log_message(f"Mengunduh {model['name']}...", "download")
                        
                        # Download model secara non-blocking dengan ThreadPoolExecutor
                        # Fungsi download_with_progress sudah didesain untuk berjalan secara non-blocking
                        download_with_progress(
                            model["url"], 
                            model_path, 
                            log_message, 
                            ui_components, 
                            model["idx"], 
                            len(model_info)
                        )
                        
                        # Tambahkan placeholder ke model_data
                        model_data[model["name"].split(".")[0]] = {
                            "path": str(model_path),
                            "size_mb": 0,  # Akan diupdate setelah download selesai
                            "status": "downloading"
                        }
                    else:
                        # Model sudah ada, tambahkan ke model_data
                        model_info = handle_existing_model(model_path, model["name"])
                        model_data[model["name"].split(".")[0]] = {
                            "path": model_info["path"],
                            "size_mb": model_info["size_mb"],
                            "status": "ready"
                        }
            
            # Sinkronkan ke Drive jika tersedia
            if is_drive_mounted_val:
                log_message(f"{ICONS.get('sync', '🔄')} Menyinkronkan model ke Google Drive...")
                model_info_dict = {"models": model_data}
                sync_local_to_drive(models_dir, drive_models_dir, model_info_dict, log_message, ui_components)
            
            # Update status
            if status_panel:
                status_panel.clear_output(wait=True)
                with status_panel:
                    if len(model_data) == len(model_info):
                        display(HTML(f"""<div style="padding:10px; background-color:{COLORS['alert_success_bg']}; 
                                color:{COLORS['alert_success_text']}; border-radius:4px; margin:5px 0;
                                border-left:4px solid {COLORS['alert_success_text']}">
                            <p style="margin:5px 0">{ICONS.get('success', '✅')} Semua model berhasil diunduh dan disinkronkan!</p>
                        </div>"""))
                    else:
                        display(HTML(f"""<div style="padding:10px; background-color:{COLORS['alert_warning_bg']}; 
                                color:{COLORS['alert_warning_text']}; border-radius:4px; margin:5px 0;
                                border-left:4px solid {COLORS['alert_warning_text']}">
                            <p style="margin:5px 0">{ICONS.get('warning', '⚠️')} Beberapa model gagal diunduh. Silakan coba lagi.</p>
                        </div>"""))
        
        # Reset progress tracking jika tersedia - one-liner style
        update_progress_ui(ui_components, 1, 1, "Selesai!")
                    
    except Exception as e:
        logger.error(f"Error dalam proses download dan sinkronisasi: {str(e)}")
        
        # Update status error
        if status_panel:
            status_panel.clear_output(wait=True)
            with status_panel:
                display(HTML(f"""<div style="padding:10px; background-color:{COLORS['alert_danger_bg']}; 
                        color:{COLORS['alert_danger_text']}; border-radius:4px; margin:5px 0;
                        border-left:4px solid {COLORS['alert_danger_text']}">
                    <p style="margin:5px 0">{ICONS.get('error', '❌')} Error: {str(e)}</p>
                </div>"""))
