"""
File: smartcash/ui/model/handlers/simple_download.py
Deskripsi: Handler sederhana untuk download dan sinkronisasi model pretrained
"""

from typing import Dict, Any, Callable, Optional
from pathlib import Path
import time
import threading
import shutil
from IPython.display import display, HTML

from smartcash.ui.utils.constants import ICONS, COLORS
from smartcash.common.logger import get_logger
from smartcash.ui.model.pretrained_initializer import is_drive_mounted, mount_drive

logger = get_logger(__name__)

def handle_download_sync_button(b, ui_components: Dict[str, Any]) -> None:
    """
    Handler sederhana untuk tombol download dan sinkronisasi model pretrained.
    
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

# Fungsi enable_button dan process_download_sync_async tidak diperlukan lagi karena tidak menggunakan threading

def process_download_sync(ui_components: Dict[str, Any]) -> None:
    """
    Memproses download dan sinkronisasi model pretrained.
    
    Args:
        ui_components: Dictionary berisi komponen UI
    """
    # Dapatkan komponen yang diperlukan
    status_panel = ui_components.get('status')
    log_output = ui_components.get('log')
    models_dir = ui_components.get('models_dir', '/content/models')
    drive_models_dir = ui_components.get('drive_models_dir', '/content/drive/MyDrive/SmartCash/models')
    
    # Bersihkan log output
    if log_output:
        log_output.clear_output(wait=True)
    
    # Fungsi untuk logging dengan timestamp
    def log_message(message: str):
        timestamp = time.strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"
        if log_output:
            with log_output:
                from IPython.display import display, HTML
                display(HTML(f"<p>{formatted_message}</p>"))
        logger.info(message)
    
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
            from smartcash.model.services.pretrained_downloader import PretrainedModelDownloader
            downloader = PretrainedModelDownloader(models_dir=models_dir)
            
            # Download YOLOv5
            try:
                log_message(f"\n{ICONS.get('download', '📥')} Memeriksa YOLOv5s...")
                yolo_info = downloader.download_yolov5()
                log_message(f"{ICONS.get('success', '✅')} Model YOLOv5s tersedia di {yolo_info['path']}")
            except Exception as e:
                log_message(f"{ICONS.get('error', '❌')} Gagal memproses YOLOv5s: {str(e)}")
            
            # Download EfficientNet-B4
            try:
                log_message(f"\n{ICONS.get('download', '📥')} Memeriksa EfficientNet-B4...")
                efficientnet_info = downloader.download_efficientnet()
                log_message(f"{ICONS.get('success', '✅')} Model EfficientNet-B4 tersedia di {efficientnet_info['path']}")
            except Exception as e:
                log_message(f"{ICONS.get('error', '❌')} Gagal memproses EfficientNet-B4: {str(e)}")
            
            # Tampilkan ringkasan informasi model
            model_info = downloader.get_model_info()
            log_message(f"\n{ICONS.get('sparkles', '✨')} Proses download selesai!\n")
            
            if 'yolov5s' in model_info.get('models', {}) and 'efficientnet_b4' in model_info.get('models', {}):
                log_message("Ringkasan model yang tersedia:")
                log_message(f"- YOLOv5s: {model_info['models']['yolov5s']['size_mb']} MB")
                log_message(f"- EfficientNet-B4: {model_info['models']['efficientnet_b4']['size_mb']} MB")
            
            # Sinkronkan ke Drive jika tersedia
            if is_drive_mounted_val:
                log_message(f"{ICONS.get('sync', '🔄')} Menyinkronkan model ke Google Drive...")
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

def sync_drive_to_local(local_dir: str, drive_dir: str, log_func: Callable = None) -> None:
    """
    Sinkronisasi model dari Google Drive ke lokal.
    
    Args:
        local_dir: Direktori lokal
        drive_dir: Direktori Google Drive
        log_func: Fungsi untuk logging
    """
    try:
        # Pastikan direktori lokal ada
        local_path = Path(local_dir)
        local_path.mkdir(parents=True, exist_ok=True)
        
        # Dapatkan daftar file di Drive
        drive_path = Path(drive_dir)
        
        # Cek file model utama
        model_files = [
            drive_path / "yolov5s.pt",
            drive_path / "efficientnet-b4_notop.h5",
            drive_path / "efficientnet_b4.pt"
        ]
        
        # Salin file dari Drive ke lokal
        for file_path in model_files:
            if file_path.exists():
                target_path = local_path / file_path.name
                
                # Salin file jika belum ada atau ukurannya berbeda
                if not target_path.exists() or target_path.stat().st_size != file_path.stat().st_size:
                    shutil.copy2(file_path, target_path)
                    if log_func:
                        log_func(f"{ICONS.get('file', '📄')} Disinkronkan: {file_path.name}")
        
        if log_func:
            log_func(f"{ICONS.get('success', '✅')} Sinkronisasi dari Drive ke lokal selesai!")
            
    except Exception as e:
        if log_func:
            log_func(f"{ICONS.get('error', '❌')} Error saat sinkronisasi dari Drive ke lokal: {str(e)}")
        else:
            logger.error(f"Error saat sinkronisasi dari Drive ke lokal: {str(e)}")

def sync_local_to_drive(local_dir: str, drive_dir: str, model_info: Optional[Dict[str, Any]] = None, log_func: Callable = None) -> None:
    """
    Sinkronisasi model dari lokal ke Google Drive.
    
    Args:
        local_dir: Direktori lokal
        drive_dir: Direktori Google Drive
        model_info: Informasi model (opsional)
        log_func: Fungsi untuk logging
    """
    try:
        # Cek apakah direktori Drive ada
        drive_path = Path(drive_dir)
        if not drive_path.exists():
            drive_path.mkdir(parents=True, exist_ok=True)
            if log_func:
                log_func(f"{ICONS.get('folder', '📁')} Membuat direktori {drive_dir}...")
        
        # Daftar file model yang perlu disinkronkan
        model_files = []
        local_path = Path(local_dir)
        
        # Jika model_info tersedia, gunakan informasi tersebut
        if model_info and 'models' in model_info:
            for model_name, model_data in model_info['models'].items():
                if 'path' in model_data:
                    model_path = Path(model_data['path'])
                    if model_path.exists():
                        model_files.append(model_path)
        
        # Jika tidak ada model_info, cari file model utama
        if not model_files:
            main_models = [
                local_path / "yolov5s.pt",
                local_path / "efficientnet-b4_notop.h5",
                local_path / "efficientnet_b4.pt"
            ]
            for file_path in main_models:
                if file_path.exists():
                    model_files.append(file_path)
        
        # Sinkronisasi file model ke Drive
        if model_files:
            if log_func:
                log_func(f"{ICONS.get('sync', '🔄')} Sinkronisasi {len(model_files)} file model ke Google Drive...")
            
            from tqdm.notebook import tqdm
            for file_path in tqdm(model_files, desc="Sinkronisasi"):
                # Hitung path relatif terhadap direktori model
                try:
                    rel_path = file_path.relative_to(local_path)
                except ValueError:
                    # Jika tidak relatif, gunakan nama file saja
                    rel_path = file_path.name
                
                drive_file_path = drive_path / rel_path
                
                # Buat direktori parent jika belum ada
                drive_file_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Salin file jika belum ada atau ukurannya berbeda
                if not drive_file_path.exists() or drive_file_path.stat().st_size != file_path.stat().st_size:
                    shutil.copy2(file_path, drive_file_path)
                    if log_func:
                        log_func(f"{ICONS.get('file', '📄')} Disinkronkan: {rel_path}")
            
            if log_func:
                log_func(f"{ICONS.get('success', '✅')} Sinkronisasi ke Google Drive selesai!")
        else:
            if log_func:
                log_func(f"{ICONS.get('warning', '⚠️')} Tidak ada file model yang ditemukan untuk disinkronkan.")
    
    except Exception as e:
        if log_func:
            log_func(f"{ICONS.get('error', '❌')} Error saat sinkronisasi ke Google Drive: {str(e)}")
        else:
            logger.error(f"Error saat sinkronisasi ke Google Drive: {str(e)}")
