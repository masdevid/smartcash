"""
File: smartcash/ui/model/pretrained_initializer.py
Deskripsi: Inisialisasi UI dan logika bisnis untuk pretrained model dengan pendekatan DRY
"""

import os
import time
import threading
from pathlib import Path
import ipywidgets as widgets
from IPython.display import display, clear_output
from typing import Dict, Any, Tuple, List, Optional, Callable

from smartcash.ui.utils.alert_utils import create_status_indicator
from smartcash.ui.utils.constants import ICONS
from smartcash.ui.utils.header_utils import create_header
from smartcash.common.logger import get_logger
from smartcash.common.environment import EnvironmentManager

logger = get_logger(__name__)

# Import model services di sini untuk menghindari circular import
# Ini akan diimpor saat diperlukan di dalam fungsi
# from smartcash.ui.model.trained.setup import setup_pretrained_models, sync_models_with_drive

def check_colab_environment() -> Tuple[bool, bool]:
    """
    Memeriksa apakah kode berjalan di Google Colab dan apakah Google Drive tersedia.
    Menggunakan EnvironmentManager untuk deteksi dan mounting.
    
    Returns:
        Tuple[bool, bool]: (is_in_colab, is_drive_available)
    """
    # Gunakan EnvironmentManager untuk deteksi environment
    env_manager = EnvironmentManager()
    
    # Cek apakah berjalan di Google Colab
    in_colab = env_manager.is_colab()
    
    # Cek apakah Drive tersedia jika di Colab
    is_drive_available = env_manager.is_drive_mounted()
    
    # Jika di Colab tapi Drive belum ter-mount, coba mount
    if in_colab and not is_drive_available:
        try:
            logger.info(f"{ICONS.get('sync', '🔄')} Mounting Google Drive...")
            success, message = env_manager.mount_drive()
            logger.info(message)
            is_drive_available = env_manager.is_drive_mounted()
        except Exception as e:
            logger.error(f"{ICONS.get('error', '❌')} Gagal mounting Google Drive: {str(e)}")
    
    return in_colab, is_drive_available

def initialize_pretrained_model_ui() -> Dict[str, Any]:
    """
    Inisialisasi UI untuk pretrained model.
    
    Returns:
        Dictionary berisi komponen UI yang telah diinisialisasi
    """
    try:
        # Buat komponen UI
        main_container = widgets.VBox()
        status_output = widgets.Output()
        log_output = widgets.Output(layout=widgets.Layout(max_height='300px', overflow='auto'))
        
        # Definisikan direktori untuk model
        models_dir = '/content/models'
        drive_models_dir = '/content/drive/MyDrive/SmartCash/models'
        
        # Buat header dengan create_header
        header = create_header(
            title="Persiapan Model Pre-trained",
            description="Download dan sinkronisasi model pre-trained YOLOv5 dan EfficientNet-B4",
            icon=ICONS.get('brain', '🧠')
        )
        
        # Tambahkan elemen ke main_container
        main_container.children = [header, status_output, log_output]
        
        # Tampilkan UI
        clear_output(wait=True)
        display(main_container)
        
        # Kumpulkan komponen UI
        ui_components = {
            'main_container': main_container,
            'status': status_output,
            'log': log_output,
            'models_dir': models_dir,
            'drive_models_dir': drive_models_dir
        }
        
        # Kembalikan komponen UI terlebih dahulu
        # Proses download akan dijalankan secara asinkron setelah UI terender
        thread = threading.Thread(target=lambda: setup_pretrained_models_ui(ui_components))
        thread.daemon = True
        thread.start()
        
        return ui_components
        
    except Exception as e:
        logger.error(f"{ICONS.get('error', '❌')} Error saat inisialisasi UI pretrained model: {str(e)}")
        
        # Buat container minimal untuk menampilkan error
        error_container = widgets.VBox([
            widgets.HTML(f"<h3>{ICONS.get('error', '❌')} Error saat inisialisasi UI pretrained model</h3>"),
            widgets.HTML(f"<p>{str(e)}</p>")
        ])
        
        display(error_container)
        return {'error_container': error_container}

def setup_pretrained_models_ui(ui_components: Dict[str, Any]) -> None:
    """
    Setup pretrained models dengan UI feedback.
    Menggunakan EnvironmentManager untuk deteksi environment dan manajemen direktori.
    Fungsi ini dijalankan secara asinkron setelah UI terender sempurna.
    
    Args:
        ui_components: Dictionary komponen UI
    """
    try:
        # Tunggu sebentar untuk memastikan UI sudah terender sempurna
        time.sleep(1)
        
        # Impor di sini untuk menghindari circular import
        from smartcash.model.services.pretrained_downloader import PretrainedModelDownloader
        
        # Setup environment
        env_manager = EnvironmentManager()
        models_dir = ui_components['models_dir']
        drive_models_dir = ui_components['drive_models_dir']
        
        # Fungsi callback untuk logging dengan timestamp
        def log_message(message: str):
            from datetime import datetime
            timestamp = datetime.now().strftime("%H:%M:%S")
            with ui_components['log']:
                print(f"[{timestamp}] {message}")
        
        # Inisialisasi downloader
        log_message(f"{ICONS.get('info', 'ℹ️')} Memulai proses persiapan model pre-trained")
        
        # Jika tidak di Colab, lewati proses download dan sinkronisasi
        if not env_manager.is_colab():
            log_message(f"{ICONS.get('warning', '⚠️')} Download dan sinkronisasi model dilewati")
            return
        
        # Cek environment Colab dan Drive
        log_message(f"{ICONS.get('search', '🔎')} Memeriksa environment...")
        in_colab = env_manager.is_colab()
        log_message(f"{ICONS.get('info', 'ℹ️')} Running di Google Colab: {in_colab}")
        
        is_drive_available = env_manager.is_drive_mounted() if in_colab else False
        
        # Jika di Colab tapi Drive belum ter-mount, coba mount
        if in_colab and not is_drive_available:
            log_message(f"{ICONS.get('sync', '🔄')} Mencoba mount Google Drive...")
            with ui_components['status']:
                clear_output(wait=True)
                display(create_status_indicator("info", 
                    f"{ICONS.get('loading', '⏳')} Mounting Google Drive..."))
            
            success, message = env_manager.mount_drive()
            log_message(message)
            is_drive_available = env_manager.is_drive_mounted()
            log_message(f"{ICONS.get('info', 'ℹ️')} Google Drive tersedia: {is_drive_available}")
        
        # Buat direktori model jika belum ada
        log_message(f"{ICONS.get('folder', '📁')} Menyiapkan direktori model: {models_dir}")
        Path(models_dir).mkdir(parents=True, exist_ok=True)
        
        # Status update - Download
        with ui_components['status']:
            clear_output(wait=True)
            display(create_status_indicator("info", 
                f"{ICONS.get('loading', '⏳')} Menyiapkan model pre-trained..."))
        
        # Lakukan sinkronisasi awal dari Drive jika di Colab
        if in_colab and is_drive_available:
            log_message(f"{ICONS.get('sync', '🔄')} Memeriksa model di Google Drive...")
            with ui_components['status']:
                clear_output(wait=True)
                display(create_status_indicator("info", 
                    f"{ICONS.get('loading', '⏳')} Sinkronisasi dari Google Drive..."))
            
            # Sinkronisasi dari Drive ke lokal
            try:
                sync_drive_to_local(models_dir, drive_models_dir, log_message)
            except Exception as e:
                log_message(f"{ICONS.get('error', '❌')} Error saat sinkronisasi dari Drive: {str(e)}")
        
        # Download model pretrained dengan callback untuk logging
        log_message(f"{ICONS.get('download', '📥')} Memulai download model pre-trained...")
        with ui_components['status']:
            clear_output(wait=True)
            display(create_status_indicator("info", 
                f"{ICONS.get('loading', '⏳')} Downloading model pre-trained..."))
        
        # Download model menggunakan downloader langsung
        try:
            downloader = PretrainedModelDownloader(models_dir=models_dir)
            
            # Download YOLOv5
            log_message(f"\n{ICONS.get('download', '📥')} Memeriksa YOLOv5s...")
            yolo_info = downloader.download_yolov5()
            log_message(f"{ICONS.get('success', '✅')} Model YOLOv5s tersedia di {yolo_info['path']}")
            
            # Download EfficientNet-B4
            log_message(f"\n{ICONS.get('download', '📥')} Memeriksa EfficientNet-B4...")
            efficientnet_info = downloader.download_efficientnet()
            log_message(f"{ICONS.get('success', '✅')} Model EfficientNet-B4 tersedia di {efficientnet_info['path']}")
            
            # Dapatkan informasi model
            model_info = downloader.get_model_info()
            
            # Tampilkan ringkasan informasi model
            if 'models' in model_info and 'yolov5s' in model_info['models'] and 'efficientnet_b4' in model_info['models']:
                log_message("\nRingkasan model yang tersedia:")
                log_message(f"- YOLOv5s: {model_info['models']['yolov5s']['size_mb']} MB")
                log_message(f"- EfficientNet-B4: {model_info['models']['efficientnet_b4']['size_mb']} MB")
        except Exception as e:
            log_message(f"{ICONS.get('error', '❌')} Error saat download model: {str(e)}")
            raise
        
        # Sinkronisasi ke Drive setelah download jika di Colab
        if in_colab and is_drive_available:
            log_message(f"{ICONS.get('sync', '🔄')} Menyinkronkan model ke Google Drive...")
            with ui_components['status']:
                clear_output(wait=True)
                display(create_status_indicator("info", 
                    f"{ICONS.get('loading', '⏳')} Sinkronisasi ke Google Drive..."))
            
            # Sinkronisasi dari lokal ke Drive
            try:
                sync_local_to_drive(models_dir, drive_models_dir, model_info, log_message)
            except Exception as e:
                log_message(f"{ICONS.get('error', '❌')} Error saat sinkronisasi ke Drive: {str(e)}")
        
        # Update status setelah selesai
        log_message(f"{ICONS.get('success', '✅')} Proses persiapan model pre-trained selesai!")
        with ui_components['status']:
            clear_output(wait=True)
            display(create_status_indicator("success", 
                f"{ICONS.get('success', '✅')} Model pre-trained berhasil disiapkan"))
        
    except Exception as e:
        logger.error(f"{ICONS.get('error', '❌')} Error saat setup pretrained model: {str(e)}")
        with ui_components['status']:
            clear_output(wait=True)
            display(create_status_indicator("error", 
                f"{ICONS.get('error', '❌')} Error saat setup pretrained model"))
        with ui_components['log']:
            print(f"{ICONS.get('error', '❌')} Error: {str(e)}")

def sync_drive_to_local(models_dir: str, drive_models_dir: str, log_func: Callable) -> None:
    """
    Sinkronisasi model dari Google Drive ke lokal.
    
    Args:
        models_dir: Direktori lokal untuk model
        drive_models_dir: Direktori Google Drive untuk model
        log_func: Fungsi untuk logging
    """
    try:
        # Cek apakah direktori Drive ada
        drive_path = Path(drive_models_dir)
        if not drive_path.exists():
            log_func(f"{ICONS.get('warning', '⚠️')} Direktori Drive {drive_models_dir} tidak ditemukan")
            return
        
        # Cek apakah ada file model di Drive
        model_files = []
        for file_path in drive_path.glob('**/*'):
            if file_path.is_file():
                model_files.append(file_path)
        
        if not model_files:
            log_func(f"{ICONS.get('warning', '⚠️')} Tidak ada file model di Drive")
            return
        
        # Salin file dari Drive ke lokal
        log_func(f"{ICONS.get('sync', '🔄')} Menyalin {len(model_files)} file dari Drive ke lokal...")
        
        for file_path in model_files:
            # Hitung path relatif terhadap direktori Drive
            rel_path = file_path.relative_to(drive_path)
            local_file_path = Path(models_dir) / rel_path
            
            # Buat direktori parent jika belum ada
            local_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Salin file jika belum ada atau ukurannya berbeda
            if not local_file_path.exists() or local_file_path.stat().st_size != file_path.stat().st_size:
                import shutil
                shutil.copy2(file_path, local_file_path)
                log_func(f"{ICONS.get('file', '📄')} Disalin dari Drive: {rel_path}")
        
        log_func(f"{ICONS.get('success', '✅')} Sinkronisasi dari Drive selesai!")
    
    except Exception as e:
        log_func(f"{ICONS.get('error', '❌')} Error saat sinkronisasi dari Drive: {str(e)}")
        raise

def sync_local_to_drive(models_dir: str, drive_models_dir: str, model_info: Dict[str, Any], log_func: Callable) -> None:
    """
    Sinkronisasi model dari lokal ke Google Drive.
    
    Args:
        models_dir: Direktori lokal untuk model
        drive_models_dir: Direktori Google Drive untuk model
        model_info: Informasi model
        log_func: Fungsi untuk logging
    """
    try:
        # Cek apakah direktori Drive ada
        drive_path = Path(drive_models_dir)
        if not drive_path.exists():
            log_func(f"{ICONS.get('folder', '📁')} Membuat direktori {drive_models_dir}...")
            drive_path.mkdir(parents=True, exist_ok=True)
        
        # Daftar file model yang perlu disinkronkan
        model_files = []
        local_path = Path(models_dir)
        
        # Jika model_info tersedia, gunakan informasi tersebut
        if model_info and 'models' in model_info:
            for model_name, model_data in model_info['models'].items():
                if 'path' in model_data:
                    model_path = Path(model_data['path'])
                    if model_path.exists():
                        model_files.append(model_path)
        
        # Jika tidak ada model_info, cari semua file di direktori model
        if not model_files:
            for file_path in local_path.glob('**/*'):
                if file_path.is_file():
                    model_files.append(file_path)
        
        # Sinkronisasi file model ke Drive
        if model_files:
            log_func(f"{ICONS.get('sync', '🔄')} Sinkronisasi {len(model_files)} file model ke Google Drive...")
            
            for file_path in model_files:
                # Hitung path relatif terhadap direktori model
                rel_path = file_path.relative_to(local_path)
                drive_file_path = drive_path / rel_path
                
                # Buat direktori parent jika belum ada
                drive_file_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Salin file jika belum ada atau ukurannya berbeda
                if not drive_file_path.exists() or drive_file_path.stat().st_size != file_path.stat().st_size:
                    import shutil
                    shutil.copy2(file_path, drive_file_path)
                    log_func(f"{ICONS.get('file', '📄')} Disinkronkan ke Drive: {rel_path}")
            
            log_func(f"{ICONS.get('success', '✅')} Sinkronisasi ke Drive selesai!")
        else:
            log_func(f"{ICONS.get('warning', '⚠️')} Tidak ada file model yang ditemukan untuk disinkronkan.")
    
    except Exception as e:
        log_func(f"{ICONS.get('error', '❌')} Error saat sinkronisasi ke Drive: {str(e)}")
        raise
