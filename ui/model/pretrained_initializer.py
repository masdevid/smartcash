"""
File: smartcash/ui/model/pretrained_initializer.py
Deskripsi: Inisialisasi UI dan logika bisnis untuk pretrained model dengan pendekatan DRY
"""

import os
from pathlib import Path
import ipywidgets as widgets
from IPython.display import display, clear_output
from typing import Dict, Any, Tuple, List, Optional

from smartcash.ui.model.trained.setup import setup_pretrained_models, sync_models_with_drive
from smartcash.ui.utils.alert_utils import create_status_indicator
from smartcash.ui.utils.constants import ICONS
from smartcash.ui.utils.header_utils import create_header
from smartcash.common.logger import get_logger
from smartcash.common.environment import EnvironmentManager

logger = get_logger(__name__)

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
        import threading
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
        import time
        time.sleep(0.5)
        
        # Gunakan EnvironmentManager untuk manajemen environment
        env_manager = EnvironmentManager()
        models_dir = ui_components['models_dir']
        drive_models_dir = ui_components['drive_models_dir']
        
        # Fungsi callback untuk logging dengan timestamp
        def log_message(message: str):
            from datetime import datetime
            timestamp = datetime.now().strftime("%H:%M:%S")
            with ui_components['log']:
                print(f"[{timestamp}] {message}")
        
        # Update status - Inisialisasi
        with ui_components['status']:
            clear_output(wait=True)
            display(create_status_indicator("info", 
                f"{ICONS.get('loading', '⏳')} Memeriksa environment dan direktori..."))
        
        log_message(f"{ICONS.get('info', 'ℹ️')} Memulai proses persiapan model pre-trained")
        
        # Cek apakah direktori parent ada
        models_parent = Path(models_dir).parent
        if not models_parent.exists():
            with ui_components['status']:
                clear_output(wait=True)
                display(create_status_indicator("warning", 
                    f"{ICONS.get('warning', '⚠️')} Direktori parent {models_parent} tidak ditemukan"))
            log_message(f"{ICONS.get('warning', '⚠️')} Direktori parent {models_parent} tidak ditemukan")
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
            
            sync_models_with_drive(models_dir, drive_models_dir, log_callback=log_message)
        
        # Download model pretrained dengan callback untuk logging
        log_message(f"{ICONS.get('download', '📥')} Memulai download model pre-trained...")
        with ui_components['status']:
            clear_output(wait=True)
            display(create_status_indicator("info", 
                f"{ICONS.get('loading', '⏳')} Downloading model pre-trained..."))
        
        model_info = setup_pretrained_models(models_dir=models_dir, log_callback=log_message)
        
        # Sinkronisasi ke Drive setelah download jika di Colab
        if model_info and in_colab and is_drive_available:
            log_message(f"{ICONS.get('sync', '🔄')} Menyinkronkan model ke Google Drive...")
            with ui_components['status']:
                clear_output(wait=True)
                display(create_status_indicator("info", 
                    f"{ICONS.get('loading', '⏳')} Sinkronisasi ke Google Drive..."))
            
            sync_models_with_drive(models_dir, drive_models_dir, model_info, log_callback=log_message)
        
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
