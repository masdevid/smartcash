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
        main_container = widgets.VBox([])
        status_output = widgets.Output()
        log_output = widgets.Output()
        
        # Definisikan direktori untuk model
        models_dir = '/content/models'
        drive_models_dir = '/content/drive/MyDrive/SmartCash/models'
        
        # Tampilkan UI
        clear_output(wait=True)
        # Tambahkan elemen ke main_container
        main_container.children = [
            widgets.HTML("<h2 style='color: #2c3e50; font-family: Arial, sans-serif;'>🧠 Persiapan Model Pre-trained</h2>"),
            widgets.HTML("<p style='color: #34495e; font-family: Arial, sans-serif;'>Download dan sinkronisasi model pre-trained YOLOv5 dan EfficientNet-B4 untuk deteksi mata uang</p>"),
            status_output,
            log_output
        ]
        # Tampilkan main_container
        display(main_container)
        
        # Kumpulkan komponen UI
        ui_components = {
            'main_container': main_container,
            'status': status_output,
            'log': log_output,
            'models_dir': models_dir,
            'drive_models_dir': drive_models_dir
        }
        
        # Jalankan proses setup model
        setup_pretrained_models_ui(ui_components)
        
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
    
    Args:
        ui_components: Dictionary komponen UI
    """
    try:
        # Gunakan EnvironmentManager untuk manajemen environment
        env_manager = EnvironmentManager()
        models_dir = ui_components['models_dir']
        drive_models_dir = ui_components['drive_models_dir']
        
        # Fungsi callback untuk logging
        def log_callback(message: str):
            with ui_components['log']:
                print(message)
        
        # Cek apakah direktori parent ada
        models_parent = Path(models_dir).parent
        if not models_parent.exists():
            with ui_components['status']:
                clear_output(wait=True)
                display(create_status_indicator("warning", 
                    f"{ICONS.get('warning', '⚠️')} Direktori parent {models_parent} tidak ditemukan"))
            log_callback(f"{ICONS.get('warning', '⚠️')} Direktori parent {models_parent} tidak ditemukan")
            log_callback(f"{ICONS.get('warning', '⚠️')} Download dan sinkronisasi model dilewati")
            return
        
        # Cek environment Colab dan Drive menggunakan EnvironmentManager
        in_colab = env_manager.is_colab()
        is_drive_available = env_manager.is_drive_mounted()
        
        # Jika di Colab tapi Drive belum ter-mount, coba mount
        if in_colab and not is_drive_available:
            success, message = env_manager.mount_drive()
            log_callback(message)
            is_drive_available = env_manager.is_drive_mounted()
        
        # Buat direktori model jika belum ada
        Path(models_dir).mkdir(parents=True, exist_ok=True)
        
        # Status update
        with ui_components['status']:
            clear_output(wait=True)
            display(create_status_indicator("info", 
                f"{ICONS.get('loading', '⏳')} Menyiapkan model pre-trained..."))
        
        # Lakukan sinkronisasi awal dari Drive jika di Colab
        if in_colab and is_drive_available:
            log_callback(f"{ICONS.get('sync', '🔄')} Sinkronisasi model dari Google Drive...")
            sync_models_with_drive(models_dir, drive_models_dir, log_callback=log_callback)
        
        # Download model pretrained dengan callback untuk logging
        model_info = setup_pretrained_models(models_dir=models_dir, log_callback=log_callback)
        
        # Sinkronisasi ke Drive setelah download jika di Colab
        if model_info and in_colab and is_drive_available:
            log_callback(f"{ICONS.get('sync', '🔄')} Sinkronisasi model ke Google Drive...")
            sync_models_with_drive(models_dir, drive_models_dir, model_info, log_callback=log_callback)
        
        # Update status setelah selesai
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
        log_callback(f"{ICONS.get('error', '❌')} Error: {str(e)}")
