"""
File: smartcash/ui/model/trained/setup.py
Deskripsi: Fungsi setup untuk model pre-trained dengan pendekatan DRY (versi yang diperbaiki)
"""

from typing import Dict, Any, Optional, Callable
from pathlib import Path
import os
from tqdm import tqdm

from smartcash.model.services.pretrained_downloader import PretrainedModelDownloader
from smartcash.ui.utils.constants import ICONS
from smartcash.common.logger import get_logger

logger = get_logger(__name__)

def setup_pretrained_models(models_dir: str = '/content/models', 
                           logger_func = None) -> Dict[str, Any]:
    """
    Download dan setup model pre-trained YOLOv5 dan EfficientNet-B4 dengan UI feedback.
    
    Args:
        models_dir: Direktori untuk menyimpan model
        logger_func: Fungsi callback untuk logging (opsional)
        
    Returns:
        Dict berisi informasi model
    """
    # Fungsi helper untuk logging
    def log_message(message: str):
        if logger_func is not None and callable(logger_func):
            logger_func(message)
        else:
            logger.info(message)
    
    # Inisialisasi downloader
    downloader = PretrainedModelDownloader(models_dir=models_dir)
    
    # Download semua model yang diperlukan
    log_message(f"{ICONS.get('rocket', '🚀')} Memulai download model pre-trained...")
    
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
    
    # Kembalikan informasi model
    return model_info

def sync_models_with_drive(models_dir: str, 
                          drive_models_dir: str, 
                          model_info: Optional[Dict[str, Any]] = None,
                          logger_func = None) -> None:
    """
    Sinkronisasi model dengan Google Drive.
    
    Args:
        models_dir: Direktori lokal untuk model
        drive_models_dir: Direktori Google Drive untuk model
        model_info: Informasi model (opsional)
        logger_func: Fungsi callback untuk logging (opsional)
    """
    # Fungsi helper untuk logging
    def log_message(message: str):
        if logger_func is not None and callable(logger_func):
            logger_func(message)
        else:
            logger.info(message)
    
    try:
        # Cek apakah direktori Drive ada
        drive_path = Path(drive_models_dir)
        if not drive_path.exists():
            log_message(f"{ICONS.get('folder', '📁')} Membuat direktori {drive_models_dir}...")
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
            log_message(f"{ICONS.get('sync', '🔄')} Sinkronisasi {len(model_files)} file model ke Google Drive...")
            
            for file_path in tqdm(model_files, desc="Sinkronisasi"):
                # Hitung path relatif terhadap direktori model
                rel_path = file_path.relative_to(local_path)
                drive_file_path = drive_path / rel_path
                
                # Buat direktori parent jika belum ada
                drive_file_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Salin file jika belum ada atau ukurannya berbeda
                if not drive_file_path.exists() or drive_file_path.stat().st_size != file_path.stat().st_size:
                    import shutil
                    shutil.copy2(file_path, drive_file_path)
                    log_message(f"{ICONS.get('file', '📄')} Disinkronkan: {rel_path}")
            
            log_message(f"{ICONS.get('success', '✅')} Sinkronisasi model selesai!")
        else:
            log_message(f"{ICONS.get('warning', '⚠️')} Tidak ada file model yang ditemukan untuk disinkronkan.")
    
    except Exception as e:
        log_message(f"{ICONS.get('error', '❌')} Error saat sinkronisasi model: {str(e)}")
