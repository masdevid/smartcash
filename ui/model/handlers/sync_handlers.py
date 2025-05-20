"""
File: smartcash/ui/model/handlers/sync_handlers.py
Deskripsi: Handler untuk sinkronisasi model pretrained antara lokal dan Google Drive
"""

from typing import Dict, Any, Callable
from pathlib import Path
import shutil
from tqdm import tqdm

from smartcash.ui.utils.constants import ICONS
from smartcash.common.logger import get_logger

logger = get_logger(__name__)

def sync_drive_to_local(models_dir: str, drive_models_dir: str, log_func: Callable) -> None:
    """
    Sinkronisasi model dari Google Drive ke lokal.
    
    Args:
        models_dir: Direktori lokal untuk model
        drive_models_dir: Direktori Google Drive untuk model
        log_func: Fungsi untuk logging
    """
    try:
        # Cek apakah direktori lokal ada
        local_path = Path(models_dir)
        if not local_path.exists():
            log_func(f"{ICONS.get('folder', '📁')} Membuat direktori {models_dir}...")
            local_path.mkdir(parents=True, exist_ok=True)
        
        # Cek apakah direktori Drive ada
        drive_path = Path(drive_models_dir)
        if not drive_path.exists():
            log_func(f"{ICONS.get('warning', '⚠️')} Direktori Drive {drive_models_dir} tidak ditemukan")
            return
        
        # Daftar file model di Drive
        model_files = []
        for file_path in drive_path.glob('**/*'):
            if file_path.is_file():
                model_files.append(file_path)
        
        if not model_files:
            log_func(f"{ICONS.get('warning', '⚠️')} Tidak ada file model di Drive")
            return
        
        # Salin file dari Drive ke lokal
        log_func(f"{ICONS.get('sync', '🔄')} Menyalin {len(model_files)} file dari Drive ke lokal...")
        
        for file_path in tqdm(model_files, desc="Sinkronisasi dari Drive"):
            # Hitung path relatif terhadap direktori Drive
            rel_path = file_path.relative_to(drive_path)
            local_file_path = local_path / rel_path
            
            # Buat direktori parent jika belum ada
            local_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Salin file jika belum ada atau ukurannya berbeda
            if not local_file_path.exists() or local_file_path.stat().st_size != file_path.stat().st_size:
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
            
            for file_path in tqdm(model_files, desc="Sinkronisasi ke Drive"):
                # Hitung path relatif terhadap direktori model
                rel_path = file_path.relative_to(local_path)
                drive_file_path = drive_path / rel_path
                
                # Buat direktori parent jika belum ada
                drive_file_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Salin file jika belum ada atau ukurannya berbeda
                if not drive_file_path.exists() or drive_file_path.stat().st_size != file_path.stat().st_size:
                    shutil.copy2(file_path, drive_file_path)
                    log_func(f"{ICONS.get('file', '📄')} Disinkronkan ke Drive: {rel_path}")
            
            log_func(f"{ICONS.get('success', '✅')} Sinkronisasi ke Drive selesai!")
        else:
            log_func(f"{ICONS.get('warning', '⚠️')} Tidak ada file model yang ditemukan untuk disinkronkan.")
    
    except Exception as e:
        log_func(f"{ICONS.get('error', '❌')} Error saat sinkronisasi ke Drive: {str(e)}")
        raise
