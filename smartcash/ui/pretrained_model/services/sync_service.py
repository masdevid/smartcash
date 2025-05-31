"""
File: smartcash/ui/pretrained_model/services/sync_service.py
Deskripsi: Layanan untuk sinkronisasi model pretrained antara lokal dan Google Drive
"""

from typing import Dict, Any, Callable, Optional, List
from pathlib import Path
import shutil
import concurrent.futures

from smartcash.ui.utils.constants import ICONS, COLORS
from smartcash.common.logger import get_logger
from smartcash.ui.pretrained_model.utils.progress import update_progress_ui

logger = get_logger(__name__)

def sync_drive_to_local(local_dir: str, drive_dir: str, log_func: Callable = None, ui_components: Optional[Dict[str, Any]] = None) -> None:
    """
    Sinkronisasi model dari Google Drive ke lokal dengan progress tracking.
    
    Args:
        local_dir: Direktori lokal
        drive_dir: Direktori Google Drive
        log_func: Fungsi untuk logging
        ui_components: Komponen UI untuk progress tracking
    """
    try:
        # Cek apakah direktori Drive ada
        drive_path = Path(drive_dir)
        if not drive_path.exists():
            if log_func:
                log_func(f"{ICONS.get('warning', '⚠️')} Direktori Drive {drive_dir} tidak ditemukan!")
            return
        
        # Cek apakah ada file model di Drive
        file_list = list(drive_path.glob('*.pt')) + list(drive_path.glob('*.h5'))
        if not file_list:
            if log_func:
                log_func(f"{ICONS.get('warning', '⚠️')} Tidak ada file model yang ditemukan di Drive!")
            return
        
        # Buat direktori lokal jika belum ada
        local_path = Path(local_dir)
        local_path.mkdir(parents=True, exist_ok=True)
        
        # Salin file model dari Drive ke lokal dengan progress tracking
        total_files = len(file_list)
        for i, file_path in enumerate(file_list):
            # Update progress jika UI components tersedia
            if ui_components:
                progress_msg = f"Sinkronisasi dari Drive: {file_path.name} ({i+1}/{total_files})"
                update_progress_ui(ui_components, i, total_files, progress_msg)
            
            # Cek apakah ini adalah file model yang valid
            if file_path.name.endswith(('.pt', '.h5', '.pth', '.weights')):
                target_path = local_path / file_path.name
                
                # Salin file jika belum ada atau ukurannya berbeda
                if not target_path.exists() or target_path.stat().st_size != file_path.stat().st_size:
                    shutil.copy2(file_path, target_path)
                    if log_func:
                        size_mb = target_path.stat().st_size / (1024 * 1024)
                        log_func(f"{ICONS.get('file', '📄')} Disinkronkan: {file_path.name} (<span style='color:{COLORS.get('alert_success_text', '#155724')}'>{size_mb:.1f} MB</span>)")
        
        # Update progress ke 100% setelah selesai
        if ui_components:
            update_progress_ui(ui_components, total_files, total_files, "Sinkronisasi dari Drive selesai!")
        
        if log_func:
            log_func(f"{ICONS.get('success', '✅')} Sinkronisasi dari Drive ke lokal selesai!")
            
    except Exception as e:
        if log_func:
            log_func(f"{ICONS.get('error', '❌')} Error saat sinkronisasi dari Drive ke lokal: {str(e)}", "error")
        else:
            logger.error(f"Error saat sinkronisasi dari Drive ke lokal: {str(e)}")

def sync_local_to_drive(local_dir: str, drive_dir: str, model_info: Optional[Dict[str, Any]] = None, log_func: Callable = None, ui_components: Optional[Dict[str, Any]] = None) -> None:
    """
    Sinkronisasi model dari lokal ke Google Drive dengan progress tracking.
    
    Args:
        local_dir: Direktori lokal
        drive_dir: Direktori Google Drive
        model_info: Informasi model (opsional)
        log_func: Fungsi untuk logging
        ui_components: Komponen UI untuk progress tracking
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
        
        # Sinkronisasi file model ke Drive dengan progress tracking
        if model_files:
            if log_func:
                log_func(f"{ICONS.get('sync', '🔄')} Sinkronisasi {len(model_files)} file model ke Google Drive...")
            
            total_files = len(model_files)
            for i, file_path in enumerate(model_files):
                # Update progress jika UI components tersedia
                if ui_components:
                    progress_msg = f"Sinkronisasi ke Drive: {file_path.name} ({i+1}/{total_files})"
                    update_progress_ui(ui_components, i, total_files, progress_msg)
                
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
                        size_mb = file_path.stat().st_size / (1024 * 1024)
                        log_func(f"{ICONS.get('file', '📄')} Disinkronkan: {rel_path} (<span style='color:{COLORS.get('alert_success_text', '#155724')}'>{size_mb:.1f} MB</span>)")
            
            # Update progress ke 100% setelah selesai
            if ui_components:
                update_progress_ui(ui_components, total_files, total_files, "Sinkronisasi ke Drive selesai!")
            
            if log_func:
                log_func(f"{ICONS.get('success', '✅')} Sinkronisasi ke Google Drive selesai!")
        else:
            if log_func:
                log_func(f"{ICONS.get('warning', '⚠️')} Tidak ada file model yang ditemukan untuk disinkronkan.")
    
    except Exception as e:
        if log_func:
            log_func(f"{ICONS.get('error', '❌')} Error saat sinkronisasi ke Google Drive: {str(e)}", "error")
        else:
            logger.error(f"Error saat sinkronisasi ke Google Drive: {str(e)}")
