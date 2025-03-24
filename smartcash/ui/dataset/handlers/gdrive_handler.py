"""
File: smartcash/ui/dataset/handlers/gdrive_handler.py
Deskripsi: Handler untuk download dataset dari Google Drive
"""

import os
import shutil
from pathlib import Path
from typing import Dict, Any, Tuple, List

def download_from_drive(ui_components: Dict[str, Any], config: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Download dataset dari Google Drive.
    
    Args:
        ui_components: Dictionary komponen UI
        config: Konfigurasi download dataset
        
    Returns:
        Tuple (success, message)
    """
    logger = ui_components.get('logger')
    
    try:
        # Update progress
        _update_progress(ui_components, 10, "Memeriksa Google Drive...")
        
        # Cek apakah Drive terpasang
        drive_path = _get_drive_path()
        if not drive_path:
            return False, "Google Drive tidak terpasang"
        
        # Get konfigurasi
        drive_folder = config.get('folder', 'SmartCash/datasets')
        output_dir = config.get('output_dir', 'data')
        
        # Pastikan drive_folder tidak dimulai dengan slash
        drive_folder = drive_folder.lstrip('/')
        
        # Path penuh ke folder dataset di Drive
        source_path = Path(drive_path) / drive_folder
        
        # Cek apakah folder ada
        if not source_path.exists():
            return False, f"Folder {drive_folder} tidak ditemukan di Google Drive"
        
        # Pastikan direktori output ada
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Update progress
        _update_progress(ui_components, 30, f"Menyiapkan copy dari {drive_folder}...")
        
        # Dapatkan daftar file dataset yang akan disalin
        files_to_copy = _get_dataset_files(source_path)
        
        if not files_to_copy:
            return False, f"Tidak ada file dataset ditemukan di {drive_folder}"
        
        # Copy file dataset dari Drive ke local
        total_files = len(files_to_copy)
        _update_progress(ui_components, 40, f"Menyalin {total_files} file...")
        
        # Salin file dengan progress tracking
        success_count = 0
        for idx, (src_file, dst_file) in enumerate(files_to_copy):
            try:
                # Buat direktori tujuan jika belum ada
                dst_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Salin file
                shutil.copy2(src_file, dst_file)
                success_count += 1
                
                # Update progress
                progress = 40 + int(50 * (idx + 1) / total_files)
                if (idx + 1) % max(1, total_files // 10) == 0:  # Update setiap 10%
                    _update_progress(ui_components, progress, f"Menyalin file {idx+1}/{total_files}...")
            except Exception as e:
                if logger: logger.warning(f"⚠️ Error saat menyalin {src_file}: {str(e)}")
        
        # Hitung statistik
        _update_progress(ui_components, 95, "Menghitung statistik dataset...")
        stats = _count_dataset_files(output_path)
        
        # Log success
        success_msg = f"Dataset berhasil disalin: {success_count}/{total_files} file ({stats['images']} gambar, {stats['labels']} label)"
        _update_progress(ui_components, 100, success_msg)
        
        return True, success_msg
    
    except Exception as e:
        if logger: logger.error(f"❌ Error saat copy dataset dari Drive: {str(e)}")
        return False, f"Error saat copy dataset dari Drive: {str(e)}"

def _get_drive_path() -> str:
    """
    Dapatkan path Google Drive.
    
    Returns:
        Path Google Drive atau None jika tidak terpasang
    """
    # Coba melalui environment manager
    try:
        from smartcash.common.environment import get_environment_manager
        env = get_environment_manager()
        if env.is_drive_mounted:
            return env.drive_path
    except ImportError:
        pass
    
    # Fallback: cek manual path standar
    drive_paths = [
        '/content/drive/MyDrive',
        '/content/drive/My Drive',
        '/gdrive/MyDrive',
        '/gdrive/My Drive'
    ]
    
    for path in drive_paths:
        if os.path.exists(path):
            return path
    
    return None

def _get_dataset_files(source_path: Path) -> List[Tuple[Path, Path]]:
    """
    Dapatkan daftar file dataset yang akan disalin dari Drive ke local.
    
    Args:
        source_path: Path sumber di Drive
        
    Returns:
        List tuple (source_file, dest_file)
    """
    files_to_copy = []
    
    # Cek jika path sumber adalah file ZIP
    if source_path.is_file() and source_path.suffix.lower() == '.zip':
        # Import dari utils dan proses melalui ZIP handler
        try:
            from smartcash.ui.dataset.handlers.zip_handler import extract_zip_dataset
            return [(source_path, Path(extract_zip_dataset(source_path)))]
        except ImportError:
            # Fallback: copy file ZIP saja
            zip_name = source_path.name
            return [(source_path, Path('data') / zip_name)]
    
    # Struktur YOLO standar
    yolo_dirs = ['train/images', 'train/labels', 'valid/images', 'valid/labels', 'test/images', 'test/labels']
    
    # Coba deteksi format dataset
    if any((source_path / d).exists() for d in yolo_dirs):
        # Format YOLO terdeteksi
        for subdir in yolo_dirs:
            src_dir = source_path / subdir
            if not src_dir.exists():
                continue
                
            dst_dir = Path('data') / subdir
            
            # Salin semua file dalam direktori
            for file_path in src_dir.glob('*.*'):
                if file_path.is_file():
                    files_to_copy.append((file_path, dst_dir / file_path.name))
    else:
        # Format tidak terdeteksi, salin semua file
        for file_path in source_path.glob('**/*.*'):
            if file_path.is_file():
                # Relative path dari source
                rel_path = file_path.relative_to(source_path)
                files_to_copy.append((file_path, Path('data') / rel_path))
    
    return files_to_copy

def _count_dataset_files(output_path: Path) -> Dict[str, int]:
    """
    Hitung jumlah file dalam dataset.
    
    Args:
        output_path: Path output dataset
        
    Returns:
        Dictionary berisi jumlah file
    """
    stats = {'images': 0, 'labels': 0, 'other': 0}
    
    # Hitung file dalam direktori
    image_exts = ['.jpg', '.jpeg', '.png', '.bmp']
    label_exts = ['.txt', '.xml', '.json']
    
    for file_path in output_path.glob('**/*.*'):
        ext = file_path.suffix.lower()
        if ext in image_exts:
            stats['images'] += 1
        elif ext in label_exts:
            stats['labels'] += 1
        else:
            stats['other'] += 1
    
    return stats

def _update_progress(ui_components: Dict[str, Any], value: int, message: str) -> None:
    """
    Update progress bar dan progress tracker.
    
    Args:
        ui_components: Dictionary komponen UI
        value: Nilai progress (0-100)
        message: Pesan progress
    """
    # Update progress bar
    progress_bar = ui_components.get('progress_bar')
    progress_message = ui_components.get('progress_message')
    
    if progress_bar:
        progress_bar.value = value
    
    if progress_message:
        progress_message.value = message
    
    # Update progress tracker
    tracker_key = 'dataset_downloader_tracker'
    if tracker_key in ui_components:
        tracker = ui_components[tracker_key]
        tracker.update(value, message)
        
    # Log ke logger
    logger = ui_components.get('logger')
    if logger and value % 20 == 0:  # Log setiap 20%
        logger.info(f"🔄 {message} ({value}%)")