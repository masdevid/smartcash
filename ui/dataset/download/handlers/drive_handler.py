"""
File: smartcash/ui/dataset/download/handlers/drive_handler.py
Deskripsi: Handler untuk download dataset dari Google Drive
"""

import os
import shutil
from pathlib import Path
from typing import Dict, Any, Tuple, List
from concurrent.futures import ThreadPoolExecutor

def process_drive_download(ui_components: Dict[str, Any]) -> None:
    """
    Proses download dataset dari Google Drive.
    
    Args:
        ui_components: Dictionary komponen UI
    """
    logger = ui_components.get('logger')
    
    # Ambil parameter dari UI components
    from smartcash.ui.dataset.download.handlers.endpoint_handler import get_endpoint_config
    config = get_endpoint_config(ui_components)
    
    try:
        # Jalankan operasi dalam thread terpisah
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(download_from_drive, ui_components, config)
            success, message = future.result()
        
        # Update UI berdasarkan hasil
        from smartcash.ui.utils.fallback_utils import show_status
        from smartcash.ui.utils.constants import ALERT_STYLES
        
        if success:
            # Sukses
            show_status(message, "success", ui_components)
            ui_components['status_panel'].value = f"""
            <div style="padding:10px; background-color:{ALERT_STYLES['success']['bg_color']}; 
                      color:{ALERT_STYLES['success']['text_color']}; border-radius:4px; margin:5px 0;
                      border-left:4px solid {ALERT_STYLES['success']['text_color']};">
                <p style="margin:5px 0">{ALERT_STYLES['success']['icon']} {message}</p>
            </div>
            """
        else:
            # Error
            show_status(message, "error", ui_components)
            ui_components['status_panel'].value = f"""
            <div style="padding:10px; background-color:{ALERT_STYLES['error']['bg_color']}; 
                      color:{ALERT_STYLES['error']['text_color']}; border-radius:4px; margin:5px 0;
                      border-left:4px solid {ALERT_STYLES['error']['text_color']};">
                <p style="margin:5px 0">{ALERT_STYLES['error']['icon']} {message}</p>
            </div>
            """
    except Exception as e:
        # Error
        error_msg = f"Error saat download dari Google Drive: {str(e)}"
        
        if logger:
            logger.error(f"❌ {error_msg}")
        
        # Update UI dengan error
        from smartcash.ui.utils.fallback_utils import show_status
        show_status(error_msg, "error", ui_components)
    finally:
        # Reset UI setelah selesai
        _reset_ui_after_download(ui_components)

def download_from_drive(ui_components: Dict[str, Any], config: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Download dataset dari Google Drive dengan dukungan backup otomatis.
    
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
        backup_existing = config.get('backup_existing', False)
        
        # Pastikan drive_folder tidak dimulai dengan slash
        drive_folder = drive_folder.lstrip('/')
        
        # Path penuh ke folder dataset di Drive
        source_path = Path(drive_path) / drive_folder
        
        # Cek apakah folder ada
        if not source_path.exists():
            return False, f"Folder {drive_folder} tidak ditemukan di Google Drive"
        
        # Coba gunakan DownloadService untuk import dari ZIP jika ada
        try:
            from smartcash.ui.utils.fallback_utils import get_dataset_manager
            download_service = get_dataset_manager({
                'data': {'dir': output_dir}
            }, logger)
            
            # Cek jika ada file ZIP di folder tersebut
            zip_files = list(source_path.glob("*.zip"))
            if zip_files and download_service:
                # Temukan file ZIP terbaru
                newest_zip = max(zip_files, key=os.path.getmtime)
                
                # Import dataset dari ZIP menggunakan service
                _update_progress(ui_components, 30, f"Mengimport dataset dari {newest_zip.name}...")
                
                result = download_service.import_from_zip(
                    zip_file=str(newest_zip),
                    target_dir=output_dir,
                    remove_zip=False,
                    show_progress=True,
                    backup_existing=backup_existing
                )
                
                if result.get('status') == 'success':
                    total_images = result.get('stats', {}).get('total_images', 0)
                    success_msg = f"Dataset berhasil diimport dari {newest_zip.name}: {total_images} gambar"
                    return True, success_msg
        except (ImportError, Exception) as e:
            # Log error tapi lanjutkan dengan implementasi fallback
            if logger:
                logger.debug(f"ℹ️ Menggunakan implementasi manual: {str(e)}")
        
        # Backup existing jika diperlukan
        if backup_existing:
            output_path = Path(output_dir)
            if output_path.exists() and any(output_path.iterdir()):
                _update_progress(ui_components, 20, "Membuat backup data yang sudah ada...")
                backup_path = _backup_existing_data(output_dir)
                if logger:
                    logger.info(f"✅ Data yang ada dibackup ke {backup_path}")
        
        # Pastikan direktori output ada
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Update progress
        _update_progress(ui_components, 30, f"Menyiapkan copy dari {drive_folder}...")
        
        # Dapatkan daftar file dataset yang akan disalin
        files_to_copy = _get_dataset_files(source_path, output_dir)
        
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
                
                # Update progress - setiap 10% file atau minimal setiap 100 file
                if (idx + 1) % max(1, min(total_files // 10, 100)) == 0:
                    progress = 40 + int(50 * (idx + 1) / total_files)
                    _update_progress(ui_components, progress, f"Menyalin file {idx+1}/{total_files}...")
            except Exception as e:
                if logger:
                    logger.warning(f"⚠️ Error saat menyalin {src_file}: {str(e)}")
        
        # Hitung statistik
        _update_progress(ui_components, 95, "Menghitung statistik dataset...")
        stats = _count_dataset_files(output_path)
        
        # Log success
        success_msg = f"Dataset berhasil disalin: {success_count}/{total_files} file ({stats['images']} gambar, {stats['labels']} label)"
        _update_progress(ui_components, 100, success_msg)
        
        return True, success_msg
    
    except Exception as e:
        if logger:
            logger.error(f"❌ Error saat copy dataset dari Drive: {str(e)}")
        return False, f"Error saat copy dataset dari Drive: {str(e)}"

def _backup_existing_data(output_dir: str) -> str:
    """
    Backup data yang sudah ada sebelum overwrite.
    
    Args:
        output_dir: Direktori output yang akan dibackup
        
    Returns:
        Path direktori backup
    """
    from datetime import datetime
    
    # Buat nama backup dengan timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(output_dir)
    backup_name = f"{output_path.name}_backup_{timestamp}"
    backup_path = output_path.parent / backup_name
    
    # Salin folder yang ada ke backup
    shutil.copytree(output_dir, backup_path)
    
    return str(backup_path)

def _get_drive_path() -> str:
    """
    Dapatkan path Google Drive dengan pendekatan multi-source.
    
    Returns:
        Path Google Drive atau None jika tidak terpasang
    """
    # Coba melalui environment manager - prioritas utama
    try:
        from smartcash.common.environment import get_environment_manager
        env = get_environment_manager()
        if env.is_drive_mounted:
            return str(env.drive_path)
    except ImportError:
        pass
    
    # Fallback: coba utilitas drive_utils
    try:
        from smartcash.ui.utils.drive_utils import detect_drive_mount
        is_mounted, drive_path = detect_drive_mount()
        if is_mounted and drive_path:
            return drive_path
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

def _get_dataset_files(source_path: Path, output_dir: str = 'data') -> List[Tuple[Path, Path]]:
    """
    Dapatkan daftar file dataset yang akan disalin dari Drive ke local.
    
    Args:
        source_path: Path sumber di Drive
        output_dir: Direktori output (default: 'data')
        
    Returns:
        List tuple (source_file, dest_file)
    """
    files_to_copy = []
    output_path = Path(output_dir)
    
    # Struktur YOLO standar
    yolo_dirs = ['train/images', 'train/labels', 'valid/images', 'valid/labels', 'test/images', 'test/labels']
    
    # Coba deteksi format dataset berdasarkan struktur folder
    if any((source_path / d).exists() for d in yolo_dirs):
        # Format YOLO terdeteksi - copy direktori yang ada
        for subdir in yolo_dirs:
            src_dir = source_path / subdir
            if not src_dir.exists():
                continue
                
            dst_dir = output_path / subdir
            
            # Salin semua file dalam direktori
            for file_path in src_dir.glob('*.*'):
                if file_path.is_file():
                    files_to_copy.append((file_path, dst_dir / file_path.name))
    else:
        # Format tidak terdeteksi, salin semua file ke struktur yang sama
        for file_path in source_path.glob('**/*.*'):
            if file_path.is_file():
                # Relative path dari source
                rel_path = file_path.relative_to(source_path)
                files_to_copy.append((file_path, output_path / rel_path))
    
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
    tracker_key = 'download_tracker'
    if tracker_key in ui_components:
        tracker = ui_components[tracker_key]
        tracker.update(value, message)
        
    # Log ke logger - hanya log pada perubahan signifikan (20% progress)
    logger = ui_components.get('logger')
    if logger and value % 20 == 0:
        logger.info(f"🔄 {message} ({value}%)")

def _reset_ui_after_download(ui_components: Dict[str, Any]) -> None:
    """Reset UI setelah proses download selesai."""
    # Sembunyikan progress bar
    if 'progress_bar' in ui_components:
        ui_components['progress_bar'].layout.visibility = 'hidden'
    if 'progress_message' in ui_components:
        progress_message = ui_components['progress_message']
        progress_message.layout.visibility = 'hidden'