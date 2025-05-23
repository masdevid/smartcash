"""
File: smartcash/ui/dataset/preprocessing/utils/drive_utils.py
Deskripsi: Utilitas untuk mengelola penyimpanan data preprocessing di Google Drive dengan symlink safety
"""

import os
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from smartcash.common.logger import get_logger
from smartcash.common.environment import get_environment_manager
from smartcash.common.constants.paths import DRIVE_PREPROCESSED_PATH, COLAB_PREPROCESSED_PATH

logger = get_logger(__name__)

def setup_drive_preprocessing_storage(ui_components: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Setup penyimpanan preprocessing di Google Drive dengan symlink.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Tuple[bool, str]: (berhasil, pesan)
    """
    env_manager = get_environment_manager()
    
    # Cek apakah Drive mounted
    if not env_manager.is_drive_mounted:
        logger.warning("🔗 Google Drive tidak terpasang, menggunakan penyimpanan lokal")
        ensure_local_preprocessing_dir()
        return False, "Google Drive tidak terpasang, menggunakan penyimpanan lokal"
    
    try:
        # Pastikan direktori Drive ada
        drive_preprocess_dir = Path(DRIVE_PREPROCESSED_PATH)
        drive_preprocess_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup symlink ke lokal
        local_preprocess_dir = Path(COLAB_PREPROCESSED_PATH)
        
        if local_preprocess_dir.exists() and not local_preprocess_dir.is_symlink():
            # Backup data lokal jika ada
            backup_path = local_preprocess_dir.parent / f"{local_preprocess_dir.name}_backup"
            if backup_path.exists():
                shutil.rmtree(backup_path)
            shutil.move(str(local_preprocess_dir), str(backup_path))
            logger.info(f"💾 Data lokal dibackup ke {backup_path}")
        
        # Hapus symlink lama jika ada
        if local_preprocess_dir.is_symlink():
            local_preprocess_dir.unlink()
        elif local_preprocess_dir.exists():
            shutil.rmtree(local_preprocess_dir)
            
        # Buat symlink baru
        local_preprocess_dir.symlink_to(drive_preprocess_dir)
        logger.info(f"🔗 Symlink preprocessing dibuat: {local_preprocess_dir} -> {drive_preprocess_dir}")
        
        return True, f"Penyimpanan Drive berhasil disetup: {drive_preprocess_dir}"
        
    except Exception as e:
        logger.error(f"❌ Error setup Drive storage: {str(e)}")
        ensure_local_preprocessing_dir()
        return False, f"Error setup Drive storage: {str(e)}"

def ensure_local_preprocessing_dir() -> Path:
    """
    Pastikan direktori preprocessing lokal ada.
    
    Returns:
        Path: Path ke direktori preprocessing
    """
    local_dir = Path(COLAB_PREPROCESSED_PATH)
    local_dir.mkdir(parents=True, exist_ok=True)
    return local_dir

def check_existing_preprocessing_data(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Cek data preprocessing yang sudah ada.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dict dengan informasi data yang sudah ada
    """
    preprocess_dir = Path(COLAB_PREPROCESSED_PATH)
    
    result = {
        'exists': False,
        'total_files': 0,
        'splits': {},
        'size_mb': 0,
        'symlink_active': False
    }
    
    if not preprocess_dir.exists():
        return result
    
    # Cek apakah symlink aktif
    result['symlink_active'] = preprocess_dir.is_symlink() and preprocess_dir.exists()
    result['exists'] = True
    
    # Hitung file dengan ThreadPoolExecutor untuk performa
    def count_files_in_dir(dir_path: Path) -> Tuple[str, int, float]:
        """Count files in directory"""
        if not dir_path.exists():
            return dir_path.name, 0, 0.0
        
        file_count = 0
        total_size = 0
        
        for item in dir_path.rglob('*'):
            if item.is_file():
                file_count += 1
                try:
                    total_size += item.stat().st_size
                except:
                    pass
        
        return dir_path.name, file_count, total_size / (1024 * 1024)  # MB
    
    # Scan split directories
    split_dirs = [d for d in preprocess_dir.iterdir() if d.is_dir()]
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(count_files_in_dir, split_dir) for split_dir in split_dirs]
        
        for future in as_completed(futures):
            split_name, file_count, size_mb = future.result()
            result['splits'][split_name] = {
                'files': file_count,
                'size_mb': size_mb
            }
            result['total_files'] += file_count
            result['size_mb'] += size_mb
    
    return result

def is_symlink_safe_to_remove(path: Path) -> bool:
    """
    Cek apakah aman untuk menghapus path (bukan symlink augmentasi).
    
    Args:
        path: Path yang akan dihapus
        
    Returns:
        bool: True jika aman dihapus
    """
    if not path.exists():
        return True
    
    # Cek apakah ini symlink augmentasi
    if path.is_symlink():
        target = path.resolve()
        if 'augment' in str(target).lower() or 'aug' in str(target).lower():
            logger.warning(f"⚠️ Symlink augmentasi terdeteksi: {path} -> {target}")
            return False
    
    return True

def safe_cleanup_preprocessing_data(ui_components: Dict[str, Any], 
                                  progress_callback: Optional[callable] = None) -> Dict[str, Any]:
    """
    Cleanup data preprocessing dengan aman (tidak menghapus symlink augmentasi).
    
    Args:
        ui_components: Dictionary komponen UI
        progress_callback: Callback untuk progress tracking
        
    Returns:
        Dict dengan hasil cleanup
    """
    preprocess_dir = Path(COLAB_PREPROCESSED_PATH)
    
    stats = {
        'deleted_files': 0,
        'deleted_dirs': 0,
        'skipped_symlinks': 0,
        'errors': []
    }
    
    if not preprocess_dir.exists():
        return stats
    
    # Collect all files and directories
    all_items = list(preprocess_dir.rglob('*'))
    files_to_delete = [item for item in all_items if item.is_file()]
    dirs_to_delete = [item for item in all_items if item.is_dir()]
    
    # Sort directories by depth (deepest first)
    dirs_to_delete.sort(key=lambda x: len(x.parts), reverse=True)
    
    total_items = len(files_to_delete) + len(dirs_to_delete)
    processed = 0
    
    # Delete files dengan ThreadPoolExecutor
    def delete_file_safe(file_path: Path) -> Tuple[bool, str]:
        """Delete file safely"""
        try:
            if is_symlink_safe_to_remove(file_path):
                file_path.unlink()
                return True, "deleted"
            else:
                return False, "skipped_symlink"
        except Exception as e:
            return False, f"error: {str(e)}"
    
    # Delete files in parallel
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(delete_file_safe, file_path) for file_path in files_to_delete]
        
        for future in as_completed(futures):
            success, result = future.result()
            processed += 1
            
            if success:
                stats['deleted_files'] += 1
            elif result == "skipped_symlink":
                stats['skipped_symlinks'] += 1
            else:
                stats['errors'].append(result)
            
            # Progress callback
            if progress_callback:
                progress_callback(processed, total_items, f"Menghapus file {processed}/{len(files_to_delete)}")
    
    # Delete directories sequentially (karena dependensi parent-child)
    for dir_path in dirs_to_delete:
        processed += 1
        
        try:
            if is_symlink_safe_to_remove(dir_path):
                if dir_path.exists() and not any(dir_path.iterdir()):  # Only delete empty dirs
                    dir_path.rmdir()
                    stats['deleted_dirs'] += 1
            else:
                stats['skipped_symlinks'] += 1
        except Exception as e:
            stats['errors'].append(f"Dir {dir_path}: {str(e)}")
        
        # Progress callback
        if progress_callback:
            progress_callback(processed, total_items, f"Membersihkan direktori {processed - len(files_to_delete)}/{len(dirs_to_delete)}")
    
    return stats

def sync_preprocessing_to_drive(ui_components: Dict[str, Any], 
                              progress_callback: Optional[callable] = None) -> Tuple[bool, str]:
    """
    Sinkronisasi data preprocessing ke Google Drive.
    
    Args:
        ui_components: Dictionary komponen UI
        progress_callback: Callback untuk progress tracking
        
    Returns:
        Tuple[bool, str]: (berhasil, pesan)
    """
    env_manager = get_environment_manager()
    
    if not env_manager.is_drive_mounted:
        return False, "Google Drive tidak terpasang"
    
    local_dir = Path(COLAB_PREPROCESSED_PATH)
    drive_dir = Path(DRIVE_PREPROCESSED_PATH)
    
    if not local_dir.exists():
        return False, "Tidak ada data preprocessing untuk disinkronkan"
    
    # Jika sudah symlink, tidak perlu sync
    if local_dir.is_symlink():
        return True, "Data sudah tersinkronisasi via symlink"
    
    try:
        # Pastikan direktori Drive ada
        drive_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy files dengan progress tracking
        all_files = list(local_dir.rglob('*'))
        files_only = [f for f in all_files if f.is_file()]
        
        synced_files = 0
        total_files = len(files_only)
        
        def sync_file(src_file: Path) -> bool:
            """Sync single file"""
            try:
                rel_path = src_file.relative_to(local_dir)
                dst_file = drive_dir / rel_path
                
                # Pastikan direktori tujuan ada
                dst_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Copy file
                shutil.copy2(src_file, dst_file)
                return True
            except Exception as e:
                logger.error(f"❌ Error sync file {src_file}: {str(e)}")
                return False
        
        # Sync dengan ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(sync_file, file_path) for file_path in files_only]
            
            for future in as_completed(futures):
                if future.result():
                    synced_files += 1
                
                # Progress callback
                if progress_callback:
                    progress_callback(synced_files, total_files, f"Sinkronisasi {synced_files}/{total_files} file")
        
        logger.info(f"✅ Berhasil sinkronisasi {synced_files}/{total_files} file ke Drive")
        return True, f"Berhasil sinkronisasi {synced_files}/{total_files} file ke Drive"
        
    except Exception as e:
        logger.error(f"❌ Error sinkronisasi ke Drive: {str(e)}")
        return False, f"Error sinkronisasi ke Drive: {str(e)}"