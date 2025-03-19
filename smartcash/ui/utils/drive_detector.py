"""
File: smartcash/ui/utils/drive_detector.py
Deskripsi: Utilitas deteksi dan pengelolaan koneksi Google Drive untuk dataset SmartCash
"""

import os
import shutil
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor
import threading

def sync_drive_to_local(
    config: Dict[str, Any], 
    env=None, 
    logger=None, 
    callback: Optional[callable] = None
) -> bool:
    """Sinkronkan data dari Google Drive ke lokal."""
    # Gunakan environment manager untuk akses drive
    try:
        from smartcash.common.environment import get_environment_manager
        env_manager = env or get_environment_manager()
        
        # Cek apakah sync diaktifkan
        if not config.get('data', {}).get('use_drive', False):
            return False
        
        # Notifikasi memulai sinkronisasi
        if logger: logger.info(f"🔄 Memulai sinkronisasi drive...")
        if callback: callback("info", "Memulai sinkronisasi dari Google Drive ke lokal")
        
        # Dapatkan path drive dan lokal dari environment manager
        if not env_manager.is_drive_mounted:
            if logger: logger.error(f"❌ Google Drive tidak ter-mount")
            if callback: callback("error", "Google Drive tidak ter-mount")
            return False
            
        drive_path = env_manager.get_path(config.get('data', {}).get('drive_path', 'data'))
        local_path = config.get('data', {}).get('local_clone_path', 'data_local')
        
        # Buat path ke objek Path
        source_path = Path(drive_path)
        target_path = Path(local_path)
        
        if not source_path.exists():
            if logger: logger.error(f"❌ Path sumber tidak ditemukan: {source_path}")
            if callback: callback("error", f"Path sumber tidak ditemukan: {source_path}")
            return False
            
        try:
            # Buat direktori lokal jika belum ada
            os.makedirs(target_path, exist_ok=True)
            
            # Mulai sinkronisasi
            if logger: logger.info(f"🔄 Menyinkronkan dari {source_path} ke {target_path}...")
            if callback: callback("info", f"Menyinkronkan {len(list(source_path.glob('**/*')))} files")
            
            # Gunakan ThreadPoolExecutor untuk sinkronisasi paralel
            with ThreadPoolExecutor(max_workers=4) as executor:
                # Sinkronisasi dataset
                futures = []
                for split in ['train', 'valid', 'test']:
                    split_dir = source_path / split
                    target_split = target_path / split
                    
                    # Skip jika split tidak ada
                    if not split_dir.exists():
                        continue
                        
                    # Buat direktori target
                    os.makedirs(target_split, exist_ok=True)
                    
                    # Salin subdirektori dengan thread terpisah
                    for subdir in ['images', 'labels']:
                        src_subdir = split_dir / subdir
                        tgt_subdir = target_split / subdir
                        
                        if src_subdir.exists():
                            # Kirim tugas ke thread pool
                            futures.append(
                                executor.submit(
                                    _copy_directory,
                                    src_subdir, 
                                    tgt_subdir, 
                                    logger
                                )
                            )
                
                # Tunggu semua tugas selesai
                for future in futures:
                    future.result()
            
            if logger: logger.success(f"✅ Sinkronisasi selesai! Dataset tersedia di {target_path}")
            if callback: callback("success", f"Sinkronisasi selesai! Dataset tersedia di {target_path}")
            return True
            
        except Exception as e:
            if logger: logger.error(f"❌ Error saat sinkronisasi: {str(e)}")
            if callback: callback("error", f"Error saat sinkronisasi: {str(e)}")
            return False
            
    except ImportError:
        if logger: logger.error(f"❌ Tidak dapat mengakses environment manager")
        if callback: callback("error", "Tidak dapat mengakses environment manager")
        return False

def _copy_directory(src_dir: Path, dst_dir: Path, logger=None) -> None:
    """Helper untuk menyalin direktori dengan ThreadPoolExecutor."""
    try:
        # Hapus target jika sudah ada
        if dst_dir.exists():
            shutil.rmtree(dst_dir)
        # Salin direktori
        shutil.copytree(src_dir, dst_dir)
        if logger: logger.debug(f"✅ Berhasil menyalin {src_dir} ke {dst_dir}")
    except Exception as e:
        if logger: logger.error(f"❌ Error saat menyalin {src_dir}: {str(e)}")

def async_sync_drive(
    config: Dict[str, Any], 
    env=None, 
    logger=None, 
    callback: Optional[callable] = None
):
    """Jalankan sinkronisasi drive secara asynchronous dengan ThreadPoolExecutor."""
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(
        sync_drive_to_local,
        config, env, logger, callback
    )
    executor.shutdown(wait=False)
    return future