"""
File: smartcash/ui/utils/drive_utils.py
Deskripsi: Utilitas yang ditingkatkan untuk deteksi dan sinkronisasi data Google Drive untuk SmartCash
"""

import os
import shutil
import time
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Callable, List
from concurrent.futures import ThreadPoolExecutor

def detect_drive_mount() -> Tuple[bool, Optional[str]]:
    """
    Deteksi apakah Google Drive terpasang di sistem.
    
    Returns:
        Tuple (drive_mounted, drive_path)
    """
    # Cek path standar di Colab
    drive_paths = [
        '/content/drive/MyDrive',
        '/content/drive',
        '/gdrive',
        '/mnt/drive'
    ]
    
    for path in drive_paths:
        if os.path.exists(path) and os.path.isdir(path):
            return True, path
    
    # Coba cek melalui environment manager
    try:
        from smartcash.common.environment import get_environment_manager
        env_manager = get_environment_manager()
        if hasattr(env_manager, 'is_drive_mounted') and env_manager.is_drive_mounted:
            return True, str(env_manager.drive_path) if hasattr(env_manager, 'drive_path') else None
    except ImportError:
        pass
    except Exception:
        pass
    
    return False, None

def resolve_drive_path(path: str, config: Dict[str, Any] = None, env = None) -> str:
    """
    Resolve path relatif ke absolute dengan mempertimbangkan Google Drive.
    
    Args:
        path: Path relatif
        config: Konfigurasi aplikasi
        env: Environment manager
        
    Returns:
        Path absolute terresolved
    """
    # Cek apakah path sudah absolute
    if os.path.isabs(path):
        return path
    
    # Dapatkan environment manager
    if env is None:
        try:
            from smartcash.common.environment import get_environment_manager
            env = get_environment_manager()
        except ImportError:
            # Fallback: Cek Google Drive
            is_mounted, drive_path = detect_drive_mount()
            if is_mounted and drive_path:
                # Gunakan drive path untuk SmartCash project
                return os.path.join(drive_path, 'SmartCash', path)
            return os.path.abspath(path)
    
    # Gunakan environment manager untuk mendapatkan path
    if env and hasattr(env, 'get_path'):
        return env.get_path(path)
    
    # Fallback: Gunakan path absolute
    return os.path.abspath(path)

def is_newer(src_path: str, dst_path: str) -> bool:
    """
    Cek apakah file source lebih baru dari file destination.
    
    Args:
        src_path: Path file source
        dst_path: Path file destination
        
    Returns:
        Boolean menunjukkan apakah source lebih baru
    """
    if not os.path.exists(dst_path):
        return True
        
    if not os.path.exists(src_path):
        return False
        
    return os.path.getmtime(src_path) > os.path.getmtime(dst_path)

def sync_drive_to_local(
    config: Dict[str, Any], 
    env=None, 
    logger=None, 
    callback: Optional[Callable[[str, str], None]] = None,
    bidirectional: bool = True
) -> Dict[str, Any]:
    """
    Sinkronisasi data antara Google Drive dan lokal dengan sinkronisasi dua arah.
    
    Args:
        config: Konfigurasi aplikasi
        env: Environment manager
        logger: Logger untuk logging
        callback: Callback function untuk update status
        bidirectional: Apakah melakukan sinkronisasi dua arah
        
    Returns:
        Dictionary berisi hasil sinkronisasi
    """
    # Gunakan environment manager untuk akses drive
    try:
        from smartcash.common.environment import get_environment_manager
        env_manager = env or get_environment_manager()
        
        # Cek apakah sync diaktifkan
        if not config.get('data', {}).get('use_drive', False):
            if logger: logger.info(f"ℹ️ Sinkronisasi Drive tidak diaktifkan dalam konfigurasi")
            if callback: callback("info", "Sinkronisasi Drive tidak diaktifkan dalam konfigurasi")
            return {"status": "skipped", "reason": "drive_sync_disabled"}
        
        # Cek apakah Drive terpasang
        if not hasattr(env_manager, 'is_drive_mounted') or not env_manager.is_drive_mounted:
            if logger: logger.error(f"❌ Google Drive tidak ter-mount")
            if callback: callback("error", "Google Drive tidak ter-mount")
            return {"status": "error", "reason": "drive_not_mounted"}
        
        # Notifikasi memulai sinkronisasi
        if logger: logger.info(f"🔄 Memulai sinkronisasi drive...")
        if callback: callback("info", "Memulai sinkronisasi data")
        
        # Struktur direktori yang akan disinkronkan
        directories = [
            ('data', config.get('data', {}).get('drive_data_dir', 'data')),
            ('runs', config.get('data', {}).get('drive_output_dir', 'runs')),
            ('logs', config.get('data', {}).get('drive_logs_dir', 'logs')),
            ('configs', 'configs'),
        ]
        
        results = {
            "synchronized": [],
            "errors": [],
            "drive_to_local": 0,
            "local_to_drive": 0,
            "status": "success"
        }
        
        # Untuk setiap direktori yang akan disinkronkan
        for local_dir, drive_dir in directories:
            # Resolve paths
            local_path = resolve_drive_path(local_dir, config, env_manager)
            drive_path = str(env_manager.drive_path / drive_dir)
            
            # Pastikan direktori ada di kedua lokasi
            os.makedirs(local_path, exist_ok=True)
            os.makedirs(drive_path, exist_ok=True)
            
            if logger: logger.info(f"🔄 Menyinkronkan: {drive_path} ⟷ {local_path}")
            if callback: callback("info", f"Menyinkronkan {drive_dir}")
            
            try:
                # Dari Drive ke lokal
                drive_to_local = sync_directory(
                    src_dir=drive_path,
                    dst_dir=local_path,
                    logger=logger,
                    recursive=True
                )
                
                # Dari lokal ke Drive (jika bidirectional)
                local_to_drive = {}
                if bidirectional:
                    local_to_drive = sync_directory(
                        src_dir=local_path,
                        dst_dir=drive_path,
                        logger=logger,
                        recursive=True
                    )
                
                results["synchronized"].append({
                    "directory": local_dir,
                    "drive_path": drive_path,
                    "local_path": local_path,
                    "drive_to_local": drive_to_local.get("copied", 0),
                    "local_to_drive": local_to_drive.get("copied", 0) if bidirectional else 0
                })
                
                results["drive_to_local"] += drive_to_local.get("copied", 0)
                results["local_to_drive"] += local_to_drive.get("copied", 0) if bidirectional else 0
                
            except Exception as e:
                error_msg = f"Error saat menyinkronkan {local_dir}: {str(e)}"
                if logger: logger.error(f"❌ {error_msg}")
                results["errors"].append({
                    "directory": local_dir,
                    "error": str(e)
                })
                
        # Log hasil sinkronisasi
        if logger:
            logger.success(f"✅ Sinkronisasi selesai! {results['drive_to_local']} file disalin dari Drive, {results['local_to_drive']} file disalin ke Drive")
        
        if callback:
            status_msg = f"Sinkronisasi selesai: {results['drive_to_local']} file dari Drive, {results['local_to_drive']} file ke Drive"
            callback("success", status_msg)
        
        return results
            
    except ImportError as e:
        error_msg = f"Tidak dapat mengakses environment manager: {str(e)}"
        if logger: logger.error(f"❌ {error_msg}")
        if callback: callback("error", error_msg)
        return {"status": "error", "reason": "environment_manager_not_found", "error": str(e)}
        
    except Exception as e:
        error_msg = f"Error tidak terduga saat sinkronisasi: {str(e)}"
        if logger: logger.error(f"❌ {error_msg}")
        if callback: callback("error", error_msg)
        return {"status": "error", "reason": "unexpected_error", "error": str(e)}

def sync_directory(
    src_dir: str,
    dst_dir: str,
    logger=None,
    recursive: bool = True,
    exclude_patterns: List[str] = None
) -> Dict[str, Any]:
    """
    Sinkronisasi isi direktori dari src_dir ke dst_dir berdasarkan timestamp.
    
    Args:
        src_dir: Direktori sumber
        dst_dir: Direktori tujuan
        logger: Logger untuk logging
        recursive: Apakah melakukan sinkronisasi rekursif
        exclude_patterns: Pattern file/direktori yang diabaikan
        
    Returns:
        Dictionary berisi hasil sinkronisasi
    """
    import fnmatch
    import re
    
    # Jika exclude_patterns None, gunakan default
    if exclude_patterns is None:
        exclude_patterns = ['*.pyc', '__pycache__', '.git', '.ipynb_checkpoints', '*.swp', '*.swo']
    
    # Pastikan direktori tujuan ada
    os.makedirs(dst_dir, exist_ok=True)
    
    result = {
        "copied": 0,
        "skipped": 0,
        "errors": 0
    }
    
    def is_excluded(path: str) -> bool:
        """Cek apakah path sesuai exclude pattern."""
        filename = os.path.basename(path)
        return any(fnmatch.fnmatch(filename, pattern) for pattern in exclude_patterns)
    
    # Sync file dari src_dir ke dst_dir
    for root, dirs, files in os.walk(src_dir):
        # Dapatkan path relatif dari src_dir
        rel_path = os.path.relpath(root, src_dir)
        if rel_path == '.':
            rel_path = ''
        
        # Filter direktori yang diabaikan (untuk os.walk)
        dirs[:] = [d for d in dirs if not is_excluded(os.path.join(root, d))]
        
        # Buat direktori tujuan jika belum ada
        dst_root = os.path.join(dst_dir, rel_path)
        os.makedirs(dst_root, exist_ok=True)
        
        # Proses setiap file
        for file in files:
            if is_excluded(file):
                result["skipped"] += 1
                continue
                
            src_file = os.path.join(root, file)
            dst_file = os.path.join(dst_root, file)
            
            try:
                # Cek apakah perlu disalin (file baru atau lebih baru)
                if not os.path.exists(dst_file) or is_newer(src_file, dst_file):
                    shutil.copy2(src_file, dst_file)
                    result["copied"] += 1
                    if logger and result["copied"] % 100 == 0:
                        logger.debug(f"📂 Disalin file: {dst_file}")
                else:
                    result["skipped"] += 1
            except Exception as e:
                if logger:
                    logger.warning(f"⚠️ Error saat salin {src_file}: {str(e)}")
                result["errors"] += 1
        
        # Berhenti rekursi jika tidak recursive
        if not recursive:
            break
    
    return result

def async_sync_drive(
    config: Dict[str, Any], 
    env=None, 
    logger=None, 
    callback: Optional[Callable[[str, str], None]] = None,
    bidirectional: bool = True
):
    """
    Jalankan sinkronisasi drive secara asynchronous dengan ThreadPoolExecutor.
    
    Args:
        config: Konfigurasi aplikasi
        env: Environment manager
        logger: Logger untuk logging
        callback: Callback function untuk update status
        bidirectional: Apakah melakukan sinkronisasi dua arah
        
    Returns:
        Future yang dapat digunakan untuk memeriksa status
    """
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(
        sync_drive_to_local,
        config, env, logger, callback, bidirectional
    )
    executor.shutdown(wait=False)
    return future