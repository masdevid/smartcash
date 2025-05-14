"""
File: smartcash/ui/setup/drive_sync_initializer.py
Deskripsi: Fungsi inisialisasi dan sinkronisasi konfigurasi antara local dan Google Drive dengan implementasi DRY
"""

import os
import threading
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

# Import konstanta pesan log dari common module
from smartcash.common.constants.log_messages import (
    STATUS_INFO, STATUS_SUCCESS, STATUS_WARNING, STATUS_ERROR,
    CONFIG_ERROR, DRIVE_NOT_MOUNTED, CONFIG_IDENTICAL, CONFIG_SYNC_SUCCESS
)

# Import utilitas IO dari common
from smartcash.common.io import (
    ensure_dir, copy_file, load_config, save_config, list_dir_recursively
)

# Lock untuk thread-safety saat sinkronisasi
_sync_lock = threading.RLock()
# Flag untuk mencegah inisialisasi berulang
_initialized = False

def initialize_configs(logger=None, ui_components: Optional[Dict[str, Any]] = None, silent: bool = True) -> Tuple[bool, str]:
    """
    Inisialisasi konfigurasi dengan pendekatan DRY.
    
    Args:
        logger: Logger opsional untuk logging
        ui_components: Dictionary berisi UI components untuk logging ke UI
        silent: Flag untuk mematikan log ke console
        
    Returns:
        Tuple (success, message)
    """
    global _initialized
    
    # Gunakan lock untuk thread-safety
    with _sync_lock:
        # Periksa apakah sudah diinisialisasi sebelumnya
        if _initialized and silent:
            return True, "Konfigurasi sudah diinisialisasi sebelumnya"
        
        try:
            # Dapatkan environment manager dari common
            from smartcash.common.environment import get_environment_manager
            env_manager = get_environment_manager()
            
            # Setup direktori konfigurasi lokal
            local_config_dir = Path("configs")
            ensure_dir(local_config_dir)
            
            # Log ke UI dan logger
            _log_to_components("Sinkronisasi konfigurasi...", "info", logger, ui_components)
            
            # Jika Drive tidak terpasang, gunakan konfigurasi default
            if not env_manager.is_drive_mounted:
                _log_to_components(DRIVE_NOT_MOUNTED, "warning", logger, ui_components)
                return _setup_default_configs(local_config_dir, logger, ui_components)
                
            # Drive terpasang - siapkan konfigurasi di Drive
            drive_config_dir = env_manager.drive_path / "configs"
            ensure_dir(drive_config_dir)
            
            # Coba gunakan config_manager untuk sinkronisasi
            try:
                # Gunakan implementasi yang ada di common/config
                from smartcash.common.config import sync_all_configs, get_config_manager
                
                # Sinkronisasi semua konfigurasi dengan 'merge' strategy
                results = sync_all_configs(
                    sync_strategy='merge',
                    create_backup=True,
                    logger=logger
                )
                
                # Hitung statistik
                success_count = len(results.get("success", []))
                skipped_count = len(results.get("skipped", []))
                failure_count = len(results.get("failure", []))
                
                status = "success" if failure_count == 0 else "warning"
                msg = f"Sinkronisasi selesai: {success_count} disinkronkan, {skipped_count} dilewati, {failure_count} gagal"
                _log_to_components(msg, status, logger, ui_components)
                
                # Tandai sudah diinisialisasi
                _initialized = True
                return failure_count == 0, msg
                
            except ImportError:
                # Fallback: Sinkronisasi manual jika config sync tidak tersedia
                _log_to_components("Menggunakan sinkronisasi manual", "info", logger, ui_components)
                return _manual_sync_configs(local_config_dir, drive_config_dir, logger, ui_components)
                
        except Exception as e:
            _log_to_components(CONFIG_ERROR.format(operation="inisialisasi", error=str(e)), "error", logger, ui_components)
            _initialized = True
            return False, f"Error saat inisialisasi konfigurasi: {str(e)}"
            
        finally:
            # Tandai sudah diinisialisasi
            _initialized = True

def _setup_default_configs(local_config_dir: Path, logger=None, ui_components=None) -> Tuple[bool, str]:
    """
    Setup konfigurasi default jika Drive tidak tersedia.
    
    Args:
        local_config_dir: Direktori konfigurasi lokal
        logger: Logger opsional
        ui_components: Dictionary UI components
        
    Returns:
        Tuple (success, message)
    """
    try:
        # Cek apakah konfigurasi lokal sudah ada
        if any(local_config_dir.glob("*.yaml")):
            _log_to_components("Menggunakan konfigurasi lokal yang ada", "info", logger, ui_components)
            return True, "Menggunakan konfigurasi lokal yang ada"
            
        # Buat konfigurasi default menggunakan common/default_config
        from smartcash.common.default_config import ensure_base_config_exists
        
        if ensure_base_config_exists():
            _log_to_components("Konfigurasi default berhasil dibuat", "success", logger, ui_components)
            return True, "Konfigurasi default berhasil dibuat"
        else:
            _log_to_components("Gagal membuat konfigurasi default", "error", logger, ui_components)
            return False, "Gagal membuat konfigurasi default"
            
    except Exception as e:
        _log_to_components(f"Error saat setup konfigurasi default: {str(e)}", "error", logger, ui_components)
        return False, f"Error saat setup konfigurasi default: {str(e)}"

def _manual_sync_configs(local_dir: Path, drive_dir: Path, logger=None, ui_components=None) -> Tuple[bool, str]:
    """
    Sinkronisasi manual antar direktori konfigurasi.
    
    Args:
        local_dir: Direktori konfigurasi lokal
        drive_dir: Direktori konfigurasi Drive
        logger: Logger opsional
        ui_components: Dictionary UI components
        
    Returns:
        Tuple (success, message)
    """
    # Cek file di kedua lokasi
    local_files = {f.name: f for f in local_dir.glob("*.yaml")}
    drive_files = {f.name: f for f in drive_dir.glob("*.yaml")}
    
    # Statistik
    stats = {"copied_to_local": 0, "copied_to_drive": 0}
    
    try:
        # Salin konfigurasi dari Drive ke lokal yang tidak ada di lokal
        for name, drive_file in drive_files.items():
            if name not in local_files:
                copy_file(drive_file, local_dir / name)
                stats["copied_to_local"] += 1
                
        # Salin konfigurasi dari lokal ke Drive yang tidak ada di Drive
        for name, local_file in local_files.items():
            if name not in drive_files:
                copy_file(local_file, drive_dir / name)
                stats["copied_to_drive"] += 1
                
        # Log hasil
        msg = f"Sinkronisasi manual: {stats['copied_to_local']} file disalin ke lokal, {stats['copied_to_drive']} file disalin ke Drive"
        _log_to_components(msg, "success", logger, ui_components)
        
        return True, msg
        
    except Exception as e:
        _log_to_components(f"Error saat sinkronisasi manual: {str(e)}", "error", logger, ui_components)
        return False, f"Error saat sinkronisasi manual: {str(e)}"

def _log_to_components(message: str, level: str, logger=None, ui_components=None):
    """
    Log pesan ke logger dan UI components dengan pendekatan DRY.
    
    Args:
        message: Pesan yang akan dilog
        level: Level log ('info', 'success', 'warning', 'error')
        logger: Logger untuk mencatat pesan
        ui_components: Dictionary berisi UI components
    """
    # Log ke UI jika tersedia
    if ui_components and 'status' in ui_components:
        from smartcash.ui.utils.ui_logger import log_to_ui
        
        # Map emoji berdasarkan level
        emoji = "✅" if level == "success" else "⚠️" if level == "warning" else "❌" if level == "error" else "ℹ️"
        log_to_ui(ui_components, message, level, emoji)
    
    # Log ke logger jika tersedia
    if logger:
        if hasattr(logger, level):
            getattr(logger, level)(message)
        elif hasattr(logger, 'info'):
            logger.info(message)