"""
File: smartcash/ui/setup/drive_connector.py
Deskripsi: Utilitas untuk menghubungkan dan mengelola Google Drive dengan pendekatan atomic
"""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

def connect_google_drive(ui_components: Dict[str, Any], silent: bool = False) -> Tuple[bool, Optional[Path]]:
    """
    Hubungkan ke Google Drive dan dapatkan path.
    
    Args:
        ui_components: Dictionary komponen UI
        silent: Jika True, tidak menampilkan output UI
        
    Returns:
        Tuple (success, drive_path)
    """
    logger = ui_components.get('logger')
    
    try:
        # Gunakan utility dari drive_utils
        from smartcash.ui.utils.drive_utils import detect_drive_mount
        is_mounted, drive_path = detect_drive_mount()
        
        if not is_mounted:
            # Update status
            _log_to_ui(ui_components, "Mounting Google Drive...", "info", "🔄")
            
            # Mount Drive dengan Google Colab
            from google.colab import drive
            drive.mount('/content/drive')
            
            # Verifikasi mounting berhasil
            import time
            time.sleep(1)  # Berikan waktu untuk mounting
            is_mounted, drive_path = detect_drive_mount()
            
            if not is_mounted:
                _log_to_ui(ui_components, "Gagal mount Google Drive", "error", "❌")
                return False, None
        
        # Path dasar Google Drive
        base_path = Path(drive_path)
        
        # Pastikan direktori SmartCash ada
        smartcash_dir = base_path / 'SmartCash'
        smartcash_dir.mkdir(parents=True, exist_ok=True)
        
        _log_to_ui(ui_components, f"Google Drive terhubung di {smartcash_dir}", "success", "✅")
        return True, smartcash_dir
    except Exception as e:
        _log_to_ui(ui_components, f"Error saat mounting Google Drive: {str(e)}", "error", "❌")
        return False, None

def create_drive_directory_structure(drive_path: Path, ui_components: Dict[str, Any]) -> bool:
    """
    Buat struktur direktori yang diperlukan di Google Drive.
    
    Args:
        drive_path: Path direktori di Google Drive
        ui_components: Dictionary komponen UI
        
    Returns:
        Boolean yang menunjukkan keberhasilan
    """
    logger = ui_components.get('logger')
    
    try:
        # Direktori utama yang diperlukan - konsolidasi one-liner dengan list comprehension
        [(drive_path / dir_name).mkdir(parents=True, exist_ok=True) for dir_name in [
            'configs',
            'data', 'data/train', 'data/train/images', 'data/train/labels',
            'data/valid', 'data/valid/images', 'data/valid/labels',
            'data/test', 'data/test/images', 'data/test/labels',
            'runs', 'runs/train', 'runs/train/weights',
            'logs', 'checkpoints'
        ]]
        
        _log_to_ui(ui_components, f"Struktur direktori berhasil dibuat di {drive_path}", "success", "✅")
        return True
    except Exception as e:
        _log_to_ui(ui_components, f"Error membuat struktur direktori: {str(e)}", "error", "❌")
        return False

def sync_configs_with_drive(ui_components: Dict[str, Any], drive_path: Path) -> bool:
    """
    Sinkronisasi konfigurasi dengan Google Drive.
    
    Args:
        ui_components: Dictionary komponen UI
        drive_path: Path direktori di Google Drive
        
    Returns:
        Boolean yang menunjukkan keberhasilan
    """
    logger = ui_components.get('logger')
    
    try:
        _log_to_ui(ui_components, "Sinkronisasi konfigurasi...", "info", "🔄")
        
        # Coba gunakan drive_sync_initializer terlebih dahulu (lebih lengkap)
        try:
            # Versi dari common (utama)
            try:
                from smartcash.common.drive_sync_initializer import initialize_configs
                success, message = initialize_configs(logger)
                _log_to_ui(ui_components, f"Sinkronisasi: {message}", "success" if success else "warning")
                return success
            except ImportError:
                # Versi dari ui.setup (fallback)
                try:
                    from smartcash.ui.setup.drive_sync_initializer import initialize_configs
                    # Hubungkan ke UI dan matikan output console
                    success, message = initialize_configs(logger=logger, ui_components=ui_components, silent=True)
                    _log_to_ui(ui_components, f"Sinkronisasi: {message}", "success" if success else "warning")
                    return success
                except ImportError:
                    pass  # Lanjut ke metode berikutnya
            
            # Gunakan config_sync jika drive_sync_initializer tidak tersedia
            from smartcash.common.config_sync import sync_all_configs
            
            # Sinkronisasi dengan strategi 'merge'
            results = sync_all_configs(
                sync_strategy='merge',
                create_backup=True,
                logger=logger
            )
            
            success_count = len(results.get("success", []))
            failure_count = len(results.get("failure", []))
            skipped_count = len(results.get("skipped", []))
            
            _log_to_ui(ui_components, 
                f"Sinkronisasi selesai: {success_count} disinkronisasi, {skipped_count} dilewati, {failure_count} gagal",
                "success" if failure_count == 0 else "warning", "✅")
            
            return True
            
        except ImportError:
            # Fallback ke sinkronisasi file sederhana
            _log_to_ui(ui_components, "Menggunakan metode sinkronisasi file sederhana", "info", "ℹ️")
            return _copy_config_files(ui_components, drive_path)
    except Exception as e:
        _log_to_ui(ui_components, f"Error saat sinkronisasi: {str(e)}", "error", "❌")
        return False

def create_symlinks_to_drive(ui_components: Dict[str, Any], drive_path: Path) -> bool:
    """
    Buat symlinks dari direktori lokal ke direktori Google Drive.
    
    Args:
        ui_components: Dictionary komponen UI
        drive_path: Path direktori di Google Drive
        
    Returns:
        Boolean yang menunjukkan keberhasilan
    """
    logger = ui_components.get('logger')
    import os, shutil
    
    try:
        # Mapping direktori yang akan dibuat symlink
        symlinks = {
            'data': drive_path / 'data',
            'configs': drive_path / 'configs',
            'runs': drive_path / 'runs',
            'logs': drive_path / 'logs',
            'checkpoints': drive_path / 'checkpoints'
        }
        
        _log_to_ui(ui_components, "Membuat symlinks...", "info", "🔗")
        
        # Buat symlinks dengan one-liner untuk setiap pasangan
        for local_name, target_path in symlinks.items():
            target_path.mkdir(parents=True, exist_ok=True)
            local_path = Path(local_name)
            
            # Handle existing directory
            if local_path.exists() and not local_path.is_symlink():
                backup_path = local_path.with_name(f"{local_name}_backup")
                _log_to_ui(ui_components, f"Memindahkan direktori lokal ke backup: {local_name} → {local_name}_backup", "info", "📦")
                
                # Hapus backup lama jika ada dan pindahkan folder saat ini ke backup
                if backup_path.exists(): shutil.rmtree(backup_path)
                local_path.rename(backup_path)
            
            # Buat symlink jika belum ada
            if not local_path.exists():
                local_path.symlink_to(target_path)
                _log_to_ui(ui_components, f"Symlink dibuat: {local_name} → {target_path}", "success", "✅")
                
        return True
    except Exception as e:
        _log_to_ui(ui_components, f"Error membuat symlinks: {str(e)}", "error", "❌")
        return False

def _copy_config_files(ui_components: Dict[str, Any], drive_path: Path) -> bool:
    """
    Salin file konfigurasi antara lokal dan Google Drive.
    
    Args:
        ui_components: Dictionary komponen UI
        drive_path: Path direktori di Google Drive
        
    Returns:
        Boolean yang menunjukkan keberhasilan
    """
    logger = ui_components.get('logger')
    import shutil
    
    try:
        local_configs_dir = Path('configs')
        drive_configs_dir = drive_path / 'configs'
        
        # Pastikan direktori ada di kedua lokasi
        local_configs_dir.mkdir(parents=True, exist_ok=True)
        drive_configs_dir.mkdir(parents=True, exist_ok=True)
        
        # Gunakan ThreadPoolExecutor untuk operasi file
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Salin file yang ada di lokal tapi tidak ada di drive
            local_to_drive_tasks = [(str(local_file), str(drive_configs_dir / local_file.name)) 
                                  for local_file in local_configs_dir.glob('*.yaml') 
                                  if not (drive_configs_dir / local_file.name).exists()]
            
            # Salin file yang ada di drive tapi tidak ada di lokal
            drive_to_local_tasks = [(str(drive_file), str(local_configs_dir / drive_file.name))
                                   for drive_file in drive_configs_dir.glob('*.yaml')
                                   if not (local_configs_dir / drive_file.name).exists()]
            
            # Helper function untuk menyalin file
            def copy_file(args):
                src, dst = args
                try:
                    shutil.copy2(src, dst)
                    return {'success': True, 'src': src, 'dst': dst}
                except Exception as e:
                    return {'success': False, 'src': src, 'dst': dst, 'error': str(e)}
            
            # Jalankan tugas secara paralel
            local_to_drive_results = list(executor.map(copy_file, local_to_drive_tasks))
            drive_to_local_results = list(executor.map(copy_file, drive_to_local_tasks))
        
        # Hitung statistik
        copied_to_drive = sum(1 for r in local_to_drive_results if r['success'])
        copied_from_drive = sum(1 for r in drive_to_local_results if r['success'])
        
        # Tampilkan hasil
        if copied_to_drive > 0:
            _log_to_ui(ui_components, f"{copied_to_drive} file disalin ke Drive", "success", "📤")
        if copied_from_drive > 0:
            _log_to_ui(ui_components, f"{copied_from_drive} file disalin dari Drive", "success", "📥")
        if copied_to_drive == 0 and copied_from_drive == 0:
            _log_to_ui(ui_components, "Tidak ada file yang perlu disinkronisasi", "info", "ℹ️")
        
        return True
    except Exception as e:
        _log_to_ui(ui_components, f"Error saat menyalin file: {str(e)}", "error", "❌")
        return False

def _log_to_ui(ui_components: Dict[str, Any], message: str, level: str = "info", emoji: str = "") -> None:
    """Log pesan ke UI dan logger."""
    logger = ui_components.get('logger')
    
    # Log ke UI
    from smartcash.ui.utils.ui_logger import log_to_ui
    log_to_ui(ui_components, message, level, emoji)
    
    # Log ke logger jika tersedia
    if logger:
        if level == "error": logger.error(message)
        elif level == "warning": logger.warning(message)
        elif level == "success": logger.success(message)
        else: logger.info(message)