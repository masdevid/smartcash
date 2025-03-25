"""
File: smartcash/ui/setup/drive_sync_initializer.py
Deskripsi: Fungsi inisialisasi dan sinkronisasi konfigurasi antara local dan Google Drive yang lebih robust
"""

import os
import shutil
import yaml
import json
import threading
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

# Lock untuk thread-safety saat sinkronisasi
_sync_lock = threading.RLock()
# Flag untuk mencegah inisialisasi berulang
_initialized = False

def copy_configs_to_drive(logger=None):
    """
    Salin semua file konfigurasi dari modul smartcash ke Google Drive.
    
    Args:
        logger: Logger opsional untuk logging
        
    Returns:
        Tuple (success, message)
    """
    try:
        # Dapatkan environment manager
        from smartcash.common.environment import get_environment_manager
        env_manager = get_environment_manager()
        
        if not env_manager.is_drive_mounted:
            if logger: logger.warning("⚠️ Google Drive tidak terpasang")
            return False, "Google Drive tidak terpasang"
            
        # Cari path konfigurasi standar dalam modul
        import smartcash
        module_path = Path(smartcash.__file__).parent
        module_config_path = module_path / "configs"
        
        if not module_config_path.exists():
            if logger: logger.warning(f"⚠️ Folder konfigurasi tidak ditemukan di modul: {module_config_path}")
            return False, f"Folder konfigurasi tidak ditemukan di modul: {module_config_path}"
            
        # Pastikan drive_path ada
        if env_manager.drive_path is None:
            if logger: logger.warning("⚠️ Drive path tidak valid (None)")
            return False, "Drive path tidak valid (None)"
            
        # Siapkan direktori di Drive
        drive_config_path = env_manager.drive_path / "configs"
        os.makedirs(drive_config_path, exist_ok=True)
        
        # Salin semua file konfigurasi
        copied_files = []
        for config_file in module_config_path.glob("*.yaml"):
            target_path = drive_config_path / config_file.name
            # Hanya salin jika file tidak ada atau berbeda
            if not target_path.exists() or _is_file_different(config_file, target_path):
                shutil.copy2(config_file, target_path)
                copied_files.append(config_file.name)
            
        msg = f"Berhasil menyalin {len(copied_files)} file konfigurasi ke Drive"
        if copied_files:
            if logger: logger.success(f"✅ {msg}")
        else:
            msg = "Semua file konfigurasi sudah ada di Drive"
            if logger: logger.info(f"ℹ️ {msg}")
            
        return True, msg
            
    except Exception as e:
        if logger: logger.error(f"❌ Error saat menyalin konfigurasi: {str(e)}")
        return False, f"Error saat menyalin konfigurasi: {str(e)}"

def _is_file_different(file1: Path, file2: Path) -> bool:
    """
    Periksa apakah dua file memiliki konten yang berbeda.
    
    Args:
        file1: Path file pertama
        file2: Path file kedua
        
    Returns:
        Boolean yang menunjukkan apakah file berbeda
    """
    try:
        # Bandingkan ukuran terlebih dahulu sebagai optimasi
        if file1.stat().st_size != file2.stat().st_size:
            return True
            
        # Bandingkan konten
        with open(file1, 'rb') as f1, open(file2, 'rb') as f2:
            for chunk1, chunk2 in zip(iter(lambda: f1.read(4096), b''), 
                                     iter(lambda: f2.read(4096), b'')):
                if chunk1 != chunk2:
                    return True
            
            # Periksa jika salah satu file lebih panjang
            if bool(f1.read(1)) != bool(f2.read(1)):
                return True
                
        return False
    except Exception:
        # Default ke berbeda jika terjadi error
        return True

def is_config_file_valid(file_path):
    """
    Cek apakah file konfigurasi valid.
    
    Args:
        file_path: Path ke file konfigurasi
        
    Returns:
        Boolean yang menunjukkan validitas file
    """
    try:
        path = Path(file_path)
        if not path.exists(): return False
        
        with open(path, 'r', encoding='utf-8') as f:
            data = None
            if path.suffix.lower() in ('.yml', '.yaml'):
                data = yaml.safe_load(f)
            elif path.suffix.lower() == '.json':
                data = json.load(f)
            return data is not None
    except Exception:
        return False

def initialize_configs(logger=None):
    """
    Inisialisasi konfigurasi dengan alur baru dan penanganan None value.
    
    Args:
        logger: Logger opsional untuk logging
        
    Returns:
        Tuple (success, message)
    """
    global _initialized
    
    # Gunakan lock untuk thread-safety
    with _sync_lock:
        # Periksa apakah sudah diinisialisasi
        if _initialized:
            return True, "Konfigurasi sudah diinisialisasi sebelumnya"
        
        try:
            # Dapatkan environment manager
            from smartcash.common.environment import get_environment_manager
            env_manager = get_environment_manager()
            
            # Setup direktori konfigurasi lokal
            local_config_dir = Path("configs")
            local_config_dir.mkdir(parents=True, exist_ok=True)
            
            # Cek apakah konfigurasi lokal ada
            local_configs_exist = list(local_config_dir.glob("*.yaml"))
            
            # Cek apakah Drive terpasang
            if env_manager.is_drive_mounted:
                # Verifikasi drive_path valid
                if env_manager.drive_path is None:
                    if logger: logger.warning("⚠️ Drive terpasang tapi path tidak valid (None)")
                    return False, "Drive terpasang tapi path tidak valid (None)"
                    
                drive_config_dir = env_manager.drive_path / "configs"
                drive_config_dir.mkdir(parents=True, exist_ok=True)
                
                # Cek apakah konfigurasi Drive ada
                drive_configs_exist = list(drive_config_dir.glob("*.yaml"))
                
                # Scenario 1: Config kosong di kedua tempat
                if not local_configs_exist and not drive_configs_exist:
                    if logger: logger.info("ℹ️ Konfigurasi kosong di lokal dan Drive. Menyalin dari modul...")
                    success, message = copy_configs_to_drive(logger)
                    if success:
                        # Salin dari Drive ke lokal
                        for config_file in drive_config_dir.glob("*.yaml"):
                            target_path = local_config_dir / config_file.name
                            if not target_path.exists():
                                shutil.copy2(config_file, target_path)
                        if logger: logger.success("✅ Konfigurasi berhasil disalin ke lokal")
                        
                    _initialized = True
                    return success, message
                    
                # Scenario 2: Config ada di lokal tapi kosong di Drive
                elif local_configs_exist and not drive_configs_exist:
                    if logger: logger.info("ℹ️ Konfigurasi kosong di Drive. Menyalin dari modul...")
                    success, message = copy_configs_to_drive(logger)
                    if success:
                        # Ganti lokal dengan Drive
                        for config_file in drive_config_dir.glob("*.yaml"):
                            target_path = local_config_dir / config_file.name
                            if not target_path.exists() or _is_file_different(config_file, target_path):
                                shutil.copy2(config_file, target_path)
                        if logger: logger.success("✅ Konfigurasi lokal diganti dengan Drive")
                    
                    _initialized = True
                    return success, message
                    
                # Scenario 3: Config kosong di lokal tapi ada di Drive
                elif not local_configs_exist and drive_configs_exist:
                    if logger: logger.info("ℹ️ Konfigurasi kosong di lokal. Menyalin dari Drive...")
                    # Salin dari Drive ke lokal
                    copied = 0
                    for config_file in drive_config_dir.glob("*.yaml"):
                        target_path = local_config_dir / config_file.name
                        if not target_path.exists():
                            shutil.copy2(config_file, target_path)
                            copied += 1
                    
                    msg = f"Konfigurasi berhasil disalin dari Drive ({copied} file)"
                    if logger: logger.success(f"✅ {msg}")
                    _initialized = True
                    return True, msg
                    
                # Scenario 4: Config ada di kedua tempat
                else:
                    if logger: logger.info("ℹ️ Konfigurasi ditemukan di Drive dan lokal. Menggunakan Drive sebagai sumber kebenaran...")
                    # Gunakan Drive sebagai sumber kebenaran
                    try:
                        from smartcash.common.config import get_config_manager
                        config_manager = get_config_manager()
                        if hasattr(config_manager, 'use_drive_as_source_of_truth'):
                            success = config_manager.use_drive_as_source_of_truth()
                            _initialized = True
                            return success, "Sinkronisasi konfigurasi selesai"
                    except Exception as e:
                        if logger: logger.warning(f"⚠️ Error saat sinkronisasi otomatis: {str(e)}")
                        # Fallback: salin manual dari Drive ke lokal
                        copied = 0
                        for config_file in drive_config_dir.glob("*.yaml"):
                            target_path = local_config_dir / config_file.name
                            if not target_path.exists() or _is_file_different(config_file, target_path):
                                shutil.copy2(config_file, target_path)
                                copied += 1
                                
                        msg = f"Konfigurasi disalin manual dari Drive ke lokal ({copied} file)"
                        if logger: logger.info(f"ℹ️ {msg}")
                        _initialized = True
                        return True, msg
                    
                    _initialized = True
                    return True, "Konfigurasi sudah ada di kedua tempat"
                    
            # Scenario 5: Drive tidak terpasang, cek lokal
            else:
                if not local_configs_exist:
                    if logger: logger.warning("⚠️ Drive tidak terpasang dan konfigurasi lokal kosong")
                    # Cari path konfigurasi standar dalam modul
                    try:
                        import smartcash
                        module_path = Path(smartcash.__file__).parent
                        module_config_path = module_path / "configs"
                        
                        if module_config_path.exists():
                            if logger: logger.info("ℹ️ Menyalin konfigurasi dari modul ke lokal...")
                            # Salin konfigurasi dari modul
                            copied = 0
                            for config_file in module_config_path.glob("*.yaml"):
                                target_path = local_config_dir / config_file.name
                                if not target_path.exists():
                                    shutil.copy2(config_file, target_path)
                                    copied += 1
                                    
                            msg = f"Konfigurasi berhasil disalin dari modul ({copied} file)"
                            if logger: logger.success(f"✅ {msg}")
                            _initialized = True
                            return True, msg
                        else:
                            if logger: logger.error("❌ Tidak dapat menemukan konfigurasi")
                            # Fallback: Buat konfigurasi default
                            from smartcash.common.default_config import ensure_base_config_exists
                            if ensure_base_config_exists():
                                _initialized = True
                                return True, "Konfigurasi default berhasil dibuat"
                            return False, "Tidak dapat menemukan konfigurasi"
                    except Exception as e:
                        if logger: logger.error(f"❌ Error saat mencari konfigurasi modul: {str(e)}")
                        _initialized = True
                        return False, f"Error saat mencari konfigurasi modul: {str(e)}"
                else:
                    if logger: logger.info("ℹ️ Menggunakan konfigurasi lokal yang ada")
                    _initialized = True
                    return True, "Menggunakan konfigurasi lokal yang ada"
                    
        except Exception as e:
            if logger: logger.error(f"❌ Error saat inisialisasi konfigurasi: {str(e)}")
            return False, f"Error saat inisialisasi konfigurasi: {str(e)}"
        
        finally:
            # Tandai sudah diinisialisasi
            _initialized = True