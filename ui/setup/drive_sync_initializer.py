"""
File: smartcash/common/drive_sync_initializer.py
Deskripsi: Fungsi inisialisasi dan sinkronisasi konfigurasi antara local dan Google Drive, dengan alur baru yang lebih konsisten
"""

import os
import shutil
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

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
            
        # Siapkan direktori di Drive
        drive_config_path = env_manager.drive_path / "configs"
        os.makedirs(drive_config_path, exist_ok=True)
        
        # Salin semua file konfigurasi
        copied_files = []
        for config_file in module_config_path.glob("*.yaml"):
            target_path = drive_config_path / config_file.name
            shutil.copy2(config_file, target_path)
            copied_files.append(config_file.name)
            
        if logger: logger.success(f"✅ Berhasil menyalin {len(copied_files)} file konfigurasi ke Drive")
        return True, f"Berhasil menyalin {len(copied_files)} file konfigurasi ke Drive"
            
    except Exception as e:
        if logger: logger.error(f"❌ Error saat menyalin konfigurasi: {str(e)}")
        return False, f"Error saat menyalin konfigurasi: {str(e)}"

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
            data = yaml.safe_load(f) if path.suffix.lower() in ('.yml', '.yaml') else json.load(f) if path.suffix.lower() == '.json' else None
            return data is not None
    except Exception:
        return False

def initialize_configs(logger=None):
    """
    Inisialisasi konfigurasi dengan alur baru.
    
    Args:
        logger: Logger opsional untuk logging
        
    Returns:
        Tuple (success, message)
    """
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
                        shutil.copy2(config_file, target_path)
                    if logger: logger.success("✅ Konfigurasi berhasil disalin ke lokal")
                    
                return success, message
                
            # Scenario 2: Config ada di lokal tapi kosong di Drive
            elif local_configs_exist and not drive_configs_exist:
                if logger: logger.info("ℹ️ Konfigurasi kosong di Drive. Menyalin dari modul...")
                success, message = copy_configs_to_drive(logger)
                if success:
                    # Ganti lokal dengan Drive
                    for config_file in drive_config_dir.glob("*.yaml"):
                        target_path = local_config_dir / config_file.name
                        shutil.copy2(config_file, target_path)
                    if logger: logger.success("✅ Konfigurasi lokal diganti dengan Drive")
                    
                return success, message
                
            # Scenario 3: Config kosong di lokal tapi ada di Drive
            elif not local_configs_exist and drive_configs_exist:
                if logger: logger.info("ℹ️ Konfigurasi kosong di lokal. Menyalin dari Drive...")
                # Salin dari Drive ke lokal
                for config_file in drive_config_dir.glob("*.yaml"):
                    target_path = local_config_dir / config_file.name
                    shutil.copy2(config_file, target_path)
                if logger: logger.success("✅ Konfigurasi berhasil disalin dari Drive")
                return True, "Konfigurasi berhasil disalin dari Drive"
                
            # Scenario 4: Config ada di kedua tempat
            else:
                if logger: logger.info("ℹ️ Konfigurasi ditemukan di Drive dan lokal. Menggunakan Drive sebagai sumber kebenaran...")
                # Gunakan Drive sebagai sumber kebenaran
                from smartcash.common.config import get_config_manager
                config_manager = get_config_manager()
                if hasattr(config_manager, 'use_drive_as_source_of_truth'):
                    success = config_manager.use_drive_as_source_of_truth()
                    return success, "Sinkronisasi konfigurasi selesai"
                return True, "Konfigurasi sudah ada di kedua tempat"
                
        # Scenario 5: Drive tidak terpasang, cek lokal
        else:
            if not local_configs_exist:
                if logger: logger.warning("⚠️ Drive tidak terpasang dan konfigurasi lokal kosong")
                # Cari path konfigurasi standar dalam modul
                import smartcash
                module_path = Path(smartcash.__file__).parent
                module_config_path = module_path / "configs"
                
                if module_config_path.exists():
                    if logger: logger.info("ℹ️ Menyalin konfigurasi dari modul ke lokal...")
                    # Salin konfigurasi dari modul
                    for config_file in module_config_path.glob("*.yaml"):
                        target_path = local_config_dir / config_file.name
                        shutil.copy2(config_file, target_path)
                    if logger: logger.success("✅ Konfigurasi berhasil disalin dari modul")
                    return True, "Konfigurasi berhasil disalin dari modul"
                else:
                    if logger: logger.error("❌ Tidak dapat menemukan konfigurasi")
                    return False, "Tidak dapat menemukan konfigurasi"
            else:
                if logger: logger.info("ℹ️ Menggunakan konfigurasi lokal yang ada")
                return True, "Menggunakan konfigurasi lokal yang ada"
                
    except Exception as e:
        if logger: logger.error(f"❌ Error saat inisialisasi konfigurasi: {str(e)}")
        return False, f"Error saat inisialisasi konfigurasi: {str(e)}"