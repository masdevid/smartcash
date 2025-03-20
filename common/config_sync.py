"""
File: smartcash/common/config_sync.py
Deskripsi: Utilitas untuk sinkronisasi konfigurasi antara lokal dan Google Drive dengan strategi resolusi konflik yang lebih komprehensif
"""

import os
import yaml
import shutil
import time
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

def get_modified_time(file_path: str) -> float:
    """Dapatkan waktu modifikasi file."""
    return os.path.getmtime(file_path) if os.path.exists(file_path) else 0

def load_yaml_config(file_path: str) -> Dict[str, Any]:
    """Load konfigurasi YAML dengan error handling."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        print(f"❌ Error saat loading {file_path}: {str(e)}")
        return {}

def save_yaml_config(config: Dict[str, Any], file_path: str) -> bool:
    """Simpan konfigurasi ke file YAML."""
    try:
        # Buat directori jika belum ada
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False)
        return True
    except Exception as e:
        print(f"❌ Error saat menyimpan {file_path}: {str(e)}")
        return False

def create_backup(file_path: str) -> str:
    """Buat backup file dengan timestamp."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    backup_path = f"{file_path}.{timestamp}.bak"
    try:
        shutil.copy2(file_path, backup_path)
        return backup_path
    except Exception as e:
        print(f"⚠️ Error saat backup {file_path}: {str(e)}")
        return ""

def deep_merge_configs(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge dua config secara rekursif dengan strategi smart:
    - Untuk nested dicts, merge rekursif
    - Untuk list, gabungkan tanpa duplikat
    - Untuk nilai skalar, overlay menang
    
    Args:
        base: Konfigurasi dasar
        overlay: Konfigurasi yang akan di-overlay
        
    Returns:
        Konfigurasi hasil merge
    """
    import copy
    result = copy.deepcopy(base)
    
    for key, value in overlay.items():
        # Jika overlay memiliki dict dan base juga, merge rekursif
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge_configs(result[key], value)
        # Jika overlay memiliki list dan base juga, gabungkan tanpa duplikat
        elif key in result and isinstance(result[key], list) and isinstance(value, list):
            # Untuk list sederhana, hilangkan duplikat
            if all(not isinstance(x, (dict, list)) for x in result[key] + value):
                unique_items = list(set(result[key] + value))
                result[key] = sorted(unique_items) if all(isinstance(x, (int, float, str)) for x in unique_items) else unique_items
            # Untuk list kompleks, gabungkan saja
            else:
                result[key] = result[key] + value
        # Selain itu, nilai overlay menang
        else:
            result[key] = copy.deepcopy(value)
    
    return result

def sync_config_with_drive(
    config_file: str,
    drive_path: Optional[str] = None,
    local_path: Optional[str] = None,
    sync_strategy: str = 'drive_priority',
    create_backup: bool = True,
    logger = None
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Sinkronisasi file konfigurasi antara lokal dan Google Drive.
    
    Args:
        config_file: Nama file konfigurasi (mis. 'base_config.yaml')
        drive_path: Path di Drive (default: 'SmartCash/configs/{config_file}')
        local_path: Path lokal (default: 'configs/{config_file}')
        sync_strategy: Strategi sinkronisasi:
            - 'drive_priority': Config Drive menang
            - 'local_priority': Config lokal menang
            - 'newest': Config terbaru menang
            - 'merge': Gabungkan config dengan strategi smart
        create_backup: Buat backup sebelum update
        logger: Logger untuk logging (opsional)
        
    Returns:
        Tuple (success, message, merged_config)
    """
    # Validasi drive path
    if not drive_path:
        try:
            from smartcash.common.environment import get_environment_manager
            env_manager = get_environment_manager()
            if not env_manager.is_drive_mounted:
                return False, "Google Drive tidak terpasang", {}
            drive_path = str(env_manager.drive_path / 'configs' / config_file)
        except Exception:
            return False, "Tidak dapat menentukan drive path", {}
    
    # Validasi local path
    if not local_path:
        local_path = os.path.join('configs', config_file)
    
    # Cek keberadaan file
    drive_exists = os.path.exists(drive_path)
    local_exists = os.path.exists(local_path)
    
    if not drive_exists and not local_exists:
        return False, f"Tidak ada file konfigurasi ditemukan untuk {config_file}", {}
    
    # Jika hanya satu file ada, salin ke yang lain
    if drive_exists and not local_exists:
        # Buat direktori lokal jika perlu
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        shutil.copy2(drive_path, local_path)
        config = load_yaml_config(local_path)
        return True, f"File konfigurasi dari Drive disalin ke lokal: {config_file}", config
    
    if local_exists and not drive_exists:
        # Buat direktori Drive jika perlu
        os.makedirs(os.path.dirname(drive_path), exist_ok=True)
        shutil.copy2(local_path, drive_path)
        config = load_yaml_config(drive_path)
        return True, f"File konfigurasi lokal disalin ke Drive: {config_file}", config
    
    # Kedua file ada, terapkan strategi sinkronisasi
    # Dapatkan timestamp dan load config
    drive_time = get_modified_time(drive_path)
    local_time = get_modified_time(local_path)
    drive_config = load_yaml_config(drive_path)
    local_config = load_yaml_config(local_path)
    
    # Buat backup jika diperlukan
    if create_backup:
        if drive_exists:
            drive_backup = create_backup(drive_path)
            if logger and drive_backup:
                logger.info(f"📦 Backup Drive config: {drive_backup}")
        if local_exists:
            local_backup = create_backup(local_path)
            if logger and local_backup:
                logger.info(f"📦 Backup local config: {local_backup}")
    
    # Terapkan strategi sinkronisasi
    result_config = {}
    message = ""
    
    if sync_strategy == 'drive_priority':
        result_config = drive_config
        shutil.copy2(drive_path, local_path)
        message = f"Konfigurasi Drive diterapkan ke lokal: {config_file}"
    
    elif sync_strategy == 'local_priority':
        result_config = local_config
        shutil.copy2(local_path, drive_path)
        message = f"Konfigurasi lokal diterapkan ke Drive: {config_file}"
    
    elif sync_strategy == 'newest':
        if drive_time > local_time:
            result_config = drive_config
            shutil.copy2(drive_path, local_path)
            message = f"Konfigurasi Drive (lebih baru) diterapkan ke lokal: {config_file}"
        else:
            result_config = local_config
            shutil.copy2(local_path, drive_path)
            message = f"Konfigurasi lokal (lebih baru) diterapkan ke Drive: {config_file}"
    
    elif sync_strategy == 'merge':
        # Merge dengan drive sebagai overlay
        result_config = deep_merge_configs(local_config, drive_config)
        
        # Simpan hasil merge ke kedua tempat
        save_yaml_config(result_config, local_path)
        save_yaml_config(result_config, drive_path)
        message = f"Konfigurasi berhasil digabungkan: {config_file}"
    
    else:
        return False, f"Strategi sinkronisasi tidak dikenal: {sync_strategy}", {}
    
    if logger:
        logger.info(f"✅ {message}")
    
    return True, message, result_config

def sync_all_configs(
    drive_configs_dir: Optional[str] = None,
    local_configs_dir: Optional[str] = None,
    sync_strategy: str = 'drive_priority',
    create_backup: bool = True,
    logger = None
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Sinkronisasi semua file konfigurasi YAML.
    
    Args:
        drive_configs_dir: Direktori konfigurasi di Drive
        local_configs_dir: Direktori konfigurasi lokal
        sync_strategy: Strategi sinkronisasi
        create_backup: Buat backup sebelum update
        logger: Logger untuk logging
        
    Returns:
        Dictionary berisi hasil sinkronisasi
    """
    # Dapatkan drive path jika tidak disediakan
    if not drive_configs_dir:
        try:
            from smartcash.common.environment import get_environment_manager
            env_manager = get_environment_manager()
            if not env_manager.is_drive_mounted:
                if logger and hasattr(logger, 'warning'):
                    logger.warning("⚠️ Google Drive tidak terpasang, tidak dapat sinkronisasi")
                return {"error": "Google Drive tidak terpasang"}
            drive_configs_dir = str(env_manager.drive_path / 'SmartCash/configs')
        except Exception as e:
            if logger and hasattr(logger, 'error'):
                logger.error(f"❌ Error mendapatkan path Drive: {str(e)}")
            return {"error": f"Tidak dapat menentukan drive path: {str(e)}"}
    
    # Tentukan direktori lokal konfigurasi
    if not local_configs_dir:
        # Coba beberapa lokasi potensial
        potential_dirs = [
            'configs',
            '/content/configs',
            '/content/smartcash/configs'
        ]
        
        for dir_path in potential_dirs:
            if os.path.exists(dir_path):
                local_configs_dir = dir_path
                break
        
        # Jika tidak ditemukan, gunakan default
        if not local_configs_dir:
            local_configs_dir = 'configs'
            os.makedirs(local_configs_dir, exist_ok=True)
    
    # Pastikan direktori ada
    os.makedirs(drive_configs_dir, exist_ok=True)
    os.makedirs(local_configs_dir, exist_ok=True)
    
    # Cari semua file YAML
    yaml_files = set()
    
    # Fungsi untuk mencari file YAML di direktori
    def find_yaml_files(directory):
        files = set()
        for ext in ['.yaml', '.yml']:
            try:
                for filename in os.listdir(directory):
                    if filename.endswith(ext):
                        files.add(filename)
            except Exception:
                pass
        return files
    
    # Gabungkan file dari direktori lokal dan Drive
    yaml_files.update(find_yaml_files(local_configs_dir))
    yaml_files.update(find_yaml_files(drive_configs_dir))
    
    # Hasil sinkronisasi
    results = {
        "success": [],
        "failure": []
    }
    
    # Proses setiap file
    for config_file in yaml_files:
        try:
            drive_path = os.path.join(drive_configs_dir, config_file)
            local_path = os.path.join(local_configs_dir, config_file)
            
            # Pastikan logger bisa dipanggil
            if logger and not hasattr(logger, 'error'):
                logger = None
            
            success, message, _ = sync_config_with_drive(
                config_file=config_file,
                drive_path=drive_path,
                local_path=local_path,
                sync_strategy=sync_strategy,
                create_backup=create_backup,
                logger=logger
            )
            
            result = {
                "file": config_file,
                "message": message
            }
            
            if success:
                results["success"].append(result)
            else:
                results["failure"].append(result)
                
        except Exception as e:
            error_message = f"❌ Error total saat sinkronisasi {config_file}: {str(e)}"
            
            # Cetak error untuk debug
            print(error_message)
            
            # Log error jika logger tersedia
            if logger and hasattr(logger, 'error'):
                logger.error(error_message)
            
            results["failure"].append({
                "file": config_file,
                "message": error_message
            })
    
    # Log summary
    if logger and hasattr(logger, 'info'):
        logger.info(f"🔄 Sinkronisasi selesai: {len(results['success'])} berhasil, {len(results['failure'])} gagal")
    
    return results