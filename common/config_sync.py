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
    """
    # Print debug info
    print(f"\n🔄 Sinkronisasi file: {config_file}")
    print(f"   Strategi: {sync_strategy}")
    
    # Tambahan pengecekan dan konversi logger
    if logger is not None:
        try:
            # Pastikan logger memiliki method yang bisa dipanggil
            if not callable(getattr(logger, 'info', None)):
                logger = None
        except Exception:
            logger = None
    
    # Dapatkan drive path jika tidak disediakan
    if not drive_path:
        try:
            from smartcash.common.environment import get_environment_manager
            env_manager = get_environment_manager()
            if not env_manager.is_drive_mounted:
                print("❌ Google Drive tidak terpasang")
                return False, "Google Drive tidak terpasang", {}
            drive_path = str(env_manager.drive_path / 'SmartCash/configs' / config_file)
        except Exception as e:
            print(f"❌ Error mendapatkan path Drive: {str(e)}")
            return False, f"Tidak dapat menentukan drive path: {str(e)}", {}
    
    # Validasi local path
    if not local_path:
        local_path = os.path.join('configs', config_file)
    
    # Cetak path untuk debug
    print(f"   Path Lokal: {local_path}")
    print(f"   Path Drive: {drive_path}")
    
    # Cek keberadaan file
    drive_exists = os.path.exists(drive_path)
    local_exists = os.path.exists(local_path)
    
    print(f"   File Lokal Ada: {local_exists}")
    print(f"   File Drive Ada: {drive_exists}")
    
    if not local_exists:
        print("❌ File lokal tidak ditemukan")
        return False, f"File lokal tidak ditemukan: {local_path}", {}
    
    # Jika file drive tidak ada, salin dari lokal
    if not drive_exists:
        try:
            # Buat direktori drive jika belum ada
            os.makedirs(os.path.dirname(drive_path), exist_ok=True)
            
            # Salin file dari lokal ke drive
            shutil.copy2(local_path, drive_path)
            
            print(f"✅ Menyalin file dari lokal ke Drive: {config_file}")
            
            # Baca konfigurasi
            config = load_yaml_config(local_path)
            return True, f"File konfigurasi lokal disalin ke Drive: {config_file}", config
        except Exception as e:
            print(f"❌ Gagal menyalin file: {str(e)}")
            return False, f"Gagal menyalin file: {str(e)}", {}
    
    # Baca konfigurasi
    drive_config = load_yaml_config(drive_path)
    local_config = load_yaml_config(local_path)
    
    # Buat backup jika diperlukan
    if create_backup:
        try:
            backup_drive = create_backup(drive_path)
            backup_local = create_backup(local_path)
            print(f"📦 Backup Drive: {backup_drive}")
            print(f"📦 Backup Lokal: {backup_local}")
        except Exception as e:
            print(f"⚠️ Gagal membuat backup: {str(e)}")
    
    # Terapkan strategi sinkronisasi
    try:
        if sync_strategy == 'drive_priority':
            result_config = drive_config
            shutil.copy2(drive_path, local_path)
            message = f"Konfigurasi Drive diterapkan ke lokal: {config_file}"
        
        elif sync_strategy == 'local_priority':
            result_config = local_config
            shutil.copy2(local_path, drive_path)
            message = f"Konfigurasi lokal diterapkan ke Drive: {config_file}"
        
        elif sync_strategy == 'newest':
            # Dapatkan waktu modifikasi
            drive_time = os.path.getmtime(drive_path)
            local_time = os.path.getmtime(local_path)
            
            if drive_time > local_time:
                result_config = drive_config
                shutil.copy2(drive_path, local_path)
                message = f"Konfigurasi Drive (lebih baru) diterapkan ke lokal: {config_file}"
            else:
                result_config = local_config
                shutil.copy2(local_path, drive_path)
                message = f"Konfigurasi lokal (lebih baru) diterapkan ke Drive: {config_file}"
        
        elif sync_strategy == 'merge':
            # Merge konfigurasi
            result_config = deep_merge_configs(local_config, drive_config)
            
            # Simpan hasil merge ke kedua tempat
            save_yaml_config(result_config, local_path)
            save_yaml_config(result_config, drive_path)
            message = f"Konfigurasi berhasil digabungkan: {config_file}"
        
        else:
            return False, f"Strategi sinkronisasi tidak dikenal: {sync_strategy}", {}
        
        print(f"✅ {message}")
        return True, message, result_config
    
    except Exception as e:
        print(f"❌ Error saat sinkronisasi: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, f"Error sinkronisasi: {str(e)}", {}
def sync_all_configs(
    drive_configs_dir: Optional[str] = None,
    local_configs_dir: Optional[str] = None,
    sync_strategy: str = 'drive_priority',
    create_backup: bool = True,
    logger = None
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Sinkronisasi semua file konfigurasi YAML.
    """
    # Print debug info
    print("🔍 Memulai sinkronisasi konfigurasi...")
    
    # Dapatkan drive path jika tidak disediakan
    if not drive_configs_dir:
        try:
            from smartcash.common.environment import get_environment_manager
            env_manager = get_environment_manager()
            if not env_manager.is_drive_mounted:
                print("⚠️ Google Drive tidak terpasang")
                return {"error": "Google Drive tidak terpasang"}
            drive_configs_dir = str(env_manager.drive_path / 'SmartCash/configs')
        except Exception as e:
            print(f"❌ Error mendapatkan path Drive: {str(e)}")
            return {"error": f"Tidak dapat menentukan drive path: {str(e)}"}
    
    # Tentukan direktori lokal konfigurasi
    if not local_configs_dir:
        # Coba beberapa lokasi potensial
        potential_dirs = [
            os.path.join(os.getcwd(), 'configs'),
            '/content/configs',
            '/content/smartcash/configs',
            os.path.join(os.getcwd(), 'smartcash/configs')
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
    
    # Print direktori yang digunakan
    print(f"📁 Direktori Lokal: {local_configs_dir}")
    print(f"📁 Direktori Drive: {drive_configs_dir}")
    
    # Cari semua file YAML di direktori lokal
    def find_yaml_files(directory):
        yaml_files = set()
        try:
            for filename in os.listdir(directory):
                if filename.lower().endswith(('.yaml', '.yml')):
                    yaml_files.add(filename)
        except Exception as e:
            print(f"❌ Error membaca direktori {directory}: {str(e)}")
        return yaml_files
    
    # Temukan file YAML di direktori lokal
    local_yaml_files = find_yaml_files(local_configs_dir)
    print(f"🔎 File YAML di {local_configs_dir}: {local_yaml_files}")
    
    # Hasil sinkronisasi
    results = {
        "success": [],
        "failure": []
    }
    
    # Proses setiap file
    for config_file in local_yaml_files:
        try:
            drive_path = os.path.join(drive_configs_dir, config_file)
            local_path = os.path.join(local_configs_dir, config_file)
            
            print(f"\n🔄 Memproses file: {config_file}")
            print(f"   Lokal: {local_path}")
            print(f"   Drive: {drive_path}")
            
            # Pastikan logger bisa dipanggil
            if logger and not callable(getattr(logger, 'error', None)):
                logger = None
            
            # Baca konten file lokal untuk debugging
            with open(local_path, 'r') as f:
                local_content = f.read()
                print(f"   Konten Lokal (pratinjau):\n{local_content[:500]}...")
            
            success, message, merged_config = sync_config_with_drive(
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
                print(f"✅ Sinkronisasi {config_file} berhasil")
            else:
                results["failure"].append(result)
                print(f"❌ Sinkronisasi {config_file} gagal")
                
        except Exception as e:
            error_message = f"❌ Error total saat sinkronisasi {config_file}: {str(e)}"
            
            # Cetak error untuk debug
            print(error_message)
            import traceback
            traceback.print_exc()
            
            # Log error jika logger tersedia
            if logger and hasattr(logger, 'error'):
                logger.error(error_message)
            
            results["failure"].append({
                "file": config_file,
                "message": error_message
            })
    
    # Print summary
    print(f"\n📊 Sinkronisasi selesai: {len(results['success'])} berhasil, {len(results['failure'])} gagal")
    
    return results