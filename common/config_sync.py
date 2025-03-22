"""
File: smartcash/common/config_sync.py
Deskripsi: Utilitas ringkas untuk sinkronisasi konfigurasi antara lokal dan Google Drive dengan perbaikan error boolean
"""

import os, shutil, yaml, json, copy
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union

def load_config_file(file_path: str) -> Dict[str, Any]:
    """Muat konfigurasi dari file YAML/JSON"""
    file_path = Path(file_path)
    if not file_path.exists(): return {}
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return (yaml.safe_load(f) or {}) if file_path.suffix.lower() in ('.yml', '.yaml') else json.load(f) if file_path.suffix.lower() == '.json' else {}
    except Exception as e:
        print(f"❌ Error saat memuat konfigurasi dari {file_path}: {str(e)}")
        return {}

def save_config_file(config: Dict[str, Any], file_path: str, create_dirs: bool = True) -> bool:
    """Simpan konfigurasi ke file YAML/JSON"""
    file_path = Path(file_path)
    
    try:
        if create_dirs and not file_path.parent.exists(): file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            if file_path.suffix.lower() in ('.yml', '.yaml'): yaml.dump(config, f, default_flow_style=False)
            elif file_path.suffix.lower() == '.json': json.dump(config, f, indent=2)
            else: yaml.dump(config, f, default_flow_style=False)
        return True
    except Exception as e:
        print(f"❌ Error saat menyimpan konfigurasi ke {file_path}: {str(e)}")
        return False

def are_configs_identical(config1: Dict[str, Any], config2: Dict[str, Any]) -> bool:
    """Periksa apakah dua konfigurasi identik"""
    return json.dumps(config1, sort_keys=True) == json.dumps(config2, sort_keys=True)

def merge_configs_smart(config1: Dict[str, Any], config2: Dict[str, Any]) -> Dict[str, Any]:
    """Gabungkan dua konfigurasi dengan strategi smart"""
    if config1 is None: return copy.deepcopy(config2)
    if config2 is None: return copy.deepcopy(config1)
    
    # Dict: gabungkan rekursif
    if isinstance(config1, dict) and isinstance(config2, dict):
        result = copy.deepcopy(config1)
        for key, value in config2.items():
            result[key] = merge_configs_smart(result[key], value) if key in result else copy.deepcopy(value)
        return result
    
    # List: gabungkan dengan filter duplikat jika perlu
    if isinstance(config1, list) and isinstance(config2, list):
        return list(set(config1 + config2)) if all(not isinstance(x, (dict, list)) for x in config1 + config2) else copy.deepcopy(config1) + copy.deepcopy(config2)
    
    # Nilai skalar: prioritaskan nilai yang tidak kosong
    return copy.deepcopy(config2) if config1 == "" or config1 is None or config1 == 0 else copy.deepcopy(config1)

def get_environment(logger=None):
    """Dapatkan environment manager dengan fallback"""
    try:
        # Gunakan import_with_fallback dari fallback_utils jika tersedia
        try:
            from smartcash.ui.utils.fallback_utils import import_with_fallback
            get_env_manager = import_with_fallback('smartcash.common.environment.get_environment_manager')
            return get_env_manager() if get_env_manager else None
        except ImportError:
            # Fallback langsung jika fungsi tidak tersedia
            from smartcash.common.environment import get_environment_manager
            return get_environment_manager()
    except Exception as e:
        if logger: logger.warning(f"⚠️ Environment manager tidak tersedia: {str(e)}")
        return None

def sync_config_with_drive(
    config_file: str,
    sync_strategy: str = 'merge',
    create_backup: bool = True,
    logger = None
) -> Tuple[bool, str, Dict[str, Any]]:
    """Sinkronisasi file konfigurasi dengan Google Drive"""
    try:
        # Setup environment dan path dengan fallback terpusat
        env_manager = get_environment(logger)
        if not env_manager or not getattr(env_manager, 'is_drive_mounted', False): 
            if logger: logger.warning("⚠️ Google Drive tidak terpasang")
            return False, "Google Drive tidak terpasang", {}
        
        # Path konfigurasi
        local_config_path = Path("configs") / config_file
        drive_config_path = env_manager.drive_path / "configs" / config_file
        
        # Validasi file ada
        if not local_config_path.exists() and not drive_config_path.exists():
            if logger: logger.warning(f"⚠️ File konfigurasi tidak ditemukan: {config_file}")
            return False, f"File konfigurasi tidak ditemukan: {config_file}", {}
        
        # Backup jika diminta - PERBAIKAN: create_backup adalah boolean, bukan callable
        if create_backup and local_config_path.exists():
            try:
                backup_dir = Path("configs/backup")
                backup_dir.mkdir(parents=True, exist_ok=True)
                
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = backup_dir / f"{local_config_path.stem}_{timestamp}{local_config_path.suffix}"
                
                shutil.copy2(local_config_path, backup_path)
                if logger: logger.info(f"✅ Backup berhasil dibuat: {backup_path}")
            except Exception as e:
                if logger: logger.warning(f"⚠️ Error saat membuat backup: {str(e)}")
        
        # Load konfigurasi
        local_config = load_config_file(local_config_path) if local_config_path.exists() else {}
        drive_config = load_config_file(drive_config_path) if drive_config_path.exists() else {}
        
        # Proses berdasarkan strategi
        if sync_strategy == 'merge':
            # Strategi merge: gabungkan dan simpan ke kedua lokasi
            merged_config = merge_configs_smart(local_config, drive_config)
            success = save_config_file(merged_config, local_config_path) and save_config_file(merged_config, drive_config_path)
            
            if success:
                if logger: logger.info(f"✅ Konfigurasi berhasil disinkronisasi dengan strategi merge: {config_file}")
                return True, f"Sinkronisasi berhasil dengan strategi merge", merged_config
            else:
                if logger: logger.warning(f"⚠️ Error saat menyimpan hasil merge: {config_file}")
                return False, "Error saat menyimpan hasil merge", {}
                
        elif sync_strategy == 'drive_priority':
            # Strategi drive priority: Drive → lokal
            if drive_config_path.exists():
                if local_config_path.exists() and are_configs_identical(local_config, drive_config):
                    if logger: logger.info(f"ℹ️ Konfigurasi sudah identik: {config_file}")
                    return True, "Konfigurasi sudah identik", drive_config
                
                success = save_config_file(drive_config, local_config_path)
                return (True, "Konfigurasi berhasil disinkronisasi dari Drive", drive_config) if success else (False, "Error saat menyimpan dari Drive", {})
            else:
                # Jika tidak ada di Drive, salin dari lokal ke Drive
                success = save_config_file(local_config, drive_config_path) if local_config_path.exists() else False
                return (True, "Konfigurasi berhasil disalin ke Drive", local_config) if success else (False, "Error saat menyalin ke Drive", {})
                
        elif sync_strategy == 'local_priority':
            # Strategi local priority: lokal → Drive
            if local_config_path.exists():
                if drive_config_path.exists() and are_configs_identical(local_config, drive_config):
                    if logger: logger.info(f"ℹ️ Konfigurasi sudah identik: {config_file}")
                    return True, "Konfigurasi sudah identik", local_config
                
                success = save_config_file(local_config, drive_config_path)
                return (True, "Konfigurasi berhasil disinkronisasi ke Drive", local_config) if success else (False, "Error saat menyimpan ke Drive", {})
            else:
                # Jika tidak ada lokal, salin dari Drive ke lokal
                success = save_config_file(drive_config, local_config_path) if drive_config_path.exists() else False
                return (True, "Konfigurasi berhasil disalin dari Drive", drive_config) if success else (False, "Error saat menyalin dari Drive", {})
        else:
            if logger: logger.warning(f"⚠️ Strategi sinkronisasi tidak valid: {sync_strategy}")
            return False, f"Strategi sinkronisasi tidak valid: {sync_strategy}", {}
    
    except Exception as e:
        if logger: logger.error(f"❌ Error saat sinkronisasi {config_file}: {str(e)}")
        return False, f"Error saat sinkronisasi {config_file}: {str(e)}", {}

def sync_all_configs(
    sync_strategy: str = 'merge',
    config_dir: str = 'configs',
    create_backup: bool = True,
    logger = None
) -> Dict[str, List[Dict[str, Any]]]:
    """Sinkronisasi semua file konfigurasi YAML/JSON dengan Google Drive"""
    results = {"success": [], "failure": [], "skipped": []}
    
    try:
        # Setup environment dengan fallback terpusat
        env_manager = get_environment(logger)
        if not env_manager or not getattr(env_manager, 'is_drive_mounted', False):
            if logger: logger.warning("⚠️ Google Drive tidak terpasang")
            return results
        
        # Setup path dan pastikan direktori ada
        local_config_dir, drive_config_dir = Path(config_dir), env_manager.drive_path / "configs"
        local_config_dir.mkdir(parents=True, exist_ok=True); drive_config_dir.mkdir(parents=True, exist_ok=True)
        
        # Kumpulkan semua file konfigurasi
        def get_yaml_json_files(dir_path): return list(dir_path.glob("*.yaml")) + list(dir_path.glob("*.yml")) + list(dir_path.glob("*.json"))
        all_config_files = set(f.name for f in get_yaml_json_files(local_config_dir)) | set(f.name for f in get_yaml_json_files(drive_config_dir))
        all_config_files = {f for f in all_config_files if not f.startswith('backup') and not f.startswith('_')}
        
        # Sinkronisasi setiap file
        for file_name in all_config_files:
            if logger: logger.info(f"🔄 Sinkronisasi {file_name}...")
            
            try:
                # PERBAIKAN: Langsung panggil sync_config_with_drive dengan parameter yang benar
                success, message, config = sync_config_with_drive(
                    config_file=file_name, 
                    sync_strategy=sync_strategy,
                    create_backup=create_backup,  # create_backup adalah boolean, bukan callable
                    logger=logger
                )
                
                if success:
                    results["skipped" if "identik" in message else "success"].append({"file": file_name, "message": message})
                else:
                    results["failure"].append({"file": file_name, "message": message})
                    
            except Exception as e:
                results["failure"].append({"file": file_name, "message": str(e)})
        
        # Log hasil
        if logger:
            counts = {k: len(v) for k, v in results.items()}
            logger.info(f"🔄 Sinkronisasi selesai: {len(all_config_files)} file diproses - {counts['success']} disinkronisasi, {counts['skipped']} dilewati, {counts['failure']} gagal")
        
        return results
    except Exception as e:
        if logger: logger.error(f"❌ Error saat sinkronisasi semua konfigurasi: {str(e)}")
        results["failure"].append({"file": "all", "message": str(e)})
        return results