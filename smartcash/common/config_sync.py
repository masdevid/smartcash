"""
File: smartcash/common/config_sync.py
Deskripsi: Utilitas ringkas untuk sinkronisasi konfigurasi antara lokal dan Google Drive dengan perbaikan error path sama
"""

import os, shutil
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union

from smartcash.common.logger import get_logger
from smartcash.common.file_utils import get_file_utils
from smartcash.common.utils import load_json, save_json
from smartcash.common.environment import get_environment_manager

def load_config_file(file_path: str) -> Dict[str, Any]:
    """Muat konfigurasi dari file YAML/JSON"""
    file_utils = get_file_utils()
    file_path = Path(file_path)
    if not file_utils.file_exists(file_path): return {}
    
    try:
        suffix = file_path.suffix.lower()
        if suffix in ('.yml', '.yaml'):
            import yaml
            with open(file_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        elif suffix == '.json':
            return load_json(file_path)
        return {}
    except Exception as e:
        logger = get_logger("config_sync")
        logger.error(f"❌ Error saat memuat konfigurasi dari {file_path}: {str(e)}")
        return {}

def save_config_file(config: Dict[str, Any], file_path: str, create_dirs: bool = True) -> bool:
    """Simpan konfigurasi ke file YAML/JSON"""
    file_utils = get_file_utils()
    file_path = Path(file_path)
    
    try:
        if create_dirs:
            file_utils.ensure_dir(file_path.parent)
        
        suffix = file_path.suffix.lower()
        if suffix in ('.yml', '.yaml'):
            import yaml
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False)
        elif suffix == '.json':
            save_json(config, file_path)
        else:
            import yaml
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False)
        return True
    except Exception as e:
        logger = get_logger("config_sync")
        logger.error(f"❌ Error saat menyimpan konfigurasi ke {file_path}: {str(e)}")
        return False

def are_configs_identical(config1: Dict[str, Any], config2: Dict[str, Any]) -> bool:
    """Periksa apakah dua konfigurasi identik"""
    import json
    return json.dumps(config1, sort_keys=True) == json.dumps(config2, sort_keys=True)

def merge_configs_smart(config1: Any, config2: Any) -> Any:
    """Gabungkan dua konfigurasi dengan strategi smart"""
    import copy
    
    # Handle None cases
    if config1 is None: return copy.deepcopy(config2)
    if config2 is None: return copy.deepcopy(config1)
    
    # Dict: gabungkan rekursif
    if isinstance(config1, dict) and isinstance(config2, dict):
        result = copy.deepcopy(config1)
        for key, value in config2.items():
            result[key] = merge_configs_smart(result.get(key), value) if key in result else copy.deepcopy(value)
        return result
    
    # List: gabungkan dengan filter duplikat jika perlu
    if isinstance(config1, list) and isinstance(config2, list):
        # Untuk list sederhana, gabungkan dengan unik
        if all(not isinstance(x, (dict, list)) for x in config1 + config2):
            # Hanya gunakan set untuk elemen yang hashable
            try:
                return list(set(config1 + config2))
            except TypeError:
                pass
        # Untuk list kompleks atau unhashable, gabungkan saja
        return copy.deepcopy(config1) + copy.deepcopy(config2)
    
    # Nilai skalar: prioritaskan nilai yang tidak kosong
    return copy.deepcopy(config2) if config1 == "" or config1 is None or config1 == 0 else copy.deepcopy(config1)

def sync_config_with_drive(
    config_file: str,
    sync_strategy: str = 'merge',
    create_backup: bool = True,
    logger = None
) -> Tuple[bool, str, Dict[str, Any]]:
    """Sinkronisasi file konfigurasi dengan Google Drive"""
    logger = logger or get_logger("config_sync")
    file_utils = get_file_utils(logger=logger)
    env_manager = get_environment_manager(logger=logger)
    
    # Verifikasi Google Drive mounted
    if not env_manager.is_drive_mounted: 
        return False, "Google Drive tidak terpasang", {}
    
    # Path konfigurasi
    local_config_path = Path("configs") / config_file
    drive_config_path = env_manager.drive_path / "configs" / config_file
    
    # PERBAIKAN: Hentikan proses jika path identik (cek realpath untuk symlink)
    if os.path.realpath(local_config_path) == os.path.realpath(drive_config_path):
        msg = f"⚠️ Path lokal sama dengan drive: {local_config_path}, gunakan path lain"
        logger.warning(msg)
        return True, msg, load_config_file(local_config_path)  # Return sukses dengan config yang ada
    
    # Validasi file ada
    if not local_config_path.exists() and not drive_config_path.exists():
        logger.warning(f"⚠️ File konfigurasi tidak ditemukan: {config_file}")
        return False, f"File konfigurasi tidak ditemukan: {config_file}", {}
    
    # Backup jika diminta
    if create_backup and local_config_path.exists():
        try:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = file_utils.ensure_dir(Path("configs/backup"))
            backup_path = backup_dir / f"{local_config_path.stem}_{timestamp}{local_config_path.suffix}"
            file_utils.copy_file(local_config_path, backup_path)
            logger.info(f"✅ Backup berhasil dibuat: {backup_path}")
        except Exception as e:
            logger.warning(f"⚠️ Error saat membuat backup: {str(e)}")
    
    # Load konfigurasi
    local_config = load_config_file(local_config_path) if local_config_path.exists() else {}
    drive_config = load_config_file(drive_config_path) if drive_config_path.exists() else {}
    
    # Proses berdasarkan strategi
    if sync_strategy == 'merge':
        # Strategi merge: gabungkan dan simpan ke kedua lokasi
        merged_config = merge_configs_smart(local_config, drive_config)
        success = save_config_file(merged_config, local_config_path) and save_config_file(merged_config, drive_config_path)
        return (True, f"Sinkronisasi berhasil dengan strategi merge", merged_config) if success else (False, "Error saat menyimpan hasil merge", {})
    elif sync_strategy == 'drive_priority':
        # Strategi drive priority: Drive → lokal
        if drive_config_path.exists():
            if local_config_path.exists() and are_configs_identical(local_config, drive_config):
                logger.info(f"ℹ️ Konfigurasi sudah identik: {config_file}")
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
                logger.info(f"ℹ️ Konfigurasi sudah identik: {config_file}")
                return True, "Konfigurasi sudah identik", local_config
            
            success = save_config_file(local_config, drive_config_path)
            return (True, "Konfigurasi berhasil disinkronisasi ke Drive", local_config) if success else (False, "Error saat menyimpan ke Drive", {})
        else:
            # Jika tidak ada lokal, salin dari Drive ke lokal
            success = save_config_file(drive_config, local_config_path) if drive_config_path.exists() else False
            return (True, "Konfigurasi berhasil disalin dari Drive", drive_config) if success else (False, "Error saat menyalin dari Drive", {})
    else:
        logger.warning(f"⚠️ Strategi sinkronisasi tidak valid: {sync_strategy}")
        return False, f"Strategi sinkronisasi tidak valid: {sync_strategy}", {}

def sync_all_configs(
    sync_strategy: str = 'merge',
    config_dir: str = 'configs',
    create_backup: bool = True,
    logger = None
) -> Dict[str, List[Dict[str, Any]]]:
    """Sinkronisasi semua file konfigurasi YAML/JSON dengan Google Drive"""
    logger = logger or get_logger("config_sync")
    file_utils = get_file_utils(logger=logger)
    env_manager = get_environment_manager(logger=logger)
    results = {"success": [], "failure": [], "skipped": []}
    
    # Verifikasi Google Drive mounted
    if not env_manager.is_drive_mounted:
        logger.warning("⚠️ Google Drive tidak terpasang")
        return results
    
    # Setup path dan pastikan direktori ada
    local_config_dir, drive_config_dir = Path(config_dir), env_manager.drive_path / "configs"
    file_utils.ensure_dir(local_config_dir)
    file_utils.ensure_dir(drive_config_dir)
    
    # Kumpulkan semua file konfigurasi
    def get_yaml_json_files(dir_path): 
        return list(dir_path.glob("*.yaml")) + list(dir_path.glob("*.yml")) + list(dir_path.glob("*.json"))
    
    all_config_files = set(f.name for f in get_yaml_json_files(local_config_dir)) | set(f.name for f in get_yaml_json_files(drive_config_dir))
    all_config_files = {f for f in all_config_files if not f.startswith('backup') and not f.startswith('_')}
    
    # Sinkronisasi setiap file
    for file_name in all_config_files:
        logger.info(f"🔄 Sinkronisasi {file_name}...")
        
        try:
            # PERBAIKAN: Jika realpath sama, lewati file
            if os.path.realpath(local_config_dir / file_name) == os.path.realpath(drive_config_dir / file_name):
                msg = f"Path lokal sama dengan drive: {file_name}, dilewati"
                logger.info(f"ℹ️ {msg}")
                results["skipped"].append({"file": file_name, "message": msg})
                continue
            
            # Panggil sync_config_with_drive dengan parameter yang benar
            success, message, config = sync_config_with_drive(
                config_file=file_name, 
                sync_strategy=sync_strategy,
                create_backup=create_backup,
                logger=logger
            )
            
            if success:
                results["skipped" if "identik" in message else "success"].append({"file": file_name, "message": message})
            else:
                results["failure"].append({"file": file_name, "message": message})
                
        except Exception as e:
            results["failure"].append({"file": file_name, "message": str(e)})
    
    # Log hasil
    counts = {k: len(v) for k, v in results.items()}
    logger.info(f"🔄 Sinkronisasi selesai: {len(all_config_files)} file diproses - {counts['success']} disinkronisasi, {counts['skipped']} dilewati, {counts['failure']} gagal")
    
    return results