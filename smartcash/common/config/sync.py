"""
File: smartcash/common/config/sync.py
Deskripsi: Utilitas untuk sinkronisasi konfigurasi antara lokal dan Google Drive
"""

import os
import copy
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union

from smartcash.common.io import (
    load_config,
    save_config,
    copy_file
)
from smartcash.common.constants.log_messages import (
    CONFIG_IDENTICAL,
    CONFIG_SYNC_SUCCESS,
    CONFIG_SYNC_ERROR,
    CONFIG_ERROR,
    FILE_BACKUP_SUCCESS,
    FILE_BACKUP_ERROR,
    DRIVE_NOT_MOUNTED,
    DRIVE_PATH_IDENTICAL,
    STATUS_INFO,
    STATUS_WARNING,
    STATUS_ERROR,
    STATUS_SUCCESS,
    OPERATION_COMPLETED
)

def are_configs_identical(config1: Dict[str, Any], config2: Dict[str, Any]) -> bool:
    """
    Periksa apakah dua konfigurasi identik.
    
    Args:
        config1: Konfigurasi pertama
        config2: Konfigurasi kedua
        
    Returns:
        Boolean yang menunjukkan keidentikan
    """
    import json
    return json.dumps(config1, sort_keys=True) == json.dumps(config2, sort_keys=True)

def merge_configs_smart(config1: Any, config2: Any) -> Any:
    """
    Gabungkan dua konfigurasi dengan strategi smart.
    
    Args:
        config1: Konfigurasi pertama
        config2: Konfigurasi kedua
        
    Returns:
        Konfigurasi gabungan
    """
    # Handle None cases
    if config1 is None: 
        return copy.deepcopy(config2)
    if config2 is None: 
        return copy.deepcopy(config1)
    
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
    """
    Sinkronisasi file konfigurasi dengan Google Drive.
    
    Args:
        config_file: Nama file konfigurasi (relatif terhadap direktori configs)
        sync_strategy: Strategi sinkronisasi ('merge', 'drive_priority', 'local_priority')
        create_backup: Buat backup sebelum sinkronisasi
        logger: Logger untuk mencatat aktivitas
        
    Returns:
        Tuple (success, message, merged_config)
    """
    # Setup logger
    if logger is None:
        try:
            from smartcash.common.logger import get_logger
            logger = get_logger("config_sync")
        except ImportError:
            import logging
            logger = logging.getLogger("config_sync")
    
    # Dapatkan environment manager
    try:
        from smartcash.common.environment import get_environment_manager
        env_manager = get_environment_manager()
    except ImportError:
        logger.error(CONFIG_ERROR.format(operation="akses environment", error="Environment manager tidak tersedia"))
        return False, "Environment manager tidak tersedia", {}
    
    # Verifikasi Google Drive mounted
    if not env_manager.is_drive_mounted: 
        logger.warning(DRIVE_NOT_MOUNTED)
        return False, "Google Drive tidak terpasang", {}
    
    # Setup path konfigurasi
    local_config_path = Path("configs") / config_file
    drive_config_path = env_manager.drive_path / "configs" / config_file
    
    # Hentikan proses jika path identik (cek realpath untuk symlink)
    if os.path.realpath(local_config_path) == os.path.realpath(drive_config_path):
        msg = DRIVE_PATH_IDENTICAL.format(path=local_config_path)
        logger.warning(msg)
        return True, msg, load_config(local_config_path, {})
    
    # Validasi file ada
    if not local_config_path.exists() and not drive_config_path.exists():
        msg = f"File konfigurasi tidak ditemukan: {config_file}"
        logger.warning(STATUS_WARNING.format(message=msg))
        return False, msg, {}
    
    # Backup jika diminta
    if create_backup and local_config_path.exists():
        try:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = Path("configs/backup")
            backup_dir.mkdir(parents=True, exist_ok=True)
            backup_path = backup_dir / f"{local_config_path.stem}_{timestamp}{local_config_path.suffix}"
            
            copy_file(local_config_path, backup_path)
            logger.info(FILE_BACKUP_SUCCESS.format(path=backup_path))
        except Exception as e:
            logger.warning(FILE_BACKUP_ERROR.format(source=local_config_path, error=str(e)))
    
    # Load konfigurasi
    local_config = load_config(local_config_path, {}) if local_config_path.exists() else {}
    drive_config = load_config(drive_config_path, {}) if drive_config_path.exists() else {}
    
    # Proses berdasarkan strategi
    if sync_strategy == 'merge':
        # Strategi merge: gabungkan dan simpan ke kedua lokasi
        merged_config = merge_configs_smart(local_config, drive_config)
        success_local = save_config(merged_config, local_config_path)
        success_drive = save_config(merged_config, drive_config_path)
        
        if success_local and success_drive:
            # logger.info(STATUS_SUCCESS.format(message="Sinkronisasi berhasil dengan strategi merge"))
            return True, "Sinkronisasi berhasil dengan strategi merge", merged_config
        else:
            logger.error(CONFIG_ERROR.format(operation="menyimpan hasil merge", error="Gagal menyimpan ke kedua lokasi"))
            return False, "Error saat menyimpan hasil merge", {}
            
    elif sync_strategy == 'drive_priority':
        # Strategi drive priority: Drive → lokal
        if drive_config_path.exists():
            if local_config_path.exists() and are_configs_identical(local_config, drive_config):
                # logger.info(CONFIG_IDENTICAL.format(name=config_file))
                return True, "Konfigurasi sudah identik", drive_config
            
            success = save_config(drive_config, local_config_path)
            if success:
                # logger.info(CONFIG_SYNC_SUCCESS.format(direction="dari Drive"))
                return True, "Konfigurasi berhasil disinkronisasi dari Drive", drive_config
            else:
                logger.error(CONFIG_ERROR.format(operation="menyimpan dari Drive", error="Gagal menulis ke file lokal"))
                return False, "Error saat menyimpan dari Drive", {}
        else:
            # Jika tidak ada di Drive, salin dari lokal ke Drive
            if local_config_path.exists():
                success = save_config(local_config, drive_config_path)
                if success:
                    # logger.info(CONFIG_SYNC_SUCCESS.format(direction="ke Drive"))
                    return True, "Konfigurasi berhasil disinkronisasi ke Drive", local_config
                else:
                    logger.error(CONFIG_ERROR.format(operation="menyalin ke Drive", error="Gagal menulis ke Drive"))
                    return False, "Error saat menyalin ke Drive", {}
            else:
                logger.warning(STATUS_WARNING.format(message="Tidak ada konfigurasi lokal atau di Drive"))
                return False, "Tidak ada konfigurasi lokal atau di Drive", {}
                
    elif sync_strategy == 'local_priority':
        # Strategi local priority: lokal → Drive
        if local_config_path.exists():
            if drive_config_path.exists() and are_configs_identical(local_config, drive_config):
                # logger.info(CONFIG_IDENTICAL.format(name=config_file))
                return True, "Konfigurasi sudah identik", local_config
            
            success = save_config(local_config, drive_config_path)
            if success:
                # logger.info(CONFIG_SYNC_SUCCESS.format(direction="ke Drive"))
                return True, "Konfigurasi berhasil disinkronisasi ke Drive", local_config
            else:
                logger.error(CONFIG_ERROR.format(operation="menyimpan ke Drive", error="Gagal menulis ke Drive"))
                return False, "Error saat menyimpan ke Drive", {}
        else:
            # Jika tidak ada lokal, salin dari Drive ke lokal
            if drive_config_path.exists():
                success = save_config(drive_config, local_config_path)
                if success:
                    # logger.info(CONFIG_SYNC_SUCCESS.format(direction="dari Drive"))
                    return True, "Konfigurasi berhasil disalin dari Drive", drive_config
                else:
                    logger.error(CONFIG_ERROR.format(operation="menyalin dari Drive", error="Gagal menulis ke file lokal"))
                    return False, "Error saat menyalin dari Drive", {}
            else:
                logger.warning(STATUS_WARNING.format(message="Tidak ada konfigurasi lokal atau di Drive"))
                return False, "Tidak ada konfigurasi lokal atau di Drive", {}
    else:
        msg = f"Strategi sinkronisasi tidak valid: {sync_strategy}"
        logger.warning(STATUS_WARNING.format(message=msg))
        return False, msg, {}

def sync_all_configs(
    sync_strategy: str = 'merge',
    config_dir: str = 'configs',
    create_backup: bool = True,
    logger = None
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Sinkronisasi semua file konfigurasi YAML/JSON dengan Google Drive.
    
    Args:
        sync_strategy: Strategi sinkronisasi ('merge', 'drive_priority', 'local_priority')
        config_dir: Direktori konfigurasi
        create_backup: Buat backup sebelum sinkronisasi
        logger: Logger untuk mencatat aktivitas
        
    Returns:
        Dictionary berisi hasil sinkronisasi
    """
    # Setup logger
    if logger is None:
        try:
            from smartcash.common.logger import get_logger
            logger = get_logger("config_sync")
        except ImportError:
            import logging
            logger = logging.getLogger("config_sync")
    
    # Dapatkan environment manager
    try:
        from smartcash.common.environment import get_environment_manager
        env_manager = get_environment_manager()
    except ImportError:
        logger.error(CONFIG_ERROR.format(operation="akses environment", error="Environment manager tidak tersedia"))
        return {"success": [], "failure": [], "skipped": []}
    
    # Verifikasi Google Drive mounted
    if not env_manager.is_drive_mounted:
        logger.warning(DRIVE_NOT_MOUNTED)
        return {"success": [], "failure": [], "skipped": []}
    
    # Setup path dan pastikan direktori ada
    local_config_dir, drive_config_dir = Path(config_dir), env_manager.drive_path / "configs"
    local_config_dir.mkdir(parents=True, exist_ok=True)
    drive_config_dir.mkdir(parents=True, exist_ok=True)
    
    # Kumpulkan semua file konfigurasi
    def get_yaml_json_files(dir_path): 
        return list(dir_path.glob("*.yaml")) + list(dir_path.glob("*.yml")) + list(dir_path.glob("*.json"))
    
    all_config_files = set(f.name for f in get_yaml_json_files(local_config_dir)) | set(f.name for f in get_yaml_json_files(drive_config_dir))
    
    # Filter file yang diawali 'backup' atau '_'
    all_config_files = {f for f in all_config_files if not f.startswith('backup') and not f.startswith('_')}
    
    # Hasil sinkronisasi
    results = {"success": [], "failure": [], "skipped": []}
    
    # Sinkronisasi setiap file
    for file_name in all_config_files:
        logger.info(STATUS_INFO.format(message=f"Sinkronisasi {file_name}..."))
        
        try:
            # Cek jika realpath sama
            if os.path.realpath(local_config_dir / file_name) == os.path.realpath(drive_config_dir / file_name):
                msg = f"Path lokal sama dengan drive: {file_name}, dilewati"
                logger.debug(STATUS_INFO.format(message=msg))
                results["skipped"].append({"file": file_name, "message": msg})
                continue
            
            # Panggil sync_config_with_drive
            success, message, _ = sync_config_with_drive(
                config_file=file_name, 
                sync_strategy=sync_strategy,
                create_backup=create_backup,
                logger=logger
            )
            
            if success:
                if "identik" in message:
                    results["skipped"].append({"file": file_name, "message": message})
                else:
                    results["success"].append({"file": file_name, "message": message})
            else:
                results["failure"].append({"file": file_name, "message": message})
                
        except Exception as e:
            results["failure"].append({"file": file_name, "message": str(e)})
    
    # Log hasil hanya jika ada perubahan atau error
    counts = {k: len(v) for k, v in results.items()}
    if counts['success'] > 0 or counts['failure'] > 0:
        logger.info(
            OPERATION_COMPLETED.format(
                operation=f"Sinkronisasi {len(all_config_files)} file",
                duration=f"{counts['success']} disinkronisasi, {counts['skipped']} dilewati, {counts['failure']} gagal"
            )
        )
    
    return results
    
    # Verifikasi Google Drive mounted
    if not env_manager.is_drive_mounted:
        logger.warning("⚠️ Google Drive tidak terpasang")
        return {"success": [], "failure": [], "skipped": []}
    
    # Setup path dan pastikan direktori ada
    local_config_dir, drive_config_dir = Path(config_dir), env_manager.drive_path / "configs"
    local_config_dir.mkdir(parents=True, exist_ok=True)
    drive_config_dir.mkdir(parents=True, exist_ok=True)
    
    # Kumpulkan semua file konfigurasi
    def get_yaml_json_files(dir_path): 
        return list(dir_path.glob("*.yaml")) + list(dir_path.glob("*.yml")) + list(dir_path.glob("*.json"))
    
    all_config_files = set(f.name for f in get_yaml_json_files(local_config_dir)) | set(f.name for f in get_yaml_json_files(drive_config_dir))
    
    # Filter file yang diawali 'backup' atau '_'
    all_config_files = {f for f in all_config_files if not f.startswith('backup') and not f.startswith('_')}
    
    # Hasil sinkronisasi
    results = {"success": [], "failure": [], "skipped": []}
    
    # Sinkronisasi setiap file
    for file_name in all_config_files:
        logger.debug(f"🔄 Sinkronisasi {file_name}...")
        
        try:
            # Cek jika realpath sama
            if os.path.realpath(local_config_dir / file_name) == os.path.realpath(drive_config_dir / file_name):
                msg = f"Path lokal sama dengan drive: {file_name}, dilewati"
                logger.debug(f"ℹ️ {msg}")
                results["skipped"].append({"file": file_name, "message": msg})
                continue
            
            # Panggil sync_config_with_drive
            success, message, _ = sync_config_with_drive(
                config_file=file_name, 
                sync_strategy=sync_strategy,
                create_backup=create_backup,
                logger=logger
            )
            
            if success:
                if "identik" in message:
                    results["skipped"].append({"file": file_name, "message": message})
                else:
                    results["success"].append({"file": file_name, "message": message})
            else:
                results["failure"].append({"file": file_name, "message": message})
                
        except Exception as e:
            results["failure"].append({"file": file_name, "message": str(e)})
    
    # Log hasil hanya jika ada perubahan atau error
    counts = {k: len(v) for k, v in results.items()}
    if counts['success'] > 0 or counts['failure'] > 0:
        logger.info(
            f"🔄 Sinkronisasi selesai: {len(all_config_files)} file diproses - "
            f"{counts['success']} disinkronisasi, {counts['skipped']} dilewati, {counts['failure']} gagal"
        )
    
    return results