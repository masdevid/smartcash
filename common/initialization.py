"""
File: smartcash/common/initialization.py
Deskripsi: Inisialisasi dan sinkronisasi konfigurasi saat aplikasi dimulai, dengan Google Drive sebagai sumber kebenaran
"""

import os
import time
from typing import Dict, Any, Tuple, List
from pathlib import Path

def initialize_config() -> Tuple[bool, Dict[str, Any]]:
    """
    Inisialisasi konfigurasi dengan sinkronisasi dari Google Drive.
    
    Returns:
        Tuple (success, config)
    """
    try:
        # Import komponen yang diperlukan
        from smartcash.common.logger import get_logger
        from smartcash.common.config import get_config_manager
        from smartcash.common.environment import get_environment_manager
        from smartcash.common.file_utils import get_file_utils
        
        # Setup logger dan managers
        logger = get_logger("initialization")
        logger.info("🚀 Memulai inisialisasi konfigurasi SmartCash")
        env_manager = get_environment_manager()
        config_manager = get_config_manager()
        file_utils = get_file_utils(logger=logger)
        
        # Deteksi dan pasang Google Drive jika di Colab
        if env_manager.is_colab and not env_manager.is_drive_mounted:
            success, message = env_manager.mount_drive()
            logger.info(message)
        
        # Pastikan direktori configs ada
        file_utils.ensure_dir("configs")
        
        # Periksa konfigurasi dasar dan salin dari Drive jika perlu
        base_config_exists = Path("configs/base_config.yaml").exists()
        drive_config_path = env_manager.drive_path / "configs/base_config.yaml" if env_manager.is_drive_mounted else None
        
        if not base_config_exists and drive_config_path and drive_config_path.exists():
            file_utils.copy_file(drive_config_path, "configs/base_config.yaml")
            logger.success(f"✅ Konfigurasi dasar berhasil disalin dari Drive")
            base_config_exists = True
        
        # Load konfigurasi dasar jika tersedia
        config = config_manager.load_config("configs/base_config.yaml") if base_config_exists else {}
        
        # Sinkronisasi dengan Drive jika tersedia
        if env_manager.is_drive_mounted:
            logger.info(f"🔄 Menggunakan Google Drive sebagai sumber kebenaran konfigurasi")
            if hasattr(config_manager, 'use_drive_as_source_of_truth'):
                success = config_manager.use_drive_as_source_of_truth()
                logger.info(f"{'✅ Konfigurasi berhasil disinkronisasi' if success else '⚠️ Beberapa file konfigurasi gagal disinkronisasi'}")
            else:
                # Fallback ke metode sinkronisasi manual
                _sync_configs_manually(logger)
                config = config_manager.load_config("configs/base_config.yaml")
        
        # Buat direktori yang diperlukan
        create_required_directories(config, env_manager, logger)
        
        logger.success(f"✅ Inisialisasi konfigurasi selesai")
        return True, config
        
    except Exception as e:
        print(f"❌ Error saat inisialisasi konfigurasi: {str(e)}")
        try:
            from smartcash.common.config import get_config_manager
            config = get_config_manager().load_config("configs/base_config.yaml") if Path("configs/base_config.yaml").exists() else {}
            return False, config
        except Exception:
            return False, {}

def _sync_configs_manually(logger):
    """Sinkronisasi manual file konfigurasi dengan Google Drive."""
    try:
        from smartcash.common.config_sync import sync_config_with_drive
        config_files = ["base_config.yaml", "colab_config.yaml", "dataset_config.yaml", 
                        "preprocessing_config.yaml", "training_config.yaml", "augmentation_config.yaml", 
                        "evaluation_config.yaml", "model_config.yaml"]
        
        for cfg_file in config_files:
            success, message, _ = sync_config_with_drive(config_file=cfg_file, sync_strategy='drive_priority', 
                                                        create_backup=True, logger=logger)
            logger.info(f"{'✅' if success else '⚠️'} {message}")
    except ImportError:
        logger.warning(f"⚠️ Modul config_sync tidak tersedia, sinkronisasi tidak dilakukan")

def create_required_directories(config: Dict[str, Any], env_manager=None, logger=None) -> None:
    """
    Buat direktori yang diperlukan berdasarkan konfigurasi.
    
    Args:
        config: Konfigurasi aplikasi
        env_manager: Environment manager
        logger: Logger untuk logging
    """
    # Daftar direktori standar dan kustom yang perlu dibuat
    directories = [
        "data/train/images", "data/train/labels", "data/valid/images", "data/valid/labels", 
        "data/test/images", "data/test/labels", "data/preprocessed/train", "data/preprocessed/valid", 
        "data/preprocessed/test", "configs", "runs/train/weights", "logs", "exports", "checkpoints"
    ]
    
    # Tambahkan direktori kustom dari konfigurasi
    data_dir = config.get('data', {}).get('dir', '')
    if data_dir:
        directories.extend([f"{data_dir}/train/images", f"{data_dir}/train/labels", 
                           f"{data_dir}/valid/images", f"{data_dir}/valid/labels",
                           f"{data_dir}/test/images", f"{data_dir}/test/labels"])
    
    processed_dir = config.get('data', {}).get('processed_dir', '')
    if processed_dir:
        directories.extend([f"{processed_dir}/train", f"{processed_dir}/valid", f"{processed_dir}/test"])
    
    # Buat semua direktori
    [os.makedirs(dir_path, exist_ok=True) for dir_path in directories]
    
    # Buat symlinks jika diperlukan dan dimungkinkan
    if env_manager and env_manager.is_colab and env_manager.is_drive_mounted and config.get('environment', {}).get('symlinks', False):
        if logger: logger.info(f"🔗 Membuat symlinks untuk integrasi Google Drive")
        env_manager.create_symlinks(lambda progress, total, message: logger.info(message) if logger else None)