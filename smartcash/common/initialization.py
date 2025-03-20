"""
File: smartcash/common/initialization.py
Deskripsi: Inisialisasi dan sinkronisasi konfigurasi saat aplikasi dimulai, dengan Google Drive sebagai sumber kebenaran
"""

import os
import time
from typing import Dict, Any, Tuple, Optional, List
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
        
        # Setup logger
        logger = get_logger("initialization")
        logger.info("🚀 Memulai inisialisasi konfigurasi SmartCash")
        
        # Dapatkan environment manager
        env_manager = get_environment_manager()
        
        # Cek ketersediaan Google Drive
        if env_manager.is_colab:
            logger.info(f"🧪 Berjalan di lingkungan Google Colab")
            
            # Cek apakah Drive sudah terpasang
            if not env_manager.is_drive_mounted:
                logger.info(f"💾 Mencoba memasang Google Drive...")
                
                try:
                    from google.colab import drive
                    drive.mount('/content/drive')
                    time.sleep(1)  # Tunggu sebentar agar Drive benar-benar terpasang
                    
                    # Update status drive
                    env_manager.detect_drive()
                    
                    if env_manager.is_drive_mounted:
                        logger.success(f"✅ Google Drive berhasil dipasang: {env_manager.drive_path}")
                    else:
                        logger.warning(f"⚠️ Gagal mendeteksi Google Drive setelah mount")
                except Exception as e:
                    logger.error(f"❌ Error saat mounting Google Drive: {str(e)}")
        
        # Dapatkan config manager
        config_manager = get_config_manager()
        logger.info(f"⚙️ Config manager siap")
        
        # Cek ketersediaan direktori configs
        Path("configs").mkdir(parents=True, exist_ok=True)
        
        # Jika tidak ada file konfigurasi lokal, coba ambil dari Drive
        base_config_exists = os.path.exists("configs/base_config.yaml")
        
        if not base_config_exists and env_manager.is_drive_mounted:
            # Cek apakah ada di Drive
            drive_config_path = env_manager.drive_path / "configs/base_config.yaml"
            
            if drive_config_path.exists():
                # Salin dari Drive ke lokal
                logger.info(f"📥 Menyalin konfigurasi dasar dari Drive")
                
                # Pastikan direktori configs ada
                os.makedirs("configs", exist_ok=True)
                
                # Salin file
                import shutil
                shutil.copy2(drive_config_path, "configs/base_config.yaml")
                
                logger.success(f"✅ Konfigurasi dasar berhasil disalin dari Drive")
                base_config_exists = True
        
        # Jika konfigurasi dasar ditemukan, load dulu
        if base_config_exists:
            config = config_manager.load_config("configs/base_config.yaml")
            logger.info(f"📝 Konfigurasi dasar dimuat")
        else:
            logger.warning(f"⚠️ File konfigurasi dasar tidak ditemukan")
            config = {}
        
        # Jika Drive terpasang, sinkronisasi semua konfigurasi
        if env_manager.is_drive_mounted:
            logger.info(f"🔄 Menggunakan Google Drive sebagai sumber kebenaran konfigurasi")
            
            # Gunakan fungsi yang telah ditambahkan ke ConfigManager
            if hasattr(config_manager, 'use_drive_as_source_of_truth'):
                success = config_manager.use_drive_as_source_of_truth()
                
                if success:
                    logger.success(f"✅ Konfigurasi berhasil disinkronisasi dengan Drive")
                else:
                    logger.warning(f"⚠️ Beberapa file konfigurasi gagal disinkronisasi")
            else:
                # Fallback: Gunakan sync_config_with_drive untuk file konfigurasi utama
                try:
                    from smartcash.common.config_sync import sync_config_with_drive
                    
                    configs_to_sync = [
                        "base_config.yaml",
                        "colab_config.yaml",
                        "dataset_config.yaml",
                        "preprocessing_config.yaml",
                        "training_config.yaml",
                        "augmentation_config.yaml",
                        "evaluation_config.yaml",
                        "model_config.yaml"
                    ]
                    
                    for cfg_file in configs_to_sync:
                        success, message, _ = sync_config_with_drive(
                            config_file=cfg_file,
                            sync_strategy='drive_priority',
                            create_backup=True,
                            logger=logger
                        )
                        
                        if success:
                            logger.info(f"✅ {message}")
                        else:
                            logger.warning(f"⚠️ {message}")
                            
                    # Reload config setelah sinkronisasi
                    config = config_manager.load_config("configs/base_config.yaml")
                    logger.info(f"📝 Konfigurasi dimuat ulang setelah sinkronisasi")
                    
                except ImportError:
                    logger.warning(f"⚠️ Modul config_sync tidak tersedia, sinkronisasi tidak dilakukan")
        
        # Buat direktori yang diperlukan
        create_required_directories(config, env_manager, logger)
        
        logger.success(f"✅ Inisialisasi konfigurasi selesai")
        return True, config
        
    except Exception as e:
        # Jika ada error, coba cetak error dan load config standar
        print(f"❌ Error saat inisialisasi konfigurasi: {str(e)}")
        
        try:
            from smartcash.common.config import get_config_manager
            config_manager = get_config_manager()
            
            if os.path.exists("configs/base_config.yaml"):
                config = config_manager.load_config("configs/base_config.yaml")
            else:
                config = {}
                
            return False, config
            
        except Exception:
            return False, {}

def create_required_directories(config: Dict[str, Any], env_manager=None, logger=None) -> None:
    """
    Buat direktori yang diperlukan berdasarkan konfigurasi.
    
    Args:
        config: Konfigurasi aplikasi
        env_manager: Environment manager
        logger: Logger untuk logging
    """
    # Daftar direktori yang perlu dibuat
    directories = [
        # Dataset directories
        "data/train/images", "data/train/labels",
        "data/valid/images", "data/valid/labels", 
        "data/test/images", "data/test/labels",
        "data/preprocessed/train", "data/preprocessed/valid", "data/preprocessed/test",
        
        # Config directory
        "configs",
        
        # Output directories
        "runs/train/weights",
        "logs",
        "exports",
        "checkpoints"
    ]
    
    # Lihat juga direktori kustom dari konfigurasi
    if 'data' in config:
        if 'dir' in config['data'] and config['data']['dir']:
            directories.extend([
                f"{config['data']['dir']}/train/images", 
                f"{config['data']['dir']}/train/labels",
                f"{config['data']['dir']}/valid/images", 
                f"{config['data']['dir']}/valid/labels",
                f"{config['data']['dir']}/test/images", 
                f"{config['data']['dir']}/test/labels",
            ])
            
        if 'processed_dir' in config['data'] and config['data']['processed_dir']:
            directories.extend([
                f"{config['data']['processed_dir']}/train",
                f"{config['data']['processed_dir']}/valid",
                f"{config['data']['processed_dir']}/test"
            ])
    
    # Buat direktori dengan error handling
    for dir_path in directories:
        try:
            os.makedirs(dir_path, exist_ok=True)
            if logger:
                logger.debug(f"📁 Membuat direktori: {dir_path}")
        except Exception as e:
            if logger:
                logger.warning(f"⚠️ Gagal membuat direktori {dir_path}: {str(e)}")
    
    # Jika Drive terpasang dan config mengizinkan, buat symlink
    if env_manager and env_manager.is_colab and env_manager.is_drive_mounted:
        create_symlinks = config.get('environment', {}).get('symlinks', False)
        
        if create_symlinks:
            if logger:
                logger.info(f"🔗 Membuat symlinks untuk integrasi Google Drive")
                
            # Mapping direktori yang akan dibuat symlink
            symlinks = {
                'data': str(env_manager.drive_path / 'data'),
                'configs': str(env_manager.drive_path / 'configs'),
                'runs': str(env_manager.drive_path / 'runs'),
                'logs': str(env_manager.drive_path / 'logs'),
                'checkpoints': str(env_manager.drive_path / 'checkpoints')
            }
            
            for local_name, target_path in symlinks.items():
                try:
                    # Pastikan direktori target ada
                    os.makedirs(target_path, exist_ok=True)
                    
                    local_path = Path(local_name)
                    
                    # Hapus direktori lokal jika sudah ada dan bukan symlink
                    if local_path.exists() and not local_path.is_symlink():
                        backup_path = local_path.with_name(f"{local_name}_backup")
                        if logger:
                            logger.info(f"🔄 Memindahkan direktori lokal ke backup: {local_name} → {local_name}_backup")
                        
                        # Hapus backup yang sudah ada
                        if backup_path.exists():
                            import shutil
                            shutil.rmtree(backup_path)
                            
                        # Pindahkan direktori lokal ke backup
                        local_path.rename(backup_path)
                    
                    # Buat symlink jika belum ada
                    if not local_path.exists():
                        local_path.symlink_to(target_path)
                        if logger:
                            logger.success(f"✅ Symlink dibuat: {local_name} → {target_path}")
                except Exception as e:
                    if logger:
                        logger.warning(f"⚠️ Gagal membuat symlink {local_name}: {str(e)}")