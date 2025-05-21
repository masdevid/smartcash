"""
File: smartcash/ui/setup/env_config/handlers/config_info_handler.py
Deskripsi: Handler untuk menampilkan informasi konfigurasi
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

from smartcash.common.utils import is_colab
from smartcash.common.config.manager import SimpleConfigManager

def log_env_info(logger: Any, base_dir: Path, config_dir: Path) -> None:
    """
    Log informasi environment dasar
    
    Args:
        logger: Logger untuk output
        base_dir: Base directory
        config_dir: Config directory
    """
    logger.success("Environment berhasil dikonfigurasi")
    logger.info(f"📁 Base directory: {base_dir}")
    logger.info(f"📁 Config directory: {config_dir}")
    
    # Cek apakah config_dir adalah symlink
    if config_dir.is_symlink():
        target = Path(config_dir).resolve()
        logger.info(f"🔗 Config directory adalah symlink ke: {target}")
        
        # Verifikasi bahwa symlink berfungsi
        if not target.exists():
            logger.warning(f"Target symlink tidak ditemukan: {target}")
            logger.info("Mencoba memperbaiki symlink...")

def log_system_info(logger: Any) -> None:
    """
    Log informasi sistem
    
    Args:
        logger: Logger untuk output
    """
    logger.info(f"📊 Informasi Environment:")
    logger.info(f"🐍 Python version: {sys.version.split()[0]}")
    
    # Cek apakah sedang berjalan di Colab
    colab_status = "Ya" if is_colab() else "Tidak"
    logger.info(f"💻 Running di Google Colab: {colab_status}")

def log_colab_config(logger: Any, config_manager: SimpleConfigManager) -> None:
    """
    Log konfigurasi Colab jika berjalan di Colab
    
    Args:
        logger: Logger untuk output
        config_manager: Config manager
    """
    if not is_colab():
        return
        
    try:
        # Ambil konfigurasi Colab
        colab_config = config_manager.get_config('colab')
        
        if not colab_config:
            return
            
        # Tampilkan informasi drive
        if 'drive' in colab_config:
            drive_config = colab_config['drive']
            logger.info(f"🗄️ Pengaturan Google Drive:")
            logger.info(f"- Sinkronisasi aktif: {drive_config.get('use_drive', False)}")
            logger.info(f"- Strategi sinkronisasi: {drive_config.get('sync_strategy', 'none')}")
            logger.info(f"- Gunakan symlinks: {drive_config.get('symlinks', False)}")
            
            # Tampilkan paths jika ada
            if 'paths' in drive_config:
                paths = drive_config['paths']
                logger.info(f"- SmartCash dir: {paths.get('smartcash_dir', 'SmartCash')}")
                logger.info(f"- Configs dir: {paths.get('configs_dir', 'configs')}")
        
        # Tampilkan informasi model jika menggunakan GPU/TPU
        if 'model' in colab_config:
            model_config = colab_config['model']
            logger.info(f"⚡ Pengaturan Hardware:")
            logger.info(f"- Gunakan GPU: {model_config.get('use_gpu', False)}")
            logger.info(f"- Gunakan TPU: {model_config.get('use_tpu', False)}")
            logger.info(f"- Precision: {model_config.get('precision', 'float32')}")
        
        # Informasi performa
        if 'performance' in colab_config:
            perf_config = colab_config['performance']
            logger.info(f"🚀 Pengaturan Performa:")
            logger.info(f"- Auto garbage collect: {perf_config.get('auto_garbage_collect', False)}")
            logger.info(f"- Simpan checkpoint ke Drive: {perf_config.get('checkpoint_to_drive', False)}")
    except Exception as e:
        logger.warning(f"Gagal memuat konfigurasi Colab: {str(e)}")

def log_available_configs(logger: Any, config_manager: SimpleConfigManager) -> None:
    """
    Log konfigurasi yang tersedia
    
    Args:
        logger: Logger untuk output
        config_manager: Config manager
    """
    try:
        # Filter config yang tidak perlu dilaporkan jika tidak ada
        ignored_configs = ['inference', 'export', 'environment']
        
        available_configs = config_manager.get_available_configs(ignored_configs)
        
        if available_configs:
            logger.info(f"📝 File Konfigurasi Tersedia:")
            for config in available_configs:
                logger.info(f"- {config}")
        else:
            logger.warning("Tidak ada file konfigurasi yang ditemukan.")
            logger.info("💡 Tip: Pastikan direktori konfigurasi berisi file .yaml")
    except Exception as e:
        logger.error(f"Gagal mendapatkan daftar konfigurasi: {str(e)}")

def display_config_info(
    ui_components: Dict[str, Any], 
    config_manager: SimpleConfigManager,
    base_dir: Path, 
    config_dir: Path
) -> None:
    """
    Tampilkan semua informasi konfigurasi
    
    Args:
        ui_components: Dictionary komponen UI
        config_manager: Config manager
        base_dir: Base directory
        config_dir: Config directory
    """
    logger = ui_components.get('logger')
    if not logger:
        return
    
    # Log informasi environment
    log_env_info(logger, base_dir, config_dir)
    
    # Log informasi sistem
    log_system_info(logger)
    
    # Log konfigurasi Colab jika berjalan di Colab
    log_colab_config(logger, config_manager)
    
    # Log konfigurasi yang tersedia
    log_available_configs(logger, config_manager) 