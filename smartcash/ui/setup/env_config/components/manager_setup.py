"""
File: smartcash/ui/setup/env_config/components/manager_setup.py
Deskripsi: Setup untuk manager environment config
"""

import os
import shutil
from pathlib import Path
from typing import Tuple, Any, Dict, Optional
import yaml

from smartcash.common.config.manager import SimpleConfigManager, get_config_manager
from smartcash.common.utils import is_colab

def setup_managers() -> Tuple[SimpleConfigManager, Path, Path]:
    """
    Setup configuration managers
    
    Returns:
        Tuple of (config_manager, base_dir, config_dir)
    """
    # Determine base directory
    if is_colab():
        base_dir = Path("/content")
        config_dir = base_dir / "configs"
        
        # Coba mendapatkan konfigurasi dari colab_config.yaml
        colab_config = {}
        repo_config_path = Path("/content/smartcash/configs/colab_config.yaml")
        
        # Dapatkan UI logger jika tersedia
        ui_logger = _get_ui_logger()
        
        if repo_config_path.exists():
            try:
                with open(repo_config_path, 'r') as f:
                    colab_config = yaml.safe_load(f) or {}
            except Exception as e:
                if ui_logger:
                    ui_logger.warning(f"Gagal memuat colab_config.yaml: {str(e)}")
                else:
                    print(f"⚠️ Gagal memuat colab_config.yaml: {str(e)}")
        
        # Dapatkan path Drive dari konfigurasi atau gunakan default
        drive_dir = 'SmartCash'
        configs_dir = 'configs'
        
        if 'drive' in colab_config and 'paths' in colab_config['drive']:
            drive_paths = colab_config['drive']['paths']
            drive_dir = drive_paths.get('smartcash_dir', drive_dir)
            configs_dir = drive_paths.get('configs_dir', configs_dir)
        
        # Pastikan direktori Drive ada
        drive_config_dir = Path(f"/content/drive/MyDrive/{drive_dir}/{configs_dir}")
        drive_config_dir.mkdir(parents=True, exist_ok=True)
        
        # Cek strategi sinkronisasi
        sync_strategy = 'drive_priority'
        use_symlinks = True
        
        if 'drive' in colab_config:
            sync_strategy = colab_config['drive'].get('sync_strategy', sync_strategy)
            use_symlinks = colab_config['drive'].get('symlinks', use_symlinks)
        
        # Copy configs dari repo ke Drive jika Drive kosong atau jika strategi adalah 'repo_priority'
        repo_configs = Path("/content/smartcash/configs")
        if repo_configs.exists() and (not any(drive_config_dir.glob("*.yaml")) or sync_strategy == 'repo_priority'):
            # Pastikan direktori Drive ada
            os.makedirs(drive_config_dir, exist_ok=True)
            # Copy file konfigurasi
            for config_file in repo_configs.glob("*.yaml"):
                shutil.copy2(config_file, drive_config_dir / config_file.name)
            
            if ui_logger:
                ui_logger.info(f"File konfigurasi disalin dari {repo_configs} ke {drive_config_dir}")
            else:
                print(f"📄 File konfigurasi disalin dari {repo_configs} ke {drive_config_dir}")
        
        # Gunakan symlinks jika dikonfigurasi
        if use_symlinks:
            # Cek apakah symlink rusak atau hilang
            if config_dir.exists() and not config_dir.is_symlink():
                # Hapus direktori jika bukan symlink
                shutil.rmtree(config_dir)
                if ui_logger:
                    ui_logger.warning(f"Direktori konfigurasi bukan symlink, akan dibuat ulang")
                else:
                    print(f"⚠️ Direktori konfigurasi bukan symlink, akan dibuat ulang")
            elif config_dir.is_symlink() and not config_dir.exists():
                # Symlink rusak, hapus
                os.unlink(config_dir)
                if ui_logger:
                    ui_logger.warning(f"Symlink konfigurasi rusak, akan dibuat ulang")
                else:
                    print(f"⚠️ Symlink konfigurasi rusak, akan dibuat ulang")
            
            # Buat symlink baru jika belum ada atau rusak
            if not config_dir.exists():
                # Pastikan direktori Drive ada sebelum membuat symlink
                if not drive_config_dir.exists():
                    drive_config_dir.mkdir(parents=True, exist_ok=True)
                    if ui_logger:
                        ui_logger.info(f"Direktori konfigurasi di Drive dibuat: {drive_config_dir}")
                    else:
                        print(f"📁 Direktori konfigurasi di Drive dibuat: {drive_config_dir}")
                
                # Buat symlink baru
                os.symlink(drive_config_dir, config_dir)
                if ui_logger:
                    ui_logger.info(f"Symlink dibuat dari {drive_config_dir} ke {config_dir}")
                else:
                    print(f"🔗 Symlink dibuat dari {drive_config_dir} ke {config_dir}")
        else:
            # Jika tidak menggunakan symlink, copy file dari Drive ke local
            if not config_dir.exists():
                config_dir.mkdir(parents=True, exist_ok=True)
                if ui_logger:
                    ui_logger.info(f"Direktori konfigurasi dibuat: {config_dir}")
                else:
                    print(f"📁 Direktori konfigurasi dibuat: {config_dir}")
            
            # Copy file dari Drive ke local
            for config_file in drive_config_dir.glob("*.yaml"):
                shutil.copy2(config_file, config_dir / config_file.name)
            
            if ui_logger:
                ui_logger.info(f"File konfigurasi disalin dari {drive_config_dir} ke {config_dir}")
            else:
                print(f"📄 File konfigurasi disalin dari {drive_config_dir} ke {config_dir}")
    else:
        # Gunakan project root sebagai base_dir
        base_dir = Path(__file__).resolve().parents[4]
        config_dir = base_dir / "configs"
        
        # Pastikan direktori config ada
        config_dir.mkdir(parents=True, exist_ok=True)
    
    # Verifikasi bahwa direktori konfigurasi ada dan dapat diakses
    if not config_dir.exists():
        # Jika masih tidak ada, buat direktori kosong sebagai fallback
        config_dir.mkdir(parents=True, exist_ok=True)
        ui_logger = _get_ui_logger()
        if ui_logger:
            ui_logger.error(f"Gagal membuat symlink atau direktori konfigurasi. Menggunakan direktori kosong sebagai fallback.")
        else:
            print(f"❌ Gagal membuat symlink atau direktori konfigurasi. Menggunakan direktori kosong sebagai fallback.")
    
    # Initialize config manager
    config_manager = get_config_manager()
    
    return config_manager, base_dir, config_dir 

def _get_ui_logger() -> Optional[Any]:
    """
    Mencoba mendapatkan UI logger jika tersedia.
    
    Returns:
        UI logger atau None jika tidak tersedia
    """
    try:
        # Coba dapatkan UI logger dari konteks global
        import builtins
        if hasattr(builtins, 'ui_components') and 'logger' in builtins.ui_components:
            logger = builtins.ui_components['logger']
            # Set level ke ERROR untuk mengurangi log yang tidak perlu
            if hasattr(logger, 'set_level'):
                import logging
                logger.set_level(logging.ERROR)
            return logger
        
        # Coba dapatkan dari modul UI
        from smartcash.ui.utils.ui_logger import get_current_ui_logger
        logger = get_current_ui_logger()
        if logger:
            # Set level ke ERROR untuk mengurangi log yang tidak perlu
            import logging
            logger.set_level(logging.ERROR)
        return logger
    except Exception:
        # Fallback ke None jika tidak ada UI logger atau terjadi error
        return None 