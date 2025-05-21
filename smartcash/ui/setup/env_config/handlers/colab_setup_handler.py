"""
File: smartcash/ui/setup/env_config/handlers/colab_setup_handler.py
Deskripsi: Handler untuk setup environment di lingkungan Google Colab
"""

import os
import shutil
from pathlib import Path
from typing import Tuple, Any, Dict, Optional, Callable
import logging
import yaml

class ColabSetupHandler:
    """
    Handler untuk operasi setup di lingkungan Google Colab
    """
    
    def __init__(self, ui_callback: Optional[Dict[str, Callable]] = None):
        """
        Inisialisasi handler untuk setup Colab
        
        Args:
            ui_callback: Dictionary callback untuk update UI
        """
        self.ui_callback = ui_callback or {}
        self.logger = logging.getLogger(__name__)
        
    def _log_message(self, message: str):
        """Log message to UI if callback exists"""
        self.logger.info(message)
        if 'log_message' in self.ui_callback:
            self.ui_callback['log_message'](message)
            
    def _update_status(self, message: str, status_type: str = "info"):
        """Update status in UI if callback exists"""
        if 'update_status' in self.ui_callback:
            self.ui_callback['update_status'](message, status_type)
            
    def load_colab_config(self) -> Dict[str, Any]:
        """
        Load konfigurasi Colab
        
        Returns:
            Dictionary konfigurasi Colab
        """
        colab_config = {}
        repo_config_path = Path("/content/smartcash/configs/colab_config.yaml")
        
        if repo_config_path.exists():
            try:
                with open(repo_config_path, 'r') as f:
                    colab_config = yaml.safe_load(f) or {}
                self._log_message(f"✅ Konfigurasi Colab dimuat dari {repo_config_path}")
            except Exception as e:
                self._log_message(f"⚠️ Gagal memuat colab_config.yaml: {str(e)}")
        else:
            self._log_message(f"ℹ️ File konfigurasi Colab tidak ditemukan di {repo_config_path}")
        
        return colab_config
    
    def get_drive_paths(self, colab_config: Dict[str, Any]) -> Tuple[str, str]:
        """
        Dapatkan path Drive dari konfigurasi
        
        Args:
            colab_config: Konfigurasi Colab
            
        Returns:
            Tuple berisi (drive_dir, configs_dir)
        """
        drive_dir = 'SmartCash'
        configs_dir = 'configs'
        
        if 'drive' in colab_config and 'paths' in colab_config['drive']:
            drive_paths = colab_config['drive']['paths']
            drive_dir = drive_paths.get('smartcash_dir', drive_dir)
            configs_dir = drive_paths.get('configs_dir', configs_dir)
        
        return drive_dir, configs_dir
    
    def get_sync_strategy(self, colab_config: Dict[str, Any]) -> Tuple[str, bool]:
        """
        Dapatkan strategi sinkronisasi dari konfigurasi
        
        Args:
            colab_config: Konfigurasi Colab
            
        Returns:
            Tuple berisi (sync_strategy, use_symlinks)
        """
        sync_strategy = 'drive_priority'
        use_symlinks = True
        
        if 'drive' in colab_config:
            sync_strategy = colab_config['drive'].get('sync_strategy', sync_strategy)
            use_symlinks = colab_config['drive'].get('symlinks', use_symlinks)
        
        return sync_strategy, use_symlinks
    
    def sync_configs(self, source_dir: Path, target_dir: Path) -> None:
        """
        Sinkronisasi konfigurasi dari source ke target
        
        Args:
            source_dir: Path direktori sumber
            target_dir: Path direktori target
        """
        # Pastikan direktori target ada
        os.makedirs(target_dir, exist_ok=True)
        
        # Copy file konfigurasi
        count = 0
        for config_file in source_dir.glob("*.yaml"):
            shutil.copy2(config_file, target_dir / config_file.name)
            count += 1
        
        self._log_message(f"✅ {count} file konfigurasi disalin dari {source_dir} ke {target_dir}")
    
    def setup_symlinks(self, config_dir: Path, drive_config_dir: Path) -> Path:
        """
        Setup symlinks untuk direktori konfigurasi
        
        Args:
            config_dir: Path direktori konfigurasi
            drive_config_dir: Path direktori konfigurasi di Drive
            
        Returns:
            Path direktori konfigurasi yang diperbarui
        """
        # Cek apakah symlink rusak atau hilang
        if config_dir.exists() and not config_dir.is_symlink():
            # Hapus direktori jika bukan symlink
            shutil.rmtree(config_dir)
            self._log_message(f"⚠️ Direktori konfigurasi bukan symlink, akan dibuat ulang")
        elif config_dir.is_symlink() and not os.path.exists(config_dir):
            # Symlink rusak, hapus
            os.unlink(config_dir)
            self._log_message(f"⚠️ Symlink konfigurasi rusak, akan dibuat ulang")
        
        # Buat symlink baru jika belum ada atau rusak
        if not config_dir.exists():
            # Pastikan direktori Drive ada sebelum membuat symlink
            if not drive_config_dir.exists():
                drive_config_dir.mkdir(parents=True, exist_ok=True)
                self._log_message(f"✅ Direktori konfigurasi di Drive dibuat: {drive_config_dir}")
            
            # Buat symlink baru
            os.symlink(drive_config_dir, config_dir)
            self._log_message(f"✅ Symlink dibuat dari {drive_config_dir} ke {config_dir}")
        
        return config_dir
    
    def setup_direct_copy(self, config_dir: Path, drive_config_dir: Path) -> Path:
        """
        Setup direktori konfigurasi dengan copy langsung
        
        Args:
            config_dir: Path direktori konfigurasi
            drive_config_dir: Path direktori konfigurasi di Drive
            
        Returns:
            Path direktori konfigurasi yang diperbarui
        """
        # Jika tidak menggunakan symlink, copy file dari Drive ke local
        if not config_dir.exists():
            config_dir.mkdir(parents=True, exist_ok=True)
            self._log_message(f"✅ Direktori konfigurasi dibuat: {config_dir}")
        
        # Copy file dari Drive ke local
        count = 0
        for config_file in drive_config_dir.glob("*.yaml"):
            shutil.copy2(config_file, config_dir / config_file.name)
            count += 1
        
        self._log_message(f"✅ {count} file konfigurasi disalin dari {drive_config_dir} ke {config_dir}")
        
        return config_dir
    
    def setup_colab_environment(self) -> Tuple[Path, Path]:
        """
        Setup environment untuk Colab
        
        Returns:
            Tuple of (base_dir, config_dir)
        """
        base_dir = Path("/content")
        config_dir = base_dir / "configs"
        
        try:
            # Coba mendapatkan konfigurasi dari colab_config.yaml
            colab_config = self.load_colab_config()
            
            # Dapatkan informasi konfigurasi drive
            drive_dir, configs_dir = self.get_drive_paths(colab_config)
            drive_config_dir = Path(f"/content/drive/MyDrive/{drive_dir}/{configs_dir}")
            drive_config_dir.mkdir(parents=True, exist_ok=True)
            
            # Dapatkan strategi sinkronisasi
            sync_strategy, use_symlinks = self.get_sync_strategy(colab_config)
            
            # Sinkronisasi dengan repo jika perlu
            repo_configs = Path("/content/smartcash/configs")
            if repo_configs.exists() and (not any(drive_config_dir.glob("*.yaml")) or sync_strategy == 'repo_priority'):
                self.sync_configs(repo_configs, drive_config_dir)
            
            # Konfigurasi direktori berdasarkan strategi
            if use_symlinks:
                config_dir = self.setup_symlinks(config_dir, drive_config_dir)
            else:
                config_dir = self.setup_direct_copy(config_dir, drive_config_dir)
                
            self._log_message(f"✅ Setup environment Colab selesai. Base: {base_dir}, Config: {config_dir}")
            
        except Exception as e:
            self._log_message(f"❌ Error saat setup Colab environment: {str(e)}")
            self._update_status(f"Error saat setup environment: {str(e)}", "error")
            
            # Fallback - buat direktori config kosong jika terjadi error
            if not config_dir.exists():
                config_dir.mkdir(parents=True, exist_ok=True)
                self._log_message(f"✅ Direktori konfigurasi fallback dibuat: {config_dir}")
        
        return base_dir, config_dir 