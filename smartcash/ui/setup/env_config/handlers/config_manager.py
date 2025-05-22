"""
File: smartcash/ui/setup/env_config/handlers/config_manager.py
Deskripsi: Handler untuk manajemen file konfigurasi - SRP untuk config operations
"""

import shutil
from typing import Dict, Any
from pathlib import Path

from smartcash.ui.utils.logger_bridge import create_ui_logger_bridge
from smartcash.common.config import get_config_manager

class ConfigFileManager:
    """Handler untuk manajemen file konfigurasi - fokus hanya pada config file operations"""
    
    def __init__(self, ui_components: Dict[str, Any]):
        """
        Inisialisasi manager dengan UI components
        
        Args:
            ui_components: Dictionary komponen UI
        """
        self.ui_components = ui_components
        self.logger = create_ui_logger_bridge(ui_components, "config_manager")
    
    def setup_config_directory(self, drive_path: Path = None) -> bool:
        """
        Setup direktori konfigurasi dengan symlink jika perlu
        
        Args:
            drive_path: Path ke direktori config di Drive (untuk Colab)
            
        Returns:
            Success status
        """
        local_config_path = Path("/content/configs") if self._is_colab() else Path.cwd() / "configs"
        
        try:
            if self._is_colab() and drive_path:
                # Setup symlink untuk Colab
                return self._setup_config_symlink(local_config_path, drive_path)
            else:
                # Setup direktori biasa untuk local
                local_config_path.mkdir(parents=True, exist_ok=True)
                self.logger.info("📝 Direktori config lokal dibuat")
                return True
                
        except Exception as e:
            self.logger.error(f"❌ Error setup config directory: {str(e)}")
            return False
    
    def _setup_config_symlink(self, local_path: Path, drive_path: Path) -> bool:
        """Setup symlink untuk direktori config"""
        try:
            # Pastikan drive path ada
            drive_path.mkdir(parents=True, exist_ok=True)
            
            # Backup dan hapus direktori lokal jika bukan symlink
            if local_path.exists() and not local_path.is_symlink():
                backup_path = Path("/content/configs_backup")
                if backup_path.exists():
                    shutil.rmtree(backup_path)
                shutil.move(local_path, backup_path)
                self.logger.info("🔄 Backup direktori config")
            
            # Buat symlink jika belum ada
            if not local_path.exists():
                local_path.symlink_to(drive_path)
                self.logger.success("🔗 Symlink config dibuat")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error creating config symlink: {str(e)}")
            return False
    
    def copy_default_configs(self) -> bool:
        """
        Copy file konfigurasi default dari repo ke direktori config
        
        Returns:
            Success status
        """
        config_files = [
            'base_config.yaml', 'colab_config.yaml', 'dataset_config.yaml',
            'model_config.yaml', 'training_config.yaml', 'augmentation_config.yaml',
            'preprocessing_config.yaml', 'hyperparameters_config.yaml'
        ]
        
        try:
            repo_config_path = Path("/content/smartcash/configs") if self._is_colab() else Path.cwd() / "smartcash/configs"
            target_config_path = Path("/content/configs") if self._is_colab() else Path.cwd() / "configs"
            
            if not repo_config_path.exists():
                self.logger.warning("⚠️ Direktori config repo tidak ditemukan")
                return False
            
            copied_count = 0
            for config_file in config_files:
                src_file = repo_config_path / config_file
                dst_file = target_config_path / config_file
                
                # Skip jika file sudah ada
                if dst_file.exists():
                    continue
                
                if src_file.exists():
                    shutil.copy2(src_file, dst_file)
                    copied_count += 1
            
            if copied_count > 0:
                self.logger.success(f"✅ Disalin {copied_count} file config")
            else:
                self.logger.info("📝 Semua config sudah ada")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error copying configs: {str(e)}")
            return False
    
    def initialize_config_manager(self) -> Any:
        """
        Inisialisasi config manager singleton
        
        Returns:
            Config manager instance atau None jika error
        """
        try:
            config_manager = get_config_manager()
            
            if config_manager:
                self.logger.success("🔧 Config manager terinisialisasi")
                return config_manager
            else:
                self.logger.error("❌ Gagal inisialisasi config manager")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Error inisialisasi config: {str(e)}")
            return None
    
    def verify_configs(self) -> Dict[str, bool]:
        """
        Verifikasi file konfigurasi yang tersedia
        
        Returns:
            Dictionary status config files
        """
        required_configs = ['base_config.yaml', 'colab_config.yaml', 'dataset_config.yaml']
        config_path = Path("/content/configs") if self._is_colab() else Path.cwd() / "configs"
        
        status = {}
        missing_configs = []
        
        for config_file in required_configs:
            file_path = config_path / config_file
            exists = file_path.exists()
            status[config_file] = exists
            
            if not exists:
                missing_configs.append(config_file)
        
        if missing_configs:
            self.logger.warning(f"⚠️ Config belum ada: {', '.join(missing_configs)}")
        else:
            self.logger.success("✅ Semua config tersedia")
        
        return status
    
    def _is_colab(self) -> bool:
        """Check apakah running di Colab"""
        try:
            import google.colab
            return True
        except ImportError:
            return False