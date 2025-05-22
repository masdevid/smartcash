"""
File: smartcash/ui/setup/env_config/handlers/environment_handler.py
Deskripsi: Handler untuk operasi terkait environment - diperbaiki dengan mengurangi duplikasi dan meningkatkan integrasi
"""

import os
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable

from smartcash.ui.setup.env_config.handlers.base_handler import BaseHandler, EnvConfigHandlerMixin

class EnvironmentHandler(BaseHandler, EnvConfigHandlerMixin):
    """
    Handler untuk operasi terkait environment - fokus pada business logic tanpa UI concerns
    """
    
    def __init__(self, ui_components: Dict[str, Any] = None, ui_callback: Optional[Dict[str, Callable]] = None):
        """
        Inisialisasi handler dengan UI components atau callbacks
        
        Args:
            ui_components: Dictionary komponen UI (preferred)
            ui_callback: Dictionary callback untuk update UI (legacy support)
        """
        super().__init__(ui_components, ui_callback)
        
        # Initialize managers
        self.env_manager = self._get_environment_manager()
        self.config_manager = self._get_config_manager()
        
        if self.env_manager:
            self._log_message(f"Environment manager initialized. Is Colab: {self.env_manager.is_colab}", "info", "🔧")
    
    def check_required_dirs(self) -> bool:
        """Check if required directories exist in Colab"""
        if not self.check_colab_environment():
            return True  # Assume OK for non-Colab
            
        required_dirs = self.get_required_directories()
        missing_dirs = []
        
        for dir_name in required_dirs:
            if not Path(f"/content/{dir_name}").exists():
                missing_dirs.append(dir_name)
        
        return len(missing_dirs) == 0
    
    def perform_setup(self) -> bool:
        """
        Perform complete environment setup dengan progress tracking
        """
        self._update_status("Memulai setup environment...", "info")
        self._log_message("Memulai setup environment...", "info", "🚀")
        
        try:
            # Step 1: Connect to drive (if needed)
            self._update_progress(0.2, "Menghubungkan ke Google Drive...")
            if not self.ensure_drive_connection():
                return False
            
            # Step 2: Setup directories
            self._update_progress(0.4, "Membuat direktori dan symlinks...")
            if not self._setup_directories():
                return False
            
            # Step 3: Setup config files
            self._update_progress(0.6, "Menyiapkan file konfigurasi...")
            if not self._setup_config_files():
                return False
            
            # Step 4: Initialize config singleton
            self._update_progress(0.8, "Menginisialisasi konfigurasi...")
            config_manager = self.initialize_config_singleton()
            if config_manager is None:
                return False
            
            # Success
            self._update_progress(1.0, "Setup selesai")
            self._update_status("Setup environment berhasil", "success")
            self._log_message("Setup environment berhasil", "success", "✅")
            
            return True
            
        except Exception as e:
            self._update_status(f"Error: {str(e)}", "error")
            self._log_message(f"Error saat setup environment: {str(e)}", "error", "❌")
            return False
    
    def _setup_directories(self) -> bool:
        """Setup directories and symlinks"""
        if not self.check_colab_environment():
            return True  # Skip for non-Colab
            
        try:
            if not self.env_manager or not self.env_manager.is_drive_mounted:
                self._log_message("Google Drive tidak terhubung", "error", "❌")
                return False
            
            # Directories to create in drive
            drive_dirs = ['data', 'exports', 'logs', 'models', 'output']
            drive_path = self.env_manager.drive_path
            
            # Create directories in drive
            for dir_name in drive_dirs:
                dir_path = Path(drive_path) / dir_name
                if not dir_path.exists():
                    os.makedirs(dir_path, exist_ok=True)
                    self._log_message(f"Dibuat direktori di Drive: {dir_name} 📁", "success", "✅")
            
            # Create symlinks in Colab
            for dir_name in drive_dirs:
                colab_path = Path(f"/content/{dir_name}")
                drive_dir_path = Path(drive_path) / dir_name
                
                # Backup existing directory if needed
                if colab_path.exists() and not colab_path.is_symlink():
                    backup_path = Path(f"/content/{dir_name}_backup")
                    if backup_path.exists():
                        shutil.rmtree(backup_path)
                    shutil.move(colab_path, backup_path)
                    self._log_message(f"Backup direktori: {dir_name} → {dir_name}_backup 🔄", "info", "📦")
                
                # Create symlink
                if not colab_path.exists():
                    colab_path.symlink_to(drive_dir_path)
                    self._log_message(f"Symlink dibuat: {dir_name} 🔗", "info", "🔗")
            
            return True
            
        except Exception as e:
            self._log_message(f"Error setup direktori: {str(e)}", "error", "❌")
            return False
    
    def _setup_config_files(self) -> bool:
        """Setup configuration files"""
        try:
            configs_dir = Path("/content/configs")
            
            # Ensure configs directory exists
            if not configs_dir.exists():
                os.makedirs(configs_dir, exist_ok=True)
                self._log_message("Direktori configs dibuat 📁", "success", "✅")
            
            # Config files to check
            config_files = [
                'dataset_config.yaml', 'training_config.yaml', 'model_config.yaml',
                'augmentation_config.yaml', 'evaluation_config.yaml', 'preprocessing_config.yaml',
                'hyperparameters_config.yaml', 'base_config.yaml', 'colab_config.yaml'
            ]
            
            # Check for missing configs
            missing_configs = []
            for config_file in config_files:
                if not (configs_dir / config_file).exists():
                    missing_configs.append(config_file)
            
            # Copy from repo if missing
            if missing_configs:
                repo_configs_dir = Path("/content/smartcash/configs")
                if repo_configs_dir.exists():
                    copied_count = 0
                    for config_file in missing_configs:
                        src_file = repo_configs_dir / config_file
                        if src_file.exists():
                            shutil.copy2(src_file, configs_dir / config_file)
                            copied_count += 1
                    
                    if copied_count > 0:
                        self._log_message(f"{copied_count} file config disalin dari repo 📋", "success", "✅")
                else:
                    self._log_message("Direktori repo config tidak ditemukan", "warning", "⚠️")
            
            return True
            
        except Exception as e:
            self._log_message(f"Error setup config files: {str(e)}", "error", "❌")
            return False
    
    def initialize_config_singleton(self):
        """Initialize config manager singleton dengan proper error handling"""
        try:
            # Use existing config manager or create new one
            if self.config_manager is None:
                self.config_manager = self._get_config_manager()
            
            if self.config_manager:
                self._log_message("Config manager berhasil diinisialisasi 🔧", "success", "✅")
                return self.config_manager
            else:
                self._log_message("Gagal menginisialisasi config manager", "error", "❌")
                return None
                
        except Exception as e:
            self._log_message(f"Error inisialisasi config: {str(e)}", "error", "❌")
            return None