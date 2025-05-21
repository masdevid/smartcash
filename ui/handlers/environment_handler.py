"""
File: smartcash/ui/handlers/environment_handler.py
Deskripsi: Handler untuk operasi terkait environment
"""

import os
import shutil
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Callable

from smartcash.common.utils import is_colab
from smartcash.common.constants.paths import COLAB_PATH
from smartcash.common.environment import get_environment_manager

class EnvironmentHandler:
    """
    Handler untuk operasi terkait environment
    """
    
    def __init__(self, ui_callback: Optional[Dict[str, Callable]] = None):
        """
        Inisialisasi handler
        
        Args:
            ui_callback: Dictionary callback untuk update UI
        """
        # Setup logger tanpa menggunakan UILogger untuk menghindari circular dependency
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            # Gunakan sys.__stdout__ untuk menghindari rekursi
            handler = logging.StreamHandler(sys.__stdout__)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        
        # Initialize environment manager singleton first
        self.env_manager = get_environment_manager()
        
        # Atur level logging untuk environment manager dan config manager
        # Hanya tampilkan error logs
        env_logger = logging.getLogger('smartcash.common.environment')
        if env_logger:
            env_logger.setLevel(logging.ERROR)
            
        config_logger = logging.getLogger('smartcash.common.config.manager')
        if config_logger:
            config_logger.setLevel(logging.ERROR)
            
        self.logger.info(f"Environment manager initialized. Is Colab: {self.env_manager.is_colab}")
        
        # Set callback functions
        self.ui_callback = ui_callback or {}
        
        # Setup required directories
        self.required_dirs = [
            'smartcash', 'yolov5', 'data', 'exports', 
            'logs', 'models', 'output', 'configs'
        ]
        
        # Setup required config files
        self.config_files = [
            'dataset_config.yaml',
            'training_config.yaml',
            'model_config.yaml',
            'augmentation_config.yaml',
            'evaluation_config.yaml',
            'preprocessing_config.yaml',
            'hyperparameters_config.yaml',
            'base_config.yaml',
            'colab_config.yaml'
        ]
    
    def _log_message(self, message: str):
        """Log message to UI if callback exists"""
        self.logger.info(message)
            
        if 'log_message' in self.ui_callback:
            self.ui_callback['log_message'](message)
    
    def _update_status(self, message: str, status_type: str = "info"):
        """Update status in UI if callback exists"""
        if 'update_status' in self.ui_callback:
            self.ui_callback['update_status'](message, status_type)
    
    def _update_progress(self, value: float, message: str = ""):
        """Update progress in UI if callback exists"""
        if 'update_progress' in self.ui_callback:
            self.ui_callback['update_progress'](value, message)
    
    def check_required_dirs(self) -> bool:
        """
        Check if required directories exist in Colab
        
        Returns:
            bool: True if all directories exist
        """
        if not is_colab():
            return False
            
        all_exist = True
        for dir_name in self.required_dirs:
            if not Path(f"{COLAB_PATH}/{dir_name}").exists():
                all_exist = False
                break
        return all_exist
    
    def connect_drive(self) -> bool:
        """
        Connect to Google Drive
        
        Returns:
            bool: True if successful
        """
        try:
            # Use environment manager to mount drive
            if not self.env_manager.is_drive_mounted:
                success, message = self.env_manager.mount_drive()
                self._update_status(message, "success" if success else "error")
                self._log_message(message)
                return success
            else:
                self._log_message("✅ Google Drive sudah terhubung")
                return True
        except Exception as e:
            self._update_status(f"Error connecting to Drive: {str(e)}", "error")
            self._log_message(f"Error connecting to Drive: {str(e)}")
            return False
    
    def setup_directories(self) -> bool:
        """
        Setup directories and symlinks
        
        Returns:
            bool: True if successful
        """
        try:
            # Directories to create in drive
            drive_dirs = ['data', 'exports', 'logs', 'models', 'output']
            
            # Create directories in drive if they don't exist
            drive_path = self.env_manager.drive_path
            for dir_name in drive_dirs:
                dir_path = Path(drive_path) / dir_name
                if not dir_path.exists():
                    os.makedirs(dir_path, exist_ok=True)
                    self._log_message(f"✅ Created directory in Drive: {dir_name}")
            
            # Create symlinks in Colab
            for dir_name in drive_dirs:
                colab_path = Path(f"{COLAB_PATH}/{dir_name}")
                drive_dir_path = Path(drive_path) / dir_name
                
                # If directory exists in Colab but is not a symlink, backup and remove
                if colab_path.exists() and not colab_path.is_symlink():
                    backup_path = Path(f"{COLAB_PATH}/{dir_name}_backup")
                    if backup_path.exists():
                        shutil.rmtree(backup_path)
                    shutil.move(colab_path, backup_path)
                    self._log_message(f"🔄 Backed up existing directory: {dir_name} to {dir_name}_backup")
                
                # Create symlink if it doesn't exist
                if not colab_path.exists():
                    colab_path.symlink_to(drive_dir_path)
                    self._log_message(f"🔗 Created symlink: {dir_name} -> {drive_dir_path}")
            
            return True
        except Exception as e:
            self._update_status(f"Error setting up directories: {str(e)}", "error")
            self._log_message(f"Error setting up directories: {str(e)}")
            return False
    
    def setup_config_files(self) -> bool:
        """
        Setup configuration files
        
        Returns:
            bool: True if successful
        """
        try:
            # Ensure configs directory exists
            configs_dir = Path(f"{COLAB_PATH}/configs")
            if not configs_dir.exists():
                os.makedirs(configs_dir, exist_ok=True)
            
            # Check for config files in configs directory
            missing_configs = []
            for config_file in self.config_files:
                if not (configs_dir / config_file).exists():
                    missing_configs.append(config_file)
            
            # If configs are missing, copy from smartcash repo
            if missing_configs:
                repo_configs_dir = Path(f"{COLAB_PATH}/smartcash/configs")
                if repo_configs_dir.exists():
                    for config_file in missing_configs:
                        src_file = repo_configs_dir / config_file
                        if src_file.exists():
                            shutil.copy2(src_file, configs_dir / config_file)
                            self._log_message(f"📄 Copied config file: {config_file}")
                        else:
                            self._log_message(f"⚠️ Config file not found in repo: {config_file}")
                else:
                    self._log_message("⚠️ Repository config directory not found")
            
            return True
        except Exception as e:
            self._update_status(f"Error setting up config files: {str(e)}", "error")
            self._log_message(f"Error setting up config files: {str(e)}")
            return False
    
    def initialize_config_singleton(self) -> bool:
        """
        Initialize singleton config manager
        
        Returns:
            bool: True if successful
        """
        try:
            # Initialize config manager at the end of setup process
            from smartcash.common.config.manager import get_config_manager
            config_manager = get_config_manager()
            self._log_message("✅ Initialized config manager singleton")
            return True
        except Exception as e:
            self._update_status(f"Error initializing config singleton: {str(e)}", "error")
            self._log_message(f"Error initializing config singleton: {str(e)}")
            return False
    
    def perform_setup(self) -> bool:
        """
        Perform complete environment setup
        
        Returns:
            bool: True if successful
        """
        try:
            self._update_status("Memulai setup environment...", "info")
            self._update_progress(0.1, "Memeriksa status environment...")
            
            # Step 1: Connect to Drive if not connected
            if not self.env_manager.is_drive_mounted:
                self._update_progress(0.2, "Menghubungkan ke Google Drive...")
                if not self.connect_drive():
                    self._update_status("Gagal menghubungkan ke Google Drive", "error")
                    return False
            else:
                self._log_message("✅ Google Drive sudah terhubung")
            
            # Step 2: Setup directories and symlinks
            self._update_progress(0.4, "Menyiapkan direktori dan symlink...")
            if not self.setup_directories():
                self._update_status("Gagal menyiapkan direktori", "error")
                return False
            
            # Step 3: Setup config files
            self._update_progress(0.6, "Menyiapkan file konfigurasi...")
            if not self.setup_config_files():
                self._update_status("Gagal menyiapkan file konfigurasi", "error")
                return False
            
            # Step 4: Initialize config singleton (at the end)
            self._update_progress(0.8, "Menginisialisasi singleton config manager...")
            if not self.initialize_config_singleton():
                self._update_status("Gagal menginisialisasi config manager", "error")
                return False
            
            # Complete
            self._update_progress(1.0, "Setup selesai")
            self._update_status("Setup environment berhasil", "success")
            self._log_message("✅ Setup environment berhasil diselesaikan")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error saat setup environment: {str(e)}")
                
            self._update_status(f"Error saat setup environment: {str(e)}", "error")
            self._log_message(f"❌ Error saat setup environment: {str(e)}")
            return False 