"""
File: smartcash/ui/setup/env_config/handlers/environment_handler.py
Deskripsi: Handler untuk operasi terkait environment (setup direktori dan file)
"""

import os
import shutil
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable

from smartcash.common.utils import is_colab
from smartcash.common.constants.paths import COLAB_PATH
from smartcash.common.environment import get_environment_manager
from smartcash.common.config import SimpleConfigManager
from smartcash.ui.utils.ui_logger_namespace import ENV_CONFIG_LOGGER_NAMESPACE
from smartcash.ui.setup.env_config.handlers.base_handler import BaseHandler
from smartcash.ui.setup.env_config.utils.config_utils import init_config_manager, ensure_config_dir

class EnvironmentHandler(BaseHandler):
    """
    Handler untuk operasi terkait environment (setup direktori dan file)
    """
    
    def __init__(self, ui_callback: Optional[Dict[str, Callable]] = None):
        """
        Inisialisasi handler
        
        Args:
            ui_callback: Dictionary callback untuk update UI
        """
        super().__init__(ui_callback, ENV_CONFIG_LOGGER_NAMESPACE)
        
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
                self._log_message(message, "success" if success else "error", "✅" if success else "❌")
                return success
            else:
                self._log_message("Google Drive sudah terhubung", "success", "✅")
                return True
        except Exception as e:
            self._update_status(f"Error connecting to Drive: {str(e)}", "error")
            self._log_message(f"Error connecting to Drive: {str(e)}", "error", "❌")
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
                    self._log_message(f"Created directory in Drive: {dir_name}", "success", "✅")
            
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
                    self._log_message(f"Backed up existing directory: {dir_name} to {dir_name}_backup", "info", "🔄")
                
                # Create symlink if it doesn't exist
                if not colab_path.exists():
                    colab_path.symlink_to(drive_dir_path)
                    self._log_message(f"Created symlink: {dir_name} -> {drive_dir_path}", "info", "🔗")
            
            return True
        except Exception as e:
            self._update_status(f"Error setting up directories: {str(e)}", "error")
            self._log_message(f"Error setting up directories: {str(e)}", "error", "❌")
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
                            self._log_message(f"Copied config file: {config_file}", "success", "✅")
            
            return True
        except Exception as e:
            self._update_status(f"Error setting up config files: {str(e)}", "error")
            self._log_message(f"Error setting up config files: {str(e)}", "error", "❌")
            return False
    
    def initialize_config_singleton(self) -> SimpleConfigManager:
        """
        Initialize config manager singleton
        
        Returns:
            SimpleConfigManager: Instance config manager atau None jika error
        """
        try:
            # Initialize config manager with proper base directory
            config_manager = init_config_manager(self.env_manager.base_dir)
            return config_manager
        except Exception as e:
            self._update_status(f"Error initializing config: {str(e)}", "error")
            self._log_message(f"Error initializing config: {str(e)}", "error", "❌")
            return None
    
    def perform_setup(self) -> bool:
        """
        Perform complete environment setup
        
        Returns:
            bool: True if all steps successful
        """
        self._update_status("Memulai setup environment...", "info")
        self._log_message("Memulai setup environment...", "info", "🚀")
        
        # Connect to drive (if not already connected)
        self._update_progress(0.1, "Menghubungkan ke Google Drive...")
        drive_success = self.connect_drive()
        if not drive_success:
            self._update_status("Error: Gagal menghubungkan ke Google Drive", "error")
            self._log_message("Error: Gagal menghubungkan ke Google Drive", "error", "❌")
            return False
        
        # Setup directories
        self._update_progress(0.3, "Membuat direktori dan symlinks...")
        dirs_success = self.setup_directories()
        if not dirs_success:
            self._update_status("Error: Gagal setup direktori", "error")
            self._log_message("Error: Gagal setup direktori", "error", "❌")
            return False
        
        # Setup config files
        self._update_progress(0.6, "Menyiapkan file konfigurasi...")
        config_success = self.setup_config_files()
        if not config_success:
            self._update_status("Error: Gagal setup file konfigurasi", "error")
            self._log_message("Error: Gagal setup file konfigurasi", "error", "❌")
            return False
        
        # Initialize config singleton
        self._update_progress(0.8, "Menginisialisasi konfigurasi...")
        config_manager = self.initialize_config_singleton()
        if config_manager is None:
            self._update_status("Error: Gagal inisialisasi konfigurasi", "error")
            self._log_message("Error: Gagal inisialisasi konfigurasi", "error", "❌")
            return False
        
        # Everything succeeded
        self._update_progress(1.0, "Setup selesai")
        self._update_status("Setup environment berhasil", "success")
        self._log_message("Setup environment berhasil", "success", "✅")
        
        return True 