"""
File: smartcash/ui/setup/env_config/components/state_manager.py
Deskripsi: Manager untuk state dan skenario environment config
"""

from typing import Dict, Any
import ipywidgets as widgets

from smartcash.ui.setup.env_config.utils.ui_utils import (
    update_status,
    set_button_state,
    log_message,
    update_progress
)

class EnvConfigStateManager:
    """
    Manager untuk state dan skenario environment config
    """
    
    def __init__(self, ui_components: Dict[str, Any], colab_manager: Any):
        """
        Inisialisasi state manager
        
        Args:
            ui_components: Dictionary UI components
            colab_manager: ColabConfigManager instance
        """
        self.ui_components = ui_components
        self.colab_manager = colab_manager
        self.tracker = ui_components.get('env_config_tracker')
        
        # Initial state
        self._update_initial_state()
    
    def _update_initial_state(self):
        """
        Update state awal berdasarkan kondisi environment
        """
        # Check if drive is already connected
        if self.colab_manager.is_drive_connected():
            set_button_state(self.ui_components['drive_button'], False, "success")
            update_status(self.ui_components, "Google Drive already connected", "success")
            log_message(self.ui_components, "Google Drive already connected")
        else:
            set_button_state(self.ui_components['drive_button'], False, "info")
            update_status(self.ui_components, "Connect to Google Drive", "info")
        
        # Check if directories are already set up
        base_dirs = [
            "/content/data",
            "/content/models",
            "/content/output",
            "/content/logs",
            "/content/exports"
        ]
        
        all_dirs_exist = all(Path(dir_path).exists() for dir_path in base_dirs)
        if all_dirs_exist:
            set_button_state(self.ui_components['directory_button'], False, "success")
            update_status(self.ui_components, "Directories already set up", "success")
            log_message(self.ui_components, "Directories already set up")
        else:
            set_button_state(self.ui_components['directory_button'], False, "info")
            update_status(self.ui_components, "Setup directories", "info")
    
    def handle_drive_connection_start(self):
        """
        Handle state saat memulai koneksi drive
        """
        set_button_state(self.ui_components['drive_button'], True, "info")
        set_button_state(self.ui_components['directory_button'], True, "info")
        update_status(self.ui_components, "Connecting to Google Drive...", "info")
        update_progress(self.tracker, 0.2, "Connecting to Google Drive...")
    
    def handle_drive_connection_success(self):
        """
        Handle state saat koneksi drive berhasil
        """
        set_button_state(self.ui_components['drive_button'], False, "success")
        set_button_state(self.ui_components['directory_button'], False, "info")
        update_status(self.ui_components, "Successfully connected to Google Drive", "success")
        update_progress(self.tracker, 1.0, "Drive connection completed")
        log_message(self.ui_components, "Successfully connected to Google Drive")
        log_message(self.ui_components, f"Drive path: {self.colab_manager.drive_base_path}")
    
    def handle_drive_connection_error(self, error: str):
        """
        Handle state saat koneksi drive gagal
        
        Args:
            error: Error message
        """
        set_button_state(self.ui_components['drive_button'], False, "danger")
        set_button_state(self.ui_components['directory_button'], False, "info")
        update_status(self.ui_components, f"Failed to connect to Google Drive: {error}", "error")
        update_progress(self.tracker, 0, "Drive connection failed")
        log_message(self.ui_components, f"Error connecting to Google Drive: {error}", "error")
    
    def handle_directory_setup_start(self):
        """
        Handle state saat memulai setup directory
        """
        set_button_state(self.ui_components['directory_button'], True, "info")
        update_status(self.ui_components, "Setting up directories...", "info")
        update_progress(self.tracker, 0.2, "Creating directories...")
    
    def handle_directory_setup_success(self):
        """
        Handle state saat setup directory berhasil
        """
        set_button_state(self.ui_components['directory_button'], False, "success")
        update_status(self.ui_components, "Successfully set up directories", "success")
        update_progress(self.tracker, 1.0, "Directory setup completed")
        log_message(self.ui_components, "Successfully set up directories")
    
    def handle_directory_setup_error(self, error: str):
        """
        Handle state saat setup directory gagal
        
        Args:
            error: Error message
        """
        set_button_state(self.ui_components['directory_button'], False, "danger")
        update_status(self.ui_components, f"Failed to set up directories: {error}", "error")
        update_progress(self.tracker, 0, "Directory setup failed")
        log_message(self.ui_components, f"Error setting up directory: {error}", "error")
    
    def handle_drive_sync_start(self):
        """
        Handle state saat memulai sinkronisasi drive
        """
        set_button_state(self.ui_components['drive_button'], True, "info")
        set_button_state(self.ui_components['directory_button'], True, "info")
        update_status(self.ui_components, "Syncing with Google Drive...", "info")
        update_progress(self.tracker, 0.2, "Syncing with Google Drive...")
    
    def handle_drive_sync_success(self):
        """
        Handle state saat sinkronisasi drive berhasil
        """
        set_button_state(self.ui_components['drive_button'], False, "success")
        set_button_state(self.ui_components['directory_button'], False, "success")
        update_status(self.ui_components, "Successfully synced with Google Drive", "success")
        update_progress(self.tracker, 1.0, "Drive sync completed")
        log_message(self.ui_components, "Successfully synced with Google Drive")
    
    def handle_drive_sync_error(self, error: str):
        """
        Handle state saat sinkronisasi drive gagal
        
        Args:
            error: Error message
        """
        set_button_state(self.ui_components['drive_button'], False, "danger")
        set_button_state(self.ui_components['directory_button'], False, "danger")
        update_status(self.ui_components, f"Failed to sync with Google Drive: {error}", "error")
        update_progress(self.tracker, 0, "Drive sync failed")
        log_message(self.ui_components, f"Error syncing with Google Drive: {error}", "error") 