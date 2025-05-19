"""
File: smartcash/ui/setup/env_config/components/state_manager.py
Deskripsi: Manager untuk state dan skenario environment config
"""

from typing import Dict, Any
import ipywidgets as widgets
from pathlib import Path

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
        # Reset UI components to initial state
        self.ui_components['status_panel'].value = ""
        self.ui_components['log_panel'].value = ""
        self.ui_components['drive_button'].disabled = False
        self.ui_components['directory_button'].disabled = False
        
        # Check if drive is already connected
        if self.colab_manager.is_drive_connected():
            self.ui_components['drive_button'].disabled = False
            log_message(self.ui_components, "Google Drive already connected")
        else:
            self.ui_components['drive_button'].disabled = False
        
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
            self.ui_components['directory_button'].disabled = False
            log_message(self.ui_components, "Directories already set up")
        else:
            self.ui_components['directory_button'].disabled = False
    
    def handle_drive_connection_start(self):
        """
        Handle state saat memulai koneksi drive
        """
        self.ui_components['drive_button'].disabled = True
        self.ui_components['directory_button'].disabled = True
        update_status(self.ui_components, "Connecting to Google Drive", "info")
        update_progress(self.tracker, 0.2, "Connecting to Google Drive...")
    
    def handle_drive_connection_success(self):
        """
        Handle state saat koneksi drive berhasil
        """
        self.ui_components['drive_button'].disabled = True
        self.ui_components['directory_button'].disabled = False
        update_status(self.ui_components, "Connected to Google Drive", "success")
        update_progress(self.tracker, 1.0, "Drive connection completed")
        log_message(self.ui_components, "Successfully connected to Google Drive")
        log_message(self.ui_components, f"Drive path: {self.colab_manager.drive_base_path}")
    
    def handle_drive_connection_error(self, error: str):
        """
        Handle state saat koneksi drive gagal
        
        Args:
            error: Error message
        """
        self.ui_components['drive_button'].disabled = False
        self.ui_components['directory_button'].disabled = False
        update_status(self.ui_components, error, "error")
        update_progress(self.tracker, 0, "Drive connection failed")
        log_message(self.ui_components, f"Error connecting to Google Drive: {error}", "error")
    
    def handle_directory_setup_start(self):
        """
        Handle state saat memulai setup directory
        """
        self.ui_components['directory_button'].disabled = True
        update_status(self.ui_components, "Setting up directories", "info")
        update_progress(self.tracker, 0.2, "Creating directories...")
    
    def handle_directory_setup_success(self):
        """
        Handle state saat setup directory berhasil
        """
        self.ui_components['directory_button'].disabled = True
        update_status(self.ui_components, "Directories setup complete", "success")
        update_progress(self.tracker, 1.0, "Directory setup completed")
        log_message(self.ui_components, "Successfully set up directories")
    
    def handle_directory_setup_error(self, error: str):
        """
        Handle state saat setup directory gagal
        
        Args:
            error: Error message
        """
        self.ui_components['directory_button'].disabled = False
        update_status(self.ui_components, error, "error")
        update_progress(self.tracker, 0, "Directory setup failed")
        log_message(self.ui_components, f"Error setting up directory: {error}", "error")
    
    def handle_drive_sync_start(self):
        """
        Handle state saat memulai sinkronisasi drive
        """
        self.ui_components['drive_button'].disabled = True
        self.ui_components['directory_button'].disabled = True
        update_status(self.ui_components, "Syncing with Google Drive", "info")
        update_progress(self.tracker, 0.2, "Syncing with Google Drive...")
    
    def handle_drive_sync_success(self):
        """
        Handle state saat sinkronisasi drive berhasil
        """
        self.ui_components['drive_button'].disabled = True
        self.ui_components['directory_button'].disabled = False
        update_status(self.ui_components, "Sync complete", "success")
        update_progress(self.tracker, 1.0, "Drive sync completed")
        log_message(self.ui_components, "Successfully synced with Google Drive")
    
    def handle_drive_sync_error(self, error: str):
        """
        Handle state saat sinkronisasi drive gagal
        
        Args:
            error: Error message
        """
        self.ui_components['drive_button'].disabled = False
        self.ui_components['directory_button'].disabled = False
        update_status(self.ui_components, error, "error")
        update_progress(self.tracker, 0, "Drive sync failed")
        log_message(self.ui_components, f"Error syncing with Google Drive: {error}", "error")

    def update_progress(self, value: int, message: str):
        """
        Update progress bar and message
        
        Args:
            value: Progress value (0-100)
            message: Progress message
        """
        self.ui_components['progress_bar'].value = value
        self.ui_components['progress_message'].value = message
        if self.tracker:
            update_progress(self.tracker, value / 100, message)
    
    def reset_progress(self):
        """Reset progress bar and message to initial state"""
        self.ui_components['progress_bar'].value = 0
        self.ui_components['progress_message'].value = ""
        if self.tracker:
            update_progress(self.tracker, 0, "") 