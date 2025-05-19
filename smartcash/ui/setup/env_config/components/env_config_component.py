"""
File: smartcash/ui/setup/env_config/components/env_config_component.py
Deskripsi: Component untuk konfigurasi environment
"""

import os
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
import asyncio
from datetime import datetime

from smartcash.common.config.manager import ConfigManager
from smartcash.common.config.colab_manager import ColabConfigManager
from smartcash.common.constants.paths import COLAB_PATH, DRIVE_PATH
from smartcash.common.constants.core import APP_NAME, DEFAULT_CONFIG_DIR
from smartcash.common.io import load_config, save_config
from smartcash.common.logger import get_logger
from smartcash.common.utils import is_colab

from smartcash.ui.setup.env_config.handlers.setup_handlers import EnvConfigHandlers
from smartcash.ui.setup.env_config.handlers.auto_check_handler import AutoCheckHandler

# Import UI components
from smartcash.ui.components.progress import ProgressComponent
from smartcash.ui.components.logs import LogComponent
from smartcash.ui.utils.alert_utils import create_info_box
from smartcash.ui.utils.header_utils import create_header
from smartcash.ui.utils.constants import COLORS, ICONS
from smartcash.ui.helpers.ui_helpers import create_spacing

logger = get_logger(__name__)

def create_env_config_ui() -> Dict[str, Any]:
    """
    Buat komponen UI untuk konfigurasi environment
    
    Returns:
        Dictionary berisi komponen UI
    """
    # Header dengan komponen standar
    header = create_header(
        "⚙️ Environment Configuration", 
        "Setup environment dan konfigurasi SmartCash"
    )
    
    # Status panel menggunakan komponen alert standar
    status_panel = widgets.HTML(
        create_info_box(
            "Environment Status", 
            "Connect to Google Drive dan setup directory",
            style="info"
        ).value
    )
    
    # Action buttons in HBox
    drive_button = widgets.Button(
        description="Connect Drive",
        button_style="success",
        icon="cloud-upload",
        tooltip="Connect to Google Drive",
        layout=widgets.Layout(width='auto', margin='0 5px')
    )
    
    directory_button = widgets.Button(
        description="Setup Directory",
        button_style="warning",
        icon="folder",
        tooltip="Setup directory structure",
        layout=widgets.Layout(width='auto', margin='0 5px')
    )
    
    action_buttons = widgets.HBox(
        [drive_button, directory_button],
        layout=widgets.Layout(
            display='flex',
            flex_flow='row',
            justify_content='center',
            width='100%',
            margin='10px 0'
        )
    )
    
    # Progress component
    progress = ProgressComponent()
    
    # Log component
    log = LogComponent()
    
    # Container utama dengan semua komponen
    main = widgets.VBox(
        [
            header,
            status_panel,
            action_buttons,
            progress.component,
            log.component
        ],
        layout=widgets.Layout(
            width='100%',
            padding='10px'
        )
    )
    
    # Struktur final komponen UI
    ui_components = {
        'ui': main,
        'status_panel': status_panel,
        'drive_button': drive_button,
        'directory_button': directory_button,
        'progress': progress,
        'log': log
    }
    
    return ui_components

class EnvConfigComponent:
    """
    Component untuk konfigurasi environment
    """
    
    def __init__(self):
        """
        Inisialisasi component
        """
        self.logger = logger
        
        # Create UI first
        self.ui_components = create_env_config_ui()
        
        # Initialize managers
        self._init_managers()
        
        # Initialize handlers
        self.handlers = EnvConfigHandlers(self)
        self.auto_check = AutoCheckHandler(self)
        
        # Setup handlers
        self.handlers.setup_handlers()
        
        # Run auto check without drive mounting
        asyncio.create_task(self.auto_check.auto_check())
    
    def _init_managers(self):
        """
        Initialize configuration managers
        """
        # Determine base directory
        if is_colab():
            self.base_dir = Path("/content")
            self.config_dir = self.base_dir / "configs"
            
            # Copy configs from smartcash/configs if not exists
            if not self.config_dir.exists():
                source_configs = Path("/content/smartcash/configs")
                if source_configs.exists():
                    shutil.copytree(source_configs, self.config_dir)
        else:
            self.base_dir = Path.home() / "SmartCash"
            self.config_dir = self.base_dir / "configs"
        
        # Initialize managers
        self.config_manager = ConfigManager(
            base_dir=str(self.base_dir),
            config_file=str(self.config_dir / "base_config.yaml")
        )
        
        self.colab_manager = ColabConfigManager(
            base_dir=str(self.base_dir),
            config_file=str(self.config_dir / "base_config.yaml")
        )
    
    async def _handle_drive_connection(self):
        """
        Handle Google Drive connection
        """
        try:
            # Update progress
            self.ui_components['progress'].update(0.2, "Connecting to Google Drive...")
            
            # Connect to drive
            await self.colab_manager.connect_to_drive()
            
            # Update progress
            self.ui_components['progress'].update(0.5, "Setting up drive configs...")
            
            # Replace configs with drive configs
            drive_configs = Path("/content/drive/MyDrive/SmartCash/configs")
            if drive_configs.exists():
                # Backup current configs
                backup_dir = self.config_dir.parent / "configs_backup"
                if self.config_dir.exists():
                    shutil.move(self.config_dir, backup_dir)
                
                # Create symlink to drive configs
                os.symlink(drive_configs, self.config_dir)
            else:
                # Create drive configs directory
                drive_configs.mkdir(parents=True)
                
                # Copy local configs to drive
                if self.config_dir.exists():
                    shutil.copytree(self.config_dir, drive_configs, dirs_exist_ok=True)
            
            # Update progress
            self.ui_components['progress'].update(1.0, "Drive connection completed")
            
            # Update status
            self._update_status()
            
            self.ui_components['log'].info("Successfully connected to Google Drive")
            self.ui_components['log'].info(f"Drive path: {self.colab_manager.drive_base_path}")
            
        except Exception as e:
            self.ui_components['log'].error(f"Error connecting to Google Drive: {str(e)}")
            self.ui_components['progress'].update(0, "Drive connection failed")
    
    async def _handle_directory_setup(self):
        """
        Handle directory setup
        """
        try:
            # Update progress
            self.ui_components['progress'].update(0.2, "Creating directories...")
            
            # Create directory structure
            if is_colab():
                # Create base directories
                base_dirs = [
                    "/content/data",
                    "/content/models",
                    "/content/output",
                    "/content/logs",
                    "/content/exports"
                ]
                
                for dir_path in base_dirs:
                    path = Path(dir_path)
                    path.mkdir(exist_ok=True)
                    self.ui_components['log'].info(f"Created directory: {dir_path}")
                
                # Create symlinks to drive if connected
                if self.colab_manager.is_drive_connected():
                    self.ui_components['progress'].update(0.5, "Setting up drive symlinks...")
                    drive_base = Path("/content/drive/MyDrive/SmartCash")
                    drive_dirs = {
                        "/content/data": drive_base / "data",
                        "/content/models": drive_base / "models",
                        "/content/output": drive_base / "output",
                        "/content/logs": drive_base / "logs",
                        "/content/exports": drive_base / "exports"
                    }
                    
                    for local_path, drive_path in drive_dirs.items():
                        # Create drive directory if not exists
                        drive_path.mkdir(parents=True, exist_ok=True)
                        
                        # Remove existing symlink if exists
                        local_path = Path(local_path)
                        if local_path.is_symlink():
                            local_path.unlink()
                        
                        # Create new symlink
                        os.symlink(drive_path, local_path)
                        self.ui_components['log'].info(f"Created symlink: {local_path} -> {drive_path}")
            
            # Update progress
            self.ui_components['progress'].update(1.0, "Directory setup completed")
            
            # Update status
            self._update_status()
            
        except Exception as e:
            self.ui_components['log'].error(f"Error setting up directory: {str(e)}")
            self.ui_components['progress'].update(0, "Directory setup failed")
    
    def _update_status(self):
        """
        Update status UI
        """
        self.ui_components['log'].info("\nCurrent Status:")
        self.ui_components['log'].info(f"Environment: {'Colab' if is_colab() else 'Local'}")
        self.ui_components['log'].info(f"Base Directory: {self.config_manager.base_dir}")
        self.ui_components['log'].info(f"Config Directory: {self.config_dir}")
        if is_colab():
            self.ui_components['log'].info(f"Drive Connected: {self.colab_manager.is_drive_connected()}")
            self.ui_components['log'].info(f"Drive Path: {self.colab_manager.drive_base_path}")
    
    def display(self):
        """
        Display component
        """
        display(self.ui_components['ui'])
