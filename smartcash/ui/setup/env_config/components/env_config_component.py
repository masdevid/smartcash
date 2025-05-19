"""
File: smartcash/ui/setup/env_config/components/env_config_component.py
Deskripsi: Component untuk konfigurasi environment
"""

import os
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

logger = get_logger(__name__)

class EnvConfigComponent:
    """
    Component untuk konfigurasi environment
    """
    
    def __init__(self):
        """
        Inisialisasi component
        """
        self.logger = logger
        
        # Determine base directory
        if is_colab():
            self.base_dir = Path("/content/SmartCash")
        else:
            self.base_dir = Path.home() / "SmartCash"
        
        # Initialize managers
        self.config_manager = ConfigManager(
            base_dir=str(self.base_dir),
            config_file=str(self.base_dir / "configs" / "base_config.yaml")
        )
        
        self.colab_manager = ColabConfigManager(
            base_dir=str(self.base_dir),
            config_file=str(self.base_dir / "configs" / "base_config.yaml")
        )
        
        # Initialize UI components
        self._init_ui_components()
        
        # Initialize handlers
        self.handlers = EnvConfigHandlers(self)
        self.auto_check = AutoCheckHandler(self)
        
        # Setup handlers
        self.handlers.setup_handlers()
        
        # Run auto check
        asyncio.create_task(self.auto_check.auto_check())
    
    def _init_ui_components(self):
        """
        Inisialisasi UI components
        """
        # Create sections
        self.drive_section = widgets.VBox([
            widgets.HTML("<h3>Google Drive Connection</h3>"),
            widgets.Button(description="Connect to Drive", button_style="primary")
        ])
        
        self.config_section = widgets.VBox([
            widgets.HTML("<h3>Configuration Sync</h3>"),
            widgets.Button(description="Sync Configurations", button_style="primary")
        ])
        
        self.progress_section = widgets.VBox([
            widgets.HTML("<h3>Progress</h3>"),
            widgets.FloatProgress(value=0.0, min=0, max=1.0)
        ])
        
        self.log_section = widgets.VBox([
            widgets.HTML("<h3>Logs</h3>"),
            widgets.Output()
        ])
        
        # Create main layout
        self.layout = widgets.VBox([
            self.drive_section,
            self.config_section,
            self.progress_section,
            self.log_section
        ])
    
    def _update_status(self):
        """
        Update status UI
        """
        with self.log_section.children[0]:
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Current Status:")
            print(f"Environment: {'Colab' if is_colab() else 'Local'}")
            print(f"Base Directory: {self.config_manager.base_dir}")
            print(f"Config File: {self.config_manager.config_file}")
            if is_colab():
                print(f"Drive Connected: {self.colab_manager.is_drive_connected()}")
                print(f"Drive Path: {self.colab_manager.drive_base_path}")
    
    def display(self):
        """
        Display component
        """
        display(self.layout)
