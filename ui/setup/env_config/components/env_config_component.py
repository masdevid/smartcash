"""
File: smartcash/ui/setup/env_config/components/env_config_component.py
Deskripsi: Komponen UI untuk konfigurasi environment
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

logger = get_logger(__name__)

class EnvConfigComponent:
    """
    Komponen UI untuk konfigurasi environment
    """
    
    def __init__(self):
        """
        Inisialisasi komponen UI
        """
        self.config_manager = ConfigManager.get_instance()
        self.colab_manager = ColabConfigManager.get_instance()
        self._setup_ui()
    
    def _setup_ui(self):
        """
        Setup UI components
        """
        # Create main container
        self.main_container = widgets.VBox()
        
        # Create header
        self.header = widgets.HTML(
            value="<h2>Environment Configuration</h2>",
            layout=widgets.Layout(margin='0 0 20px 0')
        )
        
        # Create status section
        self.status_section = widgets.VBox([
            widgets.HTML("<h3>Status</h3>"),
            widgets.HTML(id='status_text')
        ])
        
        # Create Google Drive section
        self.drive_section = widgets.VBox([
            widgets.HTML("<h3>Google Drive</h3>"),
            widgets.Button(
                description='Connect to Google Drive',
                button_style='primary',
                layout=widgets.Layout(width='auto', margin='10px 0')
            ),
            widgets.HTML(id='drive_status')
        ])
        
        # Create configuration section
        self.config_section = widgets.VBox([
            widgets.HTML("<h3>Configuration</h3>"),
            widgets.Button(
                description='Sync Configurations',
                button_style='info',
                layout=widgets.Layout(width='auto', margin='10px 0')
            ),
            widgets.HTML(id='config_status')
        ])
        
        # Create progress section
        self.progress_section = widgets.VBox([
            widgets.HTML("<h3>Progress</h3>"),
            widgets.FloatProgress(
                value=0,
                min=0,
                max=1,
                description='Progress:',
                bar_style='info',
                style={'bar_color': '#007bff'},
                layout=widgets.Layout(width='100%')
            ),
            widgets.HTML(id='progress_text')
        ])
        
        # Create log section
        self.log_section = widgets.VBox([
            widgets.HTML("<h3>Logs</h3>"),
            widgets.Output(layout=widgets.Layout(height='200px', overflow_y='auto'))
        ])
        
        # Add all sections to main container
        self.main_container.children = [
            self.header,
            self.status_section,
            self.drive_section,
            self.config_section,
            self.progress_section,
            self.log_section
        ]
        
        # Setup event handlers
        self.drive_section.children[1].on_click(self._on_connect_drive)
        self.config_section.children[1].on_click(self._on_sync_configs)
        
        # Initialize status
        self._update_status()
    
    def _update_status(self):
        """
        Update status display
        """
        is_colab = self.colab_manager.is_colab_environment()
        drive_connected = self.colab_manager.is_drive_connected()
        
        status_html = f"""
        <div style='padding: 10px; background-color: #f8f9fa; border-radius: 5px;'>
            <p><strong>Environment:</strong> {'Google Colab' if is_colab else 'Local'}</p>
            <p><strong>Google Drive:</strong> {'Connected' if drive_connected else 'Not Connected'}</p>
            <p><strong>Base Directory:</strong> {self.config_manager.base_dir}</p>
            <p><strong>Config Directory:</strong> {self.config_manager.base_dir / DEFAULT_CONFIG_DIR}</p>
        </div>
        """
        self.status_section.children[1].value = status_html
    
    async def _on_connect_drive(self, b):
        """
        Handle Google Drive connection
        """
        with self.log_section.children[0]:
            clear_output()
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Connecting to Google Drive...")
        
        try:
            # Update progress
            self.progress_section.children[1].value = 0.2
            
            # Connect to drive
            await self.colab_manager.connect_to_drive()
            
            # Update progress
            self.progress_section.children[1].value = 1.0
            
            # Update status
            self._update_status()
            
            with self.log_section.children[0]:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Successfully connected to Google Drive")
                print(f"Drive path: {self.colab_manager.drive_base_path}")
            
        except Exception as e:
            with self.log_section.children[0]:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Error connecting to Google Drive: {str(e)}")
            self.progress_section.children[1].value = 0
    
    async def _on_sync_configs(self, b):
        """
        Handle configuration synchronization
        """
        with self.log_section.children[0]:
            clear_output()
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting configuration sync...")
        
        try:
            # Update progress
            self.progress_section.children[1].value = 0.2
            
            # Get available configs
            configs = await self.colab_manager.get_available_configs()
            
            with self.log_section.children[0]:
                print(f"Found {len(configs)} configuration files")
            
            # Sync each config
            for i, config in enumerate(configs):
                progress = 0.2 + (0.6 * (i / len(configs)))
                self.progress_section.children[1].value = progress
                
                with self.log_section.children[0]:
                    print(f"Syncing {config}...")
                
                await self.colab_manager.sync_with_drive(config)
            
            # Update progress
            self.progress_section.children[1].value = 1.0
            
            # Update status
            self._update_status()
            
            with self.log_section.children[0]:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Configuration sync completed successfully")
            
        except Exception as e:
            with self.log_section.children[0]:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Error syncing configurations: {str(e)}")
            self.progress_section.children[1].value = 0
    
    def display(self):
        """
        Display the UI component
        """
        display(self.main_container)
