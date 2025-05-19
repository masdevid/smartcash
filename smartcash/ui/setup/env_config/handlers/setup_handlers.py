"""
File: smartcash/ui/setup/env_config/handlers/setup_handlers.py
Deskripsi: Setup handler untuk konfigurasi environment
"""

from typing import Dict, Any, Optional
import asyncio
from pathlib import Path
from datetime import datetime

from smartcash.common.utils import is_colab
from smartcash.common.logger import get_logger

logger = get_logger(__name__)

class EnvConfigHandlers:
    """
    Handler untuk konfigurasi environment
    """
    
    def __init__(self, component):
        """
        Inisialisasi handler
        
        Args:
            component: EnvConfigComponent instance
        """
        self.component = component
        self.logger = logger
    
    async def on_connect_drive(self, b):
        """
        Handler untuk tombol connect drive
        """
        try:
            # Update progress
            self.component.progress_section.children[1].value = 0.2
            
            # Connect to drive
            await self.component.colab_manager.connect_to_drive()
            
            # Update progress
            self.component.progress_section.children[1].value = 1.0
            
            # Update status
            self.component._update_status()
            
            with self.component.log_section.children[0]:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Successfully connected to Google Drive")
                print(f"Drive path: {self.component.colab_manager.drive_base_path}")
            
        except Exception as e:
            with self.component.log_section.children[0]:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Error connecting to Google Drive: {str(e)}")
            self.component.progress_section.children[1].value = 0
    
    async def on_sync_configs(self, b):
        """
        Handler untuk tombol sync configs
        """
        try:
            # Update progress
            self.component.progress_section.children[1].value = 0.2
            
            # Get available configs
            configs = await self.component.colab_manager.get_available_configs()
            
            with self.component.log_section.children[0]:
                print(f"Found {len(configs)} configuration files")
            
            # Sync each config
            for i, config in enumerate(configs):
                progress = 0.2 + (0.6 * (i / len(configs)))
                self.component.progress_section.children[1].value = progress
                
                with self.component.log_section.children[0]:
                    print(f"Syncing {config}...")
                
                await self.component.colab_manager.sync_with_drive(config)
            
            # Update progress
            self.component.progress_section.children[1].value = 1.0
            
            # Update status
            self.component._update_status()
            
            with self.component.log_section.children[0]:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Configuration sync completed successfully")
            
        except Exception as e:
            with self.component.log_section.children[0]:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Error syncing configurations: {str(e)}")
            self.component.progress_section.children[1].value = 0
    
    def setup_handlers(self):
        """
        Setup semua handler
        """
        # Setup drive connection handler
        self.component.drive_section.children[1].on_click(
            lambda b: asyncio.create_task(self.on_connect_drive(b))
        )
        
        # Setup sync configs handler
        self.component.config_section.children[1].on_click(
            lambda b: asyncio.create_task(self.on_sync_configs(b))
        )
