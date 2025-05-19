"""
File: smartcash/ui/setup/env_config/handlers/setup_handlers.py
Deskripsi: Setup handler untuk konfigurasi environment
"""

import asyncio
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
    
    def setup_handlers(self):
        """
        Setup semua handler
        """
        # Setup drive connection handler
        self.component.ui_components['drive_button'].on_click(
            lambda b: asyncio.create_task(self.component._handle_drive_connection())
        )
        
        # Setup directory handler
        self.component.ui_components['directory_button'].on_click(
            lambda b: asyncio.create_task(self.component._handle_directory_setup())
        )
