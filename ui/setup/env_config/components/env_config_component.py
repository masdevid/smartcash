"""
File: smartcash/ui/setup/env_config/components/env_config_component.py
Deskripsi: Component untuk konfigurasi environment
"""

from typing import Dict, Any
from IPython.display import display
import asyncio

from smartcash.common.logger import get_logger

from smartcash.ui.setup.env_config.components.ui_creator import create_env_config_ui
from smartcash.ui.setup.env_config.components.manager_setup import setup_managers
from smartcash.ui.setup.env_config.components.progress_setup import setup_progress
from smartcash.ui.setup.env_config.handlers.setup_handlers import setup_env_config_handlers
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
        
        # Create UI first
        self.ui_components = create_env_config_ui()
        
        # Initialize managers
        self.config_manager, self.colab_manager, self.base_dir, self.config_dir = setup_managers()
        
        # Setup progress tracking
        setup_progress(self.ui_components)
        
        # Initialize handlers
        setup_env_config_handlers(self.ui_components, self.colab_manager)
        self.auto_check = AutoCheckHandler(self)
        
        # Run auto check without drive mounting
        asyncio.create_task(self.auto_check.auto_check())
    
    def display(self):
        """
        Display component
        """
        display(self.ui_components['ui'])
