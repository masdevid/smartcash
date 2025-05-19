"""
File: smartcash/ui/setup/env_config/handlers/auto_check_handler.py
Deskripsi: Handler untuk auto check environment
"""

import asyncio
from datetime import datetime

from smartcash.common.utils import is_colab
from smartcash.common.logger import get_logger

logger = get_logger(__name__)

class AutoCheckHandler:
    """
    Handler untuk auto check environment
    """
    
    def __init__(self, component):
        """
        Inisialisasi handler
        
        Args:
            component: EnvConfigComponent instance
        """
        self.component = component
        self.logger = logger
    
    async def auto_check(self):
        """
        Auto check environment saat startup
        """
        try:
            # Update progress
            self.component.ui_components['progress'].value = 0.2
            
            # Check environment
            with self.component.ui_components['log']:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Checking environment...")
                print(f"Environment: {'Colab' if is_colab() else 'Local'}")
                print(f"Base Directory: {self.component.config_manager.base_dir}")
                print(f"Config Directory: {self.component.config_dir}")
            
            # Update progress
            self.component.ui_components['progress'].value = 1.0
            
            # Update status
            self.component._update_status()
            
            with self.component.ui_components['log']:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Environment check completed")
            
        except Exception as e:
            with self.component.ui_components['log']:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Error checking environment: {str(e)}")
            self.component.ui_components['progress'].value = 0
