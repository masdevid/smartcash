"""
File: smartcash/ui/setup/env_config/handlers/auto_check_handler.py
Deskripsi: Handler untuk auto check environment
"""

from datetime import datetime
import logging
from smartcash.common.utils import is_colab

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
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def auto_check(self):
        """
        Auto check environment saat startup
        """
        try:
            # Update progress
            self.component.ui_components['progress_bar'].value = 0.2
            
            # Check environment
            with self.component.ui_components['log_output']:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Checking environment...")
                print(f"Environment: {'Colab' if is_colab() else 'Local'}")
                print(f"Base Directory: {self.component.config_manager.base_dir}")
                print(f"Config Directory: {self.component.config_dir}")
            
            # Update progress
            self.component.ui_components['progress_bar'].value = 1.0
            
            # Update status
            self.component._update_status("Environment check completed", "success")
            
            with self.component.ui_components['log_output']:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Environment check completed")
            
        except Exception as e:
            self.logger.error(f"Error during auto check: {str(e)}")
            with self.component.ui_components['log_output']:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Error during auto check: {str(e)}")
            self.component._update_status(f"Error during auto check: {str(e)}", "error")
