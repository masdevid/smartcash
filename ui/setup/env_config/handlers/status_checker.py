"""
File: smartcash/ui/setup/env_config/handlers/status_checker.py
Deskripsi: Handler untuk checking initial status environment
"""

from typing import Dict, Any
from smartcash.ui.setup.env_config.utils.env_detector import detect_environment_info
from smartcash.ui.setup.env_config.utils.ui_updater import update_status_panel

class StatusChecker:
    """🔍 Handler untuk status checking environment"""
    
    def __init__(self, logger=None):
        self.logger = logger or self._create_dummy_logger()
    
    def check_initial_status(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """🔍 Check initial environment status"""
        try:
            # Detect environment
            env_info = detect_environment_info()
            
            # Generate status message
            status_msg = self._generate_status_message(env_info)
            status_type = self._determine_status_type(env_info)
            
            # Update UI
            update_status_panel(ui_components['status_panel'], status_msg, status_type)
            
            # Log status
            self.logger.info(f"🔍 Environment check: {status_msg}")
            
            return {
                'env_info': env_info,
                'status_message': status_msg,
                'status_type': status_type
            }
            
        except Exception as e:
            error_msg = f"Status check failed: {str(e)}"
            update_status_panel(ui_components['status_panel'], error_msg, "warning")
            self.logger.warning(f"⚠️ {error_msg}")
            
            return {
                'env_info': {},
                'status_message': error_msg,
                'status_type': 'warning'
            }
    
    def _generate_status_message(self, env_info: Dict[str, Any]) -> str:
        """📝 Generate status message based on environment"""
        if env_info.get('is_colab'):
            if env_info.get('drive_mounted'):
                return "Ready - Colab dengan Drive mounted"
            else:
                return "Ready - Colab environment (Drive belum mounted)"
        else:
            return "Ready - Local environment"
    
    def _determine_status_type(self, env_info: Dict[str, Any]) -> str:
        """🎯 Determine status type for styling"""
        if env_info.get('is_colab') and env_info.get('drive_mounted'):
            return "success"
        elif env_info.get('is_colab'):
            return "info"
        else:
            return "warning"
    
    def _create_dummy_logger(self):
        """📝 Create dummy logger fallback"""
        class DummyLogger:
            def info(self, msg): print(f"ℹ️ {msg}")
            def warning(self, msg): print(f"⚠️ {msg}")
            def error(self, msg): print(f"❌ {msg}")
            def success(self, msg): print(f"✅ {msg}")
        return DummyLogger()