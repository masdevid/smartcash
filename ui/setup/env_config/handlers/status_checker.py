"""
File: smartcash/ui/setup/env_config/handlers/status_checker.py
Handler for checking initial environment status and updating UI components
"""

from typing import Dict, Any, Optional
from smartcash.ui.setup.env_config.utils.env_detector import detect_environment_info
from smartcash.ui.setup.env_config.utils.ui_updater import update_status_panel

class StatusChecker:
    """🔍 Handler for environment status checking and UI updates"""
    
    def __init__(self, logger=None):
        """Initialize StatusChecker with optional logger"""
        self.logger = logger or self._create_dummy_logger()
        self._last_status = {}
    
    def check_initial_status(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check and update the initial environment status
        
        Args:
            ui_components: Dictionary containing UI components to update
            
        Returns:
            Dictionary containing environment info and status
        """
        self.logger.debug("Starting initial environment status check...")
        
        try:
            # 1. Detect environment information
            self.logger.debug("Detecting environment information...")
            env_info = detect_environment_info()
            
            # 2. Generate status message and type
            status_msg = self._generate_status_message(env_info)
            status_type = self._determine_status_type(env_info)
            
            # 3. Update UI components
            self._update_ui_components(ui_components, status_msg, status_type)
            
            # 4. Log and store status
            self.logger.info(f"✅ Environment status: {status_msg}")
            self._last_status = {
                'env_info': env_info,
                'status_message': status_msg,
                'status_type': status_type,
                'success': True
            }
            
            return self._last_status
            
        except Exception as e:
            error_msg = f"❌ Status check failed: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            
            # Try to update UI even in case of error
            try:
                if 'status_panel' in ui_components:
                    update_status_panel(
                        ui_components['status_panel'], 
                        error_msg, 
                        "error"
                    )
            except Exception as ui_error:
                self.logger.error(f"Failed to update status panel: {str(ui_error)}")
            
            self._last_status = {
                'env_info': {},
                'status_message': error_msg,
                'status_type': 'error',
                'success': False,
                'error': str(e)
            }
            
            return self._last_status
    
    def _update_ui_components(self, 
                           ui_components: Dict[str, Any], 
                           status_msg: str, 
                           status_type: str) -> None:
        """
        Update all relevant UI components with the current status
        
        Args:
            ui_components: Dictionary of UI components
            status_msg: Status message to display
            status_type: Type of status (success, warning, error, info)
        """
        try:
            # Update status panel if available
            if 'status_panel' in ui_components:
                update_status_panel(
                    ui_components['status_panel'], 
                    status_msg, 
                    status_type
                )
                self.logger.debug("Updated status panel with current status")
                
            # Update setup summary if available
            if 'setup_summary' in ui_components:
                try:
                    from smartcash.ui.setup.env_config.components.setup_summary import update_setup_summary
                    update_setup_summary(ui_components['setup_summary'], status_msg, status_type)
                    self.logger.debug("Updated setup summary with current status")
                except Exception as e:
                    self.logger.warning(f"Failed to update setup summary: {str(e)}")
                    
        except Exception as e:
            self.logger.error(f"Error updating UI components: {str(e)}", exc_info=True)
            raise
    
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