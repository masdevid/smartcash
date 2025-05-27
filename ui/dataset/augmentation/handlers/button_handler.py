"""
File: smartcash/ui/dataset/augmentation/handlers/button_handler.py
Deskripsi: Button handler dengan unified logging dan service integration
"""

from typing import Dict, Any
from smartcash.ui.utils.button_state_manager import get_button_state_manager
from smartcash.ui.dataset.augmentation.utils.ui_logger_utils import log_to_ui

class ButtonHandler:
    """Button handler dengan service integration"""
    
    def __init__(self, ui_components: Dict[str, Any]):
        self.ui_components = ui_components
        self.manager = self._get_or_create_manager()
    
    def _get_or_create_manager(self):
        """Get existing or create button state manager"""
        existing_manager = self.ui_components.get('button_state_manager')
        if existing_manager and hasattr(existing_manager, 'can_start_operation'):
            return existing_manager
        
        manager = get_button_state_manager(self.ui_components)
        self.ui_components['button_state_manager'] = manager
        return manager
    
    def execute_operation(self, operation_name: str, operation_func):
        """Execute operation dengan service integration"""
        try:
            can_start, message = self.manager.can_start_operation(operation_name)
            if not can_start:
                log_to_ui(self.ui_components, message, 'warning', "⚠️ ")
                return
            
            self._reset_log_area_only()
            
            with self.manager.operation_context(operation_name):
                operation_func()
                log_to_ui(self.ui_components, f"{operation_name.title()} berhasil", 'success', "✅ ")
                
        except Exception as e:
            error_msg = f"{operation_name.title()} gagal: {str(e)}"
            log_to_ui(self.ui_components, error_msg, 'error', "❌ ")
    
    def execute_config(self, config_name: str, config_func):
        """Execute config operation"""
        try:
            with self.manager.config_context(config_name):
                config_func()
                log_to_ui(self.ui_components, f"{config_name.title()} berhasil", 'success', "✅ ")
                
        except Exception as e:
            error_msg = f"{config_name.title()} gagal: {str(e)}"
            log_to_ui(self.ui_components, error_msg, 'error', "❌ ")
    
    def _reset_log_area_only(self):
        """Reset hanya log output area"""
        tracker = self.ui_components.get('tracker')
        if tracker and hasattr(tracker, 'reset'):
            tracker.reset()
        elif 'reset_all' in self.ui_components:
            self.ui_components['reset_all']()

# Factory function
create_button_handler = lambda ui_components: ButtonHandler(ui_components)