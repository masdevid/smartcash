
# =============================================================================
# File: smartcash/ui/setup/dependency/handlers/base_handler.py - FIXED
# Deskripsi: Base handler dengan logger_bridge dan error_handler integration
# =============================================================================

from typing import Dict, Any
from smartcash.ui.pretrained.utils import (
    with_error_handling,
    log_errors,
    get_logger,
    safe_ui_operation
)

# Initialize logger
logger = get_logger()

class BaseDependencyHandler:
    """Base handler dengan centralized error handling dan logger_bridge"""
    
    @with_error_handling(component="dependency", operation="BaseDependencyHandler")
    def __init__(self, ui_components: Dict[str, Any]):
        self.ui_components = ui_components
        self.logger_bridge = ui_components.get('logger_bridge')
        logger.info("Initialized BaseDependencyHandler")
    
    def log_info(self, message: str) -> None:
        """Log info message"""
        if self.logger_bridge and hasattr(self.logger_bridge, 'info'):
            self.logger_bridge.info(message)
    
    def log_success(self, message: str) -> None:
        """Log success message"""
        if self.logger_bridge and hasattr(self.logger_bridge, 'success'):
            self.logger_bridge.success(message)
    
    def log_warning(self, message: str) -> None:
        """Log warning message"""  
        if self.logger_bridge and hasattr(self.logger_bridge, 'warning'):
            self.logger_bridge.warning(message)
    
    def log_error(self, message: str) -> None:
        """Log error message"""
        if self.logger_bridge and hasattr(self.logger_bridge, 'error'):
            self.logger_bridge.error(message)
    
    @safe_ui_operation
    def disable_ui_during_operation(self) -> None:
        """Disable UI components safely"""
        buttons = ['install_btn', 'check_updates_btn', 'uninstall_btn', 'add_custom_btn']
        for btn_key in buttons:
            if btn_key in self.ui_components:
                self.ui_components[btn_key].disabled = True
        
        # Disable checkboxes
        for key, component in self.ui_components.items():
            if key.startswith('pkg_') and hasattr(component, 'disabled'):
                component.disabled = True
        
        if 'custom_packages_input' in self.ui_components:
            self.ui_components['custom_packages_input'].disabled = True
    
    @safe_ui_operation
    def enable_ui_after_operation(self) -> None:
        """Enable UI components safely"""
        buttons = ['install_btn', 'check_updates_btn', 'uninstall_btn', 'add_custom_btn']
        for btn_key in buttons:
            if btn_key in self.ui_components:
                self.ui_components[btn_key].disabled = False
        
        # Enable checkboxes (except required)
        for key, component in self.ui_components.items():
            if key.startswith('pkg_') and hasattr(component, 'disabled'):
                from .defaults import get_package_by_key
                package_key = key.replace('pkg_', '')
                package = get_package_by_key(package_key)
                component.disabled = package.get('required', False) if package else False
        
        if 'custom_packages_input' in self.ui_components:
            self.ui_components['custom_packages_input'].disabled = False
    
    @safe_ui_operation
    def clear_log_accordion(self) -> None:
        """Clear and open log accordion"""
        if 'log_output' in self.ui_components:
            self.ui_components['log_output'].clear_output()
        if 'log_accordion' in self.ui_components:
            self.ui_components['log_accordion'].selected_index = 0
    
    def setup_progress(self, total_steps: int, operation_name: str) -> None:
        """Setup progress trackers"""
        if 'main_progress' in self.ui_components:
            self.ui_components['main_progress'].value = 0
            self.ui_components['main_progress'].max = total_steps
            self.ui_components['main_progress'].description = f'{operation_name}:'
        
        if 'step_progress' in self.ui_components:
            self.ui_components['step_progress'].value = 0
            self.ui_components['step_progress'].description = 'Preparing...'
    
    def update_main_progress(self, current: int, status: str = None) -> None:
        """Update main progress"""
        if 'main_progress' in self.ui_components:
            self.ui_components['main_progress'].value = current
            if status:
                self.ui_components['main_progress'].description = status
    
    def update_step_progress(self, value: int, description: str = None) -> None:
        """Update step progress"""
        if 'step_progress' in self.ui_components:
            self.ui_components['step_progress'].value = value
            if description:
                self.ui_components['step_progress'].description = description
    
    def complete_progress(self, success: bool = True) -> None:
        """Complete progress bars"""
        if 'main_progress' in self.ui_components:
            self.ui_components['main_progress'].value = self.ui_components['main_progress'].max
        if 'step_progress' in self.ui_components:
            self.ui_components['step_progress'].value = 100
            self.ui_components['step_progress'].description = 'Completed ✅' if success else 'Failed ❌'
    
    def log_to_summary(self, message: str) -> None:
        """Log to summary panel"""
        if 'summary_output' in self.ui_components:
            # Update status panel content
            pass