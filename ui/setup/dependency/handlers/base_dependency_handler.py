"""
File: smartcash/ui/setup/dependency/handlers/base_dependency_handler.py
Description: Base handler for dependency module with centralized error handling
"""

from typing import Dict, Any, Optional
from smartcash.ui.handlers.base_handler import BaseHandler


class BaseDependencyHandler(BaseHandler):
    """Base handler for dependency module with centralized error handling."""
    
    def __init__(self, ui_components: Optional[Dict[str, Any]] = None, **kwargs):
        """Initialize base dependency handler with centralized error handling.
        
        Args:
            ui_components: Dictionary containing UI components
            **kwargs: Additional arguments passed to parent class
        """
        super().__init__(**kwargs)
        self.ui_components = ui_components or {}
        self.module_name = "dependency"
        
    def disable_ui_during_operation(self) -> None:
        """Disable UI components safely during operations."""
        try:
            # Use parent class method to disable buttons
            buttons = ['install_btn', 'check_updates_btn', 'uninstall_btn', 'add_custom_btn']
            self.set_buttons_state(self.ui_components, enabled=False, button_keys=buttons)
            
            # Disable checkboxes
            for key, component in self.ui_components.items():
                if key.startswith('pkg_') and hasattr(component, 'disabled'):
                    component.disabled = True
            
            if 'custom_packages_input' in self.ui_components:
                self.ui_components['custom_packages_input'].disabled = True
                
            self.logger.debug("UI components disabled for operation")
        except Exception as e:
            self.handle_error(e, "Failed to disable UI components", exc_info=True)
    
    def enable_ui_after_operation(self) -> None:
        """Enable UI components safely after operations."""
        try:
            # Use parent class method to enable buttons
            buttons = ['install_btn', 'check_updates_btn', 'uninstall_btn', 'add_custom_btn']
            self.set_buttons_state(self.ui_components, enabled=True, button_keys=buttons)
            
            # Enable checkboxes (except required)
            for key, component in self.ui_components.items():
                if key.startswith('pkg_') and hasattr(component, 'disabled'):
                    from .defaults import get_package_by_key
                    package_key = key.replace('pkg_', '')
                    package = get_package_by_key(package_key)
                    component.disabled = package.get('required', False) if package else False
            
            if 'custom_packages_input' in self.ui_components:
                self.ui_components['custom_packages_input'].disabled = False
                
            self.logger.debug("UI components enabled after operation")
        except Exception as e:
            self.handle_error(e, "Failed to enable UI components", exc_info=True)
    
    def clear_log_accordion(self) -> None:
        """Clear and open log accordion with centralized error handling."""
        try:
            # Use parent class method to clear UI outputs
            self.clear_ui_outputs(self.ui_components, ['log_output', 'log_accordion'])
            
            # Open log accordion
            if 'log_accordion' in self.ui_components:
                self.ui_components['log_accordion'].selected_index = 0
                
            self.logger.debug("Log accordion cleared and opened")
        except Exception as e:
            self.handle_error(e, "Failed to clear log accordion", exc_info=True)
    
    def setup_progress(self, total_steps: int, operation_name: str) -> None:
        """Setup dual progress trackers for overall and per-stage progress.
        
        Args:
            total_steps: Total number of steps in the operation
            operation_name: Name of the operation being performed
        """
        try:
            # Map UI components to expected progress tracker components
            progress_components = self.ui_components
            
            # If we have a progress_tracker component, use it
            if 'progress_tracker' in self.ui_components:
                # Use parent class method to initialize progress tracker
                # This will handle both primary and secondary progress bars
                from smartcash.ui.components.progress_tracker.progress_tracker import ProgressTracker
                tracker = self.ui_components['progress_tracker']
                
                if isinstance(tracker, ProgressTracker):
                    tracker.reset()
                    tracker.show()
                    tracker.set_title(operation_name)
                    tracker.set_total_steps(total_steps)
                    return
            
            # Otherwise use our custom progress bars
            if 'main_progress' in self.ui_components:
                self.ui_components['main_progress'].value = 0
                self.ui_components['main_progress'].max = total_steps
                self.ui_components['main_progress'].description = f'{operation_name}:'
            
            if 'step_progress' in self.ui_components:
                self.ui_components['step_progress'].value = 0
                self.ui_components['step_progress'].max = 100
                self.ui_components['step_progress'].description = 'Preparing...'
                
            self.logger.info(f"Starting operation: {operation_name} with {total_steps} steps")
        except Exception as e:
            self.handle_error(e, f"Failed to setup progress", exc_info=True)
    
    def update_main_progress(self, current: int, status: str = None) -> None:
        """Update main (overall) progress bar.
        
        Args:
            current: Current progress value
            status: Optional status message
        """
        try:
            # If we have a progress_tracker component, use parent class method
            if 'progress_tracker' in self.ui_components:
                from smartcash.ui.components.progress_tracker.progress_tracker import ProgressTracker
                tracker = self.ui_components['progress_tracker']
                
                if isinstance(tracker, ProgressTracker):
                    # Use parent class method to update primary progress
                    tracker.update_primary(current, status or '')
                    return
            
            # Otherwise use our custom progress bars
            if 'main_progress' in self.ui_components:
                self.ui_components['main_progress'].value = current
                if status:
                    self.ui_components['main_progress'].description = f'{status}'
                
                # Log progress update
                max_value = self.ui_components['main_progress'].max
                percentage = int((current / max_value) * 100) if max_value > 0 else 0
                self.logger.info(f"Main progress: {percentage}% ({current}/{max_value}) {status or ''}")
        except Exception as e:
            self.handle_error(e, f"Failed to update main progress", exc_info=True)
    
    def update_step_progress(self, value: int, description: str = None) -> None:
        """Update step (per-stage) progress bar.
        
        Args:
            value: Current progress value (0-100)
            description: Optional step description
        """
        try:
            # Ensure value is between 0-100
            value = max(0, min(100, value))
            
            # If we have a progress_tracker component, use parent class method
            if 'progress_tracker' in self.ui_components:
                from smartcash.ui.components.progress_tracker.progress_tracker import ProgressTracker
                tracker = self.ui_components['progress_tracker']
                
                if isinstance(tracker, ProgressTracker):
                    # Use parent class method to update step progress
                    tracker.update_step(value, description or '')
                    return
            
            # Otherwise use our custom progress bars
            if 'step_progress' in self.ui_components:
                self.ui_components['step_progress'].value = value
                if description:
                    self.ui_components['step_progress'].description = description
                    self.logger.info(f"Step progress: {value}% - {description}")
                else:
                    self.logger.debug(f"Step progress: {value}%")
        except Exception as e:
            self.handle_error(e, f"Failed to update step progress", exc_info=True)
    
    def complete_progress(self, success: bool = True) -> None:
        """Complete both progress bars and update status.
        
        Args:
            success: Whether the operation was successful
        """
        try:
            # If we have a progress_tracker component, use parent class method
            if 'progress_tracker' in self.ui_components:
                from smartcash.ui.components.progress_tracker.progress_tracker import ProgressTracker
                tracker = self.ui_components['progress_tracker']
                
                if isinstance(tracker, ProgressTracker):
                    # Use parent class method to complete progress
                    message = "Operation completed successfully" if success else "Operation completed with errors"
                    if success:
                        tracker.complete(message)
                    else:
                        tracker.error(message)
                    return
            
            # Otherwise use our custom progress bars
            # Complete main progress
            if 'main_progress' in self.ui_components:
                self.ui_components['main_progress'].value = self.ui_components['main_progress'].max
                
            # Complete step progress
            if 'step_progress' in self.ui_components:
                self.ui_components['step_progress'].value = 100
                status_text = 'Completed ✅' if success else 'Failed ❌'
                self.ui_components['step_progress'].description = status_text
                
            # Log completion status
            if success:
                self.logger.info("Operation completed successfully")
            else:
                self.logger.warning("Operation completed with errors")
                
        except Exception as e:
            self.handle_error(e, "Failed to complete progress", exc_info=True)
            
    def update_dual_progress(self, main_value: int, step_value: int, main_desc: str = None, step_desc: str = None) -> None:
        """Update both main and step progress bars in one call.
        
        Args:
            main_value: Current main progress value
            step_value: Current step progress value (0-100)
            main_desc: Optional main progress description
            step_desc: Optional step progress description
        """
        try:
            # If we have a progress_tracker component, use parent class method directly
            if 'progress_tracker' in self.ui_components:
                # Get max value for main progress
                main_max = 1  # Default fallback
                if 'main_progress' in self.ui_components and hasattr(self.ui_components['main_progress'], 'max'):
                    main_max = self.ui_components['main_progress'].max
                
                # Use parent class method to update dual progress
                super().update_dual_progress(
                    self.ui_components,
                    overall_value=main_value,
                    overall_max=main_max,
                    current_value=step_value,
                    current_max=100,  # Step progress is always percentage-based (0-100)
                    overall_message=main_desc,
                    current_message=step_desc
                )
                return
            
            # Otherwise use our custom implementation
            # Update main progress
            self.update_main_progress(main_value, main_desc)
            
            # Update step progress
            self.update_step_progress(step_value, step_desc)
            
        except Exception as e:
            self.handle_error(e, "Failed to update dual progress", exc_info=True)
    
    def log_to_summary(self, message: str) -> None:
        """Log message to summary panel with centralized error handling.
        
        Args:
            message: Message to log
        """
        try:
            # Use parent class method to update status panel
            self.update_status_panel(self.ui_components, message, status_type='info', title="Summary")
            
            # Also log to regular logger
            self.logger.info(f"Summary: {message}")
        except Exception as e:
            self.handle_error(e, "Failed to log to summary panel", exc_info=True)
