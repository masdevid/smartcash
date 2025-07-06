"""
File: smartcash/ui/handlers/base_handler.py

Base handler that centralizes common functionality for UI handlers:
- Centralized logging
- Error handling using error_handler.py
"""

from typing import Dict, Any, List, Callable, Literal, TypeVar
import logging
from abc import ABC
from datetime import datetime

from smartcash.ui.logger import get_module_logger
from smartcash.ui.core.errors.handlers import (
    handle_ui_errors, 
    safe_execute,
    create_error_response,
    ErrorContext
)
from smartcash.ui.core.decorators.ui_decorators import (
    safe_ui_operation,
    safe_widget_operation,
    safe_progress_operation,
    safe_component_access
)
# Note: We don't import status_panel, dialog, or progress_tracker modules here to avoid circular imports
# They're imported locally in methods that need them

# Progress tracker level type
ProgressLevel = Literal['single', 'dual', 'triple']

# Type variable for generic return types
T = TypeVar('T')

class BaseHandler(ABC):
    """⚠️ DEPRECATED: This class is deprecated and will be removed in a future version.
    Please use smartcash.ui.core.handlers.BaseHandler instead.
    
    Base handler with centralized logging, error handling, and UI utilities.
    
    This is a compatibility layer that forwards all calls to the new implementation in core.
    """
    
    @handle_ui_errors(error_component_title="Handler Initialization Error", log_error=True)
    def __init__(self, module_name: str, parent_module: str = None):
        """Initialize the base handler.
        
        Args:
            module_name: Name of the module
            parent_module: Optional parent module name
        """
        warnings.warn(
            "The 'BaseHandler' class is deprecated and will be removed in a future version. "
            "Please use 'smartcash.ui.core.handlers.BaseHandler' instead.",
            DeprecationWarning,
            stacklevel=2
        )
        
        # Import the new implementation
        from smartcash.ui.core.handlers import BaseHandler as CoreBaseHandler
        
        # Initialize the core handler
        self._core_handler = CoreBaseHandler(module_name, parent_module)
        
        # Set up forwarding for attribute access
        self.module_name = module_name
        self.parent_module = parent_module
        self.full_module_name = f"{parent_module}.{module_name}" if parent_module else module_name
        
        # Get logger from core handler
        self.logger = self._core_handler.logger
        
        # Log deprecation warning
        self.logger.warning(
            "This handler is deprecated. Please update to use smartcash.ui.core.handlers.BaseHandler"
        )
    
    def setup_log_redirection(self, ui_components: Dict[str, Any]) -> None:
        """Set up log redirection to log_accordion component.
        
        Args:
            ui_components: Dictionary of UI components containing 'log_output' or 'log_accordion'
        """
        if not ui_components:
            return
            
        # Store UI components for later use
        self._log_ui_components = ui_components
        
        # Remove existing handler if any
        if self._log_handler:
            self.logger.removeHandler(self._log_handler)
            self._log_handler = None
            
        # Create custom log handler
        class LogAccordionHandler(logging.Handler):
            def __init__(self, ui_components):
                super().__init__()
                self.ui_components = ui_components
                
            def emit(self, record):
                try:
                    # Get log level and message
                    level_name = record.levelname.lower()
                    message = self.format(record)
                    
                    # Map logging levels to log_accordion levels
                    level_map = {
                        'debug': 'debug',
                        'info': 'info',
                        'warning': 'warning',
                        'error': 'error',
                        'critical': 'critical'
                    }
                    level = level_map.get(level_name, 'info')
                    
                    # Update log_accordion
                    self._update_log_accordion(self.ui_components, message, level)
                except Exception:
                    # Fallback to sys.stderr
                    self.handleError(record)
                    
        # Create and add handler
        self._log_handler = LogAccordionHandler(ui_components)
        self._log_handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(self._log_handler)
        
    def _update_log_accordion(self, ui_components: Dict[str, Any], message: str, level: str = 'info') -> None:
        """Update log accordion with message.
        
        Args:
            ui_components: Dictionary of UI components
            message: Log message
            level: Log level (debug, info, warning, error, critical)
        """
        try:
            # Try log_output first (main target)
            if 'log_output' in ui_components and hasattr(ui_components['log_output'], 'append_log'):
                ui_components['log_output'].append_log(message=message, level=level)
                return
                
            # Try log_accordion as fallback
            if 'log_accordion' in ui_components and hasattr(ui_components['log_accordion'], 'append_log'):
                ui_components['log_accordion'].append_log(message=message, level=level)
                return
                
            # Try using update_log function from log_accordion module
            try:
                from smartcash.ui.components.log_accordion import update_log
                update_log(
                    ui_components=ui_components,
                    message=message,
                    level=level,
                    expand=False
                )
            except ImportError:
                # Fallback to status panel
                self._update_status_panel(ui_components, message, level)
                
        except Exception as e:
            # Last resort: print to stderr
            print(f"Error updating log accordion: {str(e)}\nOriginal message: {message}", file=sys.stderr)
    
    # ===== Error Handling Utilities =====
    
    def handle_error(self, error: Exception, context: str = None, ui_components: Dict[str, Any] = None,
                    error_title: str = "Error", include_traceback: bool = True) -> Dict[str, Any]:
        """Handle errors with consistent logging and UI feedback.
        
        Args:
            error: The exception that occurred
            context: Context information about where the error occurred
            ui_components: Optional UI components to update with error
            error_title: Title for the error component
            include_traceback: Whether to include traceback in the error display
            
        Returns:
            Error response dictionary
        """
        error_context = ErrorContext(
            component=self.__class__.__name__,
            operation=context or "execution"
        )
        
        # Log the error
        self.logger.error(
            f"Error in {error_context.component}.{error_context.operation}: {str(error)}",
            exc_info=True
        )
        
        # Create error response
        error_response = create_error_response(
            error_message=str(error),
            error=error,
            title=error_title,
            include_traceback=include_traceback
        )
        
        # Update UI components if provided
        if ui_components:
            self._update_status_panel(ui_components, str(error), "error")
            
            # Update error display area if available
            if 'error_area' in ui_components and hasattr(ui_components['error_area'], 'update'):
                ui_components['error_area'].update(error_response)
        
        return error_response
    
    # ===== Confirmation Dialog Utilities =====
    
    def show_confirmation_dialog(self, ui_components: Dict[str, Any], message: str, 
                              callback: Callable = None, timeout_seconds: int = 120,
                              title: str = "Confirmation Required", 
                              confirm_text: str = "Confirm", 
                              cancel_text: str = "Cancel",
                              danger_mode: bool = False) -> None:
        """Show a confirmation dialog with timeout.
        
        Args:
            ui_components: Dictionary of UI components
            message: Confirmation message to display
            callback: Optional callback to execute when confirmed
            timeout_seconds: Timeout in seconds (default: 120)
            title: Title for the confirmation dialog
            confirm_text: Text for the confirm button
            cancel_text: Text for the cancel button
            danger_mode: Whether to use danger styling (red) for the dialog
        """
        try:
            # Import dialog module here to avoid circular imports
            from smartcash.ui.components.dialog.confirmation_dialog import show_confirmation_dialog, create_confirmation_area
            
            # Ensure confirmation area exists
            if 'confirmation_area' not in ui_components:
                self.logger.debug("Creating confirmation area")
                create_confirmation_area(ui_components)
            
            # Set confirmation state for tracking
            self._confirmation_state = {
                'pending': True,
                'message': message,
                'timestamp': datetime.now(),
                'timeout_seconds': timeout_seconds,
                'callback': callback
            }
            
            # Define callbacks for confirm and cancel actions
            def on_confirm():
                self._confirmation_state['pending'] = False
                if callback:
                    try:
                        callback()
                    except Exception as e:
                        self.logger.error(f"Error in confirmation callback: {str(e)}", exc_info=True)
            
            def on_cancel():
                self._confirmation_state['pending'] = False
                self.logger.debug("Confirmation dialog canceled")
            
            # Show the confirmation dialog using the dialog component
            show_confirmation_dialog(
                ui_components=ui_components,
                title=title,
                message=message,
                on_confirm=on_confirm,
                on_cancel=on_cancel,
                confirm_text=confirm_text,
                cancel_text=cancel_text,
                danger_mode=danger_mode
            )
            
            self.logger.debug(f"Displayed confirmation dialog: {message}")
        except Exception as e:
            self.logger.error(f"Error showing confirmation dialog: {str(e)}", exc_info=True)
    
    def handle_confirmation_result(self, ui_components: Dict[str, Any], confirmed: bool = False) -> None:
        """Handle confirmation dialog result.
        
        Args:
            ui_components: Dictionary of UI components
            confirmed: Whether the action was confirmed
        """
        if not self._confirmation_state.get('pending', False):
            self.logger.warning("No pending confirmation to handle")
            return
            
        try:
            # Import dialog module here to avoid circular imports
            from smartcash.ui.components.dialog.confirmation_dialog import clear_dialog_area
            
            # Clear confirmation area
            clear_dialog_area(ui_components)
                
            # Execute callback if confirmed
            if confirmed and self._confirmation_state.get('callback'):
                self._confirmation_state['callback']()
                
            # Reset confirmation state
            self._confirmation_state = {'pending': False}
            
            self.logger.debug(f"Handled confirmation result: {confirmed}")
        except Exception as e:
            self.logger.error(f"Error handling confirmation result: {str(e)}", exc_info=True)
    
    def is_confirmation_pending(self, ui_components: Dict[str, Any] = None) -> bool:
        """Check if a confirmation dialog is pending.
        
        Args:
            ui_components: Optional dictionary of UI components to check dialog visibility
            
        Returns:
            bool: True if a confirmation is pending, False otherwise
        """
        # First check internal state
        if not self._confirmation_state.get('pending', False):
            return False
            
        # Check for timeout
        if 'timestamp' in self._confirmation_state and 'timeout_seconds' in self._confirmation_state:
            elapsed = (datetime.now() - self._confirmation_state['timestamp']).total_seconds()
            if elapsed > self._confirmation_state['timeout_seconds']:
                self._confirmation_state['pending'] = False
                return False
        
        # If ui_components provided, check dialog visibility
        if ui_components:
            try:
                from smartcash.ui.components.dialog.confirmation_dialog import is_dialog_visible
                if not is_dialog_visible(ui_components):
                    self._confirmation_state['pending'] = False
                    return False
            except Exception as e:
                self.logger.debug(f"Error checking dialog visibility: {str(e)}")
                
        return True
    
    # ===== Button State Management =====
    
    def set_buttons_state(self, ui_components: Dict[str, Any], enabled: bool = True, 
                        button_keys: List[str] = None) -> None:
        """Enable or disable buttons in the UI.
        
        Args:
            ui_components: Dictionary of UI components
            enabled: Whether buttons should be enabled
            button_keys: Optional list of specific button keys to update
        """
        try:
            # Find all button components
            buttons = {}
            if button_keys:
                # Use specified button keys
                buttons = {k: ui_components.get(k) for k in button_keys if k in ui_components}
            else:
                # Find all button-like components
                for key, component in ui_components.items():
                    if (key.endswith('_button') or key.endswith('_btn') or 
                        (hasattr(component, 'description') and hasattr(component, 'disabled'))):
                        buttons[key] = component
            
            # Update button states
            for key, button in buttons.items():
                if button and hasattr(button, 'disabled'):
                    button.disabled = not enabled
                    
            self.logger.debug(f"Set {len(buttons)} buttons to {'enabled' if enabled else 'disabled'}")
        except Exception as e:
            self.logger.error(f"Error setting button states: {str(e)}", exc_info=True)
    
    def disable_all_buttons(self, ui_components: Dict[str, Any], button_keys: List[str] = None) -> None:
        """Disable all buttons in the UI.
        
        Args:
            ui_components: Dictionary of UI components
            button_keys: Optional list of specific button keys to disable
        """
        self.set_buttons_state(ui_components, False, button_keys)
    
    def enable_all_buttons(self, ui_components: Dict[str, Any], button_keys: List[str] = None) -> None:
        """Enable all buttons in the UI.
        
        Args:
            ui_components: Dictionary of UI components
            button_keys: Optional list of specific button keys to enable
        """
        self.set_buttons_state(ui_components, True, button_keys)
    
    # ===== Status Panel Management =====
    
    def update_status_panel(self, ui_components: Dict[str, Any], message: str, 
                          status_type: str = 'info', title: str = "Status Update") -> None:
        """Update status panel with proper error handling.
        
        Args:
            ui_components: Dictionary of UI components
            message: Status message to display
            status_type: Status type (info, success, warning, error)
            title: Title for the status update
        """
        self._update_status_panel(ui_components, message, status_type, title)
    
    @safe_ui_operation(operation_name="update_status_panel", log_level="error")
    def _update_status_panel(self, ui_components: Dict[str, Any], message: str, 
                           status_type: str = 'info', title: str = "Status Update") -> None:
        """Update status panel with safe fallback.
        
        Args:
            ui_components: Dictionary of UI components
            message: Status message to display
            status_type: Status type (info, success, warning, error)
            title: Title for the status update
        """
        # Import status panel module here to avoid circular imports
        from smartcash.ui.components.status_panel import update_status_panel
        
        if 'status_panel' in ui_components:
            # Use the dedicated update_status_panel function if available
            update_status_panel(ui_components['status_panel'], message, status_type)
        elif 'logger' in ui_components:
            # Fallback to logger if available
            log_method = getattr(ui_components['logger'], status_type, ui_components['logger'].info)
            log_method(f"Status: {message}")
            
        # Log to handler logger as well
        log_method = getattr(self.logger, status_type, self.logger.info)
        log_method(f"Status update: {message}")
    
    # ===== UI Component Management =====
    
    @safe_widget_operation(widget_key_arg=0)
    def clear_ui_outputs(self, ui_components: Dict[str, Any], output_keys: List[str] = None) -> None:
        """Clear UI outputs with safe widget access.
        
        Args:
            ui_components: Dictionary of UI components
            output_keys: Optional list of specific output keys to clear
        """
        default_output_keys = ['log_output', 'status_panel', 'confirmation_area', 'error_area', 'output', 'log_accordion', 'progress_tracker']
        keys_to_clear = output_keys or default_output_keys
        
        for key in keys_to_clear:
            widget = ui_components.get(key)
            if not widget:
                continue
                
            self._clear_single_ui_component(key, widget, ui_components)
    
    @safe_component_access(component_type="UI component")
    def _clear_single_ui_component(self, key: str, widget: Any, ui_components: Dict[str, Any]) -> None:
        """Clear a single UI component safely.
        
        Args:
            key: Component key in UI components
            widget: The widget to clear
            ui_components: Full UI components dictionary (needed for some components)
        """
        # Handle log_accordion component
        if key == 'log_output' or key == 'log_accordion':
            if hasattr(widget, 'clear_logs'):
                widget.clear_logs()
                return
        
        # Handle dialog components
        if key == 'confirmation_area' or key == 'error_area':
            self._clear_dialog_area(ui_components)
            return
        
        # Handle status_panel component
        if key == 'status_panel':
            self._clear_status_panel(widget)
            return
        
        # Handle progress_tracker component
        if key == 'progress_tracker':
            self._clear_progress_tracker(widget)
            return
        
        # Default fallback to clear_output for standard widgets
        if hasattr(widget, 'clear_output'):
            widget.clear_output(wait=True)
        elif hasattr(widget, 'value') and isinstance(widget.value, str):
            widget.value = ""
    
    @safe_component_access(component_type="dialog area")
    def _clear_dialog_area(self, ui_components: Dict[str, Any]) -> None:
        """Clear dialog area safely.
        
        Args:
            ui_components: Dictionary of UI components
        """
        # Import locally to avoid circular imports
        from smartcash.ui.components.dialog.confirmation_dialog import clear_dialog_area
        clear_dialog_area(ui_components)
    
    @safe_component_access(component_type="status panel")
    def _clear_status_panel(self, status_panel: Any) -> None:
        """Clear status panel safely.
        
        Args:
            status_panel: Status panel widget
        """
        # Import locally to avoid circular imports
        from smartcash.ui.components.status_panel import update_status_panel
        update_status_panel(status_panel, "", "info")
    
    @safe_component_access(component_type="progress tracker")
    def _clear_progress_tracker(self, tracker: Any) -> None:
        """Clear progress tracker safely.
        
        Args:
            tracker: Progress tracker widget
        """
        # Import locally to avoid circular imports
        from smartcash.ui.components.progress_tracker.progress_tracker import ProgressTracker
        if isinstance(tracker, ProgressTracker):
            tracker.reset()
            tracker.hide()
    
    @safe_widget_operation(widget_key_arg=0)
    def reset_progress_bars(self, ui_components: Dict[str, Any], progress_keys: List[str] = None) -> None:
        """Reset progress bars with safe widget access.
        
        Args:
            ui_components: Dictionary of UI components
            progress_keys: Optional list of specific progress keys to reset
        """
        default_progress_keys = ['progress_bar', 'progress_container', 'current_progress', 'progress_tracker']
        keys_to_reset = progress_keys or default_progress_keys
        
        for key in keys_to_reset:
            widget = ui_components.get(key)
            if widget:
                self._reset_single_progress_widget(key, widget)                
    
    @safe_component_access(component_type="progress widget")
    def _reset_single_progress_widget(self, key: str, widget: Any) -> None:
        """Reset a single progress widget safely.
        
        Args:
            key: Widget key in UI components
            widget: The widget to reset
        """
        # Handle ProgressTracker component
        if key == 'progress_tracker':
            # Import locally to avoid circular imports
            from smartcash.ui.components.progress_tracker.progress_tracker import ProgressTracker
            if isinstance(widget, ProgressTracker):
                widget.reset()
                return
        
        # Try to hide widget
        if hasattr(widget, 'layout'):
            widget.layout.visibility = 'hidden'
            widget.layout.display = 'none'
        
        # Try to reset value
        if hasattr(widget, 'value'):
            widget.value = 0
        
        # Try to reset progress tracker
        if hasattr(widget, 'reset'):
            widget.reset()
    
    @safe_progress_operation(progress_key="progress_tracker")
    def update_progress(self, ui_components: Dict[str, Any], value: float, 
                      max_value: float = 100, message: str = None, 
                      level: str = 'primary') -> None:
        """Update progress bar with proper error handling.
        
        Args:
            ui_components: Dictionary of UI components
            value: Current progress value (0-100)
            max_value: Maximum progress value (default: 100)
            message: Optional progress message
            level: Progress level for multi-level trackers ('primary', 'overall', 'current', 'step')
        """
        # Calculate percentage value (0-100)
        percentage = int((value / max_value) * 100) if max_value > 0 else 0
        percentage = max(0, min(100, percentage))  # Ensure within bounds
        
        # Handle ProgressTracker component
        if 'progress_tracker' in ui_components:
            # Import locally to avoid circular imports
            from smartcash.ui.components.progress_tracker.progress_tracker import ProgressTracker
            tracker = ui_components['progress_tracker']
            
            if isinstance(tracker, ProgressTracker):
                # Show tracker if not visible
                if not tracker.ui_manager.is_visible:
                    tracker.show()
                
                # Update appropriate level
                if level == 'primary':
                    tracker.update_primary(percentage, message or '')
                elif level == 'overall':
                    tracker.update_overall(percentage, message or '')
                elif level == 'current':
                    tracker.update_current(percentage, message or '')
                elif level == 'step':
                    tracker.update_step(percentage, message or '')
                else:
                    # Default to primary
                    tracker.update_primary(percentage, message or '')
                
                return  # Early return if ProgressTracker handled
        
        # Fallback to standard progress bar
        if 'progress_bar' in ui_components and hasattr(ui_components['progress_bar'], 'value'):
            ui_components['progress_bar'].value = percentage
            
            # Make visible if needed
            if hasattr(ui_components['progress_bar'], 'layout'):
                ui_components['progress_bar'].layout.visibility = 'visible'
                ui_components['progress_bar'].layout.display = 'block'
        
        # Update progress message if available
        if message and 'progress_message' in ui_components and hasattr(ui_components['progress_message'], 'value'):
            ui_components['progress_message'].value = message
            
        # Update status panel with progress
        if message:
            self._update_status_panel(
                ui_components, 
                f"{message} ({int(value)}/{int(max_value)})", 
                "info"
            )
            
        self.logger.debug(f"Progress update: {percentage}% ({value}/{max_value}) - {message or ''}")

    
    @safe_progress_operation(progress_key="progress_tracker")
    def complete_progress(self, ui_components: Dict[str, Any], message: str = "Operation completed successfully!") -> None:
        """Mark progress as complete.
        
        Args:
            ui_components: Dictionary of UI components
            message: Completion message
        """
        # Handle ProgressTracker component
        if 'progress_tracker' in ui_components:
            # Import locally to avoid circular imports
            from smartcash.ui.components.progress_tracker.progress_tracker import ProgressTracker
            tracker = ui_components['progress_tracker']
            
            if isinstance(tracker, ProgressTracker):
                tracker.complete(message)
                return  # Early return if ProgressTracker handled
        
        # Fallback to standard progress bar
        if 'progress_bar' in ui_components and hasattr(ui_components['progress_bar'], 'value'):
            ui_components['progress_bar'].value = 100
        
        # Update status panel with completion message
        self._update_status_panel(ui_components, message, "success")
    
    # ===== Progress Tracker Wrapper Methods =====
    
    @safe_progress_operation(progress_key="progress_tracker")
    def update_single_progress(self, ui_components: Dict[str, Any], value: int, max_value: int, 
                               message: str = None) -> None:
        """Update single-level progress tracker.
        
        Args:
            ui_components: Dictionary of UI components
            value: Current progress value
            max_value: Maximum progress value
            message: Optional progress message
        """
        # Handle ProgressTracker component
        if 'progress_tracker' in ui_components:
            # Import locally to avoid circular imports
            from smartcash.ui.components.progress_tracker.progress_tracker import ProgressTracker
            tracker = ui_components['progress_tracker']
            
            if isinstance(tracker, ProgressTracker):
                if not tracker.ui_manager.is_visible:
                    tracker.show()
                
                # Update primary progress only
                tracker.update_primary(value, message)
                return
        
        # Fallback to standard progress update
        self.update_progress(ui_components, value, max_value, message)
    
    @safe_progress_operation(progress_key="progress_tracker")
    def update_dual_progress(self, ui_components: Dict[str, Any], 
                            overall_value: int, overall_max: int, 
                            current_value: int, current_max: int,
                            overall_message: str = None, current_message: str = None) -> None:
        """Update dual-level progress tracker with overall and current progress.
        
        Args:
            ui_components: Dictionary of UI components
            overall_value: Overall progress value
            overall_max: Overall maximum value
            current_value: Current step progress value
            current_max: Current step maximum value
            overall_message: Optional overall progress message
            current_message: Optional current step message
        """
        # Handle ProgressTracker component
        if 'progress_tracker' in ui_components:
            # Import locally to avoid circular imports
            from smartcash.ui.components.progress_tracker.progress_tracker import ProgressTracker
            tracker = ui_components['progress_tracker']
            
            if isinstance(tracker, ProgressTracker):
                if not tracker.ui_manager.is_visible:
                    tracker.show()
                
                # Update both overall and current progress
                if overall_message:
                    tracker.update_primary(overall_value, overall_message)
                else:
                    tracker.update_primary(overall_value)
                    
                if current_message:
                    tracker.update_current(current_value, current_message)
                else:
                    tracker.update_current(current_value)
                return
        
        # Fallback to standard progress update (just show overall)
        self.update_progress(ui_components, overall_value, overall_max, overall_message or current_message)
    
    @safe_progress_operation(progress_key="progress_tracker")
    def update_triple_progress(self, ui_components: Dict[str, Any], 
                             overall_value: int, overall_max: int, 
                             current_value: int, current_max: int,
                             step_value: int, step_max: int,
                             overall_message: str = None, 
                             current_message: str = None,
                             step_message: str = None) -> None:
        """Update triple-level progress tracker with overall, current, and step progress.
        
        Args:
            ui_components: Dictionary of UI components
            overall_value: Overall progress value
            overall_max: Overall maximum value
            current_value: Current task progress value
            current_max: Current task maximum value
            step_value: Step progress value
            step_max: Step maximum value
            overall_message: Optional overall progress message
            current_message: Optional current task message
            step_message: Optional step message
        """
        # Handle ProgressTracker component
        if 'progress_tracker' in ui_components:
            # Import locally to avoid circular imports
            from smartcash.ui.components.progress_tracker.progress_tracker import ProgressTracker
            tracker = ui_components['progress_tracker']
            
            if isinstance(tracker, ProgressTracker):
                if not tracker.ui_manager.is_visible:
                    tracker.show()
                
                # Update all three progress levels
                if overall_message:
                    tracker.update_primary(overall_value, overall_message)
                else:
                    tracker.update_primary(overall_value)
                    
                if current_message:
                    tracker.update_current(current_value, current_message)
                else:
                    tracker.update_current(current_value)
                    
                if step_message:
                    tracker.update_step(step_value, step_message)
                else:
                    tracker.update_step(step_value)
                return
        
        # Fallback to standard progress update (just show overall)
        self.update_progress(ui_components, overall_value, overall_max, 
                           overall_message or current_message or step_message)

    @safe_progress_operation(progress_key="progress_tracker")
    def error_progress(self, ui_components: Dict[str, Any], message: str) -> None:
        """Mark progress as error.
        
        Args:
            ui_components: Dictionary of UI components
            message: Error message
        """
        # Handle ProgressTracker component
        if 'progress_tracker' in ui_components:
            # Import locally to avoid circular imports
            from smartcash.ui.components.progress_tracker.progress_tracker import ProgressTracker
            tracker = ui_components['progress_tracker']
            
            if isinstance(tracker, ProgressTracker):
                tracker.error(message)
                return  # Early return if ProgressTracker handled
        
        # Update status panel with error message
        self._update_status_panel(ui_components, message, "error")
        
        self.logger.error(f"Progress error: {message}")
    
    # ===== Utility Methods =====
    
    def log_info(self, message: str, *args, **kwargs) -> None:
        """Helper function untuk logging info dalam Bahasa Indonesia.
        
        Args:
            message: Pesan log dalam Bahasa Indonesia
            *args: Argumen tambahan untuk logger
            **kwargs: Keyword argumen tambahan untuk logger
        """
        self.logger.info(message, *args, **kwargs)
    
    def log_error(self, message: str, *args, **kwargs) -> None:
        """Helper function untuk logging error dalam Bahasa Indonesia.
        
        Args:
            message: Pesan error dalam Bahasa Indonesia
            *args: Argumen tambahan untuk logger
            **kwargs: Keyword argumen tambahan untuk logger
        """
        self.logger.error(message, *args, **kwargs)
    
    def log_warning(self, message: str, *args, **kwargs) -> None:
        """Helper function untuk logging warning dalam Bahasa Indonesia.
        
        Args:
            message: Pesan warning dalam Bahasa Indonesia
            *args: Argumen tambahan untuk logger
            **kwargs: Keyword argumen tambahan untuk logger
        """
        self.logger.warning(message, *args, **kwargs)
    
    def log_debug(self, message: str, *args, **kwargs) -> None:
        """Helper function untuk logging debug dalam Bahasa Indonesia.
        
        Args:
            message: Pesan debug dalam Bahasa Indonesia
            *args: Argumen tambahan untuk logger
            **kwargs: Keyword argumen tambahan untuk logger
        """
        self.logger.debug(message, *args, **kwargs)
    
    def is_success_response(self, response: Dict[str, Any]) -> bool:
        """Check if API response indicates success.
        
        Args:
            response: API response dictionary
            
        Returns:
            True if response indicates success, False otherwise
        """
        if not isinstance(response, dict):
            return False
            
        # Check for 'status' key with value 'success' for consistency across all handlers
        # Note: This aligns with the engine implementation standard
        return response.get('status', '') == 'success'
    
    def with_busy_cursor(self, ui_components: Dict[str, Any]):
        """Context manager for showing busy cursor during operations.
        
        Args:
            ui_components: Dictionary of UI components
            
        Usage:
            with handler.with_busy_cursor(ui_components):
                # Do something that takes time
        """
        class BusyCursorContext:
            def __init__(self, handler, components):
                self.handler = handler
                self.components = components
                
            def __enter__(self):
                self.handler.disable_all_buttons(self.components)
                self.handler._update_status_panel(self.components, "Processing...", "info")
                return self
                
            def __exit__(self, exc_type, exc_val, exc_tb):
                self.handler.enable_all_buttons(self.components)
                if exc_val:
                    self.handler._update_status_panel(
                        self.components, 
                        f"Error: {str(exc_val)}", 
                        "error"
                    )
                    return False
                return True
                
        return BusyCursorContext(self, ui_components)
    
    @safe_ui_operation(operation_name="show_info_dialog", log_level="error")
    def show_info_dialog(self, ui_components: Dict[str, Any], message: str, 
                        title: str = "Information", ok_text: str = "OK") -> None:
        """Show an information dialog.
        
        Args:
            ui_components: Dictionary of UI components
            message: Information message to display
            title: Title for the information dialog
            ok_text: Text for the OK button
        """
        # Import dialog module here to avoid circular imports
        from smartcash.ui.components.dialog.confirmation_dialog import show_info_dialog
        
        # Show info dialog using the dialog component
        show_info_dialog(
            ui_components=ui_components,
            title=title,
            message=message,
            on_ok=None,  # No callback needed for simple info
            ok_text=ok_text
        )
            
        self.logger.info(f"Info: {message}")
        
        # Note: If an exception occurs, the decorator will handle it and
        # the following fallback code will not be executed
        # Fallback to status panel is now handled in the except block of the decorator
