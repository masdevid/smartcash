"""
Operation Container Component

This module provides a unified interface for operation-related UI components,
including dialogs, progress tracking, and logging.
"""

from typing import Dict, Any, Optional, Literal, Callable, Union
from ipywidgets import widgets
from smartcash.ui.core.errors.enums import ErrorLevel
from smartcash.ui.components.progress_tracker.types import ProgressLevel, ProgressConfig
from smartcash.ui.components.log_accordion import LogLevel, LogAccordion
from smartcash.ui.components.progress_tracker.progress_tracker import ProgressTracker
from smartcash.ui.core.errors.handlers import CoreErrorHandler
from smartcash.ui.components.dialog import SimpleDialog
from smartcash.ui.components.base_component import BaseUIComponent

def create_operation_container(
    show_progress: bool = True,
    show_dialog: bool = True,
    show_logs: bool = True,
    log_module_name: str = "Operation",
    log_namespace_filter: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """Create an operation container with the specified components.
    
    Args:
        show_progress: Whether to show the progress tracker
        show_dialog: Whether to enable dialog functionality
        show_logs: Whether to show the log accordion
        log_module_name: Name to display in the log accordion header
        log_namespace_filter: Optional namespace prefix to filter logs (e.g. 'preprocess')
        **kwargs: Additional arguments to pass to OperationContainer
        
    Returns:
        Dict containing the container and its components
    """
    container = OperationContainer(
        show_progress=show_progress,
        show_dialog=show_dialog,
        show_logs=show_logs,
        log_module_name=log_module_name,
        log_namespace_filter=log_namespace_filter,
        **kwargs
    )
    
    return {
        'container': container.widget,
        'progress_tracker': container.progress_tracker if show_progress else None,
        'dialog': container.dialog if container._show_dialog_enabled else None,
        'log_accordion': container.log_accordion if show_logs else None,
        'show_dialog': container.show_dialog,
        'show_info': container.show_info,
        'clear_dialog': container.clear_dialog,
        'update_progress': container.update_progress,
        'update_triple_progress': container.update_triple_progress,
        'complete_triple_progress': container.complete_triple_progress,
        'error_triple_progress': container.error_triple_progress,
        'log': container.log
    }


class OperationContainer(BaseUIComponent):
    """A container for operation-related UI components.
    
    This class provides a unified interface for managing operation UI elements:
    - Action buttons
    - Dialogs
    - Progress tracking
    - Logging
    
    Attributes:
        container: The main VBox widget that holds all UI components
    """
    
    def __init__(self,
                 component_name: str = "operation_container",
                 progress_levels: Literal['single', 'dual', 'triple'] = 'single',
                 show_progress: bool = True,
                 show_logs: bool = True,
                 show_dialog: bool = True,
                 progress_config: Optional[ProgressConfig] = None,
                 log_module_name: str = "Operation",
                 log_height: str = "150px",
                 log_namespace_filter: Optional[str] = None,
                 **kwargs):
        """Initialize the OperationContainer.
        
        Args:
            component_name: Unique name for this component
            progress_levels: Number of progress levels to show ('single', 'dual', or 'triple')
            show_progress: Whether to show the progress tracker
            show_logs: Whether to show the log accordion
            show_dialog: Whether to enable dialog functionality
            progress_config: Configuration for the progress tracker
            log_module_name: Name of the module for logging
            log_height: Height of the log accordion
            log_namespace_filter: Optional namespace prefix to filter logs (e.g. 'preprocess')
            **kwargs: Additional arguments for BaseUIComponent (logger, error_handler)
        """
        # Filter out unexpected kwargs for BaseUIComponent
        base_kwargs = {k: v for k, v in kwargs.items() 
                      if k in ('logger', 'error_handler')}
        
        super().__init__(component_name, **base_kwargs)
        
        # Store configuration
        self.progress_levels = progress_levels
        self.show_progress = show_progress
        self.show_logs = show_logs
        self._show_dialog_enabled = show_dialog  # Renamed to avoid method conflict
        self.log_module_name = log_module_name
        self.log_height = log_height
        self.log_namespace_filter = log_namespace_filter
        # Initialize components
        self.progress_tracker = None
        self.progress_bars = {}
        self.log_accordion = None
        self.dialog = None
        
        # Create main container widget first
        self._create_container()
        
        # Initialize dialog if enabled (must be before creating UI components)
        if self._show_dialog_enabled:
            self._init_dialog()
            
        # Create UI components and add them to container
        self._create_ui_components(progress_config)
        
    def _create_container(self) -> None:
        """Create the main container widget."""
        # Create layout with validated properties
        layout_kwargs = {
            'width': '100%',
            'margin': '10px 0',
            'padding': '10px',
            'border': '1px solid #e0e0e0',
            'border_radius': '5px',
            'flex': '1 1 auto',
            'display': 'flex',
            'flex_flow': 'column',
            'position': 'relative',
            'min_height': '0',
            'align_items': 'stretch'
        }

        # Create and validate the layout
        layout = widgets.Layout(**{
            k: self._validate_layout_value(v, k, 'stretch' if k == 'align_items' else v)
            for k, v in layout_kwargs.items()
        })

        # Initialize empty tuple for children
        children = ()
        
        # 1. Add progress tracker if enabled (top)
        if self.progress_tracker and hasattr(self.progress_tracker, 'container') and self.progress_tracker.container is not None:
            children += (self.progress_tracker.container,)

        # 2. Add dialog (middle) - will be populated when needed
        if hasattr(self, 'dialog') and self.dialog is not None:
            children += (self.dialog.container,)

        # 3. Add log accordion if enabled (bottom) - use helper for DRY approach
        if self.log_accordion:
            log_widget = self._get_log_accordion_widget()
            if log_widget is not None:
                children += (log_widget,)

        # Create the main container with all valid components
        self.container = widgets.VBox(children=children, layout=layout)
        
    @property
    def widget(self) -> widgets.VBox:
        """Get the main widget container."""
        return self.container
    
    def _create_ui_components(self, progress_config: Optional[ProgressConfig] = None) -> None:
        """Create and initialize UI components.
        
        This method is called by BaseUIComponent.initialize() and should set up
        all UI components used by this container.
        
        Args:
            progress_config: Optional configuration for the progress tracker
        """
        # Create progress tracker if enabled
        if self.show_progress:
            # Create appropriate progress config based on progress_levels
            if not progress_config:
                from smartcash.ui.components.progress_tracker.types import ProgressLevel
                level_mapping = {
                    'single': ProgressLevel.SINGLE,
                    'dual': ProgressLevel.DUAL,
                    'triple': ProgressLevel.TRIPLE
                }
                level = level_mapping.get(self.progress_levels, ProgressLevel.SINGLE)
                progress_config = ProgressConfig(level=level, operation="Operasi")
            
            # Create progress tracker with the specified number of levels
            self.progress_tracker = ProgressTracker(
                component_name=f"{self.component_name}_progress",
                config=progress_config
            )
            # Initialize the progress tracker
            self.progress_tracker.initialize()
            
            # Show progress tracker by default for operation containers
            self.progress_tracker.show()
            
            # Initialize progress bars based on the number of levels
            levels = {
                'single': ['primary'],
                'dual': ['primary', 'secondary'],
                'triple': ['primary', 'secondary', 'tertiary']
            }.get(self.progress_levels, ['primary'])
            
            for level in levels:
                self.progress_bars[level] = {
                    'value': 0,
                    'message': '',
                    'visible': True
                }
        
        # Create log accordion if enabled
        if self.show_logs:
            self.log_accordion = LogAccordion(
                component_name=f"{self.component_name}_log_accordion",
                module_name=self.log_module_name,
                height=self.log_height,
                namespace_filter=self.log_namespace_filter,

            )
            self.log_accordion.initialize()
        
        # Create main container
        self._create_container()
    
    def _validate_layout_value(self, value: str, prop_name: str, default: str) -> str:
        """Validate and sanitize layout property values.
        
        Args:
            value: The value to validate
            prop_name: The name of the property being validated
            default: The default value to use if validation fails
            
        Returns:
            A valid value for the specified property
        """
        if value is None:
            return default
            
        value = str(value).lower()
        
        if prop_name == 'align_items':
            valid_values = ['flex-start', 'flex-end', 'center', 'baseline', 'stretch', 'inherit', 'initial', 'unset']
            if value not in valid_values:
                try:
                    self.logger.warning(f"Invalid {prop_name} value: '{value}'. Using default: '{default}'")
                except (TypeError, AttributeError):
                    # Fallback if logger is not properly initialized
                    import logging
                    logging.getLogger(__name__).warning(f"Invalid {prop_name} value: '{value}'. Using default: '{default}'")
                return default
                
        return value
        
    def _create_container(self) -> None:
        """Create the main container widget.
        
        Components are ordered as:
        1. Progress Tracker (top)
        2. Dialog Area (middle)
        3. Log Accordion (bottom)
        """
        # Create layout with validated properties
        layout_kwargs = {
            'width': '100%',
            'margin': '10px 0',
            'padding': '10px',
            'border': '1px solid #e0e0e0',
            'border_radius': '5px',
            'flex': '1 1 auto',
            'display': 'flex',
            'flex_flow': 'column',
            'position': 'relative',
            'min_height': '0',
            'align_items': 'stretch'
        }

        # Create and validate the layout
        layout = widgets.Layout(**{
            k: self._validate_layout_value(v, k, 'stretch' if k == 'align_items' else v)
            for k, v in layout_kwargs.items()
        })

        # Initialize empty tuple for children
        children = ()
        
        # 1. Add progress tracker if enabled (top)
        if self.progress_tracker and hasattr(self.progress_tracker, 'container') and self.progress_tracker.container is not None:
            children += (self.progress_tracker.container,)

        # 2. Add dialog (middle) - will be populated when needed
        if hasattr(self, 'dialog') and self.dialog is not None:
            children += (self.dialog.container,)

        # 3. Add log accordion if enabled (bottom) - use helper for DRY approach
        if self.log_accordion:
            log_widget = self._get_log_accordion_widget()
            if log_widget is not None:
                children += (log_widget,)

        # Create the main container with all valid components
        self.container = widgets.VBox(children=children, layout=layout)
    
    def _init_dialog(self) -> None:
        """Initialize the dialog component."""
        self.dialog = SimpleDialog(
            component_name=f"{self.component_name}_dialog"
        )
        self.dialog.initialize()
    
    def _get_log_accordion_widget(self):
        """Get log accordion widget in a DRY way, handling different component types.
        
        Returns:
            Widget or None if not available
        """
        if not self.log_accordion:
            return None
            
        try:
            # Method 1: Try to get container property directly
            if hasattr(self.log_accordion, 'container'):
                widget = getattr(self.log_accordion, 'container', None)
                if widget is not None:
                    return widget
            
            # Method 2: Try initialized property check then container
            if hasattr(self.log_accordion, '_initialized'):
                if not self.log_accordion._initialized:
                    # Initialize if not already done
                    if hasattr(self.log_accordion, 'initialize'):
                        self.log_accordion.initialize()
                
                # Now try container again
                if hasattr(self.log_accordion, 'container'):
                    widget = getattr(self.log_accordion, 'container', None)
                    if widget is not None:
                        return widget
            
            # Method 3: Try show() method as fallback (but avoid during initialization)
            if hasattr(self.log_accordion, 'show') and callable(self.log_accordion.show):
                # Only call show() if we're not in initialization phase
                widget = self.log_accordion.show()
                if widget is not None:
                    return widget
                    
        except Exception:
            # Silently handle any errors to avoid breaking initialization
            pass
            
        return None
    
    # ===== Progress Tracking Methods =====
    
    def update_progress(self, 
                      progress: int, 
                      message: str = "", 
                      level: str = "primary",
                      level_label: str = None) -> None:
        """Update progress with the given message and level.
        
        Args:
            progress: Progress percentage (0-100)
            message: Progress message
            level: Progress level (primary, secondary, tertiary)
            level_label: Optional label for the progress level
        """
        if not self.progress_tracker:
            return
            
        # Ensure progress bars are initialized if level doesn't exist
        if level not in self.progress_bars:
            # Auto-initialize progress bars if not already done
            if not self.progress_bars:
                levels = {
                    'single': ['primary'],
                    'dual': ['primary', 'secondary'],
                    'triple': ['primary', 'secondary', 'tertiary']
                }.get(self.progress_levels, ['primary'])
                
                for lvl in levels:
                    self.progress_bars[lvl] = {
                        'value': 0,
                        'message': '',
                        'visible': True
                    }
            
            # If level still doesn't exist after setup, return
            if level not in self.progress_bars:
                self.debug(f"Progress level '{level}' not found in configured levels: {list(self.progress_bars.keys())}")
                return
        
        # Ensure progress tracker is visible when updating progress
        if not self.progress_tracker._initialized:
            self.progress_tracker.initialize()
        
        # Prevent multiple progress tracker instances within operation container
        if hasattr(self.progress_tracker, 'container') and self.progress_tracker.container:
            # Check if already visible to prevent duplication
            if (self.progress_tracker.container.layout.display == 'none' and 
                not getattr(self.progress_tracker, '_display_active', False)):
                self.progress_tracker.show()
            
        # Update progress value and message
        self.progress_bars[level].update({
            'value': max(0, min(100, progress)),
            'message': message,
            'visible': True
        })
        
        # Update the progress tracker
        self._update_progress_bars()
    
    def set_progress_visibility(self, level: str, visible: bool) -> None:
        """Show or hide a specific progress bar.
        
        Args:
            level: Progress level (primary, secondary, tertiary)
            visible: Whether to show or hide the progress bar
        """
        if level in self.progress_bars:
            self.progress_bars[level]['visible'] = visible
            self._update_progress_bars()
    
    def complete_progress(self, 
                        message: str = "Completed!", 
                        level: str = "primary") -> None:
        """Mark progress as complete for a specific level.
        
        Args:
            message: Completion message
            level: Progress level (primary, secondary, tertiary)
        """
        self.update_progress(100, message, level)
    
    def error_progress(self, 
                      message: str = "An error occurred!", 
                      level: str = "primary") -> None:
        """Mark progress as errored for a specific level.
        
        Args:
            message: Error message
            level: Progress level (primary, secondary, tertiary)
        """
        if level in self.progress_bars:
            self.progress_bars[level].update({
                'error': True,
                'message': message
            })
            self._update_progress_bars()
    
    def reset_progress(self, level: str = None) -> None:
        """Reset progress for a specific level or all levels.
        
        Args:
            level: Progress level to reset (None for all levels)
        """
        if level is None:
            for lvl in self.progress_bars:
                self.progress_bars[lvl].update({
                    'value': 0,
                    'message': '',
                    'error': False,
                    'visible': lvl == 'primary'  # Only primary visible by default
                })
        elif level in self.progress_bars:
            self.progress_bars[level].update({
                'value': 0,
                'message': '',
                'error': False
            })
        
        self._update_progress_bars()
    
    def update_triple_progress(self, 
                              overall_step: int = None, overall_message: str = "",
                              phase_step: int = None, phase_message: str = "",
                              current_step: int = None, current_message: str = "") -> None:
        """Update triple progress display for granular operation tracking.
        
        Args:
            overall_step: Overall progress value (0-100)
            overall_message: Overall progress message
            phase_step: Phase progress value (0-100) 
            phase_message: Phase progress message
            current_step: Current step progress value (0-100)
            current_message: Current step progress message
        """
        try:
            # Update overall progress (primary level)
            if overall_step is not None:
                self.update_progress(overall_step, overall_message, 'primary')
            
            # Update phase progress (secondary level)
            if phase_step is not None:
                self.update_progress(phase_step, phase_message, 'secondary')
            
            # Update current progress (tertiary level)
            if current_step is not None:
                self.update_progress(current_step, current_message, 'tertiary')
                
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Error in update_triple_progress: {e}")
    
    def complete_triple_progress(self, message: str = "Completed!") -> None:
        """Complete triple progress tracking by setting all levels to 100%.
        
        Args:
            message: Completion message
        """
        try:
            self.update_triple_progress(
                overall_step=100, overall_message=message,
                phase_step=100, phase_message=message,
                current_step=100, current_message=message
            )
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Error in complete_triple_progress: {e}")
    
    def error_triple_progress(self, message: str = "An error occurred!") -> None:
        """Set triple progress to error state.
        
        Args:
            message: Error message
        """
        try:
            # Set error state for all three levels
            for level in ['primary', 'secondary', 'tertiary']:
                if level in self.progress_bars:
                    self.progress_bars[level].update({
                        'error': True,
                        'message': f"❌ {message}"
                    })
            self._update_progress_bars()
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Error in error_triple_progress: {e}")
    
    def _update_progress_bars(self) -> None:
        """Update the progress tracker with current progress values."""
        if not self.progress_tracker:
            return
        
        try:
            # Ensure progress tracker is properly initialized and visible
            if not hasattr(self.progress_tracker, '_initialized') or not self.progress_tracker._initialized:
                self.progress_tracker.initialize()
            
            # Show progress tracker if any bars are visible
            has_visible_bars = any(data.get('visible', False) for data in self.progress_bars.values())
            
            if has_visible_bars:
                # Ensure tracker is visible with duplicate prevention
                if hasattr(self.progress_tracker, 'container') and self.progress_tracker.container:
                    if (self.progress_tracker.container.layout.display == 'none' and 
                        not getattr(self.progress_tracker, '_display_active', False)):
                        self.progress_tracker.show()
                
                # Update each visible progress bar
                for level, data in self.progress_bars.items():
                    if data.get('visible', False):
                        try:
                            if data.get('error', False):
                                # Show error state
                                if hasattr(self.progress_tracker, 'error'):
                                    self.progress_tracker.error(data.get('message', 'An error occurred'))
                                else:
                                    # Fallback to set_progress with error styling
                                    self.progress_tracker.set_progress(
                                        data.get('value', 0), 
                                        f"❌ {data.get('message', 'Error')}",
                                        level
                                    )
                            else:
                                # Normal progress update
                                self.progress_tracker.set_progress(
                                    data.get('value', 0), 
                                    data.get('message', ''),
                                    level
                                )
                        except Exception as e:
                            # Log error but don't break the UI
                            import logging
                            logger = logging.getLogger(__name__)
                            logger.error(f"Error updating progress bar {level}: {e}")
            else:
                # No visible bars - consider hiding tracker
                # But keep it visible if it was explicitly shown
                pass
                
        except Exception as e:
            # Comprehensive error handling for progress tracker updates
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Error in _update_progress_bars: {e}")
    
    # ===== Dialog Methods =====
    
    def show_dialog(self, 
                   title: str, 
                   message: str, 
                   on_confirm: Optional[Callable] = None,
                   on_cancel: Optional[Callable] = None,
                   confirm_text: str = "Confirm",
                   cancel_text: str = "Cancel",
                   danger_mode: bool = False) -> bool:
        """Show a confirmation dialog using SimpleDialog component.
        
        Args:
            title: Dialog title
            message: Dialog message
            on_confirm: Callback when user confirms
            on_cancel: Callback when user cancels
            confirm_text: Text for confirm button
            cancel_text: Text for cancel button
            danger_mode: If True, shows the confirm button in danger color
            
        Returns:
            True if dialog was shown successfully, False otherwise
        """
        if not self.dialog:
            # Log fallback if dialog is not available
            self.log(f"Dialog requested: {title} - {message}", LogLevel.INFO)
            if on_confirm:
                on_confirm()
            return False
        
        try:
            # Enhanced callback wrappers with logging
            def enhanced_confirm():
                try:
                    self.log(f"Dialog confirmed: {title}", LogLevel.INFO)
                    if on_confirm:
                        on_confirm()
                except Exception as e:
                    self.log(f"Error in dialog confirm callback: {e}", LogLevel.ERROR)
                    
            def enhanced_cancel():
                try:
                    self.log(f"Dialog cancelled: {title}", LogLevel.INFO)
                    if on_cancel:
                        on_cancel()
                except Exception as e:
                    self.log(f"Error in dialog cancel callback: {e}", LogLevel.ERROR)
            
            # Use the dedicated SimpleDialog component
            self.dialog.show_confirmation(
                title=title,
                message=message,
                on_confirm=enhanced_confirm,
                on_cancel=enhanced_cancel,
                confirm_text=confirm_text,
                cancel_text=cancel_text,
                danger_mode=danger_mode
            )
            
            # Log dialog display
            self.log(f"Showing dialog: {title}", LogLevel.DEBUG)
            return True
            
        except Exception as e:
            self.log(f"Error showing dialog '{title}': {e}", LogLevel.ERROR)
            return False
    
    def show_info(self, 
                 title: str, 
                 message: str, 
                 on_ok: Optional[Callable] = None,
                 ok_text: str = "OK",
                 info_type: str = "info") -> bool:
        """Show an info dialog using SimpleDialog component.
        
        Args:
            title: Dialog title
            message: Dialog message
            on_ok: Callback when user clicks OK
            ok_text: Text for OK button
            info_type: Type of info dialog (info, success, warning, error)
            
        Returns:
            True if dialog was shown successfully, False otherwise
        """
        if not self.dialog:
            # Log fallback if dialog is not available
            self.log(f"Info dialog requested: {title} - {message}", LogLevel.INFO)
            if on_ok:
                on_ok()
            return False
        
        try:
            # Enhanced callback wrapper with logging
            def enhanced_ok():
                try:
                    self.log(f"Info dialog acknowledged: {title}", LogLevel.INFO)
                    if on_ok:
                        on_ok()
                except Exception as e:
                    self.log(f"Error in info dialog callback: {e}", LogLevel.ERROR)
            
            # Use the dedicated SimpleDialog component
            self.dialog.show_info(
                title=title,
                message=message,
                on_ok=enhanced_ok,
                ok_text=ok_text,
                info_type=info_type
            )
            
            # Log dialog display
            self.log(f"Showing info dialog: {title}", LogLevel.DEBUG)
            return True
            
        except Exception as e:
            self.log(f"Error showing info dialog '{title}': {e}", LogLevel.ERROR)
            return False
    
    # Alias for backward compatibility
    show_info_dialog = show_info
    
    def clear_dialog(self) -> bool:
        """Clear any currently displayed dialog.
        
        Returns:
            True if dialog was cleared successfully, False otherwise
        """
        if not self.dialog:
            return False
            
        # Use SimpleDialog's hide method
        self.dialog.hide()
        
        # Log dialog clearing
        self.log("Dialog cleared", LogLevel.DEBUG)
        return True
    
    def is_dialog_visible(self) -> bool:
        """Check if a dialog is currently visible.
        
        Returns:
            True if any dialog is visible, False otherwise
        """
        if not self.dialog:
            return False
            
        # Use SimpleDialog's is_visible method
        return self.dialog.is_visible()
    
    # ===== Logging Methods =====
    
    def log(self, message: str, level: LogLevel = LogLevel.INFO, namespace: str = None) -> None:
        """Log a message with the specified level.
        
        Args:
            message: The message to log
            level: The log level (default: INFO)
            namespace: Optional namespace for categorizing logs
        """
        if not self.log_accordion:
            return
            
        try:
            # Ensure log accordion is properly initialized
            if not hasattr(self.log_accordion, '_initialized') or not self.log_accordion._initialized:
                self.log_accordion.initialize()
            
            # Apply namespace filtering if configured
            if self.log_namespace_filter and namespace:
                # Only show logs that match the namespace filter
                if not namespace.lower().startswith(self.log_namespace_filter.lower()):
                    return  # Skip logs that don't match the filter
            
            # Add timestamp and namespace formatting if provided
            if namespace:
                formatted_message = f"[{namespace}] {message}"
            else:
                formatted_message = message
            
            # Log the message
            self.log_accordion.log(formatted_message, level)
            
            # Auto-expand log accordion for important messages
            if level in [LogLevel.WARNING, LogLevel.ERROR]:
                if hasattr(self.log_accordion, 'expand'):
                    self.log_accordion.expand()
                    
        except Exception as e:
            # Fallback logging to console if log accordion fails
            import logging
            fallback_logger = logging.getLogger(__name__)
            
            # Safely handle LogLevel enum or string
            try:
                if hasattr(level, 'value'):
                    level_value = str(level.value).upper()
                elif hasattr(level, 'name'):
                    level_value = level.name.upper()
                else:
                    level_value = str(level).upper()
                
                fallback_logger.log(getattr(logging, level_value, logging.INFO), f"[FALLBACK] {message}")
            except Exception as level_error:
                # Final fallback with INFO level
                fallback_logger.info(f"[FALLBACK] {message}")
                fallback_logger.error(f"Level normalization error: {level_error}")
            
            fallback_logger.error(f"Log accordion error: {e}")
    
    # Alias for backward compatibility
    log_message = log
    
    def debug(self, message: str) -> None:
        """Log a debug message."""
        self.log(message, LogLevel.DEBUG)
    
    def log_debug(self, message: str, namespace: str = None) -> None:
        """Log a debug message with optional namespace (alias for debug with namespace support)."""
        self.log(message, LogLevel.DEBUG, namespace)
    
    def info(self, message: str) -> None:
        """Log an info message."""
        self.log(message, LogLevel.INFO)
    
    def warning(self, message: str) -> None:
        """Log a warning message."""
        self.log(message, LogLevel.WARNING)
    
    def error(self, message: str) -> None:
        """Log an error message."""
        self.log(message, LogLevel.ERROR)
    
    # NOTE: update_status() removed from operation_container
    # Status updates should go through operation_mixin → header_container
    # This maintains proper separation of concerns
    
    def clear_logs(self) -> None:
        """Clear all log messages from the log accordion."""
        if self.log_accordion and hasattr(self.log_accordion, 'clear'):
            try:
                self.log_accordion.clear()
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"Error clearing logs: {e}")
    
    def get_log_count(self) -> int:
        """Get the total number of log entries.
        
        Returns:
            Number of log entries, or 0 if log accordion is not available
        """
        if self.log_accordion and hasattr(self.log_accordion, 'get_log_count'):
            try:
                return self.log_accordion.get_log_count()
            except Exception:
                pass
        return 0
    
    def export_logs(self) -> str:
        """Export all logs as a formatted string.
        
        Returns:
            Formatted string containing all log entries
        """
        if self.log_accordion and hasattr(self.log_accordion, 'export_logs'):
            try:
                return self.log_accordion.export_logs()
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"Error exporting logs: {e}")
        return ""
    
    def critical(self, message: str) -> None:
        """Log a critical message."""
        self.log(message, LogLevel.CRITICAL)
    
    # ===== Integration and Health Check Methods =====
    
    def validate_integration(self) -> Dict[str, Any]:
        """Validate integration of all operation container components.
        
        Returns:
            Dictionary with validation results for each component
        """
        results = {
            'overall_status': 'healthy',
            'components': {},
            'issues': [],
            'recommendations': []
        }
        
        # Check progress tracker
        if self.show_progress:
            progress_status = self._validate_progress_tracker()
            results['components']['progress_tracker'] = progress_status
            if not progress_status['healthy']:
                results['issues'].extend(progress_status['issues'])
                results['overall_status'] = 'issues'
        
        # Check log accordion
        if self.show_logs:
            log_status = self._validate_log_accordion()
            results['components']['log_accordion'] = log_status
            if not log_status['healthy']:
                results['issues'].extend(log_status['issues'])
                results['overall_status'] = 'issues'
        
        # Check dialog system
        if self._show_dialog_enabled:
            dialog_status = self._validate_dialog_system()
            results['components']['dialog_system'] = dialog_status
            if not dialog_status['healthy']:
                results['issues'].extend(dialog_status['issues'])
                results['overall_status'] = 'issues'
        
        # Check container layout
        container_status = self._validate_container_layout()
        results['components']['container_layout'] = container_status
        if not container_status['healthy']:
            results['issues'].extend(container_status['issues'])
            results['overall_status'] = 'issues'
        
        return results
    
    def _validate_progress_tracker(self) -> Dict[str, Any]:
        """Validate progress tracker component."""
        status = {'healthy': True, 'issues': [], 'details': {}}
        
        try:
            if not self.progress_tracker:
                status['healthy'] = False
                status['issues'].append("Progress tracker not initialized")
                return status
            
            # Check initialization
            if not hasattr(self.progress_tracker, '_initialized') or not self.progress_tracker._initialized:
                status['issues'].append("Progress tracker not properly initialized")
                status['healthy'] = False
            
            # Check container availability
            if not hasattr(self.progress_tracker, 'container') or not self.progress_tracker.container:
                status['issues'].append("Progress tracker container not available")
                status['healthy'] = False
            
            # Check progress bars configuration
            expected_levels = {
                'single': ['primary'],
                'dual': ['primary', 'secondary'],
                'triple': ['primary', 'secondary', 'tertiary']
            }.get(self.progress_levels, ['primary'])
            
            for level in expected_levels:
                if level not in self.progress_bars:
                    status['issues'].append(f"Progress bar level '{level}' not configured")
                    status['healthy'] = False
            
            status['details'] = {
                'initialized': getattr(self.progress_tracker, '_initialized', False),
                'levels_configured': list(self.progress_bars.keys()),
                'expected_levels': expected_levels,
                'visible': hasattr(self.progress_tracker, 'container') and 
                          self.progress_tracker.container and 
                          self.progress_tracker.container.layout.display != 'none'
            }
            
        except Exception as e:
            status['healthy'] = False
            status['issues'].append(f"Progress tracker validation error: {e}")
        
        return status
    
    def _validate_log_accordion(self) -> Dict[str, Any]:
        """Validate log accordion component."""
        status = {'healthy': True, 'issues': [], 'details': {}}
        
        try:
            if not self.log_accordion:
                status['healthy'] = False
                status['issues'].append("Log accordion not initialized")
                return status
            
            # Check initialization
            if not hasattr(self.log_accordion, '_initialized') or not self.log_accordion._initialized:
                status['issues'].append("Log accordion not properly initialized")
                status['healthy'] = False
            
            # Check logging capability
            if not hasattr(self.log_accordion, 'log') or not callable(self.log_accordion.log):
                status['issues'].append("Log accordion missing log method")
                status['healthy'] = False
            
            status['details'] = {
                'initialized': getattr(self.log_accordion, '_initialized', False),
                'module_name': self.log_module_name,
                'height': self.log_height,
                'log_count': self.get_log_count()
            }
            
        except Exception as e:
            status['healthy'] = False
            status['issues'].append(f"Log accordion validation error: {e}")
        
        return status
    
    def _validate_dialog_system(self) -> Dict[str, Any]:
        """Validate dialog system component."""
        status = {'healthy': True, 'issues': [], 'details': {}}
        
        try:
            if not self.dialog:
                status['healthy'] = False
                status['issues'].append("Dialog not initialized")
                return status
            
            # Check dialog initialization
            if not hasattr(self.dialog, '_initialized') or not self.dialog._initialized:
                status['issues'].append("Dialog not properly initialized")
                status['healthy'] = False
            
            status['details'] = {
                'dialog_available': self.dialog is not None,
                'dialog_initialized': getattr(self.dialog, '_initialized', False),
                'dialog_visible': self.is_dialog_visible()
            }
            
        except Exception as e:
            status['healthy'] = False
            status['issues'].append(f"Dialog system validation error: {e}")
        
        return status
    
    def _validate_container_layout(self) -> Dict[str, Any]:
        """Validate container layout and structure."""
        status = {'healthy': True, 'issues': [], 'details': {}}
        
        try:
            if not self.container:
                status['healthy'] = False
                status['issues'].append("Main container not initialized")
                return status
            
            # Check container children
            expected_children = 0
            if self.show_progress and self.progress_tracker:
                expected_children += 1
            if self._show_dialog_enabled and self.dialog:
                expected_children += 1
            if self.show_logs and self.log_accordion:
                expected_children += 1
            
            actual_children = len(self.container.children)
            if actual_children != expected_children:
                status['issues'].append(f"Container children mismatch: expected {expected_children}, got {actual_children}")
                status['healthy'] = False
            
            status['details'] = {
                'expected_children': expected_children,
                'actual_children': actual_children,
                'show_progress': self.show_progress,
                'show_logs': self.show_logs,
                'show_dialog': self._show_dialog_enabled
            }
            
        except Exception as e:
            status['healthy'] = False
            status['issues'].append(f"Container layout validation error: {e}")
        
        return status
        
    def clear_logs(self) -> None:
        """Clear all log messages."""
        if self.log_accordion:
            self.log_accordion.clear()
    
    # ===== Convenience Methods =====
    
    def show_info_dialog(self, title: str, message: str, on_ok: Optional[Callable] = None) -> bool:
        """Convenience method for showing info dialogs."""
        return self.show_info(title, message, on_ok, "OK", "info")
    
    def show_success_dialog(self, title: str, message: str, on_ok: Optional[Callable] = None) -> bool:
        """Convenience method for showing success dialogs."""
        return self.show_info(title, message, on_ok, "OK", "success")
    
    def show_warning_dialog(self, title: str, message: str, on_ok: Optional[Callable] = None) -> bool:
        """Convenience method for showing warning dialogs."""
        return self.show_info(title, message, on_ok, "OK", "warning")
    
    def show_error_dialog(self, title: str, message: str, on_ok: Optional[Callable] = None) -> bool:
        """Convenience method for showing error dialogs."""
        return self.show_info(title, message, on_ok, "OK", "error")
    
    # ===== Logging Bridge =====
    
    def setup_logging_bridge(self) -> None:
        """Setup logging bridge to capture backend service logs.
        
        This method configures the Python logging system to redirect
        logs from backend services to the operation container.
        """
        try:
            import logging
            from smartcash.ui.components.log_accordion import LogLevel
            
            # Check if bridge is already setup
            if hasattr(self, '_logging_bridge_setup') and self._logging_bridge_setup:
                return
            
            # Get root logger to capture all backend logs
            root_logger = logging.getLogger()
            
            # Create custom handler that redirects to operation container
            class OperationContainerHandler(logging.Handler):
                def __init__(self, operation_container):
                    super().__init__()
                    self.operation_container = operation_container
                    
                def emit(self, record):
                    try:
                        # Format the message
                        message = self.format(record)
                        
                        # Map Python logging levels to UI log levels
                        level_mapping = {
                            logging.DEBUG: LogLevel.DEBUG,
                            logging.INFO: LogLevel.INFO,
                            logging.WARNING: LogLevel.WARNING,
                            logging.ERROR: LogLevel.ERROR,
                            logging.CRITICAL: LogLevel.ERROR
                        }
                        
                        ui_level = level_mapping.get(record.levelno, LogLevel.INFO)
                        
                        # Extract namespace from logger name and map to module namespace
                        namespace = None
                        if record.name and record.name != 'root':
                            # Map backend logger names to module namespaces
                            logger_name = record.name.lower()
                            if 'preprocessor' in logger_name:
                                namespace = 'preprocessing'
                            elif 'augmentor' in logger_name:
                                namespace = 'augmentation'
                            elif 'backbone' in logger_name:
                                namespace = 'backbone'
                            elif 'trainer' in logger_name:
                                namespace = 'training'
                            elif 'evaluator' in logger_name:
                                namespace = 'evaluation'
                            elif 'visualizer' in logger_name:
                                namespace = 'visualization'
                            else:
                                # Use the module name from the operation container if available
                                namespace = getattr(self.operation_container, 'log_namespace_filter', None) or record.name
                        
                        # Log to operation container with proper namespace
                        self.operation_container.log(message, ui_level, namespace)
                        
                    except Exception:
                        # Avoid infinite recursion by not logging errors here
                        pass
            
            # Create and configure the handler
            self._ui_handler = OperationContainerHandler(self)
            self._ui_handler.setLevel(logging.DEBUG)  # Capture all levels
            
            # Set a simple formatter
            formatter = logging.Formatter('%(message)s')
            self._ui_handler.setFormatter(formatter)
            
            # Add handler to root logger to capture ALL Python logging
            root_logger.addHandler(self._ui_handler)
            
            # Aggressively disable console output from all existing loggers
            self._disable_console_handlers()
            
            # Set root logger to propagate to our handler
            root_logger.setLevel(logging.DEBUG)
            
            # Activate UI mode for SmartCashLogger
            self._activate_smartcash_ui_mode()
            
            # Mark bridge as setup
            self._logging_bridge_setup = True
            
            self.log("✅ Logging bridge setup complete - console output disabled", LogLevel.DEBUG, "operation_container")
            
        except Exception as e:
            # Fallback logging
            import logging
            fallback_logger = logging.getLogger(__name__)
            fallback_logger.error(f"Failed to setup logging bridge: {e}")
    
    def capture_logs(self, logger_instance=None) -> None:
        """Capture logs from a specific logger instance or setup general capture.
        
        Args:
            logger_instance: Specific logger to capture (optional)
        """
        try:
            import logging
            from smartcash.ui.components.log_accordion import LogLevel
            
            # If no specific logger provided, setup general bridge
            if logger_instance is None:
                self.setup_logging_bridge()
                return
            
            # Ensure we have a UI handler
            if not hasattr(self, '_ui_handler'):
                self.setup_logging_bridge()
            
            if not hasattr(self, '_captured_loggers'):
                self._captured_loggers = []
            
            # Handle different types of logger instances
            if hasattr(logger_instance, 'logger') and hasattr(logger_instance.logger, 'addHandler'):
                # This is a SmartCashLogger or similar wrapper
                if logger_instance not in self._captured_loggers:
                    logger_instance.logger.addHandler(self._ui_handler)
                    self._captured_loggers.append(logger_instance)
                    self.log(f"✅ Captured SmartCashLogger: {getattr(logger_instance, 'name', 'unknown')}", 
                            LogLevel.DEBUG, "operation_container")
                            
            elif hasattr(logger_instance, 'addHandler'):
                # This is a standard Python logger
                if logger_instance not in self._captured_loggers:
                    logger_instance.addHandler(self._ui_handler)
                    self._captured_loggers.append(logger_instance)
                    self.log(f"✅ Captured Python logger: {logger_instance.name}", 
                            LogLevel.DEBUG, "operation_container")
                            
            elif isinstance(logger_instance, str):
                # Logger name provided, get the actual logger
                actual_logger = logging.getLogger(logger_instance)
                if actual_logger not in self._captured_loggers:
                    actual_logger.addHandler(self._ui_handler)
                    self._captured_loggers.append(actual_logger)
                    self.log(f"✅ Captured logger by name: {logger_instance}", 
                            LogLevel.DEBUG, "operation_container")
            
        except Exception as e:
            # Fallback logging
            import logging
            fallback_logger = logging.getLogger(__name__)
            fallback_logger.error(f"Failed to capture logger: {e}")
    
    def _disable_console_handlers(self) -> None:
        """Disable console handlers across all active loggers to prevent log leakage."""
        try:
            import logging
            
            # Get all active loggers
            loggers_to_check = [logging.getLogger()] # Start with root logger
            
            # Add all named loggers
            for name in logging.Logger.manager.loggerDict:
                logger = logging.getLogger(name)
                if isinstance(logger, logging.Logger):
                    loggers_to_check.append(logger)
            
            # Disable console handlers in all loggers
            for logger in loggers_to_check:
                if hasattr(logger, 'handlers'):
                    for handler in logger.handlers[:]:
                        if isinstance(handler, logging.StreamHandler):
                            # Check if it's a console handler (stdout/stderr)
                            if hasattr(handler, 'stream') and hasattr(handler.stream, 'name'):
                                if handler.stream.name in ['<stdout>', '<stderr>']:
                                    # Disable by setting level very high
                                    handler.setLevel(logging.CRITICAL + 1)
                                    
                                    self.log(f"Disabled console handler for logger: {logger.name}", 
                                            LogLevel.DEBUG, "operation_container")
            
        except Exception as e:
            import logging
            fallback_logger = logging.getLogger(__name__)
            fallback_logger.error(f"Failed to disable console handlers: {e}")
    
    def _activate_smartcash_ui_mode(self) -> None:
        """Activate UI mode for SmartCashLogger to redirect console output."""
        try:
            from smartcash.common.logger import SmartCashLogger
            
            # Activate UI mode globally for all SmartCashLogger instances
            SmartCashLogger.set_ui_mode(True, self._ui_handler)
            
            self.log("SmartCashLogger UI mode activated - console output redirected", 
                    LogLevel.DEBUG, "operation_container")
                
        except ImportError:
            pass  # SmartCashLogger not available
        except Exception as e:
            import logging
            fallback_logger = logging.getLogger(__name__)
            fallback_logger.error(f"Failed to activate SmartCashLogger UI mode: {e}")
    
    def remove_logging_bridge(self) -> None:
        """Remove the logging bridge and restore normal logging."""
        try:
            import logging
            
            # Remove handler from root logger
            if hasattr(self, '_ui_handler'):
                root_logger = logging.getLogger()
                root_logger.removeHandler(self._ui_handler)
                
                # Remove from captured loggers
                if hasattr(self, '_captured_loggers'):
                    for logger_instance in self._captured_loggers:
                        if hasattr(logger_instance, 'logger'):
                            logger_instance.logger.removeHandler(self._ui_handler)
                    self._captured_loggers.clear()
                
                del self._ui_handler
            
            # Deactivate SmartCashLogger UI mode
            try:
                from smartcash.common.logger import SmartCashLogger
                SmartCashLogger.set_ui_mode(False, None)
            except (ImportError, Exception):
                pass
            
            # Reset bridge status
            self._logging_bridge_setup = False
            
        except Exception as e:
            # Fallback logging
            import logging
            fallback_logger = logging.getLogger(__name__)
            fallback_logger.error(f"Failed to remove logging bridge: {e}")
    
    # ===== Cleanup =====
    
    def clear(self) -> None:
        """Clear all components."""
        self.clear_logs()
        self.reset_progress()
        self.clear_dialog()
    
    def close(self) -> None:
        """Clean up resources."""
        self.clear()
        self.remove_logging_bridge()  # Clean up logging bridge
        if hasattr(self, 'container') and self.container:
            self.container.close()
