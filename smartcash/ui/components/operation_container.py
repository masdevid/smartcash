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
from smartcash.ui.components.dialog import (
    show_confirmation_dialog,
    show_info_dialog,
    clear_dialog_area,
    is_dialog_visible
)

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
        'dialog_area': container.dialog_area if container._show_dialog_enabled else None,
        'log_accordion': container.log_accordion if show_logs else None,
        'show_dialog': container.show_dialog,
        'show_info_dialog': container.show_info_dialog,
        'clear_dialog': container.clear_dialog,
        'update_progress': container.update_progress,
        'log_message': container.log_message
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
                 log_entry_style: str = 'compact',
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
            log_entry_style: Style of log entries ('compact' or 'default')
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
        self.log_entry_style = log_entry_style
        
        # Initialize components
        self.progress_tracker = None
        self.progress_bars = {}
        self.log_accordion = None
        self.dialogs = {}
        self.dialog_area = None
        
        # Create main container widget first
        self._create_container()
        
        # Initialize dialog area if enabled (must be before creating UI components)
        if self._show_dialog_enabled:
            self._init_dialog_area()
            
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

        # 2. Add dialog area (middle) - will be populated when needed
        if hasattr(self, 'dialog_area') and self.dialog_area is not None:
            children += (self.dialog_area,)

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
            # Create progress tracker with the specified number of levels
            self.progress_tracker = ProgressTracker(
                component_name=f"{self.component_name}_progress",
                config=progress_config or ProgressConfig()
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
                log_entry_style=self.log_entry_style
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
            'padding': '0',
            'border': '1px solid #ddd',
            'border_radius': '8px',
            'display': 'flex',
            'flex_flow': 'column',
            'align_items': 'stretch',
            'overflow': 'hidden'
        }

        # Create container with layout
        self.container.children = []
        self.container.layout = widgets.Layout(**layout_kwargs)

        # 1. Add progress tracker if enabled (top)
        if self.progress_tracker and hasattr(self.progress_tracker, 'container') and self.progress_tracker.container is not None:
            self.container.children += [self.progress_tracker.container]

        # 2. Add dialog area (middle) - will be populated when needed
        if hasattr(self, 'dialog_area') and self.dialog_area is not None:
            self.container.children += [self.dialog_area]

        # 3. Add log accordion if enabled (bottom) - use helper for DRY approach
        if self.log_accordion:
            log_widget = self._get_log_accordion_widget()
            if log_widget is not None:
                self.container.children += [log_widget]
        
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

        # 2. Add dialog area (middle) - will be populated when needed
        if hasattr(self, 'dialog_area') and self.dialog_area is not None:
            children += (self.dialog_area,)

        # 3. Add log accordion if enabled (bottom) - use helper for DRY approach
        if self.log_accordion:
            log_widget = self._get_log_accordion_widget()
            if log_widget is not None:
                children += (log_widget,)

        # Create the main container with all valid components
        self.container = widgets.VBox(children=children, layout=layout)
    
    def _init_dialog_area(self) -> None:
        """Initialize the dialog area."""
        self.dialog_area = widgets.Output(
            layout=widgets.Layout(
                width='100%',
                display='none',  # Hidden by default
                margin='10px 0',
                padding='10px',
                border_radius='5px',
                background='rgba(255, 255, 255, 0.95)',
                box_shadow='0 4px 6px rgba(0, 0, 0, 0.1)',
                border='1px solid #e0e0e0',
                z_index=1000  # Ensure it appears above other content
            )
        )
        
        # The dialog area will be added to the container in _create_container
    
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
        if not self.progress_tracker or level not in self.progress_bars:
            return
        
        # Ensure progress tracker is visible when updating progress
        if not self.progress_tracker._initialized:
            self.progress_tracker.initialize()
        
        # Show progress tracker if it's hidden
        if hasattr(self.progress_tracker, 'container') and self.progress_tracker.container:
            if self.progress_tracker.container.layout.display == 'none':
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
                # Ensure tracker is visible
                if hasattr(self.progress_tracker, 'container') and self.progress_tracker.container:
                    if self.progress_tracker.container.layout.display == 'none':
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
                                        level,
                                        f"❌ {data.get('message', 'Error')}"
                                    )
                            else:
                                # Normal progress update
                                self.progress_tracker.set_progress(
                                    data.get('value', 0), 
                                    level,
                                    data.get('message', '')
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
        """Show a confirmation dialog with enhanced integration.
        
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
        if not self.dialog_area:
            # Log fallback if dialog area is not available
            self.log(f"Dialog requested: {title} - {message}", LogLevel.INFO)
            if on_confirm:
                on_confirm()
            return False
        
        try:
            # Clear any existing dialog
            self.clear_dialog()
            
            # Enhanced callback wrappers with logging
            def enhanced_confirm():
                try:
                    self.log(f"Dialog confirmed: {title}", LogLevel.INFO)
                    if on_confirm:
                        on_confirm()
                except Exception as e:
                    self.log(f"Error in dialog confirm callback: {e}", LogLevel.ERROR)
                finally:
                    self.clear_dialog()
                    
            def enhanced_cancel():
                try:
                    self.log(f"Dialog cancelled: {title}", LogLevel.INFO)
                    if on_cancel:
                        on_cancel()
                except Exception as e:
                    self.log(f"Error in dialog cancel callback: {e}", LogLevel.ERROR)
                finally:
                    self.clear_dialog()
            
            # Create dialog with enhanced callbacks
            dialog = SimpleDialog(
                component_name=f"dialog_{title}",
                title=title,
                message=message,
                on_confirm=enhanced_confirm,
                on_cancel=enhanced_cancel,
                confirm_text=confirm_text,
                cancel_text=cancel_text,
                danger_mode=danger_mode
            )
            
            # Store dialog reference and display
            self.dialogs[title] = dialog
            self.dialog_area.children = (dialog,)
            self.dialog_area.layout.display = 'flex'
            
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
                 ok_text: str = "OK") -> bool:
        """Show an info dialog with enhanced integration.
        
        Args:
            title: Dialog title
            message: Dialog message
            on_ok: Callback when user clicks OK
            ok_text: Text for OK button
            
        Returns:
            True if dialog was shown successfully, False otherwise
        """
        if not self.dialog_area:
            # Log fallback if dialog area is not available
            self.log(f"Info dialog requested: {title} - {message}", LogLevel.INFO)
            if on_ok:
                on_ok()
            return False
        
        try:
            # Clear any existing dialog
            self.clear_dialog()
            
            # Enhanced callback wrapper with logging
            def enhanced_ok():
                try:
                    self.log(f"Info dialog acknowledged: {title}", LogLevel.INFO)
                    if on_ok:
                        on_ok()
                except Exception as e:
                    self.log(f"Error in info dialog callback: {e}", LogLevel.ERROR)
                finally:
                    self.clear_dialog()
            
            # Use dialog area with enhanced callbacks
            with self.dialog_area:
                clear_dialog_area({'dialog_area': self.dialog_area})
                self.dialog_area.layout.display = 'flex'
                
                show_info_dialog(
                    {'dialog_area': self.dialog_area},
                    title=title,
                    message=message,
                    on_ok=enhanced_ok,
                    ok_text=ok_text
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
        if not self.dialog_area:
            return False
            
        # Hide any visible dialog
        for dialog in self.dialogs.values():
            if hasattr(dialog, 'hide'):
                dialog.hide()
        
        # Clear dialog area
        self.dialog_area.children = ()
        self.dialog_area.layout.display = 'none'
        
        # Log dialog clearing
        self.log("Dialog cleared", LogLevel.DEBUG)
        return True
    
    def is_dialog_visible(self) -> bool:
        """Check if a dialog is currently visible.
        
        Returns:
            True if any dialog is visible, False otherwise
        """
        if not self.dialog_area:
            return False
            
        # Check if dialog area has any children and is visible
        return bool(self.dialog_area.children) and self.dialog_area.layout.display != 'none'
    
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
            fallback_logger.log(getattr(logging, level.value.upper(), logging.INFO), f"[FALLBACK] {message}")
            fallback_logger.error(f"Log accordion error: {e}")
    
    # Alias for backward compatibility
    log_message = log
    
    def debug(self, message: str) -> None:
        """Log a debug message."""
        self.log(message, LogLevel.DEBUG)
    
    def info(self, message: str) -> None:
        """Log an info message."""
        self.log(message, LogLevel.INFO)
    
    def warning(self, message: str) -> None:
        """Log a warning message."""
        self.log(message, LogLevel.WARNING)
    
    def error(self, message: str) -> None:
        """Log an error message."""
        self.log(message, LogLevel.ERROR)
    
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
            if not self.dialog_area:
                status['healthy'] = False
                status['issues'].append("Dialog area not initialized")
                return status
            
            # Check dialog area layout
            if not hasattr(self.dialog_area, 'layout'):
                status['issues'].append("Dialog area missing layout")
                status['healthy'] = False
            
            status['details'] = {
                'dialog_area_available': self.dialog_area is not None,
                'active_dialogs': len(self.dialogs),
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
            if self._show_dialog_enabled and self.dialog_area:
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
    
    # ===== Additional Dialog Methods =====
    
    def close_dialog(self, title: str) -> None:
        """Close a dialog by title.
        
        Args:
            title: Title of the dialog to close
        """
        if title in self.dialogs:
            self.dialogs[title].close()
            del self.dialogs[title]
    
    # ===== Cleanup =====
    
    def clear(self) -> None:
        """Clear all components."""
        self.clear_logs()
        self.reset_progress()
        self.clear_dialog()
        if hasattr(self, 'info_panel') and self.info_panel:
            self.info_panel.clear_output()
        for dialog in list(self.dialogs.values()):
            dialog.close()
        self.dialogs.clear()
    
    def close(self) -> None:
        """Clean up resources."""
        self.clear()
        if hasattr(self, 'container') and self.container:
            self.container.close()
