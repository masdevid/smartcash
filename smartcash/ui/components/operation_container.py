"""
Operation Container Component

This module provides a unified interface for operation-related UI components,
including action buttons, dialogs, progress tracking, and logging.
"""

from typing import Dict, Any, Optional, List, Callable, Union, Literal
import ipywidgets as widgets
from IPython.display import display

# Import components
from smartcash.ui.components.progress_tracker.progress_tracker import ProgressTracker, ProgressConfig
from smartcash.ui.components.progress_tracker.types import ProgressLevel, ProgressConfig
from smartcash.ui.components.log_accordion import LogAccordion, LogLevel
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
    **kwargs
) -> Dict[str, Any]:
    """Create an operation container with the specified components.
    
    Args:
        show_progress: Whether to show the progress tracker
        show_dialog: Whether to enable dialog functionality
        show_logs: Whether to show the log accordion
        log_module_name: Name to display in the log accordion header
        **kwargs: Additional arguments to pass to OperationContainer
        
    Returns:
        Dict containing the container and its components
    """
    container = OperationContainer(
        show_progress=show_progress,
        show_dialog=show_dialog,
        show_logs=show_logs,
        log_module_name=log_module_name,
        **kwargs
    )
    
    return {
        'container': container.container,
        'progress_tracker': container.progress_tracker if show_progress else None,
        'dialog': container.dialog if show_dialog else None,
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
    """
    
    def __init__(self,
                 component_name: str = "operation_container",
                 progress_levels: Literal['single', 'dual', 'triple'] = 'single',
                 show_progress: bool = True,
                 show_logs: bool = True,
                 progress_config: Optional[ProgressConfig] = None,
                 log_module_name: str = "Operation",
                 log_height: str = "300px",
                 **kwargs):
        """Initialize the OperationContainer.
        
        Args:
            component_name: Unique name for this component
            progress_levels: Number of progress levels to show ('single', 'dual', or 'triple')
            show_progress: Whether to show the progress tracker
            show_logs: Whether to show the log accordion
            progress_config: Configuration for the progress tracker
            log_module_name: Name of the module for logging
            log_height: Height of the log accordion
            **kwargs: Additional arguments for BaseUIComponent
        """
        super().__init__(component_name, **kwargs)
        
        # Store configuration
        self.progress_levels = progress_levels
        self.show_progress = show_progress
        self.show_logs = show_logs
        self.log_module_name = log_module_name
        self.log_height = log_height
        
        # Initialize components
        self.progress_tracker = None
        self.progress_bars = {}
        self.log_accordion = None
        self.dialogs = {}
        self.dialog_area = None
        
        # Create UI components
        self._create_components(progress_config)
        
        # Initialize dialog area (must be before creating container)
        self._init_dialog_area()
    
    def _create_components(self, progress_config: Optional[ProgressConfig] = None) -> None:
        """Create and initialize UI components."""
        # Create progress tracker if enabled
        if self.show_progress:
            # Create progress tracker with the specified number of levels
            self.progress_tracker = ProgressTracker(
                component_name=f"{self.component_name}_progress",
                config=progress_config or ProgressConfig()
            )
            
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
                component_name=f"{self.component_name}_logs",
                module_name=self.log_module_name,
                height=self.log_height
            )
        
        # Create main container
        self._create_container()
    
    def _create_container(self) -> None:
        """Create the main container widget.
        
        Components are ordered as:
        1. Progress Tracker (top)
        2. Dialog Area (middle)
        3. Log Accordion (bottom)
        """
        children = []
        
        # 1. Add progress tracker if enabled (top)
        if self.progress_tracker:
            children.append(self.progress_tracker.container)
        
        # 2. Add dialog area (middle) - will be populated when needed
        if hasattr(self, 'dialog_area'):
            children.append(self.dialog_area)
        
        # 3. Add log accordion if enabled (bottom)
        if self.log_accordion:
            children.append(self.log_accordion.container)
        
        # Create the main container with all components in order
        self.container = widgets.VBox(
            children=children,
            layout=widgets.Layout(
                width='100%',
                margin='10px 0',
                padding='10px',
                border='1px solid #e0e0e0',
                border_radius='5px',
                flex='1 1 auto',
                display='flex',
                flex_flow='column',
                position='relative'
            )
        )
    
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
            
        for level, data in self.progress_bars.items():
            if data['visible']:
                if data.get('error', False):
                    self.progress_tracker.error_progress(data['message'], level)
                else:
                    self.progress_tracker.update_progress(
                        data['value'], 
                        data['message'], 
                        level
                    )
            else:
                # Hide the progress bar
                self.progress_tracker.hide_progress(level)
    
    # ===== Dialog Methods =====
    
    def show_dialog(self, 
                   title: str, 
                   message: str, 
                   on_confirm: Optional[Callable] = None,
                   on_cancel: Optional[Callable] = None,
                   confirm_text: str = "Confirm",
                   cancel_text: str = "Cancel",
                   danger_mode: bool = False) -> None:
        """Show a confirmation dialog.
        
        Args:
            title: Dialog title
            message: Dialog message
            on_confirm: Callback when user confirms
            on_cancel: Callback when user cancels
            confirm_text: Text for confirm button
            cancel_text: Text for cancel button
            danger_mode: If True, shows the confirm button in danger color
        """
        if not self.dialog_area:
            return
            
        with self.dialog_area:
            clear_dialog_area({'dialog_area': self.dialog_area})
            self.dialog_area.layout.display = 'flex'
            
            def wrapped_confirm():
                if on_confirm:
                    on_confirm()
                self.clear_dialog()
                
            def wrapped_cancel():
                if on_cancel:
                    on_cancel()
                self.clear_dialog()
            
            show_confirmation_dialog(
                {'dialog_area': self.dialog_area},
                title=title,
                message=message,
                on_confirm=wrapped_confirm,
                on_cancel=wrapped_cancel,
                confirm_text=confirm_text,
                cancel_text=cancel_text,
                danger_mode=danger_mode
            )
    
    def show_info(self, 
                 title: str, 
                 message: str, 
                 on_ok: Optional[Callable] = None,
                 ok_text: str = "OK") -> None:
        """Show an info dialog.
        
        Args:
            title: Dialog title
            message: Dialog message
            on_ok: Callback when user clicks OK
            ok_text: Text for OK button
        """
        if not self.dialog_area:
            return
            
        with self.dialog_area:
            clear_dialog_area({'dialog_area': self.dialog_area})
            self.dialog_area.layout.display = 'flex'
            
            def wrapped_ok():
                if on_ok:
                    on_ok()
                self.clear_dialog()
            
            show_info_dialog(
                {'dialog_area': self.dialog_area},
                title=title,
                message=message,
                on_ok=wrapped_ok,
                ok_text=ok_text
            )
    
    def clear_dialog(self) -> None:
        """Clear any currently displayed dialog."""
        if self.dialog_area:
            clear_dialog_area({'dialog_area': self.dialog_area})
            self.dialog_area.layout.display = 'none'
    
    def is_dialog_visible(self) -> bool:
        """Check if a dialog is currently visible."""
        if not self.dialog_area:
            return False
        return is_dialog_visible({'dialog_area': self.dialog_area})
    
    # ===== Logging Methods =====
    
    def log(self, message: str, level: LogLevel = LogLevel.INFO) -> None:
        """Log a message with the specified level.
        
        Args:
            message: The message to log
            level: The log level (default: INFO)
        """
        if self.log_accordion:
            self.log_accordion.log(message, level)
    
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
    
    def critical(self, message: str) -> None:
        """Log a critical message."""
        self.log(message, LogLevel.CRITICAL)
        
    def clear_logs(self) -> None:
        """Clear all log messages."""
        if self.log_accordion:
            self.log_accordion.clear()
    
    # ===== Dialogs =====
    
    def show_dialog(self, 
                   title: str, 
                   content: Union[str, widgets.Widget],
                   buttons: Optional[List[Dict[str, Any]]] = None) -> None:
        """Show a modal dialog.
        
        Args:
            title: Dialog title
            content: Dialog content (text or widget)
            buttons: List of button definitions, each with 'label' and 'callback'
        """
        # Create dialog content
        if isinstance(content, str):
            content_widget = widgets.HTML(value=content)
        else:
            content_widget = content
        
        # Create dialog
        dialog = widgets.Output(layout={"border": "1px solid black"})
        
        with dialog:
            print(title)
            print("-" * len(title))
            display(content_widget)
            
            # Add buttons if provided
            if buttons:
                button_row = widgets.HBox([
                    widgets.Button(description=btn["label"], 
                                 button_style=btn.get("style", ""),
                                 layout=widgets.Layout(margin='0 5px'))
                    for btn in buttons
                ])
                
                # Add button click handlers
                for btn_widget, btn_def in zip(button_row.children, buttons):
                    if "callback" in btn_def:
                        btn_widget.on_click(lambda _, b=btn_def: b["callback"]())
                
                display(button_row)
        
        # Store dialog reference
        self.dialogs[title] = dialog
        
        # Show the dialog
        display(dialog)
    
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
