"""
Footer container component for consistent UI footers across the application.

This module provides a flexible footer container component that combines
progress tracking, logs, and information panels with a consistent layout.
It uses flexbox for responsive design and reuses existing components.
"""

from typing import Dict, Any, Optional, List, Union, Callable
import ipywidgets as widgets
from IPython.display import HTML

# Import shared components
from smartcash.ui.components.progress_tracker.progress_tracker import ProgressTracker
from smartcash.ui.components.progress_tracker.progress_config import ProgressConfig
from smartcash.ui.components.log_accordion import LogLevel
from smartcash.ui.components.info.info_component import InfoAccordion, InfoBox

class FooterContainer:
    """Flexible footer container with progress, logs, and info panels.
    
    This class provides a footer container with progress tracking, logs, and
    information panels that can be configured and updated dynamically.
    It uses flexbox layout for responsive design.
    """
    
    def __init__(self,
                 show_progress: bool = True,
                 show_logs: bool = True,
                 show_info: bool = True,
                 show_tips: bool = False,
                 progress_config: Optional[ProgressConfig] = None,
                 log_module_name: str = "SmartCash",
                 log_height: str = "200px",
                 info_title: str = "Information",
                 info_content: str = "",
                 info_style: str = "info",
                 info_box_path: Optional[str] = None,
                 tips_title: str = "Tips",
                 tips_content: Optional[List[Dict[str, str]]] = None,
                 style: Optional[Dict[str, str]] = None):
        """Initialize the footer container.
        
        Args:
            show_progress: Whether to show the progress tracker
            show_logs: Whether to show the log accordion
            show_info: Whether to show the info panel
            show_tips: Whether to show the tips panel
            progress_config: Configuration for the progress tracker
            log_module_name: Name of the module for logging
            log_height: Height of the log accordion
            info_title: Title for the info panel
            info_content: Content for the info panel (string or HTML widget)
            info_style: Style for the info panel ('info', 'success', 'warning', 'error')
            info_box_path: Optional path to info box module
            tips_title: Title for the tips panel
            tips_content: List of tips to display
            style: CSS styles for the container
        """
        super().__init__()
        
        # Store visibility flags
        self.show_progress = show_progress
        self.show_logs = show_logs
        self.show_info = show_info
        self.show_tips = show_tips
        self._tips_visible = False
        self.tips_title = tips_title
        self.tips_content = tips_content or []
        
        # Store style
        self.style = style or {}
        
        # Store info box path for later use
        self.info_box_path = info_box_path
        
        # Initialize components and container
        self.progress_tracker = None
        self.progress_component = None
        self.log_accordion = None
        self.log_output = None
        self.info_panel = None
        self.tips_panel = None
        self._info_content = info_content  # Store raw content for testing
        self.container = None  # Initialize container to avoid attribute errors
        self._creating_container = False  # Flag to prevent recursion
        
        # Initialize progress component with layout
        if show_progress:
            self.progress_component = widgets.HBox(layout=widgets.Layout(display='flex' if show_progress else 'none'))
        
        # Create components
        self._create_components(
            progress_config=progress_config,
            log_module_name=log_module_name,
            log_height=log_height,
            info_title=info_title,
            info_content=info_content,
            info_style=info_style
        )
        
        # Initialize tips if enabled
        if show_tips:
            self._init_tips(tips_title, tips_content)
    
    def _create_components(self, 
                         progress_config: Optional[ProgressConfig],
                         log_module_name: str,
                         log_height: str,
                         info_title: str,
                         info_content: str,
                         info_style: str):
        """Create and initialize all child components.
        
        Args:
            progress_config: Configuration for the progress tracker
            log_module_name: Name of the module for logging
            log_height: Height of the log accordion
            info_title: Title for the info panel
            info_content: Content for the info panel (string or HTML widget)
            info_style: Style for the info panel
        """
        # Create progress tracker if enabled
        if self.show_progress:
            if self.progress_tracker is None:
                self.progress_tracker = ProgressTracker(config=progress_config)
                self.progress_component = self.progress_tracker.container
                if hasattr(self.progress_component, 'layout'):
                    self.progress_component.layout.display = 'flex' if self.show_progress else 'none'
        else:
            self.progress_tracker = None
            self.progress_component = None
        
        # Create log accordion if enabled
        if self.show_logs:
            if self.log_accordion is None:
                self.log_output = widgets.Output(layout={
                    'border': '1px solid #e0e0e0',
                    'height': log_height,
                    'overflow_y': 'auto',
                    'display': 'flex' if self.show_logs else 'none'
                })
                self.log_accordion = InfoAccordion(
                    title=f"{log_module_name} Logs",
                    content=self.log_output,
                    style='info',
                    open_by_default=False
                )
                if hasattr(self.log_accordion, 'layout'):
                    self.log_accordion.layout.display = 'flex' if self.show_logs else 'none'
        else:
            self.log_accordion = None
            self.log_output = None
        
        # Create info panel if enabled
        if self.show_info:
            if self.info_panel is None:
                # Store raw content for testing
                self._info_content = info_content
                
                # Store raw content for testing
                self._info_content = info_content
                content_to_use = info_content
                
                # Handle info box path if provided
                if self.info_box_path:
                    try:
                        # Dynamically import the info box module
                        import importlib
                        info_box_module = importlib.import_module(self.info_box_path)
                        
                        # Get the content from the module
                        if hasattr(info_box_module, 'get_info_content'):
                            content_widget = info_box_module.get_info_content()
                            if hasattr(content_widget, 'value'):
                                content_to_use = content_widget.value
                                self._info_content = content_to_use
                            else:
                                content_to_use = str(content_widget)
                                self._info_content = content_to_use
                        else:
                            error_msg = f"Info box module {self.info_box_path} does not have get_info_content() function"
                            content_to_use = error_msg
                            info_style = 'warning'
                            self._info_content = error_msg
                    except Exception as e:
                        error_msg = f"Failed to load info box from {self.info_box_path}: {str(e)}"
                        content_to_use = error_msg
                        info_style = 'warning'
                        self._info_content = error_msg
                
                # Create the info panel with default title if none provided
                self.info_panel = InfoBox(
                    content=content_to_use,
                    title=info_title or "Information",
                    style=info_style
                )
                # Ensure layout exists before accessing it
                if hasattr(self.info_panel, 'layout'):
                    self.info_panel.layout = self.info_panel.layout or {}
                    self.info_panel.layout.display = 'flex' if self.show_info else 'none'
        else:
            self.info_panel = None
            self._info_content = ""
        
        # Initialize tips panel if not already done
        if self.show_tips and self.tips_panel is None:
            self.tips_panel = widgets.HTML()
            self.tips_panel.layout = widgets.Layout(
                display='flex' if getattr(self, '_tips_visible', False) else 'none'
            )
        
        # Create container after components are initialized
        self._create_container()
    
    def _update_container_children(self):
        """Update the container's children based on current visibility states."""
        if not hasattr(self, 'container') or self.container is None:
            self.container = widgets.VBox(
                layout=widgets.Layout(
                    width='100%',
                    display='flex',
                    flex_flow='column',
                    align_items='stretch',
                    margin='0',
                    padding='0',
                    border='none',
                    **self.style
                )
            )
        
        components = []
        
        # Add progress component if enabled and exists
        if self.show_progress and self.progress_component is not None:
            widget = self.progress_component
            if hasattr(widget, 'show'):
                widget = widget.show()
            components.append(widget)
        
        # Add log accordion if enabled and exists
        if self.show_logs and self.log_accordion is not None:
            widget = self.log_accordion
            if hasattr(widget, 'show'):
                widget = widget.show()
            components.append(widget)
        
        # Add info panel if enabled and exists
        if self.show_info and self.info_panel is not None:
            widget = self.info_panel
            if hasattr(widget, 'show'):
                widget = widget.show()
            components.append(widget)
        
        # Add tips panel if enabled and exists
        if self.show_tips and self.tips_panel is not None:
            widget = self.tips_panel
            if hasattr(widget, 'show'):
                widget = widget.show()
            components.append(widget)
        
        # Update container children with proper widget instances
        self.container.children = tuple(components)
    
    def _create_container(self):
        """Create the main container with all components."""
        # Initialize container if it doesn't exist
        if not hasattr(self, 'container') or self.container is None:
            self.container = widgets.VBox(
                layout=widgets.Layout(
                    width='100%',
                    display='flex',
                    flex_flow='column',
                    align_items='stretch',
                    margin='0',
                    padding='0',
                    border='none',
                    **self.style
                )
            )
        
        # Update container children
        self._update_container_children()
    
    def update_progress_status(self, message: str, level: str = 'main') -> None:
        """Update the progress tracker status message.
        
        Args:
            message: New status message
            level: Progress level to update ('main', 'step', etc.)
        """
        if self.progress_tracker:
            self.progress_tracker.update_status(message, level)
    
    def update_progress(self, value: float, total: float = 100.0, level: str = 'main') -> None:
        """Update the progress tracker value.
        
        Args:
            value: Current progress value
            total: Total progress value
            level: Progress level to update ('main', 'step', etc.)
        """
        # Update internal progress state (as a percentage 0-100)
        if total > 0:
            self._progress = (value / total) * 100.0
        else:
            self._progress = 0.0
            
        if self.progress_tracker:
            # Check if the progress tracker has a set_progress method (newer version)
            if hasattr(self.progress_tracker, 'set_progress'):
                self.progress_tracker.set_progress(value, total, level)
            # Fallback to update method (older version)
            elif hasattr(self.progress_tracker, 'update'):
                self.progress_tracker.update(level, value, message=f"{value/total*100:.1f}%" if total > 0 else "0%")
    
    def get_progress(self) -> float:
        """Get the current progress value.
        
        Returns:
            float: Current progress value (0-100)
        """
        # Return the internal progress state
        return self._progress
    
    def complete(self) -> None:
        """Mark the current progress as complete (100%).
        
        This is a convenience method for test compatibility.
        """
        self._progress = 100.0  # Explicitly set internal state
        if self.progress_tracker:
            # Try different methods to update progress
            if hasattr(self.progress_tracker, 'set_progress'):
                self.progress_tracker.set_progress(100, 100, 'main')
            elif hasattr(self.progress_tracker, 'update'):
                self.progress_tracker.update('main', 100, message="100%")
            
    def reset(self) -> None:
        """Reset the progress to 0%.
        
        This is a convenience method for test compatibility.
        """
        self._progress = 0.0  # Explicitly set internal state
        if self.progress_tracker:
            # Try different methods to update progress
            if hasattr(self.progress_tracker, 'set_progress'):
                self.progress_tracker.set_progress(0, 100, 'main')
            elif hasattr(self.progress_tracker, 'update'):
                self.progress_tracker.update('main', 0, message="0%")
    
    def log(self, message: str, level: LogLevel = LogLevel.INFO) -> None:
        """Log a message to the log accordion.
        
        Args:
            message: The message to log
            level: The log level (default: INFO)
        """
        if self.log_output:
            with self.log_output:
                print(f"[{level.name}] {message}")
    
    def update_info(self, title: str, content: Union[str, widgets.HTML], style: str = 'info') -> None:
        """Update the info panel with new content.
        
        Args:
            title: Title for the info panel
            content: Content to display (string or HTML widget)
            style: Style for the info panel ('info', 'success', 'warning', 'error')
        """
        # Ensure info_panel exists
        if self.info_panel is None:
            from smartcash.ui.components.info.info_component import InfoBox
            self.info_panel = InfoBox(
                title=title,
                content="",
                style=style
            )
        
        # Store the content for testing
        if isinstance(content, widgets.HTML):
            self._info_content = content.value if hasattr(content, 'value') else str(content)
            content_widget = content
        else:
            self._info_content = str(content)
            content_widget = widgets.HTML(value=str(content))
        
        # Update the info panel
        if hasattr(self.info_panel, 'update_content'):
            # For InfoBox component
            self.info_panel.update_content(self._info_content)
            if hasattr(self.info_panel, 'update_style'):
                self.info_panel.update_style(style)
        elif hasattr(self.info_panel, 'content'):
            # For widgets with content attribute
            self.info_panel.content = content_widget
            if hasattr(self.info_panel, 'style'):
                self.info_panel.style = style
        
        # Update title if supported
        if hasattr(self.info_panel, 'title'):
            self.info_panel.title = title
            
        # Ensure the panel is visible and has a layout
        if not hasattr(self.info_panel, 'layout') or self.info_panel.layout is None:
            self.info_panel.layout = widgets.Layout()
            
        # Ensure the panel is visible
        self.show_component('info', True)
        
        # For testing purposes, make sure we can access the raw content
        if not hasattr(self, '_info_content'):
            if isinstance(content, widgets.HTML):
                self._info_content = content.value if hasattr(content, 'value') else str(content)
            else:
                self._info_content = str(content)
                
    def show_component(self, component_name: str, visible: bool = True):
        """Show or hide a component in the footer.
        
        Args:
            component_name: Name of the component to show/hide ('progress', 'logs', 'info', 'tips')
            visible: Whether to show (True) or hide (False) the component
        """
        component_name = component_name.lower()
        component = None
        
        # Update the appropriate visibility flag and get the component
        if component_name == 'progress':
            self.show_progress = visible
            # Ensure progress component exists and has a layout
            if self.progress_component is None:
                self.progress_component = widgets.HBox()
                self.progress_component.layout = widgets.Layout(display='flex' if visible else 'none')
            component = self.progress_component
        elif component_name == 'logs':
            self.show_logs = visible
            component = self.log_accordion
            # Ensure log accordion exists
            if component is None and visible and hasattr(self, '_create_log_accordion'):
                self._create_log_accordion()
                component = self.log_accordion
        elif component_name == 'info':
            self.show_info = visible
            component = self.info_panel
            # Ensure info panel exists
            if component is None and visible:
                from smartcash.ui.components.info.info_component import InfoBox
                self.info_panel = InfoBox(title="", content="", style="info")
                component = self.info_panel
        elif component_name == 'tips':
            self.show_tips = visible
            self._tips_visible = visible
            component = self.tips_panel
            # Ensure tips panel exists if needed
            if component is None and visible and hasattr(self, '_init_tips'):
                self._init_tips()
                component = self.tips_panel
        else:
            return
            
        # Update the component's visibility if it exists
        if component is not None:
            # Ensure the component has a layout
            if not hasattr(component, 'layout') or component.layout is None:
                component.layout = widgets.Layout()
            
            # Set the display style
            if hasattr(component.layout, 'display'):
                component.layout.display = 'flex' if visible else 'none'
        
        # Update the container if it exists
        if hasattr(self, 'container') and self.container is not None:
            self._update_container_children()
    
    def set_tips_visible(self, visible: bool):
        """Set the visibility of the tips panel.
        
        Args:
            visible: Whether the tips panel should be visible
        """
        self.show_tips = True  # Keep the component in the layout
        self._tips_visible = visible  # Control visibility with display style
        
        if self.tips_panel is None and visible:
            self.tips_panel = widgets.HTML()
            self.tips_panel.layout = widgets.Layout(
                display='flex' if visible else 'none',
                visibility='visible' if visible else 'hidden'
            )
        elif self.tips_panel is not None:
            self.tips_panel.layout.display = 'flex' if visible else 'none'
            
        # Update the container to reflect changes
        self._create_container()
    
    def add_class(self, class_name: str) -> None:
        """Add a CSS class to the container.
        
        Args:
            class_name: CSS class name to add
        """
        self.container.add_class(class_name)
    
    def remove_class(self, class_name: str) -> None:
        """Remove a CSS class from the container.
        
        Args:
            class_name: CSS class name to remove
        """
        if hasattr(self, 'container') and hasattr(self.container, 'remove_class'):
            self.container.remove_class(class_name)
    
    @property
    def info_content(self) -> str:
        """Get the current info content as a string.
        
        Returns:
            str: The raw string content of the info panel
        """
        # Return the stored raw content if available
        if hasattr(self, '_info_content') and self._info_content is not None:
            return self._info_content
            
        # Fallback to extracting from the info panel
        if self.info_panel is not None:
            # For InfoBox with content attribute
            if hasattr(self.info_panel, 'content'):
                content = self.info_panel.content
                if hasattr(content, 'value'):
                    return content.value
                return str(content) if content is not None else ""
            # For widgets with value attribute
            if hasattr(self.info_panel, 'value'):
                value = self.info_panel.value
                return value if isinstance(value, str) else str(value)
                
        return ""

def create_footer_container(
    show_progress: bool = False,
    show_logs: bool = False,
    show_info: bool = False,
    show_tips: bool = False,
    progress_config: Optional[ProgressConfig] = None,
    log_module_name: str = "Module",
    log_height: str = "200px",
    info_title: str = "Information",
    info_content: str = "",
    info_style: str = "info",
    info_box_path: Optional[str] = None,
    tips_title: str = "Tips",
    tips_content: Optional[List[Dict[str, str]]] = None,
    style: Optional[Dict] = None,
) -> FooterContainer:
    """Create a footer container with the specified components.
    
    Args:
        show_progress: Whether to show the progress tracker
        show_logs: Whether to show the log accordion
        show_info: Whether to show the info panel
        progress_config: Configuration for the progress tracker
        log_module_name: Name of the module for the log accordion
        log_height: Height of the log output
        info_title: Title for the info panel
        info_content: Initial content for the info panel
        info_style: Style for the info panel
        info_box_path: Optional path to a module containing a get_info_content() function
        style: Additional CSS styles for the container
        
    Returns:
        FooterContainer: The created footer container
    """
    footer = FooterContainer(
        show_progress=show_progress,
        show_logs=show_logs,
        show_info=show_info,
        show_tips=show_tips,
        progress_config=progress_config,
        log_module_name=log_module_name,
        log_height=log_height,
        info_title=info_title,
        info_content=info_content,
        info_style=info_style,
        info_box_path=info_box_path,
        tips_title=tips_title,
        tips_content=tips_content,
        style=style or {}
    )
    
    # Ensure the container is created
    if not hasattr(footer, 'container') or footer.container is None:
        footer._create_container()
    
    # Make sure the container is visible and has a layout
    if hasattr(footer, 'container') and footer.container is not None:
        if not hasattr(footer.container, 'layout') or footer.container.layout is None:
            footer.container.layout = widgets.Layout()
        footer.container.layout.display = 'flex'
    
    # Initialize tips panel if needed
    if hasattr(footer, 'show_tips') and footer.show_tips and footer.tips_panel is None:
        footer._init_tips(footer.tips_title, footer.tips_content)
        
    # Ensure the info panel is created if show_info is True
    if footer.show_info and footer.info_panel is None:
        footer.info_panel = InfoBox(
            title=footer.info_title if hasattr(footer, 'info_title') else "Information",
            content=footer._info_content if hasattr(footer, '_info_content') else "",
            style=footer.info_style if hasattr(footer, 'info_style') else "info"
        )
        
    return footer
