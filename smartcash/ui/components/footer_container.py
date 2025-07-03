"""
Footer container component for consistent UI footers across the application.

This module provides a flexible footer container component that combines
progress tracking, logs, and information panels with a consistent layout.
It uses flexbox for responsive design and reuses existing components.
"""

from typing import Dict, Any, Optional, List, Union, Callable
import importlib
import ipywidgets as widgets
from IPython.display import HTML

# Import shared components
from smartcash.ui.components.progress_tracker.progress_tracker import ProgressTracker
from smartcash.ui.components.progress_tracker.progress_config import ProgressConfig
from smartcash.ui.components.log_accordion import create_log_accordion
from smartcash.ui.components.log_accordion import LogLevel
from smartcash.ui.components.info_accordion import create_info_accordion
from smartcash.ui.components.closeable_tips_panel import create_closeable_tips_panel

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
                 log_module_name: str = 'Process',
                 log_height: str = '250px',
                 tips_title: str = "💡 Tips & Requirements",
                 tips_content: Optional[List[Union[str, List[str]]]] = None,
                 info_box_path: Optional[str] = None,
                 info_title: str = 'Information',
                 info_content: str = '',
                 info_style: str = 'info',
                 **style_options):
        """Initialize the footer container.
        
        Args:
            show_progress: Whether to show the progress tracker
            show_logs: Whether to show the log accordion
            show_info: Whether to show the info panel
            show_tips: Whether to show the closeable tips panel above the info box
            progress_config: Configuration for the progress tracker
            log_module_name: Name for the log accordion
            log_height: Height for the log accordion
            tips_title: Title for the tips panel
            tips_content: List of tips for the tips panel
            info_box_path: Path to info box file in smartcash/ui/info_boxes (e.g., 'preprocessing_info')
                           If provided, this takes precedence over info_content
            info_title: Title for the info panel
            info_content: HTML content for the info panel (used if info_box_path is None)
            info_style: Style for the info panel ('info', 'success', 'warning', 'error')
            **style_options: Additional styling options
        """
        self.show_progress = show_progress
        self.show_logs = show_logs
        self.show_info = show_info
        self.show_tips = show_tips
        self.info_box_path = info_box_path
        self.tips_title = tips_title
        self.tips_content = tips_content
        
        # Track progress state internally
        self._progress = 0.0
        
        # Default style options
        self.style = {
            'width': '100%',
            'margin_top': '16px',
            'padding_top': '8px',
            'border_top': '1px solid #e0e0e0'
        }
        
        # Update with custom style options
        self.style.update(style_options)
        
        # Create components
        self._create_components(
            progress_config=progress_config,
            log_module_name=log_module_name,
            log_height=log_height,
            info_title=info_title,
            info_content=info_content,
            info_style=info_style
        )
        
        # Create the container
        self._create_container()
    
    def _create_components(self, 
                          progress_config: Optional[ProgressConfig],
                          log_module_name: str,
                          log_height: str,
                          info_title: str,
                          info_content: str,
                          info_style: str):
        """Create the footer components."""
        # Create progress tracker if enabled
        if self.show_progress:
            self.progress_tracker = ProgressTracker(config=progress_config)
            self.progress_component = self.progress_tracker.container
        else:
            self.progress_tracker = None
            self.progress_component = None
        
        # Create log accordion if enabled
        if self.show_logs:
            log_components = create_log_accordion(
                module_name=log_module_name,
                height=log_height,
                width='100%',
                auto_scroll=True,
                enable_deduplication=True
            )
            # The legacy create_log_accordion returns 'accordion' as the key instead of 'log_accordion'
            self.log_accordion = log_components['accordion']
            # Try to get the output widget from the accordion's children
            if hasattr(self.log_accordion, 'children') and len(self.log_accordion.children) > 0:
                box = self.log_accordion.children[0]
                if hasattr(box, 'children') and len(box.children) > 0:
                    vbox = box.children[0]
                    if hasattr(vbox, 'children') and len(vbox.children) > 0:
                        # The output widget is typically the last child
                        self.log_output = vbox.children[-1]
                        return
            
            # Fallback to using the accordion itself if we can't find the output widget
            self.log_output = widgets.Output()
            
            # Add the output widget to the accordion if it's not already there
            if hasattr(self.log_accordion, 'children') and len(self.log_accordion.children) > 0:
                box = self.log_accordion.children[0]
                if hasattr(box, 'children') and len(box.children) > 0:
                    vbox = box.children[0]
                    if hasattr(vbox, 'children') and len(vbox.children) > 0:
                        vbox.children = list(vbox.children[:-1]) + [self.log_output]
        else:
            self.log_accordion = None
            self.log_output = None
            
        # Create closeable tips panel if enabled
        if self.show_tips:
            tips_components = create_closeable_tips_panel(
                title=self.tips_title,
                tips=self.tips_content,
                margin="0 0 16px 0"  # Add margin below tips panel
            )
            self.tips_panel = tips_components['container']
            self.set_tips_visible = tips_components['set_visible']
        else:
            self.tips_panel = None
            self.set_tips_visible = lambda visible: None  # No-op function
        
        # Create info panel if enabled
        if self.show_info:
            # Map info_style to an appropriate icon
            style_to_icon = {
                'info': 'info',
                'success': 'check_circle',
                'warning': 'warning',
                'error': 'error'
            }
            icon = style_to_icon.get(info_style, 'info')
            
            # Convert string content to HTML widget if needed
            if isinstance(info_content, str):
                info_content = widgets.HTML(value=info_content)
            
            # Load info content from info_boxes if path is provided
            if self.info_box_path:
                try:
                    # Import the module dynamically
                    module_path = f'smartcash.ui.info_boxes.{self.info_box_path}'
                    module = importlib.import_module(module_path)
                    
                    # Get the info box function (convention: get_*_info)
                    function_name = f'get_{self.info_box_path}'
                    if not hasattr(module, function_name):
                        function_name = 'get_info'
                    
                    # Call the function to get the content
                    info_content = getattr(module, function_name)()
                    
                    # Convert string content to HTML widget if needed
                    if isinstance(info_content, str):
                        info_content = widgets.HTML(value=info_content)
                    
                    # Create the info panel with the loaded content and extract the container widget
                    info_accordion = create_info_accordion(
                        title=info_title,
                        content=info_content,
                        icon=icon
                    )
                    self.info_panel = info_accordion['container']
                except (ImportError, AttributeError) as e:
                    print(f"Warning: Could not load info box '{self.info_box_path}': {e}")
                    # Create error accordion with the error message
                    error_message = widgets.HTML(f"<p>Could not load info box: {e}</p>")
                    error_accordion = create_info_accordion(
                        title=f"{info_title} (Error)",
                        content=error_message,
                        icon='warning'
                    )
                    self.info_panel = error_accordion['container']
            else:
                # Use provided content and extract the container widget
                info_accordion = create_info_accordion(
                    title=info_title,
                    content=info_content,
                    icon=icon
                )
                self.info_panel = info_accordion['container']
        else:
            self.info_panel = None
    
    def _create_container(self):
        """Create the main container with all components using vertical flexbox layout."""
        # Filter out None components
        components = []
        if self.progress_component:
            components.append(self.progress_component)
        if self.log_accordion:
            components.append(self.log_accordion)
        
        # For info section, we may need to combine tips and info panel
        if self.show_info and self.show_tips:
            # If both tips and info are shown, create a container for them
            info_section_components = []
            if self.tips_panel:
                info_section_components.append(self.tips_panel)
            if self.info_panel:
                info_section_components.append(self.info_panel)
                
            # Create a container for tips and info
            info_section = widgets.VBox(
                info_section_components,
                layout=widgets.Layout(
                    width='100%',
                    display='flex',
                    flex_direction='column',
                    gap='0px'
                )
            )
            components.append(info_section)
        else:
            # Add components individually if only one is shown
            if self.tips_panel:
                components.append(self.tips_panel)
            if self.info_panel:
                components.append(self.info_panel)
        
        # Always use vertical layout
        self.container = widgets.VBox(
            components,
            layout=widgets.Layout(
                width=self.style.get('width', '100%'),
                margin_top=self.style.get('margin_top', '16px'),
                padding_top=self.style.get('padding_top', '8px'),
                border_top=self.style.get('border_top', '1px solid #e0e0e0'),
                display='flex',
                flex_direction='column',
                gap='8px'
            )
        )
    
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
    
    def log(self, message: str, level: str = 'info') -> None:
        """Add a log message to the log accordion.
        
        Args:
            message: Log message
            level: Log level ('debug', 'info', 'success', 'warning', 'error', 'critical')
        """
        if self.log_output:
            # For legacy log accordion, we need to find the output widget inside the Accordion
            if hasattr(self.log_output, 'children') and len(self.log_output.children) > 0:
                # Get the first child which should be the Box containing the output
                box = self.log_output.children[0]
                if hasattr(box, 'children') and len(box.children) > 0:
                    # Get the VBox inside the Box
                    vbox = box.children[0]
                    if hasattr(vbox, 'children') and len(vbox.children) > 0:
                        # Find the output widget (should be the last child)
                        output_widget = vbox.children[-1]
                        if hasattr(output_widget, 'append_stdout'):
                            output_widget.append_stdout(f"[{level.upper()}] {message}\n")
                            return
            
            # For newer log accordion with LogLevel
            if hasattr(LogLevel, level.upper()):
                log_level = LogLevel[level.upper()]
                if hasattr(self.log_output, 'append_log'):
                    self.log_output.append_log(message, log_level)
                    return
            
            # Fallback to simple logging
            print(f"[{level.upper()}] {message}")
    
    def update_info(self, title: str, content: Union[str, widgets.Widget], style: str = 'info', info_box_path: Optional[str] = None) -> None:
        """Update the info panel content.
        
        Args:
            title: New title for the info panel
            content: New content for the info panel (used if info_box_path is None)
            style: Style for the info panel ('info', 'success', 'warning', 'error')
            info_box_path: Path to info box file in smartcash/ui/info_boxes (e.g., 'preprocessing_info')
                           If provided, this takes precedence over content
        """
        if self.info_panel:
            # Determine if we should use info box path or direct content
            if info_box_path:
                try:
                    # Import the module dynamically
                    module_path = f'smartcash.ui.info_boxes.{info_box_path}'
                    module = importlib.import_module(module_path)
                    
                    # Get the info box function (convention: get_*_info)
                    function_name = f'get_{info_box_path}'
                    if not hasattr(module, function_name):
                        # Try without the 'get_' prefix
                        function_name = info_box_path
                    
                    if hasattr(module, function_name):
                        # Call the function to get the info accordion
                        new_info_panel = getattr(module, function_name)(open_by_default=self.info_panel.selected_index == 0)
                    else:
                        # Fallback to creating a basic info panel with error message
                        error_content = widgets.HTML(value=f"<p>Error: Could not find info box function in {module_path}</p>")
                        new_info_panel = create_info_accordion(
                            title=title,
                            content=error_content,
                            icon='warning'
                        )
                except ImportError as e:
                    # Fallback to creating a basic info panel with error message
                    error_content = widgets.HTML(value=f"<p>Error: Could not load info box from {info_box_path}: {str(e)}</p>")
                    new_info_panel = create_info_accordion(
                        title=title,
                        content=error_content,
                        icon='warning'
                    )
            else:
                # Use provided content
                new_info_panel = create_info_accordion(
                    title=title,
                    content=content,
                    style=style,
                    open_by_default=self.info_panel.selected_index == 0
                )
            
            # Replace the old info panel in the container
            for i, child in enumerate(self.container.children):
                if child is self.info_panel:
                    new_children = list(self.container.children)
                    new_children[i] = new_info_panel
                    self.container.children = tuple(new_children)
                    break
            
            # Update the reference
            self.info_panel = new_info_panel
    
    def show_component(self, component_type: str, show: bool = True) -> None:
        """Show or hide a component.
        
        Args:
            component_type: Type of component ('progress', 'logs', 'info', 'tips')
            show: Whether to show the component
        """
        component = None
        if component_type == 'progress' and self.progress_component:
            component = self.progress_component
        elif component_type == 'logs' and self.log_accordion:
            component = self.log_accordion
        elif component_type == 'info' and self.info_panel:
            component = self.info_panel
        elif component_type == 'tips' and self.tips_panel:
            # Use the set_visible function for tips panel
            self.set_tips_visible(show)
            return
        
        if component:
            component.layout.display = 'flex' if show else 'none'
            
    def update_tips(self, title: str, tips: List[Union[str, List[str]]], visible: bool = True) -> None:
        """Update the tips panel content.
        
        Args:
            title: New title for the tips panel
            tips: New tips content
            visible: Whether to show the tips panel after update
        """
        if self.tips_panel:
            # Create a new tips panel with updated content
            tips_components = create_closeable_tips_panel(
                title=title,
                tips=tips,
                margin="0 0 16px 0",
                initially_visible=visible
            )
            
            # Find the tips panel in the container structure
            # This is more complex because tips might be in a nested container
            if self.show_info and self.show_tips:
                # Tips is in a nested container with info panel
                for i, child in enumerate(self.container.children):
                    # Look for the VBox containing both tips and info
                    if isinstance(child, widgets.VBox) and len(child.children) > 0:
                        if child.children[0] is self.tips_panel:
                            # Replace the tips panel in the VBox
                            new_children = list(child.children)
                            new_children[0] = tips_components['container']
                            child.children = tuple(new_children)
                            break
            else:
                # Tips is directly in the main container
                for i, child in enumerate(self.container.children):
                    if child is self.tips_panel:
                        new_children = list(self.container.children)
                        new_children[i] = tips_components['container']
                        self.container.children = tuple(new_children)
                        break
            
            # Update the references
            self.tips_panel = tips_components['container']
            self.set_tips_visible = tips_components['set_visible']
            
            # Update visibility
            self.set_tips_visible(visible)
    
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
        self.container.remove_class(class_name)

def create_footer_container(
    show_progress: bool = True,
    show_logs: bool = True,
    show_info: bool = True,
    show_tips: bool = False,
    progress_config: Optional[ProgressConfig] = None,
    log_module_name: str = 'Process',
    log_height: str = '250px',
    tips_title: str = "💡 Tips & Requirements",
    tips_content: Optional[List[Union[str, List[str]]]] = None,
    info_box_path: Optional[str] = None,
    info_title: str = 'Information',
    info_content: str = '',
    info_style: str = 'info',
    **style_options
) -> FooterContainer:
    """Create a footer container with progress, logs, info panels, and optional tips panel.
    
    Args:
        show_progress: Whether to show the progress tracker
        show_logs: Whether to show the log accordion
        show_info: Whether to show the info panel
        show_tips: Whether to show the closeable tips panel above the info box
        progress_config: Configuration for the progress tracker
        log_module_name: Name for the log accordion
        log_height: Height for the log accordion
        tips_title: Title for the tips panel
        tips_content: List of tips for the tips panel
        info_box_path: Path to info box file in smartcash/ui/info_boxes (e.g., 'preprocessing_info')
                       If provided, this takes precedence over info_content
        info_title: Title for the info panel
        info_content: HTML content for the info panel (used if info_box_path is None)
        info_style: Style for the info panel ('info', 'success', 'warning', 'error')
        **style_options: Additional styling options
        
    Returns:
        FooterContainer instance
    """
    return FooterContainer(
        show_progress=show_progress,
        show_logs=show_logs,
        show_info=show_info,
        show_tips=show_tips,
        progress_config=progress_config,
        log_module_name=log_module_name,
        log_height=log_height,
        tips_title=tips_title,
        tips_content=tips_content,
        info_box_path=info_box_path,
        info_title=info_title,
        info_content=info_content,
        info_style=info_style,
        **style_options
    )
