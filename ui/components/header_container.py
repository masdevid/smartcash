"""
Header container component for consistent UI headers across the application.

This module provides a flexible header container component that combines
a title header and status panel with methods to update their content.
It reuses the existing header and status panel components for consistency.
"""

from typing import Dict, Any, Optional, List, Union, Callable
import ipywidgets as widgets
from IPython.display import HTML

# Import shared components
from smartcash.ui.components.header.header import create_header
from smartcash.ui.components.status_panel import create_status_panel, update_status_panel

class HeaderContainer:
    """Flexible header container with title and status panel.
    
    This class provides a header container with a title section and status panel
    that can be updated dynamically. It provides methods to update the title,
    subtitle, and status message. It reuses the existing header and status panel
    components for consistency.
    """
    
    def __init__(self, 
                 title: str = "",
                 subtitle: str = "",
                 icon: str = "",
                 status_message: str = "",
                 status_type: str = "info",
                 show_status_panel: bool = True,
                 **style_options):
        """Initialize the header container.
        
        Args:
            title: Main title text
            subtitle: Subtitle text
            icon: Optional emoji or icon for the header
            status_message: Initial status message
            status_type: Status type (info, success, warning, error)
            show_status_panel: Whether to show the status panel
            **style_options: Additional styling options
        """
        self.title = title
        self.subtitle = subtitle
        self.icon = icon
        self.status_message = status_message
        self.status_type = status_type
        self.show_status_panel = show_status_panel
        
        # Default style options
        self.style = {
            'margin_bottom': '16px',
            'padding_bottom': '8px',
            'border_bottom': '1px solid #e0e0e0'
        }
        
        # Update with custom style options
        self.style.update(style_options)
        
        # Create the components
        self._create_header()
        self._create_status_panel()
        
        # Create the container
        self._create_container()
    
    def _create_header(self):
        """Create the header component using the shared header component."""
        # Use the shared header component
        self.header = create_header(
            title=self.title,
            description=self.subtitle,
            icon=self.icon
        )
    
    def _create_status_panel(self):
        """Create the status panel component using the shared status panel component."""
        # Use the shared status panel component
        self.status_panel = create_status_panel(
            message=self.status_message,
            status_type=self.status_type,
            layout={'display': 'block' if self.show_status_panel else 'none'}
        )
    
    def _create_container(self):
        """Create the main container with header and status panel."""
        self.container = widgets.VBox(
            [self.header, self.status_panel],
            layout=widgets.Layout(
                margin_bottom=self.style.get('margin_bottom', '16px'),
                padding_bottom=self.style.get('padding_bottom', '8px'),
                border_bottom=self.style.get('border_bottom', '1px solid #e0e0e0')
            )
        )
    
    def update_title(self, title: str, subtitle: Optional[str] = None, icon: Optional[str] = None) -> None:
        """Update the header title, subtitle, and/or icon.
        
        Args:
            title: New title text
            subtitle: New subtitle text (or None to keep current)
            icon: New icon (or None to keep current)
        """
        self.title = title
        if subtitle is not None:
            self.subtitle = subtitle
        if icon is not None:
            self.icon = icon
        
        # Recreate the header and container
        self._create_header()
        self._create_container()
    
    def update_status(self, message: str, status_type: str = "info", show: bool = True) -> None:
        """Update the status panel message and type.
        
        Args:
            message: New status message
            status_type: Status type (info, success, warning, error)
            show: Whether to show the status panel
        """
        self.status_message = message
        self.status_type = status_type
        self.show_status_panel = show
        
        # Update the existing status panel using the shared update function
        update_status_panel(self.status_panel, message, status_type)
        
        # Update visibility
        self.status_panel.layout.display = 'block' if show else 'none'
    
    def show_status(self, show: bool = True) -> None:
        """Show or hide the status panel.
        
        Args:
            show: Whether to show the status panel
        """
        self.show_status_panel = show
        self.status_panel.layout.display = 'block' if show else 'none'
    
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

def create_header_container(
    title: str = "",
    subtitle: str = "",
    icon: str = "",
    status_message: str = "",
    status_type: str = "info",
    show_status_panel: bool = True,
    **style_options
) -> HeaderContainer:
    """Create a header container with title and status panel.
    
    Args:
        title: Main title text
        subtitle: Subtitle text
        icon: Optional emoji or icon for the header
        status_message: Initial status message
        status_type: Status type (info, success, warning, error)
        show_status_panel: Whether to show the status panel
        **style_options: Additional styling options
        
    Returns:
        HeaderContainer instance
    """
    return HeaderContainer(
        title=title,
        subtitle=subtitle,
        icon=icon,
        status_message=status_message,
        status_type=status_type,
        show_status_panel=show_status_panel,
        **style_options
    )
