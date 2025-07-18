"""
Header container component for consistent UI headers across the application.

This module provides a flexible header container component that provides
a consistent header with title and optional subtitle and icon.
"""

from typing import Dict, Any, Optional
import ipywidgets as widgets

# Import shared components
from smartcash.ui.components.header.header import create_header

class HeaderContainer:
    """Flexible header container with title and optional subtitle.
    
    This class provides a consistent header component that can display
    a title, optional subtitle, and icon. It's used across the application
    to maintain a consistent look and feel for section headers.
    """
    
    def __init__(self, 
                 title: str = "",
                 subtitle: str = "",
                 icon: str = "",
                 **style_options):
        """Initialize the header container.
        
        Args:
            title: Main title text
            subtitle: Subtitle text
            icon: Optional emoji or icon for the header
            **style_options: Additional styling options
        """
        self.title = title
        self.subtitle = subtitle
        self.icon = icon
        
        # Default style options
        self.style = {
            'margin_bottom': '16px',
            'padding_bottom': '8px',
            'border_bottom': '1px solid #e0e0e0'
        }
        
        # Update with custom style options
        self.style.update(style_options)
        
        # Create the header
        self._create_header()
        
        # Create the container with just the header
        self._create_container()
    
    def _create_header(self):
        """Create the header component using the shared header component."""
        # Use the shared header component
        self.header = create_header(
            title=self.title,
            description=self.subtitle,
            icon=self.icon
        )
    
    def _create_container(self):
        """Create the main container with just the header."""
        self.container = widgets.VBox(
            [self.header],
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
    **style_options
) -> HeaderContainer:
    """Create a header container with title.
    
    Args:
        title: Main title text
        subtitle: Subtitle text
        icon: Optional emoji or icon for the header
        **style_options: Additional styling options
        
    Returns:
        HeaderContainer instance
    """
    return HeaderContainer(
        title=title,
        subtitle=subtitle,
        icon=icon,
        **style_options
    )
