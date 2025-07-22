"""
Header container component for consistent UI headers across the application.

This module provides a flexible header container component that provides
a consistent header with title, optional subtitle, and environment indicator.
"""

from typing import Optional
import ipywidgets as widgets

# Import shared components
from smartcash.ui.components.header.header import create_header
from smartcash.ui.components.indicator_panel import create_indicator_panel

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
                 show_environment: bool = True,
                 environment: str = "local",
                 config_path: str = "config.yaml",
                 **style_options):
        """Initialize the header container.
        
        Args:
            title: Main title text
            subtitle: Subtitle text
            icon: Optional emoji or icon for the header
            show_environment: Whether to show the environment indicator
            environment: Current environment ('colab', 'local', etc.)
            config_path: Path to the config file being used
            **style_options: Additional styling options
        """
        self.title = title
        self.subtitle = subtitle
        self.icon = icon
        self.show_environment = show_environment
        self.environment = environment
        self.config_path = config_path
        
        # Default style options
        self.style = {
            'margin_bottom': '16px',
            'padding_bottom': '8px',
            'border_bottom': '1px solid #e0e0e0',
            'environment_margin_top': '8px'
        }
        
        # Update with custom style options
        self.style.update(style_options)
        
        # Create the header and environment indicator
        self._create_header()
        if self.show_environment:
            self._create_environment_indicator()
        
        # Create the container with just the header
        self._create_container()
    
    def _create_header(self):
        """Create the header component."""
        # Create the main header
        self.header = create_header(
            title=self.title,
            description=self.subtitle,
            icon=self.icon
        )
        
        # Create the header container
        self.header_container = widgets.VBox(
            [self.header],
            layout=widgets.Layout(
                width='100%',
                margin='0',
                padding='0',
                border_bottom=self.style['border_bottom'],
                padding_bottom=self.style['padding_bottom'],
                margin_bottom=self.style['margin_bottom']
            )
        )
        
        # Initialize children with just the header
        self.container = widgets.VBox(
            [self.header_container],
            layout=widgets.Layout(
                width='100%',
                margin='0',
                padding='0',
                border='none'
            )
        )
    
    def _create_container(self):
        """Create the main container widget - already handled in _create_header."""
        # Container is already created in _create_header method
        # This method exists to satisfy the calling code but doesn't need to do anything
        pass
    
    def _create_environment_indicator(self):
        """Create and add the environment indicator."""
        # Create the indicator panel
        self.environment_indicator = create_indicator_panel(
            environment=self.environment,
            config_path=self.config_path
        )
        
        # Add it below the header
        if len(self.container.children) == 1:  # Only header exists
            self.container.children = list(self.container.children) + [self.environment_indicator.show()]
        else:  # Update existing indicator
            self.container.children = list(self.container.children)[:1] + [self.environment_indicator.show()]
    
    def update(self, 
               title: Optional[str] = None, 
               subtitle: Optional[str] = None, 
               icon: Optional[str] = None,
               environment: Optional[str] = None,
               config_path: Optional[str] = None):
        """Update the header content and environment indicator.
        
        Args:
            title: New title text (optional)
            subtitle: New subtitle text (optional)
            icon: New icon (optional)
            environment: New environment value (optional)
            config_path: New config path (optional)
        """
        update_header = False
        
        # Update header content if needed
        if title is not None:
            self.title = title
            update_header = True
        if subtitle is not None:
            self.subtitle = subtitle
            update_header = True
        if icon is not None:
            self.icon = icon
            update_header = True
            
        # Update header if any header content changed
        if update_header:
            self._create_header()
        
        # Update environment indicator if needed
        if environment is not None or config_path is not None:
            if hasattr(self, 'environment_indicator'):
                self.environment_indicator.update(
                    environment=environment if environment is not None else self.environment,
                    config_path=config_path if config_path is not None else self.config_path
                )
            elif self.show_environment:
                if environment is not None:
                    self.environment = environment
                if config_path is not None:
                    self.config_path = config_path
                self._create_environment_indicator()
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
    show_environment: bool = True,
    environment: str = "local",
    config_path: str = "config.yaml",
    **style_options
) -> HeaderContainer:
    """Create a header container with title and optional environment indicator.
    
    Args:
        title: Main title text
        subtitle: Subtitle text
        icon: Optional emoji or icon for the header
        show_environment: Whether to show the environment indicator
        environment: Current environment ('colab', 'local', etc.)
        config_path: Path to the config file being used
        **style_options: Additional styling options
        
    Returns:
        HeaderContainer instance with optional environment indicator
    """
    return HeaderContainer(
        title=title,
        subtitle=subtitle,
        icon=icon,
        show_environment=show_environment,
        environment=environment,
        config_path=config_path,
        **style_options
    )
