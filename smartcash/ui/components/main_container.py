"""
Main container component for consistent UI layout across the application.

This module provides a flexible main container component that can be used
to create a consistent layout for different parts of the application.
"""

from typing import Dict, Any, Optional, List, Union, Sequence
import ipywidgets as widgets

# Import utility functions

class MainContainer:
    """Flexible main container component with consistent styling.
    
    This class provides a container with consistent styling that can be used
    to create a unified look and feel across the application. It accepts
    different container sections (header, form, action, footer) and arranges
    them in a vertical layout with consistent styling.
    """
    
    def __init__(self, 
                 header_container: Optional[widgets.Widget] = None,
                 form_container: Optional[widgets.Widget] = None,
                 action_container: Optional[widgets.Widget] = None,
                 footer_container: Optional[widgets.Widget] = None,
                 progress_container: Optional[widgets.Widget] = None,
                 log_container: Optional[widgets.Widget] = None,
                 **style_options):
        """Initialize the main container with optional sections.
        
        Args:
            header_container: Optional header section
            form_container: Optional form/input section
            action_container: Optional action buttons section
            footer_container: Optional footer section
            progress_container: Optional progress tracking section
            log_container: Optional log display section
            **style_options: Additional styling options
        """
        self.containers = {
            'header': header_container,
            'form': form_container,
            'action': action_container,
            'progress': progress_container,
            'log': log_container,
            'footer': footer_container
        }
        
        # Default style options
        self.style = {
            'width': '100%',
            'max_width': '1280px',
            'margin': '0 auto',
            'padding': '8px',
            'border': '1px solid #e0e0e0',
            'border_radius': '8px',
            'box_shadow': '0 2px 4px rgba(0,0,0,0.05)'
        }
        
        # Update with custom style options
        self.style.update(style_options)
        
        # Create the main container
        self._create_container()
    
    def _create_container(self):
        """Create the main container with all sections."""
        # Filter out None containers
        children = [container for container in self.containers.values() if container is not None]
        
        # Create the main container
        self.container = widgets.VBox(
            children,
            layout=widgets.Layout(**self.style)
        )
    
    def update_section(self, section_name: str, new_content: widgets.Widget) -> None:
        """Update a section of the container.
        
        Args:
            section_name: Name of the section to update ('header', 'form', etc.)
            new_content: New widget to replace the current section
        """
        if section_name not in self.containers:
            raise ValueError(f"Invalid section name: {section_name}")
        
        # Update the container dictionary
        self.containers[section_name] = new_content
        
        # Rebuild the container
        self._create_container()
    
    def get_section(self, section_name: str) -> Optional[widgets.Widget]:
        """Get a section of the container.
        
        Args:
            section_name: Name of the section to get
            
        Returns:
            The section widget or None if not found
        """
        return self.containers.get(section_name)
    
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

def create_main_container(
    header_container: Optional[widgets.Widget] = None,
    form_container: Optional[widgets.Widget] = None,
    action_container: Optional[widgets.Widget] = None,
    footer_container: Optional[widgets.Widget] = None,
    progress_container: Optional[widgets.Widget] = None,
    log_container: Optional[widgets.Widget] = None,
    clean_stray_widgets: bool = True,
    **style_options
) -> MainContainer:
    """Create a main container with consistent styling.
    
    Args:
        header_container: Optional header section
        form_container: Optional form/input section
        action_container: Optional action buttons section
        footer_container: Optional footer section
        progress_container: Optional progress tracking section
        log_container: Optional log display section
        **style_options: Additional styling options
        
    Returns:
        MainContainer instance
    """
    # Create the main container
    container = MainContainer(
        header_container=header_container,
        form_container=form_container,
        action_container=action_container,
        progress_container=progress_container,
        log_container=log_container,
        footer_container=footer_container,
        **style_options
    )
    return container
