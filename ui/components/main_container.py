"""
Main container component for consistent UI layout across the application.

This module provides a flexible main container component that can be used
to create a consistent layout for different parts of the application.
"""

from typing import Dict, Any, Optional, List, Union, Sequence, TypedDict, Literal
import ipywidgets as widgets

# Type definitions
ContainerType = Literal['header', 'form', 'action', 'operation', 'footer', 'custom']

class ContainerConfig(TypedDict, total=False):
    """Configuration for a container component."""
    type: ContainerType
    component: widgets.Widget
    order: int
    name: str
    visible: bool

class MainContainer:
    """Flexible main container component with consistent styling and ordering.
    
    This class provides a container with consistent styling that can be used
    to create a unified look and feel across the application. It supports:
    - Flexible ordering of components
    - Custom components with configurable order
    - Show/hide individual components
    - Backward compatibility with legacy container types
    """
    
    def __init__(
        self,
        *,
        # Legacy container parameters (for backward compatibility)
        header_container: Optional[widgets.Widget] = None,
        form_container: Optional[widgets.Widget] = None,
        action_container: Optional[widgets.Widget] = None,
        footer_container: Optional[widgets.Widget] = None,
        operation_container: Optional[widgets.Widget] = None,
        # New flexible container system
        components: Optional[List[ContainerConfig]] = None,
        **style_options
    ):
        """Initialize the main container with flexible component ordering.
        
        Args:
            # Legacy parameters (for backward compatibility)
            header_container: Optional header section
            form_container: Optional form/input section
            action_container: Optional action buttons section
            footer_container: Optional footer section
            operation_container: Optional container for operation-related components
            
            # New flexible container system
            components: List of container configurations with explicit ordering
            
            **style_options: Additional styling options
        """
        # Initialize containers dictionary
        self.containers: Dict[str, ContainerConfig] = {}
        self._component_order: List[str] = []
        
        # Process legacy container parameters if no components are provided
        if components is None:
            components = []
            
            if header_container:
                components.append({
                    'type': 'header',
                    'component': header_container,
                    'order': 0,
                    'visible': True
                })
                
            if form_container:
                components.append({
                    'type': 'form',
                    'component': form_container,
                    'order': 1,
                    'visible': True
                })
                
            if action_container:
                components.append({
                    'type': 'action',
                    'component': action_container,
                    'order': 2,
                    'visible': True
                })
                
            if operation_container:
                components.append({
                    'type': 'operation',
                    'component': operation_container,
                    'order': 3,
                    'visible': True
                })
                
            if footer_container:
                components.append({
                    'type': 'footer',
                    'component': footer_container,
                    'order': 4,
                    'visible': True
                })
        
        # Process components
        for i, component in enumerate(components):
            config = component.copy()
            
            # Set default values
            if 'order' not in config:
                config['order'] = i
                
            if 'visible' not in config:
                config['visible'] = True
                
            # Generate a unique name if not provided
            if 'name' not in config:
                if 'type' in config:
                    config['name'] = f"{config['type']}_{i}"
                else:
                    config['name'] = f"custom_{i}"
            
            # Add to containers dictionary
            self.containers[config['name']] = config
            
            # Track order if visible
            if config['visible'] and config['component'] is not None:
                self._component_order.append(config['name'])
        
        # Sort components by order
        self._component_order.sort(key=lambda name: self.containers[name].get('order', 0))
        
        # Default style options
        self.style = {
            'container_padding': '20px',
            'container_margin': '0 auto',
            'max_width': '1200px',
            'background': '#ffffff',
            'border_radius': '8px',
            'box_shadow': '0 2px 4px rgba(0,0,0,0.1)',
            'gap': '20px',
            'operation_gap': '15px',  # Gap for operation container
        }
        
        # Update with custom style options
        self.style.update(style_options)
        
        # Create the main container
        self._create_container()
    
    def _create_container(self) -> None:
        """Create the main container with all visible components."""
        # Get visible components in order
        children = [
            self.containers[name]['component']
            for name in self._component_order
            if self.containers[name].get('visible', True) and 
               self.containers[name]['component'] is not None
        ]
        
        # Create the main container
        self.container = widgets.VBox(
            children,
            layout=widgets.Layout(
                padding=self.style.get('container_padding', '20px'),
                margin=self.style.get('container_margin', '0 auto'),
                max_width=self.style.get('max_width', '1200px'),
                background=self.style.get('background', '#ffffff'),
                border_radius=self.style.get('border_radius', '8px'),
                box_shadow=self.style.get('box_shadow', '0 2px 4px rgba(0,0,0,0.1)'),
                gap=self.style.get('gap', '20px'),
                width='auto',
                display='flex',
                flex_flow='column',
                align_items='stretch'
            )
        )
    
    def add_component(self, component: widgets.Widget, component_type: ContainerType = 'custom', 
                     name: Optional[str] = None, order: Optional[int] = None, 
                     visible: bool = True) -> str:
        """Add a new component to the container.
        
        Args:
            component: The widget to add
            component_type: Type of the component (header, form, action, etc.)
            name: Optional unique name for the component
            order: Optional order index (lower numbers come first)
            visible: Whether the component should be visible
            
        Returns:
            str: The name assigned to the component
        """
        if name is None:
            # Generate a unique name if not provided
            base_name = f"{component_type}_"
            counter = 0
            while f"{base_name}{counter}" in self.containers:
                counter += 1
            name = f"{base_name}{counter}"
        
        if order is None:
            # Default to the end if no order is specified
            order = len(self.containers)
        
        # Add the component
        self.containers[name] = {
            'type': component_type,
            'component': component,
            'order': order,
            'name': name,
            'visible': visible
        }
        
        # Update the component order
        if visible:
            self._update_component_order()
        
        # Rebuild the container
        self._create_container()
        
        return name
    
    def remove_component(self, name: str) -> None:
        """Remove a component from the container.
        
        Args:
            name: Name of the component to remove
        """
        if name in self.containers:
            del self.containers[name]
            if name in self._component_order:
                self._component_order.remove(name)
            self._create_container()
    
    def set_component_visibility(self, name: str, visible: bool) -> None:
        """Show or hide a component.
        
        Args:
            name: Name of the component
            visible: Whether the component should be visible
        """
        if name in self.containers:
            self.containers[name]['visible'] = visible
            self._update_component_order()
            self._create_container()
    
    def set_component_order(self, name: str, order: int) -> None:
        """Set the order of a component.
        
        Args:
            name: Name of the component
            order: New order index (lower numbers come first)
        """
        if name in self.containers:
            self.containers[name]['order'] = order
            self._update_component_order()
            self._create_container()
    
    def _update_component_order(self) -> None:
        """Update the internal component order based on visibility and order."""
        self._component_order = [
            name for name, config in self.containers.items()
            if config.get('visible', True) and config['component'] is not None
        ]
        self._component_order.sort(key=lambda name: self.containers[name].get('order', 0))
    
    def get_component(self, name: str) -> Optional[widgets.Widget]:
        """Get a component by name.
        
        Args:
            name: Name of the component
            
        Returns:
            The component widget, or None if not found
        """
        return self.containers.get(name, {}).get('component')
    
    def update_section(self, section_name: str, new_content: widgets.Widget) -> None:
        """Legacy method: Update a section of the container.
        
        Args:
            section_name: Name of the section to update ('header', 'form', etc.)
            new_content: New content widget
        """
        if section_name in self.containers:
            self.containers[section_name]['component'] = new_content
            self._create_container()
    
    def get_section(self, section_name: str) -> Optional[widgets.Widget]:
        """Get a section of the container.
        
        Args:
            section_name: Name of the section to get
            
        Returns:
            The section widget or None if not found
        """
        return self.containers.get(section_name, {}).get('component')
    
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

    def show(self) -> widgets.Widget:
        """Display the container."""
        return self.container

def create_main_container(
    *,
    # Legacy container parameters (for backward compatibility)
    header_container: Optional[widgets.Widget] = None,
    form_container: Optional[widgets.Widget] = None,
    action_container: Optional[widgets.Widget] = None,
    footer_container: Optional[widgets.Widget] = None,
    operation_container: Optional[widgets.Widget] = None,
    # New flexible container system
    components: Optional[List[ContainerConfig]] = None,
    **style_options
) -> MainContainer:
    """Create a main container with flexible component ordering.
    
    This function provides two ways to create a container:
    1. Legacy way: Pass individual container components
    2. New way: Pass a list of component configurations with explicit ordering
    
    Args:
        # Legacy parameters (for backward compatibility)
        header_container: Optional header section
        form_container: Optional form/input section
        action_container: Optional action buttons section
        footer_container: Optional footer section
        operation_container: Optional container for operation-related components
        
        # New flexible container system
        components: List of container configurations with explicit ordering
        
        **style_options: Additional styling options
        
    Returns:
        MainContainer instance with the specified components
        
    Example (new way):
        components = [
            {'type': 'header', 'component': header, 'order': 0},
            {'type': 'form', 'component': form, 'order': 1},
            {'type': 'action', 'component': actions, 'order': 2},
            {'type': 'operation', 'component': operations, 'order': 3},
            {'type': 'footer', 'component': footer, 'order': 4}
        ]
        container = create_main_container(components=components)
    """
    return MainContainer(
        header_container=header_container,
        form_container=form_container,
        action_container=action_container,
        footer_container=footer_container,
        operation_container=operation_container,
        components=components,
        **style_options
    )
