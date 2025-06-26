"""
Parent Component Management for Config Cell

This module handles the creation and management of parent components
that can contain and orchestrate child components in a hierarchical structure.
"""
from typing import Dict, Any, Optional, List, Type, TypeVar, Callable, Union
import ipywidgets as widgets
from IPython.display import display

from smartcash.common.logger import get_logger
from smartcash.ui.config_cell.components.component_registry import component_registry
from smartcash.ui.config_cell.components.ui_factory import (
    create_config_cell_ui,
    create_parent_container
)
from smartcash.ui.config_cell.handlers.config_handler import ConfigCellHandler
from smartcash.ui.config_cell.handlers.error_handler import handle_ui_errors

logger = get_logger(__name__)
T = TypeVar('T', bound=ConfigCellHandler)

class ParentComponentManager:
    """Manages parent-child relationships between UI components."""
    
    def __init__(self, parent_id: str, title: str = None):
        """Initialize with a unique parent ID and optional title.
        
        Args:
            parent_id: Unique identifier for this parent component
            title: Optional title for the parent component
        """
        self.parent_id = parent_id
        self.title = title or f"Parent_{parent_id}"
        self.children: List[str] = []
        self.components: Dict[str, Any] = {}
        self._initialize_container()
    
    def _initialize_container(self):
        """Initialize the parent container using the UI factory."""
        parent_ui = create_parent_container(title=self.title)
        self.container = parent_ui['container']
        self.content_area = parent_ui['content_area']
    
    @handle_ui_errors
    def add_child_component(
        self, 
        child_id: str, 
        component: Union[Dict[str, Any], str],
        handler: Optional[ConfigCellHandler] = None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Add a child component to this parent.
        
        Args:
            child_id: Unique identifier for the child component
            component: Either a module name (str) or a pre-created component dict
            handler: Optional config handler instance for the child
            config: Optional configuration for the component
            **kwargs: Additional arguments passed to create_config_cell_ui
            
        Returns:
            Dict[str, Any]: The created component dictionary
            
        Raises:
            ValueError: If component creation fails
        """
        if isinstance(component, str):
            # Create component using the factory function
            component = create_config_cell_ui(
                module_name=component,
                handler=handler,
                config=config,
                **kwargs
            )
        
        if not component or 'container' not in component:
            raise ValueError("Invalid component: must contain 'container' key")
        
        # Store the handler if provided
        if handler is not None:
            component['handler'] = handler
        
        # Register the component
        component_registry.register_component(
            component_id=child_id,
            component=component,
            parent_id=self.parent_id
        )
        
        # Add to children list
        self.children.append(child_id)
        
        # Add to content area
        current_children = list(self.content_area.children)
        current_children.append(component['container'])
        self.content_area.children = current_children
        
        return component
    
    def get_child_component(self, child_id: str) -> Optional[Dict[str, Any]]:
        """Get a child component by ID.
        
        Args:
            child_id: ID of the child component to retrieve
            
        Returns:
            The child component dictionary or None if not found
        """
        return component_registry.get_component(child_id)
    
    def remove_child_component(self, child_id: str) -> bool:
        """Remove a child component.
        
        Args:
            child_id: ID of the child component to remove
            
        Returns:
            bool: True if removed, False if not found
        """
        if child_id not in self.children:
            return False
            
        # Remove from container
        child_comp = self.get_child_component(child_id)
        if child_comp and 'container' in child_comp:
            current_children = list(self.container.children)
            if child_comp['container'] in current_children:
                current_children.remove(child_comp['container'])
                self.container.children = current_children
        
        # Clean up registry
        component_registry.unregister_component(child_id)
        self.children.remove(child_id)
        return True
    
    def display(self):
        """Display the parent component and all its children."""
        display(self.container)


def create_parent_component(
    parent_id: str,
    title: Optional[str] = None,
    children: Optional[List[Dict[str, Any]]] = None
) -> ParentComponentManager:
    """Create a new parent component manager.
    
    Args:
        parent_id: Unique identifier for the parent component
        title: Optional title for the parent component
        children: Optional list of child component configurations with keys:
            - id: Unique identifier for the child
            - module: Module name (passed to create_config_cell_ui)
            - handler: Optional config handler instance
            - config: Optional configuration dictionary
            - **kwargs: Additional arguments for create_config_cell_ui
            
    Returns:
        ParentComponentManager instance
    """
    parent = ParentComponentManager(parent_id, title)
    
    # Add children if provided
    if children:
        for child in children:
            parent.add_child_component(
                child_id=child['id'],
                component=child.get('module', child.get('component', '')),
                handler=child.get('handler'),
                config=child.get('config'),
                **{k: v for k, v in child.items() 
                   if k not in ('id', 'module', 'component', 'handler', 'config')}
            )
    
    return parent
