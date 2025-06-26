"""
Component registry and hierarchy management for config cell UI.

This module handles the registration and management of UI components in a hierarchical structure,
allowing for parent-child relationships between components.
"""
from typing import Dict, Any, Optional, List, Set
import ipywidgets as widgets

class ComponentRegistry:
    """Manages registration and lookup of UI components in a hierarchical structure."""
    
    def __init__(self):
        """Initialize the component registry."""
        self._components: Dict[str, Dict[str, Any]] = {}
        self._parent_map: Dict[str, str] = {}
        self._children_map: Dict[str, Set[str]] = {}
    
    def register_component(
        self, 
        component_id: str,
        component: Dict[str, Any],
        parent_id: Optional[str] = None
    ) -> None:
        """Register a component with optional parent.
        
        Args:
            component_id: Unique identifier for the component
            component: Component dictionary containing UI elements
            parent_id: Optional parent component ID
        """
        self._components[component_id] = component
        
        if parent_id:
            self._parent_map[component_id] = parent_id
            if parent_id not in self._children_map:
                self._children_map[parent_id] = set()
            self._children_map[parent_id].add(component_id)
    
    def get_component(self, component_id: str) -> Optional[Dict[str, Any]]:
        """Get a component by ID.
        
        Args:
            component_id: ID of the component to retrieve
            
        Returns:
            The component dictionary or None if not found
        """
        return self._components.get(component_id)
    
    def get_parent(self, component_id: str) -> Optional[Dict[str, Any]]:
        """Get the parent of a component.
        
        Args:
            component_id: ID of the child component
            
        Returns:
            The parent component dictionary or None if no parent
        """
        parent_id = self._parent_map.get(component_id)
        return self.get_component(parent_id) if parent_id else None
    
    def get_children(self, component_id: str) -> List[Dict[str, Any]]:
        """Get all children of a component.
        
        Args:
            component_id: ID of the parent component
            
        Returns:
            List of child component dictionaries
        """
        children_ids = self._children_map.get(component_id, set())
        return [self.get_component(cid) for cid in children_ids if cid in self._components]
    
    def unregister_component(self, component_id: str) -> bool:
        """Unregister a component and all its children.
        
        Args:
            component_id: ID of the component to unregister
            
        Returns:
            True if component was found and removed, False otherwise
        """
        if component_id not in self._components:
            return False
            
        # Recursively remove children
        for child_id in list(self._children_map.get(component_id, [])):
            self.unregister_component(child_id)
            
        # Remove from parent's children
        parent_id = self._parent_map.pop(component_id, None)
        if parent_id and parent_id in self._children_map:
            self._children_map[parent_id].discard(component_id)
            
        # Clean up
        self._components.pop(component_id, None)
        self._children_map.pop(component_id, None)
        
        return True

# Global registry instance
component_registry = ComponentRegistry()
