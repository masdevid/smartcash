"""
File: smartcash/ui/core/shared/ui_component_manager.py
Deskripsi: UI component manager dengan fail-fast principle dan silent fail untuk reset
"""

import weakref
from typing import Dict, Any, Optional, List, Type, Union, Callable
from threading import Lock
from dataclasses import dataclass
from smartcash.ui.utils.ui_logger import get_module_logger

@dataclass
class ComponentInfo:
    """Component registration info."""
    name: str
    instance: Any
    component_type: Type
    module: str
    references: int = 0
    metadata: Dict[str, Any] = None

class ComponentRegistry:
    """Global registry untuk shared components dengan fail-fast operations."""
    
    _instance = None
    _lock = Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._components: Dict[str, ComponentInfo] = {}
            self._type_index: Dict[Type, set] = {}
            self._weak_refs: Dict[str, weakref.ref] = {}
            self._lock = Lock()
            self._initialized = True
    
    def register(self, name: str, component: Any, component_type: Type = None,
                module: str = None, metadata: Dict[str, Any] = None) -> None:
        """Register component dengan fail-fast validation."""
        if not name:
            raise ValueError("Component name cannot be empty")
        
        if component is None:
            raise ValueError(f"Component '{name}' cannot be None")
        
        with self._lock:
            if component_type is None:
                component_type = type(component)
            
            info = ComponentInfo(
                name=name,
                instance=component,
                component_type=component_type,
                module=module or "unknown",
                metadata=metadata or {}
            )
            
            # Update existing or create new
            if name in self._components:
                self._components[name].references += 1
            else:
                self._components[name] = info
                
                # Update type index
                if component_type not in self._type_index:
                    self._type_index[component_type] = set()
                self._type_index[component_type].add(name)
                
                # Create weak reference
                self._weak_refs[name] = weakref.ref(
                    component, lambda ref: self._on_component_deleted(name)
                )
    
    def get(self, name: str) -> Optional[Any]:
        """Get component dengan reference counting."""
        with self._lock:
            if info := self._components.get(name):
                info.references += 1
                return info.instance
            return None
    
    def get_by_type(self, component_type: Type) -> List[Any]:
        """Get all components of specific type."""
        with self._lock:
            components = []
            if names := self._type_index.get(component_type):
                for name in names:
                    if comp := self.get(name):
                        components.append(comp)
            return components
    
    def unregister(self, name: str) -> bool:
        """Unregister component."""
        with self._lock:
            if info := self._components.get(name):
                # Remove from type index
                if info.component_type in self._type_index:
                    self._type_index[info.component_type].discard(name)
                    if not self._type_index[info.component_type]:
                        del self._type_index[info.component_type]
                
                # Remove component
                del self._components[name]
                
                # Remove weak ref
                if name in self._weak_refs:
                    del self._weak_refs[name]
                
                return True
            return False
    
    def _on_component_deleted(self, name: str) -> None:
        """Callback when component is garbage collected."""
        self.unregister(name)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        with self._lock:
            return {
                'total_components': len(self._components),
                'total_references': sum(info.references for info in self._components.values()),
                'weak_refs': len(self._weak_refs)
            }

class UIComponentManager:
    """Manager untuk UI component lifecycle dengan fail-fast operations."""
    
    def __init__(self, module_name: str):
        if not module_name:
            raise ValueError("Module name cannot be empty")
        
        self.module_name = module_name
        from smartcash.ui.core.shared.logger import get_enhanced_logger
        self.logger = get_enhanced_logger(f"smartcash.ui.{module_name}")
        
        self._components: Dict[str, Any] = {}
        self._component_validators: Dict[str, Callable] = {}
        self._registry = ComponentRegistry()
        
        self.logger.debug(f"🎯 Initialized UIComponentManager for {module_name}")
    
    def add_component(self, name: str, component: Any, shared: bool = False,
                     validator: Optional[Callable[[Any], bool]] = None) -> None:
        """Add component dengan fail-fast validation."""
        if not name:
            raise ValueError("Component name cannot be empty")
        
        if component is None:
            raise ValueError(f"Component '{name}' cannot be None")
        
        # Validate if validator provided
        if validator:
            if not validator(component):
                raise ValueError(f"Component '{name}' failed validation")
            self._component_validators[name] = validator
        
        # Store component
        self._components[name] = component
        
        # Register if shared
        if shared:
            self._registry.register(
                f"{self.module_name}.{name}",
                component,
                module=self.module_name
            )
        
        self.logger.debug(f"✅ Added component '{name}' (shared: {shared})")
    
    def get_component(self, name: str, default: Any = None) -> Any:
        """Get component dengan fallback ke registry."""
        # Check local first
        if name in self._components:
            return self._components[name]
        
        # Check registry dengan module prefix
        full_name = f"{self.module_name}.{name}"
        if comp := self._registry.get(full_name):
            return comp
        
        # Check registry tanpa prefix
        if comp := self._registry.get(name):
            return comp
        
        return default
    
    def remove_component(self, name: str) -> bool:
        """Remove component dari local dan registry."""
        removed = False
        
        # Remove local
        if name in self._components:
            del self._components[name]
            removed = True
        
        # Remove dari registry
        full_name = f"{self.module_name}.{name}"
        if self._registry.unregister(full_name):
            removed = True
        
        # Remove validator
        if name in self._component_validators:
            del self._component_validators[name]
        
        if removed:
            self.logger.debug(f"🗑️ Removed component '{name}'")
        
        return removed
    
    def has_component(self, name: str) -> bool:
        """Check if component exists."""
        return self.get_component(name) is not None
    
    def list_components(self) -> List[str]:
        """List all component names."""
        return list(self._components.keys())
    
    def add_components(self, components: Dict[str, Any], shared: bool = False) -> None:
        """Add multiple components dengan fail-fast validation."""
        if not components:
            raise ValueError("Components dictionary cannot be empty")
        
        for name, component in components.items():
            self.add_component(name, component, shared)
        
        self.logger.info(f"📦 Added {len(components)} components")
    
    def update_component_safely(self, name: str, update_func: Callable[[Any], None]) -> None:
        """Update component dengan fail-fast validation."""
        component = self.get_component(name)
        if component is None:
            raise RuntimeError(f"Component '{name}' not found for update")
        
        try:
            update_func(component)
        except Exception as e:
            raise RuntimeError(f"Failed to update component '{name}': {str(e)}")
    
    def reset_components_silently(self, component_names: Optional[List[str]] = None) -> Dict[str, bool]:
        """Reset components dengan silent fail untuk missing components."""
        results = {}
        
        # Determine target components
        if component_names is None:
            target_components = list(self._components.keys())
        else:
            target_components = component_names
        
        for component_name in target_components:
            try:
                # Silent fail: skip if component doesn't exist
                if component_name not in self._components:
                    self.logger.debug(f"🔇 Component '{component_name}' not found, skipping")
                    results[component_name] = False
                    continue
                
                component = self._components[component_name]
                
                # Silent fail: skip if component is None
                if component is None:
                    self.logger.debug(f"🔇 Component '{component_name}' is None, skipping")
                    results[component_name] = False
                    continue
                
                # Reset component
                success = self._reset_component(component)
                results[component_name] = success
                
                if success:
                    self.logger.debug(f"✅ Reset component '{component_name}'")
                else:
                    self.logger.debug(f"🔇 No reset method for '{component_name}'")
                    
            except Exception as e:
                # Log error but continue with other components
                self.logger.error(f"❌ Error resetting '{component_name}': {e}")
                results[component_name] = False
        
        return results
    
    def _reset_component(self, component: Any) -> bool:
        """Reset component dengan available methods."""
        # Try common reset methods
        if hasattr(component, 'reset') and callable(component.reset):
            component.reset()
            return True
        elif hasattr(component, 'clear') and callable(component.clear):
            component.clear()
            return True
        elif hasattr(component, 'clear_output') and callable(component.clear_output):
            component.clear_output(wait=True)
            return True
        elif hasattr(component, 'clear_logs') and callable(component.clear_logs):
            component.clear_logs()
            return True
        elif hasattr(component, 'value'):
            if isinstance(component.value, str):
                component.value = ""
                return True
            elif isinstance(component.value, (int, float)):
                component.value = 0
                return True
            elif isinstance(component.value, bool):
                component.value = False
                return True
            elif isinstance(component.value, list):
                component.value = []
                return True
            elif isinstance(component.value, dict):
                component.value = {}
                return True
        
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get manager statistics."""
        return {
            'module': self.module_name,
            'local_components': len(self._components),
            'validators': len(self._component_validators),
            'registry_stats': self._registry.get_stats()
        }
    
    def cleanup(self) -> None:
        """Cleanup manager resources."""
        count = len(self._components)
        self._components.clear()
        self._component_validators.clear()
        self.logger.info(f"🧹 Cleaned up UIComponentManager: {count} components removed")
        
    def cleanup_stray_widgets(self, widget_types: List[str] = None) -> Dict[str, int]:
        """Clean up stray widgets that might be causing UI issues.
        
        This method finds and hides orphaned widgets that aren't properly attached to
        the widget hierarchy, which can cause duplicate UI elements to appear.
        
        Args:
            widget_types: List of widget type names to clean up. If None, cleans up
                         common problematic widgets like Accordion, Output, etc.
                         
        Returns:
            Dictionary with count of hidden widgets by type
        """
        if widget_types is None:
            # Default list of widget types that commonly cause issues when orphaned
            widget_types = ['Accordion', 'Output', 'VBox', 'HBox', 'Tab']
            
        import gc
        results = {}
        
        for widget_type in widget_types:
            results[widget_type] = 0
            
        for obj in gc.get_objects():
            try:
                # Safe way to check if object is likely a widget without triggering warnings
                # First check if it has a __module__ attribute that contains 'ipywidgets'
                is_likely_widget = False
                try:
                    if hasattr(obj, '__module__') and 'ipywidget' in getattr(obj, '__module__', ''):
                        is_likely_widget = True
                except:
                    pass
                    
                # Skip if not likely a widget
                if not is_likely_widget:
                    continue
                    
                # Now check for layout attribute which all widgets should have
                if not hasattr(obj, 'layout'):
                    continue
                    
                # Get widget type name safely
                widget_type_name = None
                try:
                    widget_type_name = obj.__class__.__name__
                except:
                    continue
                    
                # Skip if not in our target types
                if widget_type_name not in widget_types:
                    continue
                    
                # Check if it's an orphaned widget (not attached to any parent)
                is_orphaned = True
                if hasattr(obj, '_parent') and obj._parent is not None:
                    is_orphaned = False
                    
                # Hide orphaned widgets
                if is_orphaned:
                    obj.layout.display = 'none'
                    results[widget_type_name] += 1
                    
            except Exception:
                # Skip any objects that cause errors during inspection
                continue
                
        # Log results
        for widget_type, count in results.items():
            if count > 0:
                self.logger.info(f"🧹 Hidden {count} orphaned {widget_type} widgets")
                
        return results

# Global instance for convenience
_component_manager_instance = None

def get_component_manager(module_name: str = "default") -> UIComponentManager:
    """Get or create a global UIComponentManager instance.
    
    Args:
        module_name: Name of the module using the component manager.
        
    Returns:
        UIComponentManager: The global component manager instance.
    """
    global _component_manager_instance
    if _component_manager_instance is None:
        _component_manager_instance = UIComponentManager(module_name)
    return _component_manager_instance


def cleanup_stray_widgets(widget_types: List[str] = None, module_name: str = "default") -> Dict[str, int]:
    """Convenience function to clean up stray widgets.
    
    This is a global helper function that gets the component manager and calls
    cleanup_stray_widgets on it. This makes it easy to clean up stray widgets
    from anywhere in the codebase without having to get the component manager first.
    
    Args:
        widget_types: List of widget type names to clean up
        module_name: Name of the module using the component manager
        
    Returns:
        Dictionary with count of hidden widgets by type
    """
    manager = get_component_manager(module_name)
    return manager.cleanup_stray_widgets(widget_types)