"""
File: smartcash/ui/core/ui_module_factory.py
Description: Factory for creating and managing UIModule instances with lifecycle management.
"""

from typing import Dict, Any, Optional, List, Type, Union, Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import threading
import weakref
from pathlib import Path

from smartcash.ui.logger import get_module_logger
from smartcash.ui.core.ui_module import UIModule, ModuleStatus, SharedMethodRegistry
from smartcash.ui.core.errors import SmartCashUIError, ErrorContext

logger = get_module_logger("smartcash.ui.core.factory")

class FactoryMode(Enum):
    """Factory operation modes."""
    DEVELOPMENT = "development"  # Create new instances freely
    PRODUCTION = "production"   # Strict instance management
    TESTING = "testing"         # Isolated instances for testing

@dataclass
class ModuleTemplate:
    """Template for creating UIModule instances."""
    module_name: str
    parent_module: Optional[str]
    default_config: Dict[str, Any]
    required_components: List[str]
    required_operations: List[str]
    auto_initialize: bool = True
    description: str = ""

class UIModuleFactory:
    """Factory for creating and managing UIModule instances.
    
    Features:
    - 🏭 Centralized instance creation and management
    - 📋 Module templates for consistent configuration
    - 🔄 Lifecycle management (create, initialize, cleanup)
    - 🧵 Thread-safe operations
    - 💾 Instance registry with weak references
    - 🚀 Lazy loading and auto-initialization
    - 🔧 Configuration inheritance and merging
    """
    
    _instances: Dict[str, weakref.ReferenceType] = {}
    _templates: Dict[str, ModuleTemplate] = {}
    _factory_mode = FactoryMode.DEVELOPMENT
    _lock = threading.RLock()
    
    @classmethod
    def set_mode(cls, mode: FactoryMode) -> None:
        """Set factory operation mode.
        
        Args:
            mode: Factory mode to set
        """
        with cls._lock:
            cls._factory_mode = mode
            logger.info(f"🏭 Factory mode set to: {mode.value}")
    
    @classmethod
    def get_mode(cls) -> FactoryMode:
        """Get current factory mode.
        
        Returns:
            Current FactoryMode
        """
        return cls._factory_mode
    
    # === Template Management ===
    
    @classmethod
    def register_template(cls, template: ModuleTemplate, overwrite: bool = False) -> None:
        """Register a module template.
        
        Args:
            template: ModuleTemplate instance
            overwrite: Allow overwriting existing template
            
        Raises:
            ValueError: If template already exists and overwrite=False
        """
        template_key = cls._get_template_key(template.module_name, template.parent_module)
        
        with cls._lock:
            if template_key in cls._templates and not overwrite:
                raise ValueError(f"Template '{template_key}' already exists. Use overwrite=True to replace.")
            
            cls._templates[template_key] = template
            logger.debug(f"📋 Registered template: {template_key}")
    
    @classmethod
    def get_template(cls, module_name: str, parent_module: str = None) -> Optional[ModuleTemplate]:
        """Get a module template.
        
        Args:
            module_name: Module name
            parent_module: Parent module name
            
        Returns:
            ModuleTemplate or None if not found
        """
        template_key = cls._get_template_key(module_name, parent_module)
        return cls._templates.get(template_key)
    
    @classmethod
    def list_templates(cls) -> Dict[str, ModuleTemplate]:
        """List all registered templates.
        
        Returns:
            Dictionary of template_key -> ModuleTemplate
        """
        with cls._lock:
            return cls._templates.copy()
    
    @classmethod
    def remove_template(cls, module_name: str, parent_module: str = None) -> bool:
        """Remove a module template.
        
        Args:
            module_name: Module name
            parent_module: Parent module name
            
        Returns:
            True if template was removed, False if not found
        """
        template_key = cls._get_template_key(module_name, parent_module)
        
        with cls._lock:
            if template_key in cls._templates:
                del cls._templates[template_key]
                logger.debug(f"🗑️ Removed template: {template_key}")
                return True
            return False
    
    @classmethod
    def _get_template_key(cls, module_name: str, parent_module: str = None) -> str:
        """Generate template key."""
        return f"{parent_module}.{module_name}" if parent_module else module_name
    
    # === Instance Management ===
    
    @classmethod
    def create_module(cls,
                     module_name: str,
                     parent_module: str = None,
                     config: Dict[str, Any] = None,
                     auto_initialize: bool = None,
                     force_new: bool = False,
                     use_template: bool = True) -> UIModule:
        """Create or get a UIModule instance.
        
        Args:
            module_name: Module name
            parent_module: Parent module name
            config: Configuration dictionary
            auto_initialize: Auto-initialize module (None = use template/default)
            force_new: Force creation of new instance
            use_template: Use registered template if available
            
        Returns:
            UIModule instance
            
        Raises:
            SmartCashUIError: If creation fails
        """
        instance_key = cls._get_instance_key(module_name, parent_module)
        
        with cls._lock:
            # Check for existing instance
            if not force_new and instance_key in cls._instances:
                existing_ref = cls._instances[instance_key]
                existing_instance = existing_ref()
                
                if existing_instance is not None:
                    logger.debug(f"📦 Reusing existing instance: {instance_key}")
                    
                    # Update config if provided
                    if config:
                        existing_instance.update_config(**config)
                    
                    return existing_instance
                else:
                    # Weak reference is dead, remove it
                    del cls._instances[instance_key]
            
            # Create new instance
            try:
                # Get template if available and requested
                template = None
                if use_template:
                    template = cls.get_template(module_name, parent_module)
                
                # Merge configuration
                final_config = {}
                if template:
                    final_config.update(template.default_config)
                if config:
                    final_config.update(config)
                
                # Determine auto_initialize
                final_auto_initialize = auto_initialize
                if final_auto_initialize is None:
                    final_auto_initialize = template.auto_initialize if template else False
                
                # Create instance
                instance = UIModule(
                    module_name=module_name,
                    parent_module=parent_module,
                    config=final_config,
                    auto_initialize=final_auto_initialize
                )
                
                # Apply template requirements if available
                if template:
                    cls._apply_template_requirements(instance, template)
                
                # Register instance with weak reference
                cls._instances[instance_key] = weakref.ref(
                    instance, 
                    lambda ref: cls._cleanup_instance_ref(instance_key)
                )
                
                logger.info(f"🏭 Created UIModule: {instance_key} (template={template is not None})")
                return instance
                
            except Exception as e:
                error_msg = f"Failed to create UIModule {instance_key}: {e}"
                logger.error(error_msg)
                raise SmartCashUIError(
                    error_msg,
                    context=ErrorContext(
                        component="UIModuleFactory",
                        operation="create_module",
                        details={
                            'module_name': module_name,
                            'parent_module': parent_module,
                            'use_template': use_template,
                            'force_new': force_new
                        }
                    )
                )
    
    @classmethod
    def get_module(cls, module_name: str, parent_module: str = None) -> Optional[UIModule]:
        """Get an existing UIModule instance.
        
        Args:
            module_name: Module name
            parent_module: Parent module name
            
        Returns:
            UIModule instance or None if not found
        """
        instance_key = cls._get_instance_key(module_name, parent_module)
        
        with cls._lock:
            if instance_key in cls._instances:
                instance_ref = cls._instances[instance_key]
                instance = instance_ref()
                
                if instance is not None:
                    return instance
                else:
                    # Clean up dead reference
                    del cls._instances[instance_key]
        
        return None
    
    @classmethod
    def list_instances(cls) -> Dict[str, UIModule]:
        """List all active UIModule instances.
        
        Returns:
            Dictionary of instance_key -> UIModule
        """
        active_instances = {}
        dead_refs = []
        
        with cls._lock:
            for instance_key, instance_ref in cls._instances.items():
                instance = instance_ref()
                if instance is not None:
                    active_instances[instance_key] = instance
                else:
                    dead_refs.append(instance_key)
            
            # Clean up dead references
            for dead_key in dead_refs:
                del cls._instances[dead_key]
        
        return active_instances
    
    @classmethod
    def cleanup_instance(cls, module_name: str, parent_module: str = None) -> bool:
        """Cleanup a specific UIModule instance.
        
        Args:
            module_name: Module name
            parent_module: Parent module name
            
        Returns:
            True if instance was found and cleaned up
        """
        instance_key = cls._get_instance_key(module_name, parent_module)
        
        with cls._lock:
            if instance_key in cls._instances:
                instance_ref = cls._instances[instance_key]
                instance = instance_ref()
                
                if instance is not None:
                    try:
                        instance.cleanup()
                        logger.debug(f"🧹 Cleaned up instance: {instance_key}")
                    except Exception as e:
                        logger.error(f"Error cleaning up instance {instance_key}: {e}")
                
                del cls._instances[instance_key]
                return True
        
        return False
    
    @classmethod
    def cleanup_all_instances(cls) -> int:
        """Cleanup all UIModule instances.
        
        Returns:
            Number of instances cleaned up
        """
        with cls._lock:
            instance_keys = list(cls._instances.keys())
            cleaned_count = 0
            
            for instance_key in instance_keys:
                instance_ref = cls._instances[instance_key]
                instance = instance_ref()
                
                if instance is not None:
                    try:
                        instance.cleanup()
                        cleaned_count += 1
                    except Exception as e:
                        logger.error(f"Error cleaning up instance {instance_key}: {e}")
                
                del cls._instances[instance_key]
            
            logger.info(f"🧹 Cleaned up {cleaned_count} UIModule instances")
            return cleaned_count
    
    @classmethod
    def _get_instance_key(cls, module_name: str, parent_module: str = None) -> str:
        """Generate instance key."""
        return f"{parent_module}.{module_name}" if parent_module else module_name
    
    @classmethod
    def _cleanup_instance_ref(cls, instance_key: str) -> None:
        """Cleanup callback for weak reference."""
        with cls._lock:
            if instance_key in cls._instances:
                del cls._instances[instance_key]
                logger.debug(f"🗑️ Cleaned up dead reference: {instance_key}")
    
    @classmethod
    def _apply_template_requirements(cls, instance: UIModule, template: ModuleTemplate) -> None:
        """Apply template requirements to instance."""
        try:
            # Register required operations as placeholders
            for operation_name in template.required_operations:
                if operation_name not in instance.list_operations():
                    placeholder_op = lambda: f"Placeholder operation: {operation_name}"
                    instance.register_operation(operation_name, placeholder_op)
            
            # Note: Required components would be registered by the actual module implementation
            # We can't create placeholder components without knowing their types
            
        except Exception as e:
            logger.warning(f"Failed to apply template requirements: {e}")
    
    # === Batch Operations ===
    
    @classmethod
    def create_modules_from_config(cls, config_dict: Dict[str, Dict[str, Any]]) -> Dict[str, UIModule]:
        """Create multiple modules from configuration.
        
        Args:
            config_dict: Dictionary of module_key -> module_config
            
        Returns:
            Dictionary of module_key -> UIModule instance
        """
        created_modules = {}
        
        for module_key, module_config in config_dict.items():
            try:
                # Parse module key
                if '.' in module_key:
                    parent_module, module_name = module_key.split('.', 1)
                else:
                    parent_module, module_name = None, module_key
                
                # Extract factory-specific config
                factory_config = module_config.pop('_factory', {})
                
                # Create module
                instance = cls.create_module(
                    module_name=module_name,
                    parent_module=parent_module,
                    config=module_config,
                    **factory_config
                )
                
                created_modules[module_key] = instance
                
            except Exception as e:
                logger.error(f"Failed to create module {module_key}: {e}")
        
        logger.info(f"🏭 Created {len(created_modules)} modules from config")
        return created_modules
    
    @classmethod
    def initialize_all_modules(cls, module_keys: List[str] = None) -> Dict[str, bool]:
        """Initialize multiple modules.
        
        Args:
            module_keys: List of module keys to initialize (None = all)
            
        Returns:
            Dictionary of module_key -> success_status
        """
        instances = cls.list_instances()
        
        if module_keys:
            instances = {k: v for k, v in instances.items() if k in module_keys}
        
        results = {}
        for module_key, instance in instances.items():
            try:
                if not instance.is_ready():
                    instance.initialize()
                results[module_key] = True
            except Exception as e:
                logger.error(f"Failed to initialize module {module_key}: {e}")
                results[module_key] = False
        
        success_count = sum(results.values())
        logger.info(f"🚀 Initialized {success_count}/{len(results)} modules")
        return results
    
    # === Factory State Management ===
    
    @classmethod
    def get_factory_stats(cls) -> Dict[str, Any]:
        """Get factory statistics.
        
        Returns:
            Dictionary with factory statistics
        """
        active_instances = cls.list_instances()
        
        stats = {
            'mode': cls._factory_mode.value,
            'active_instances': len(active_instances),
            'registered_templates': len(cls._templates),
            'instances_by_status': {},
            'shared_methods_count': len(SharedMethodRegistry.list_methods()),
            'created_at': datetime.now().isoformat()
        }
        
        # Count instances by status
        for instance in active_instances.values():
            status = instance.get_status().value
            stats['instances_by_status'][status] = stats['instances_by_status'].get(status, 0) + 1
        
        return stats
    
    @classmethod
    def reset_factory(cls) -> None:
        """Reset factory to initial state."""
        with cls._lock:
            # Cleanup all instances
            cls.cleanup_all_instances()
            
            # Clear templates
            cls._templates.clear()
            
            # Reset mode
            cls._factory_mode = FactoryMode.DEVELOPMENT
            
            logger.info("🔄 Factory reset to initial state")
    
    # === Context Manager Support ===
    
    @classmethod
    def module_context(cls, 
                      module_name: str, 
                      parent_module: str = None,
                      **kwargs) -> 'ModuleContext':
        """Create a context manager for a UIModule.
        
        Args:
            module_name: Module name
            parent_module: Parent module name
            **kwargs: Additional arguments for create_module
            
        Returns:
            ModuleContext instance
        """
        return ModuleContext(cls, module_name, parent_module, **kwargs)


class ModuleContext:
    """Context manager for UIModule instances."""
    
    def __init__(self, factory_cls: Type[UIModuleFactory], 
                 module_name: str, 
                 parent_module: str = None,
                 cleanup_on_exit: bool = True,
                 **kwargs):
        self.factory_cls = factory_cls
        self.module_name = module_name
        self.parent_module = parent_module
        self.cleanup_on_exit = cleanup_on_exit
        self.kwargs = kwargs
        self.module: Optional[UIModule] = None
    
    def __enter__(self) -> UIModule:
        self.module = self.factory_cls.create_module(
            module_name=self.module_name,
            parent_module=self.parent_module,
            **self.kwargs
        )
        return self.module
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cleanup_on_exit and self.module:
            self.factory_cls.cleanup_instance(
                module_name=self.module_name,
                parent_module=self.parent_module
            )


# === Convenience Functions ===

def create_module(module_name: str, 
                 parent_module: str = None,
                 **kwargs) -> UIModule:
    """Convenience function to create a UIModule.
    
    Args:
        module_name: Module name
        parent_module: Parent module name
        **kwargs: Additional arguments for UIModuleFactory.create_module
        
    Returns:
        UIModule instance
    """
    return UIModuleFactory.create_module(
        module_name=module_name,
        parent_module=parent_module,
        **kwargs
    )

def get_module(module_name: str, parent_module: str = None) -> Optional[UIModule]:
    """Convenience function to get a UIModule.
    
    Args:
        module_name: Module name
        parent_module: Parent module name
        
    Returns:
        UIModule instance or None
    """
    return UIModuleFactory.get_module(module_name, parent_module)

def create_template(module_name: str,
                   parent_module: str = None,
                   default_config: Dict[str, Any] = None,
                   required_components: List[str] = None,
                   required_operations: List[str] = None,
                   **kwargs) -> ModuleTemplate:
    """Convenience function to create a ModuleTemplate.
    
    Args:
        module_name: Module name
        parent_module: Parent module name
        default_config: Default configuration
        required_components: Required component names
        required_operations: Required operation names
        **kwargs: Additional ModuleTemplate arguments
        
    Returns:
        ModuleTemplate instance
    """
    return ModuleTemplate(
        module_name=module_name,
        parent_module=parent_module,
        default_config=default_config or {},
        required_components=required_components or [],
        required_operations=required_operations or [],
        **kwargs
    )