"""
Validation mixin for UI modules.

Provides standard validation decorators and checks for common operations.
"""

from typing import Dict, Any, Optional, Callable
from functools import wraps


def requires_initialization(func: Callable) -> Callable:
    """
    Decorator that ensures the module is initialized before execution.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not getattr(self, '_is_initialized', False):
            if hasattr(self, 'initialize'):
                if not self.initialize():
                    raise RuntimeError("Module initialization failed")
            else:
                raise RuntimeError("Module not initialized and no initialize method available")
        
        return func(self, *args, **kwargs)
    
    return wrapper


def requires_config_handler(func: Callable) -> Callable:
    """
    Decorator that ensures the config handler is available before execution.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not getattr(self, '_config_handler', None):
            raise RuntimeError("Config handler not available")
        
        return func(self, *args, **kwargs)
    
    return wrapper


def requires_operation_manager(func: Callable) -> Callable:
    """
    Decorator that ensures the operation manager is available before execution.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not getattr(self, '_operation_manager', None):
            raise RuntimeError("Operation manager not available")
        
        return func(self, *args, **kwargs)
    
    return wrapper


def requires_ui_components(func: Callable) -> Callable:
    """
    Decorator that ensures UI components are available before execution.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not getattr(self, '_ui_components', None):
            raise RuntimeError("UI components not available")
        
        return func(self, *args, **kwargs)
    
    return wrapper


class ValidationMixin:
    """
    Mixin providing common validation functionality.
    
    This mixin provides:
    - Validation decorators for common checks
    - Component validation methods
    - Configuration validation
    - State validation
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._validation_errors: Dict[str, str] = {}
    
    def validate_initialization(self) -> bool:
        """
        Validate that the module is properly initialized.
        
        Returns:
            True if initialization is valid
        """
        try:
            errors = []
            
            # Check basic initialization
            if not getattr(self, '_is_initialized', False):
                errors.append("Module not initialized")
            
            # Check config handler
            if not getattr(self, '_config_handler', None):
                errors.append("Config handler not available")
            
            # Check UI components
            if not getattr(self, '_ui_components', None):
                errors.append("UI components not available")
            
            # Store errors
            if errors:
                self._validation_errors['initialization'] = "; ".join(errors)
                return False
            
            return True
            
        except Exception as e:
            self._validation_errors['initialization'] = str(e)
            return False
    
    def validate_config(self, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Validate configuration.
        
        Args:
            config: Configuration to validate (uses current if None)
            
        Returns:
            Validation result dictionary
        """
        try:
            if config is None:
                config = getattr(self, '_merged_config', {})
            
            # Basic validation
            if not isinstance(config, dict):
                return {
                    'valid': False,
                    'message': 'Configuration must be a dictionary'
                }
            
            # Use config handler validation if available
            if hasattr(self, '_config_handler') and self._config_handler:
                if hasattr(self._config_handler, 'validate_config'):
                    return self._config_handler.validate_config(config)
            
            # Basic validation passed
            return {
                'valid': True,
                'message': 'Configuration is valid'
            }
            
        except Exception as e:
            return {
                'valid': False,
                'message': f'Configuration validation failed: {str(e)}'
            }
    
    def validate_ui_components(self) -> Dict[str, Any]:
        """
        Validate UI components.
        
        Returns:
            Validation result dictionary
        """
        try:
            if not hasattr(self, '_ui_components') or not self._ui_components:
                return {
                    'valid': False,
                    'message': 'UI components not available'
                }
            
            errors = []
            
            # Check required components
            required_components = getattr(self, '_required_components', [])
            for component_name in required_components:
                if component_name not in self._ui_components:
                    errors.append(f"Required component missing: {component_name}")
                elif self._ui_components[component_name] is None:
                    errors.append(f"Required component is None: {component_name}")
            
            # Check component types
            for component_name, component in self._ui_components.items():
                if component is not None:
                    # Check if component has expected attributes
                    if component_name.endswith('_container'):
                        if not hasattr(component, 'children') and not hasattr(component, 'layout'):
                            errors.append(f"Component {component_name} does not appear to be a valid widget")
            
            if errors:
                return {
                    'valid': False,
                    'message': "; ".join(errors)
                }
            
            return {
                'valid': True,
                'message': 'UI components are valid'
            }
            
        except Exception as e:
            return {
                'valid': False,
                'message': f'UI component validation failed: {str(e)}'
            }
    
    def validate_operation_manager(self) -> Dict[str, Any]:
        """
        Validate operation manager.
        
        Returns:
            Validation result dictionary
        """
        try:
            if not hasattr(self, '_operation_manager') or not self._operation_manager:
                return {
                    'valid': False,
                    'message': 'Operation manager not available'
                }
            
            errors = []
            
            # Check required methods
            required_methods = ['initialize', 'cleanup', 'log']
            for method_name in required_methods:
                if not hasattr(self._operation_manager, method_name):
                    errors.append(f"Operation manager missing method: {method_name}")
            
            # Check if initialized
            if hasattr(self._operation_manager, 'is_initialized'):
                if not self._operation_manager.is_initialized():
                    errors.append("Operation manager not initialized")
            
            if errors:
                return {
                    'valid': False,
                    'message': "; ".join(errors)
                }
            
            return {
                'valid': True,
                'message': 'Operation manager is valid'
            }
            
        except Exception as e:
            return {
                'valid': False,
                'message': f'Operation manager validation failed: {str(e)}'
            }
    
    def validate_all(self) -> Dict[str, Any]:
        """
        Validate all module components.
        
        Returns:
            Comprehensive validation result
        """
        results = {
            'initialization': self.validate_initialization(),
            'config': self.validate_config(),
            'ui_components': self.validate_ui_components(),
            'operation_manager': self.validate_operation_manager()
        }
        
        # Overall validation
        all_valid = all(
            result.get('valid', False) if isinstance(result, dict) else result
            for result in results.values()
        )
        
        return {
            'valid': all_valid,
            'results': results,
            'message': 'All validations passed' if all_valid else 'Some validations failed'
        }
    
    def get_validation_errors(self) -> Dict[str, str]:
        """
        Get current validation errors.
        
        Returns:
            Dictionary of validation errors
        """
        return self._validation_errors.copy()
    
    def clear_validation_errors(self) -> None:
        """Clear all validation errors."""
        self._validation_errors.clear()
    
    def add_validation_error(self, key: str, message: str) -> None:
        """
        Add a validation error.
        
        Args:
            key: Error key
            message: Error message
        """
        self._validation_errors[key] = message
    
    def has_validation_errors(self) -> bool:
        """
        Check if there are any validation errors.
        
        Returns:
            True if there are validation errors
        """
        return len(self._validation_errors) > 0