"""
Mock implementation of core errors module for testing.
This needs to be imported before any module that uses handle_errors.
"""
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union, cast
from enum import Enum, auto

# Type variable for generic function typing
F = TypeVar('F', bound=Callable[..., Any])

# Mock ErrorLevel enum
class ErrorLevel(Enum):
    DEBUG = auto()
    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()

# Mock ErrorContext
class ErrorContext:
    def __init__(self, **kwargs):
        self.details = kwargs or {}

# Mock SmartCashUIError
class SmartCashUIError(Exception):
    """Base exception class for SmartCash UI errors."""
    def __init__(self, message: str, error_code: Optional[str] = None, **kwargs):
        self.message = message
        self.error_code = error_code
        self.extra = kwargs
        super().__init__(self.message)

    def __str__(self):
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message


class UIComponentError(SmartCashUIError):
    """Exception raised for errors in UI components."""
    def __init__(self, message: str, component_name: Optional[str] = None, **kwargs):
        self.component_name = component_name
        error_code = f"UI_COMPONENT_ERROR_{component_name.upper()}" if component_name else "UI_COMPONENT_ERROR"
        super().__init__(message=message, error_code=error_code, **kwargs)

# Mock CoreErrorHandler
class CoreErrorHandler:
    def __init__(self, module_name: str, **kwargs):
        self.module_name = module_name
        self.kwargs = kwargs
        
    def handle_error(self, error_msg: str, level: ErrorLevel = ErrorLevel.ERROR, **kwargs):
        level_name = level.name if hasattr(level, 'name') else level
        print(f"[{level_name}] {error_msg}")
        
    def __call__(self, error_msg: str, level: ErrorLevel = ErrorLevel.ERROR, **kwargs):
        self.handle_error(error_msg, level, **kwargs)

# Helper function to create a decorated function
def _create_wrapper(func: F, **decorator_kwargs: Any) -> F:
    """Create a wrapped function with error handling.
    
    Args:
        func: The function to wrap
        **decorator_kwargs: Decorator arguments (error_msg, level, reraise, etc.)
    """
    # Extract the parameters we care about
    error_msg = decorator_kwargs.get('error_msg')
    level = decorator_kwargs.get('level', ErrorLevel.ERROR)
    reraise = decorator_kwargs.get('reraise', True)
    log_error = decorator_kwargs.get('log_error', True)
    
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Format error message
            msg = error_msg or getattr(func, '__doc__', f"Error in {func.__name__}")
            formatted_msg = f"{msg}: {str(e)}"
            
            # Log the error if enabled
            if log_error:
                level_name = level.name if hasattr(level, 'name') else level
                print(f"[{level_name}] {formatted_msg}")
            
            # Re-raise if configured to do so
            if reraise:
                raise
            
            # Otherwise continue execution
            return None
    
    return cast(F, wrapper)

# Mock handle_errors decorator with support for both @handle_errors and @handle_errors(...)
def handle_errors(
    func_or_error_msg: Optional[Union[F, str]] = None,
    level: ErrorLevel = ErrorLevel.ERROR,
    reraise: bool = True,
    log_error: bool = True,
    create_ui: bool = False,
    handler: Optional[Any] = None,
    context_attr: Optional[str] = None,
    error_msg: Optional[str] = None,
    **kwargs: Any
) -> Union[Callable[[F], F], F]:
    """Mock handle_errors decorator for testing with support for both direct and factory usage.
    
    Supports both:
    - @handle_errors
    - @handle_errors(...with params...)
    """
    # If called as @handle_errors(...) with parameters
    def decorator(func: F) -> F:
        # Build kwargs for the wrapper
        wrapper_kwargs = {
            'level': level,
            'reraise': reraise,
            'log_error': log_error,
            'create_ui': create_ui,
            'handler': handler,
            'context_attr': context_attr,
            'error_msg': error_msg,
            **kwargs
        }
        
        # If first argument is a string, it's the error message
        if isinstance(func_or_error_msg, str) and not error_msg:
            wrapper_kwargs['error_msg'] = func_or_error_msg
        
        return _create_wrapper(func, **wrapper_kwargs)
    
    # Handle the case where we're called with a string message directly
    if isinstance(func_or_error_msg, str):
        return lambda f: handle_errors(
            level=level,
            reraise=reraise,
            log_error=log_error,
            create_ui=create_ui,
            handler=handler,
            context_attr=context_attr,
            error_msg=func_or_error_msg,
            **kwargs
        )(f)
    
    return decorator

# Mock handle_component_validation
def handle_component_validation(
    component: Any, 
    component_name: str,
    required_attrs: Optional[list] = None
) -> bool:
    """Mock handle_component_validation function for testing."""
    if component is None:
        print(f"[ERROR] Component '{component_name}' is None")
        return False
    
    if required_attrs is None:
        required_attrs = []
    
    missing_attrs = [attr for attr in required_attrs if not hasattr(component, attr)]
    
    if missing_attrs:
        print(f"[ERROR] Component '{component_name}' is missing required attributes: {', '.join(missing_attrs)}")
        return False
    
    return True

# Mock validate_ui_components
def validate_ui_components(
    components: Dict[str, Any],
    required_components: Optional[List[str]] = None,
    required_attrs: Optional[Dict[str, List[str]]] = None
) -> Dict[str, bool]:
    """Mock validate_ui_components function for testing.
    
    Args:
        components: Dictionary of component name to component object
        required_components: List of component names that must be present
        required_attrs: Dict mapping component names to lists of required attributes
        
    Returns:
        Dict mapping component names to validation status (True/False)
    """
    if required_components is None:
        required_components = []
    if required_attrs is None:
        required_attrs = {}
        
    results = {}
    
    # Check all required components exist
    for comp_name in required_components:
        if comp_name not in components:
            print(f"[ERROR] Required component '{comp_name}' is missing")
            results[comp_name] = False
    
    # Validate each component
    for comp_name, component in components.items():
        if component is None:
            print(f"[ERROR] Component '{comp_name}' is None")
            results[comp_name] = False
            continue
            
        # Check required attributes if specified
        if comp_name in required_attrs:
            missing_attrs = [
                attr for attr in required_attrs[comp_name] 
                if not hasattr(component, attr)
            ]
            
            if missing_attrs:
                print(f"[ERROR] Component '{comp_name}' is missing required attributes: {', '.join(missing_attrs)}")
                results[comp_name] = False
                continue
                
        results[comp_name] = True
    
    return results

# Mock ErrorComponent class
class ErrorComponent:
    """Mock ErrorComponent class for testing."""
    
    def __init__(self, error_msg: str, level: str = "error", **kwargs):
        self.error_msg = error_msg
        self.level = level
        self.kwargs = kwargs
        
    def __str__(self):
        return f"<MockErrorComponent: {self.level.upper()}: {self.error_msg}>"
    
    def display(self):
        """Display the error component."""
        print(f"[{self.level.upper()}] {self.error_msg}")
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'error_msg': self.error_msg,
            'level': self.level,
            **self.kwargs
        }

def create_error_component(error_msg: str, level: str = "error", **kwargs) -> ErrorComponent:
    """Mock create_error_component function for testing."""
    return ErrorComponent(error_msg, level, **kwargs)

# Mock safe_component_operation
def safe_component_operation(
    component_name: str,
    operation: str,
    error_component: Optional[Any] = None,
    **context_kwargs
) -> Callable[[Callable], Callable]:
    """Mock safe_component_operation decorator for testing."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                print(f"[ERROR] Component operation failed: {component_name}.{operation}: {e}")
                if error_component is not None:
                    return error_component
                raise
        return wrapper
    return decorator

# Mock get_error_handler
def get_error_handler(module_name: str, **kwargs):
    """Mock get_error_handler function for testing."""
    return MagicMock()

# Mock the errors module
mock_errors = {
    'ErrorLevel': ErrorLevel,
    'ErrorContext': ErrorContext,
    'SmartCashUIError': SmartCashUIError,
    'UIComponentError': UIComponentError,
    'CoreErrorHandler': CoreErrorHandler,
    'ErrorComponent': ErrorComponent,
    'create_error_component': create_error_component,
    'handle_errors': handle_errors,
    'handle_component_validation': handle_component_validation,
    'validate_ui_components': validate_ui_components,
    'safe_component_operation': safe_component_operation,
    'get_error_handler': get_error_handler,
    # Add other exports from the real errors module as needed
}

# Ensure the module has a __file__ attribute
import os
import sys
mock_errors['__file__'] = os.path.abspath(__file__)

try:
    import sys
    sys.modules['smartcash.ui.core.errors'] = type('MockErrorsModule', (), mock_errors)()
    
    # Also update any direct imports
    import smartcash.ui.core.errors as errors
    for name, value in mock_errors.items():
        setattr(errors, name, value)
        
except ImportError:
    pass
