"""
Input validation and safe operation utilities for SmartCash UI Core.

This module provides functions for validating inputs and safely performing
operations with proper error handling.
"""
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union
from functools import wraps

from .handlers import get_error_handler
from .enums import ErrorLevel

# Type variables for generic function typing
T = TypeVar('T')
F = TypeVar('F', bound=Callable[..., Any])


def validate_not_none(
    value: Any, 
    name: str, 
    error_msg: Optional[str] = None,
    level: ErrorLevel = ErrorLevel.ERROR
) -> None:
    """
    Validate that a value is not None.
    
    Args:
        value: The value to validate.
        name: The name of the parameter for error messages.
        error_msg: Custom error message. If None, a default will be used.
        level: The error level to use if validation fails.
        
    Raises:
        ValueError: If the value is None.
    """
    if value is None:
        msg = error_msg or f"{name} cannot be None"
        get_error_handler().handle_error(
            msg,
            level=level,
            fail_fast=True,
            param_name=name,
            value=value
        )


def validate_type(
    value: Any, 
    expected_type: Union[Type, tuple[Type, ...]], 
    name: str,
    error_msg: Optional[str] = None,
    level: ErrorLevel = ErrorLevel.ERROR
) -> None:
    """
    Validate that a value is of the expected type.
    
    Args:
        value: The value to validate.
        expected_type: The expected type or tuple of types.
        name: The name of the parameter for error messages.
        error_msg: Custom error message. If None, a default will be used.
        level: The error level to use if validation fails.
        
    Raises:
        TypeError: If the value is not of the expected type.
    """
    if not isinstance(value, expected_type):
        type_names = [t.__name__ for t in expected_type] if isinstance(expected_type, tuple) else [expected_type.__name__]
        expected = " or ".join(type_names)
        actual = type(value).__name__
        
        msg = error_msg or f"{name} must be of type {expected}, got {actual}"
        
        get_error_handler().handle_error(
            msg,
            level=level,
            fail_fast=True,
            param_name=name,
            expected_type=expected,
            actual_type=actual,
            value=value
        )


def validate_condition(
    condition: bool,
    error_msg: str,
    level: ErrorLevel = ErrorLevel.ERROR,
    **context: Any
) -> None:
    """
    Validate that a condition is True.
    
    Args:
        condition: The condition to validate.
        error_msg: The error message to use if validation fails.
        level: The error level to use if validation fails.
        **context: Additional context to include with the error.
        
    Raises:
        ValueError: If the condition is False.
    """
    if not condition:
        get_error_handler().handle_error(
            error_msg,
            level=level,
            fail_fast=True,
            **context
        )


def handle_component_validation(
    component: Any, 
    component_name: str,
    required_attrs: Optional[List[str]] = None
) -> bool:
    """
    Validate a UI component has the required attributes.
    
    Args:
        component: The component to validate.
        component_name: The name of the component for error messages.
        required_attrs: List of required attribute names.
        
    Returns:
        bool: True if the component is valid, False otherwise.
    """
    if component is None:
        get_error_handler().handle_error(
            f"Component '{component_name}' is None",
            level=ErrorLevel.ERROR,
            fail_fast=False,
            component_name=component_name
        )
        return False
    
    if not required_attrs:
        return True
    
    missing_attrs = [attr for attr in required_attrs if not hasattr(component, attr)]
    
    if missing_attrs:
        get_error_handler().handle_error(
            f"Component '{component_name}' is missing required attributes: {', '.join(missing_attrs)}",
            level=ErrorLevel.ERROR,
            fail_fast=False,
            component_name=component_name,
            missing_attrs=missing_attrs,
            required_attrs=required_attrs
        )
        return False
    
    return True


def safe_component_operation(
    component: Any,
    operation: str,
    *args: Any,
    component_name: Optional[str] = None,
    default: Any = None,
    **kwargs: Any
) -> Any:
    """
    Safely perform an operation on a UI component.
    
    Args:
        component: The component to operate on.
        operation: The name of the operation/method to call.
        *args: Positional arguments to pass to the operation.
        component_name: Optional name of the component for error messages.
        default: Default value to return if the operation fails.
        **kwargs: Keyword arguments to pass to the operation.
        
    Returns:
        The result of the operation, or the default value if it fails.
    """
    comp_name = component_name or str(component)
    
    if not hasattr(component, operation):
        get_error_handler().handle_error(
            f"Component '{comp_name}' has no operation '{operation}'",
            level=ErrorLevel.ERROR,
            fail_fast=False,
            component=component,
            operation=operation
        )
        return default
    
    try:
        method = getattr(component, operation)
        return method(*args, **kwargs)
    except Exception as e:
        get_error_handler().handle_error(
            f"Error in component '{comp_name}' operation '{operation}': {str(e)}",
            level=ErrorLevel.ERROR,
            fail_fast=False,
            component=component,
            operation=operation,
            error=str(e)
        )
        return default


def validate_ui_components(components: Dict[str, Any]) -> bool:
    """
    Validate a dictionary of UI components.
    
    Args:
        components: Dictionary of component_name: component pairs.
        
    Returns:
        bool: True if all components are valid, False otherwise.
    """
    if not components:
        get_error_handler().handle_error(
            "No components provided for validation",
            level=ErrorLevel.WARNING,
            fail_fast=False
        )
        return False
    
    all_valid = True
    
    for name, component in components.items():
        if component is None:
            get_error_handler().handle_error(
                f"Component '{name}' is None",
                level=ErrorLevel.ERROR,
                fail_fast=False,
                component_name=name
            )
            all_valid = False
    
    return all_valid
