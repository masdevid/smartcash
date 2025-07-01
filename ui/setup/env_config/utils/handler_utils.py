"""
Common utilities for env_config handlers.

This module provides utilities to reduce code duplication across handlers,
including error handling and result formatting.
"""

from typing import Dict, Any, Callable, Awaitable
from functools import wraps
import logging

from smartcash.ui.utils.ui_logger import UILogger


def with_common_error_handling(
    func: Callable[..., Awaitable[Dict[str, Any]]]
) -> Callable[..., Awaitable[Dict[str, Any]]]:
    """Decorator to add common error handling to handler methods.
    
    This decorator ensures consistent error handling and response formatting
    across all handler methods that return a result dictionary.
    """
    @wraps(func)
    async def wrapper(self, *args, **kwargs) -> Dict[str, Any]:
        try:
            result = await func(self, *args, **kwargs)
            
            # Ensure result has required fields
            if not isinstance(result, dict):
                result = {'result': result}
                
            if 'status' not in result:
                result['status'] = 'success'
                
            return result
            
        except Exception as e:
            error_msg = f"Error in {func.__name__}: {str(e)}"
            
            # Log the error if the handler has a logger
            if hasattr(self, '_logger') and isinstance(self._logger, (logging.Logger, UILogger)):
                self._logger.error(error_msg, exc_info=True)
                
            # If there's an error handler, use it
            if hasattr(self, '_handle_error'):
                self._handle_error(e, error_msg, operation=func.__name__)
                
            return {
                'status': 'error',
                'message': error_msg,
                'error': str(e),
                'operation': func.__name__
            }
            
    return wrapper


def standardize_result(
    result: Any,
    operation: str,
    **additional_fields
) -> Dict[str, Any]:
    """Standardize the result dictionary with common fields.
    
    Args:
        result: The result to standardize (can be a dict or any other value)
        operation: Name of the operation that produced the result
        **additional_fields: Additional fields to include in the result
        
    Returns:
        A standardized result dictionary
    """
    if isinstance(result, dict):
        result_dict = result
    else:
        result_dict = {'result': result}
    
    # Ensure required fields exist
    result_dict.setdefault('status', 'success')
    result_dict.setdefault('operation', operation)
    
    # Add timestamp if not present
    if 'timestamp' not in result_dict:
        from datetime import datetime
        result_dict['timestamp'] = datetime.utcnow().isoformat()
    
    # Add any additional fields
    result_dict.update(additional_fields)
    
    return result_dict
