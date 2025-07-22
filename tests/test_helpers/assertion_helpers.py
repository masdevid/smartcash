"""
File: tests/test_helpers/assertion_helpers.py
Description: Custom assertion helpers for tests.
"""
from typing import Any, Dict, List, Sequence, Optional, Callable
from unittest.mock import call


def assert_called_with_any_order(
    mock_obj: Any,
    expected_calls: List[call],
    strict: bool = False
) -> None:
    """Assert that the mock was called with the expected calls in any order.
    
    Args:
        mock_obj: The mock object to check
        expected_calls: List of expected call objects
        strict: If True, checks that exactly these calls were made (no extra calls)
    """
    mock_calls = mock_obj.mock_calls
    
    # Check if all expected calls are present in any order
    for expected in expected_calls:
        assert expected in mock_calls, f"Expected call {expected} not found in {mock_calls}"
    
    # If strict, check that no extra calls were made
    if strict:
        assert len(mock_calls) == len(expected_calls), \
            f"Expected exactly {len(expected_calls)} calls, got {len(mock_calls)}"


def assert_dict_subset(
    actual: Dict[Any, Any],
    expected_subset: Dict[Any, Any],
    path: str = ""
) -> None:
    """Assert that the actual dictionary contains all items from the expected subset.
    
    Args:
        actual: The actual dictionary to check
        expected_subset: The expected subset of key-value pairs
        path: Current path in the dictionary (used for recursive calls)
    """
    for key, expected_value in expected_subset.items():
        current_path = f"{path}.{key}" if path else key
        
        assert key in actual, f"Key '{current_path}' not found in actual dict"
        
        if isinstance(expected_value, dict):
            # Recursively check nested dictionaries
            assert isinstance(actual[key], dict), \
                f"Expected dict at '{current_path}', got {type(actual[key])}"
            assert_dict_subset(actual[key], expected_value, current_path)
        else:
            assert actual[key] == expected_value, \
                f"Mismatch at '{current_path}': expected {expected_value}, got {actual[key]}"


def assert_async_called_with(mock_obj: Any, *args, **kwargs) -> None:
    """Assert that an async function was called with the specified arguments.
    
    Args:
        mock_obj: The mock async function
        *args: Expected positional arguments
        **kwargs: Expected keyword arguments
    """
    # For async functions, the call is stored in the mock's await_args_list
    call_found = False
    
    for call_item in mock_obj.await_args_list:
        call_args = call_item[0]  # Get the (args, kwargs) tuple
        call_kwargs = call_item[1] if len(call_item) > 1 else {}
        
        if (call_args == args and call_kwargs == kwargs):
            call_found = True
            break
    
    assert call_found, f"Expected call with args={args}, kwargs={kwargs} not found"
