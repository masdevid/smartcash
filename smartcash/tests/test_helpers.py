"""
Test helpers and utilities for SmartCash test suite.

This module provides shared test fixtures, mocks, and utilities to reduce
duplication across test modules.
"""
import sys
import os
import inspect
import asyncio
from typing import Any, Dict, List, Optional, Type, TypeVar, Union, Callable
from unittest.mock import MagicMock, AsyncMock, patch, PropertyMock, create_autospec
import pytest
import ipywidgets as widgets
from IPython.display import display

# Import common test utilities
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Type variables for generic typing
T = TypeVar('T')
WidgetType = TypeVar('WidgetType', bound=widgets.Widget)

# Common widget mocks used across multiple test modules
class MockWidget:
    """Base mock widget class with common widget properties."""
    def __init__(self, **kwargs):
        self.layout = widgets.Layout()
        self.style = {}
        self.children = []
        self.value = ""
        self.visible = True
        self.disabled = False
        self.__dict__.update(kwargs)
    
    def add_class(self, class_name):
        """Mock add_class method."""
        if not hasattr(self, '_dom_classes'):
            self._dom_classes = []
        if class_name not in self._dom_classes:
            self._dom_classes.append(class_name)
    
    def remove_class(self, class_name):
        """Mock remove_class method."""
        if hasattr(self, '_dom_classes') and class_name in self._dom_classes:
            self._dom_classes.remove(class_name)
    
    def __getitem__(self, key):
        """Allow dict-style access to attributes."""
        return getattr(self, key, None)
    
    def __setitem__(self, key, value):
        """Allow dict-style setting of attributes."""
        setattr(self, key, value)

class MockErrorContext:
    """Mock error context for testing."""
    def __init__(self, error_handler=None):
        self.error_handler = error_handler or MockErrorHandler()
        self.raised = False
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val is not None:
            self.raised = True
            self.error_handler.handle_error(exc_val)
            return True  # Suppress the exception
        return False

class MockErrorHandler:
    """Mock error handler for testing."""
    def __init__(self, **kwargs):
        self.handled_errors = []
        self.show_to_user = True
        self.logger = MagicMock()
        self.__dict__.update(kwargs)
        
    def handle_error(self, error, context=None, show_to_user=True, log_level="error"):
        """Mock handle_error method."""
        error_info = {
            'error': str(error),
            'context': context,
            'show_to_user': show_to_user,
            'log_level': log_level
        }
        self.handled_errors.append(error_info)
        self.logger.log(log_level, f"Error: {error}", exc_info=True)
        
    def get_context(self):
        """Get a new error context."""
        return MockErrorContext(self)

class MockProgressTracker(MockWidget):
    """Mock progress tracker for testing."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.progress = 0.0
        self.description = ""
        self.bar_style = ""
        self.max = 100
        self.min = 0
        self.orientation = "horizontal"
        self._observers = {}
    
    def update(self, progress=None, description=None, bar_style=None):
        """Mock update method."""
        if progress is not None:
            self.progress = progress
        if description is not None:
            self.description = description
        if bar_style is not None:
            self.bar_style = bar_style
    
    def observe(self, callback, names='value'):
        """Mock observe method."""
        if names not in self._observers:
            self._observers[names] = []
        self._observers[names].append(callback)
    
    def unobserve(self, callback, names='value'):
        """Mock unobserve method."""
        if names in self._observers and callback in self._observers[names]:
            self._observers[names].remove(callback)

# Common fixtures
@pytest.fixture
def mock_error_context():
    """Fixture providing a mock error context."""
    return MockErrorContext()


@pytest.fixture
def mock_error_handler():
    """Fixture providing a mock error handler."""
    return MockErrorHandler()

@pytest.fixture
def mock_progress_tracker():
    """Fixture providing a mock progress tracker."""
    return MockProgressTracker()

@pytest.fixture
def mock_widget():
    """Fixture providing a basic mock widget."""
    return MockWidget()

# Helper functions
def create_mock_widget(widget_class=MockWidget, **kwargs):
    """Helper to create a mock widget with the given class and attributes."""
    return widget_class(**kwargs)

def patch_module(module_name, **attrs):
    """Helper to patch a module with the given attributes."""
    return patch.dict(f'sys.modules.{module_name}.__dict__', attrs, clear=False)

# ============================================================================
# Widget Testing Utilities
# ============================================================================

def assert_has_trait(widget: widgets.Widget, trait_name: str, value: Any = None) -> None:
    """Assert that a widget has a trait with an optional value check.
    
    Args:
        widget: The widget to check
        trait_name: Name of the trait to check
        value: If provided, assert that the trait has this value
    """
    assert hasattr(widget, trait_name), f"Widget is missing trait: {trait_name}"
    if value is not None:
        assert getattr(widget, trait_name) == value, \
            f"Expected {trait_name}={value}, got {getattr(widget, trait_name)}"

def assert_widget_visible(widget: widgets.Widget, expected: bool = True) -> None:
    """Assert that a widget's visibility matches the expected value."""
    if hasattr(widget, 'visible'):
        assert widget.visible == expected, \
            f"Widget visibility={widget.visible}, expected {expected}"
    elif hasattr(widget, 'layout') and hasattr(widget.layout, 'visibility'):
        visibility = 'visible' if expected else 'hidden'
        assert widget.layout.visibility == visibility, \
            f"Widget visibility={widget.layout.visibility}, expected {visibility}"
    else:
        pytest.fail("Widget does not have visible attribute or visibility in layout")

def assert_has_class(widget: widgets.Widget, class_name: str) -> None:
    """Assert that a widget has the specified CSS class."""
    if hasattr(widget, '_dom_classes'):
        assert class_name in widget._dom_classes, \
            f"Widget is missing CSS class: {class_name}"
    else:
        pytest.fail("Widget does not support CSS classes (_dom_classes attribute)")

# ============================================================================
# Async Testing Utilities
# ============================================================================

def async_test(coro):
    """
    Wrapper for async test functions to run them in an event loop.

    Usage:
        @async_test
        async def test_something():
            result = await some_async_function()
            assert result == expected
    """
    def wrapper(*args, **kwargs):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro(*args, **kwargs))
        finally:
            loop.close()
    return wrapper

def run_async(coro):
    """Run a coroutine and return its result."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()

# ============================================================================
# Mock Widget Factories
# ============================================================================

def create_mock_widget(
    widget_class: Type[WidgetType] = widgets.Widget,
    **kwargs
) -> WidgetType:
    """
    Create a mock widget with the specified class and attributes.
    
    Args:
        widget_class: The widget class to mock (default: widgets.Widget)
        **kwargs: Attributes to set on the widget
        
    Returns:
        A mock widget instance
    """
    widget = create_autospec(widget_class, instance=True)
    
    # Set default attributes
    widget.layout = MagicMock(spec=widgets.Layout)
    widget.style = MagicMock()
    widget.value = None
    widget.children = []
    widget.observe = MagicMock()
    
    # Apply any provided attributes
    for key, value in kwargs.items():
        setattr(widget, key, value)
    
    return widget

# ============================================================================
# Common Mock Objects
# ============================================================================

class MockButton(MockWidget):
    """Mock button widget for testing."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.description = kwargs.get('description', '')
        self.disabled = kwargs.get('disabled', False)
        self.button_style = kwargs.get('button_style', '')
        self.tooltip = kwargs.get('tooltip', '')
        self.icon = kwargs.get('icon', '')
        self.on_click = MagicMock()
        self.click = MagicMock()

class MockText(MockWidget):
    """Mock text widget for testing."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.value = kwargs.get('value', '')
        self.placeholder = kwargs.get('placeholder', '')
        self.disabled = kwargs.get('disabled', False)

class MockSelect(MockWidget):
    """Mock select/dropdown widget for testing."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.options = kwargs.get('options', [])
        self.value = kwargs.get('value', None)
        self.disabled = kwargs.get('disabled', False)

# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_widget():
    """Fixture providing a basic mock widget."""
    return MockWidget()

@pytest.fixture
def mock_vbox():
    """Fixture providing a mock VBox widget."""
    return create_mock_widget(widgets.VBox)

@pytest.fixture
def mock_hbox():
    """Fixture providing a mock HBox widget."""
    return create_mock_widget(widgets.HBox)

@pytest.fixture
def mock_button():
    """Fixture providing a mock Button widget."""
    return MockButton()

@pytest.fixture
def mock_text():
    """Fixture providing a mock Text widget."""
    return MockText()

@pytest.fixture
def mock_select():
    """Fixture providing a mock Select/Dropdown widget."""
    return MockSelect()

@pytest.fixture
def mock_ui_components():
    """Fixture providing a dictionary of common UI components."""
    return {
        'header': MockWidget(),
        'footer': MockWidget(),
        'main_content': MockWidget(),
        'sidebar': MockWidget(),
        'status_bar': MockWidget(),
        'progress_bar': MockProgressTracker(),
        'error_handler': MockErrorHandler(),
        'logger': MagicMock()
    }

# ============================================================================
# Patch Helpers
# ============================================================================

def patch_widget(widget_class_path: str, **kwargs):
    """
    Patch a widget class with the specified attributes.
    
    Args:
        widget_class_path: Dotted path to the widget class to patch
        **kwargs: Attributes and methods to add to the mock
        
    Returns:
        A mock class that can be used as a context manager or decorator
    """
    return patch(widget_class_path, **kwargs)

def patch_display():
    """
    Patch IPython.display to prevent actual display during tests.
    
    Returns:
        A mock that can be used as a context manager or decorator
    """
    return patch('IPython.display.display')

def patch_import(module_name: str, **attrs):
    """
    Patch a module import with the specified attributes.
    
    Args:
        module_name: Name of the module to patch
        **attrs: Attributes to add to the mock module
        
    Returns:
        A mock module that can be used as a context manager or decorator
    """
    return patch.dict('sys.modules', {module_name: MagicMock(**attrs)})

def patch_class(klass, **kwargs):
    """
    Patch a class with the specified attributes and methods.
    
    Args:
        klass: The class to patch
        **kwargs: Attributes and methods to add to the mock class
        
    Returns:
        A mock class that can be used as a context manager or decorator
    """
    return patch.object(klass, **kwargs)

# ============================================================================
# Assertion Helpers
# ============================================================================

def assert_widget_children(widget, expected_children_count=None, child_types=None):
    """
    Assert that a widget has the expected children.
    
    Args:
        widget: The widget to check
        expected_children_count: Expected number of children (optional)
        child_types: List of expected child types (optional)
    """
    if hasattr(widget, 'children'):
        if expected_children_count is not None:
            assert len(widget.children) == expected_children_count, \
                f"Expected {expected_children_count} children, got {len(widget.children)}"
                
        if child_types is not None:
            assert len(widget.children) == len(child_types), \
                f"Mismatch between children count ({len(widget.children)}) and expected types count ({len(child_types)})"
                
            for child, expected_type in zip(widget.children, child_types):
                assert isinstance(child, expected_type), \
                    f"Expected child of type {expected_type.__name__}, got {type(child).__name__}"

def assert_has_call_with_substring(mock, substring):
    """
    Assert that a mock was called with a string containing the given substring.
    
    Args:
        mock: The mock object to check
        substring: Substring to search for in the call arguments
    """
    call_args_list = [str(call) for call in mock.call_args_list]
    assert any(substring in call for call in call_args_list), \
        f"No call found containing substring: {substring}\nCalls: {call_args_list}"

def assert_has_call_with_dict_containing(mock, expected_dict):
    """
    Assert that a mock was called with a dictionary containing the expected key-value pairs.
    
    Args:
        mock: The mock object to check
        expected_dict: Dictionary of expected key-value pairs
    """
    for call in mock.call_args_list:
        args, kwargs = call
        all_args = {}
        
        # Combine all positional and keyword arguments
        for arg in args:
            if isinstance(arg, dict):
                all_args.update(arg)
        all_args.update(kwargs)
        
        # Check if all expected key-value pairs are present
        if all(k in all_args and all_args[k] == v for k, v in expected_dict.items()):
            return
            
    # If we get here, no matching call was found
    assert False, f"No call found with dictionary containing: {expected_dict}"
