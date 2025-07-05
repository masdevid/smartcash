"""
Test configuration and fixtures for UI component tests.

This module provides common test fixtures and utilities for testing UI components
that use ipywidgets with proper traitlets support.
"""
import pytest
from unittest.mock import MagicMock, patch, DEFAULT
from ipywidgets import Widget, VBox, HTML, Button, Output, HBox
from traitlets import List, Unicode, Instance, default, Int, Bool
from typing import Optional, List as TypingList, Any, Dict

class BaseTestWidget(Widget):
    """Base test widget with common functionality for all test widgets."""
    # Common traits for all widgets
    value = Unicode('', help="The value of the widget").tag(sync=True)
    description = Unicode('', help="Description of the widget").tag(sync=True)
    disabled = Bool(False, help="Enable or disable user changes").tag(sync=True)
    visible = Bool(True, help="Show or hide the widget").tag(sync=True)
    
    def __init__(self, **kwargs):
        # Initialize the widget with the default values
        super().__init__(**kwargs)
        
        # Set up layout with default values
        self.layout = HBox().layout
        self.layout.width = None
        self.layout.height = None
        self.layout.margin = None
        self.layout.padding = None
        self.layout.display = 'flex' if self.visible else 'none'
        self.layout.visibility = 'visible' if self.visible else 'hidden'
        
        # Set up mock methods
        self.observe = MagicMock()
        self.unobserve = MagicMock()
        
        # Set any additional attributes from kwargs
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    # Make the widget subscriptable for testing convenience
    def __setitem__(self, key, value):
        setattr(self, key, value)
    
    def __getitem__(self, key):
        return getattr(self, key, None)

class TestVBox(BaseTestWidget):
    """Test VBox widget with proper traitlets support."""
    # Override the children trait to be more permissive for testing
    children = List(trait=Instance(Widget), help="List of widget children").tag(
        sync=True
    )
    
    def __init__(self, children=None, **kwargs):
        # Store the container reference for OperationContainer compatibility
        self.container = self
        
        # Initialize with empty children to avoid traitlets errors
        kwargs['children'] = children or []
        super().__init__(**kwargs)
        
        # Set up layout
        self.layout.flex_flow = 'column'
        self.layout.align_items = 'stretch'
    
    @property
    def _model_name(self):
        return 'VBoxModel'

class TestHTML(BaseTestWidget):
    """Test HTML widget for displaying HTML content."""
    value = Unicode('', help="The HTML content as a string").tag(sync=True)
    
    def __init__(self, value='', **kwargs):
        super().__init__(**kwargs)
        self.value = value
    
    @property
    def _model_name(self):
        return 'HTMLModel'

class TestButton(BaseTestWidget):
    """Test Button widget with click handling."""
    button_style = Unicode('', help="Use a predefined styling for the button").tag(sync=True)
    icon = Unicode('', help="Font-awesome icon name").tag(sync=True)
    tooltip = Unicode('', help="Tooltip caption").tag(sync=True)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.click = MagicMock()
        self.on_click = MagicMock()
        
        # Set default button style if not provided
        if 'button_style' not in kwargs:
            self.button_style = ''
    
    def click(self):
        """Simulate a button click."""
        self.click()
    
    @property
    def _model_name(self):
        return 'ButtonModel'

class TestOutput(BaseTestWidget):
    """Test Output widget for capturing and displaying output."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.clear_output = MagicMock()
        self.append_stdout = MagicMock()
        self.append_stderr = MagicMock()
        self.outputs = []
    
    def clear_output(self, *args, **kwargs):
        """Clear the output area."""
        self.clear_output(*args, **kwargs)
        self.outputs = []
    
    def append_stdout(self, output):
        """Append text to stdout."""
        self.append_stdout(output)
        self.outputs.append(('stdout', output))
    
    def append_stderr(self, output):
        """Append text to stderr."""
        self.append_stderr(output)
        self.outputs.append(('stderr', output))
    
    @property
    def _model_name(self):
        return 'OutputModel'

# Dictionary to track widget creation counts
widget_counts = {
    'VBox': 0,
    'HTML': 0,
    'Button': 0,
    'Output': 0,
    'HBox': 0
}

@pytest.fixture(autouse=True)
def mock_ipywidgets():
    """
    Fixture to mock ipywidgets for all tests with proper traitlets support.
    
    This ensures consistent widget mocking across all component tests.
    """
    # Reset widget counts before each test
    global widget_counts
    widget_counts = {k: 0 for k in widget_counts}
    
    def create_widget(widget_class, name):
        """Helper to create a widget and track its creation."""
        def _create_widget(*args, **kwargs):
            widget_counts[name] += 1
            return widget_class(*args, **kwargs)
        return _create_widget
    
    with (
        patch('ipywidgets.VBox', side_effect=create_widget(TestVBox, 'VBox')) as vbox_mock,
        patch('ipywidgets.HTML', side_effect=create_widget(TestHTML, 'HTML')) as html_mock,
        patch('ipywidgets.Button', side_effect=create_widget(TestButton, 'Button')) as button_mock,
        patch('ipywidgets.Output', side_effect=create_widget(TestOutput, 'Output')) as output_mock,
        patch('ipywidgets.HBox', side_effect=create_widget(TestVBox, 'HBox')) as hbox_mock
    ):
        # Yield the mock objects for assertions if needed
        yield {
            'VBox': vbox_mock,
            'HTML': html_mock,
            'Button': button_mock,
            'Output': output_mock,
            'HBox': hbox_mock,
            'counts': widget_counts
        }

@pytest.fixture
def mock_widgets():
    """Fixture that provides access to the test widget classes."""
    return {
        'VBox': TestVBox,
        'HTML': TestHTML,
        'Button': TestButton,
        'Output': TestOutput,
        'HBox': TestVBox  # Reuse TestVBox for HBox since they're similar
    }
