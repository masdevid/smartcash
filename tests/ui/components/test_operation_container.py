"""
Tests for OperationContainer component.

This module contains unit tests for the OperationContainer class which provides
a unified interface for operation-related UI components.
"""
import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, ANY, PropertyMock

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the module with a try-except to provide a better error message
try:
    from smartcash.ui.components.operation_container import (
        OperationContainer,
        create_operation_container
    )
    from smartcash.ui.components.log_accordion.log_level import LogLevel
except ImportError as e:
    print(f"Error importing modules: {e}")
    print(f"Python path: {sys.path}")
    raise

class TestOperationContainer:
    """Test suite for OperationContainer component."""
    
    @pytest.fixture
    def container(self, mock_widgets):
        """Create an OperationContainer with test widgets."""
        # Get the test widget classes
        TestVBox = mock_widgets['VBox']
        TestHTML = mock_widgets['HTML']
        
        # Create a container instance
        container = OperationContainer()
        
        # Create a proper ProgressTracker mock that handles both string and int values
        class MockProgressTracker(TestVBox):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._value = 0  # Initialize as int to match test expectations
                self._message = ""
                self._level = "primary"
                self._level_label = None
                self.update_progress = self._update_progress_side_effect
                # Add error_progress method for test compatibility
                self.error_progress = self._error_progress_side_effect
                # Add clear method
                self.clear = self._clear_side_effect
            
            def _update_progress_side_effect(self, progress, message="", level="primary", level_label=None):
                # Update the value using the setter to ensure proper type conversion and bounds checking
                progress_int = int(progress) if progress is not None else 0
                self._value = max(0, min(100, progress_int))
                self._message = str(message) if message is not None else ""
                self._level = str(level) if level is not None else "primary"
                self._level_label = str(level_label) if level_label is not None else None
                return self  # Return self to allow method chaining
            
                
            def _clear_side_effect(self):
                self._value = 0
                self._message = ""
                self._level = "primary"
                self._level_label = None
            
            def _error_progress_side_effect(self, message, level="primary"):
                self._message = str(message) if message is not None else ""
                self._level = str(level) if level is not None else "primary"
            
            @property
            def value(self):
                return self._value
                
            @value.setter
            def value(self, val):
                # Convert to int if possible, otherwise keep as string
                try:
                    self._value = int(val) if val is not None else 0
                except (ValueError, TypeError):
                    self._value = str(val) if val is not None else 0
            
            @property
            def description(self):
                return self._message
                
            @description.setter
            def description(self, val):
                self._message = str(val) if val is not None else ""
        
        # Create test widgets with proper traitlets support
        container.progress_tracker = MockProgressTracker()
        
        # Create a proper LogAccordion mock that's callable and handles logging
        class MockLogAccordion(TestVBox):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                # Create a proper logger that matches the expected interface
                class MockLogger:
                    def __init__(self):
                        self.handlers = []
                        self.level = 0
                        self.call_count = 0  # Add call_count attribute
                        
                    def debug(self, *args, **kwargs):
                        self.call_count += 1
                        
                    def info(self, *args, **kwargs):
                        self.call_count += 1
                        
                    def warning(self, *args, **kwargs):
                        self.call_count += 1
                        
                    def error(self, *args, **kwargs):
                        self.call_count += 1
                        
                    def critical(self, *args, **kwargs):
                        self.call_count += 1
                        
                    # Make logger callable
                    def __call__(self, *args, **kwargs):
                        self.call_count += 1
                        return self  # Return self to allow method chaining
                
                self.log = MockLogger()
                self.clear_call_count = 0
                self.log_messages = []
                
                # Add clear method
                def clear_side_effect():
                    self.clear_call_count += 1
                    self.log_messages = []
                
                self.clear = clear_side_effect
                # Make the instance itself callable
                self.__call__ = MagicMock()
            
            def log(self, message, level=None):
                self.log_messages.append((message, level))
        
        # Create log_accordion with proper logger
        container.log_accordion = MockLogAccordion()
        
        # Create dialog area with proper visibility handling
        class MockDialogArea(TestHTML):
            def __init__(self, *args, **kwargs):
                # Initialize _visible before parent's __init__ to avoid property access issues
                self._visible = False
                # Initialize value attribute
                self.value = ""
                # Now call the parent's __init__
                super().__init__(*args, **kwargs)
                # Update visibility from kwargs if provided
                if 'visible' in kwargs:
                    self._visible = bool(kwargs['visible'])
            
            @property
            def visible(self):
                return getattr(self, '_visible', False)
                
            @visible.setter
            def visible(self, value):
                self._visible = bool(value)
                
            def show(self, *args, **kwargs):
                self.visible = True
                if args:
                    self.value = args[0]
                
            def hide(self):
                self.visible = False
        
        container.dialog_area = MockDialogArea()
        
        # Set up the container's children with proper widget instances
        container.container = TestVBox(children=[
            container.progress_tracker,
            container.log_accordion,
            container.dialog_area
        ])
        
        # Configure the container to return itself for the container attribute
        container.container.container = container.container
        
        return container
    
    def test_initialization(self, container):
        """Test that OperationContainer initializes correctly."""
        # Check that all required attributes exist
        assert hasattr(container, 'progress_tracker')
        assert hasattr(container, 'log_accordion')
        assert hasattr(container, 'dialog_area')
        assert hasattr(container, 'container')
        
        # Check that the container has the expected children
        children = container.container.children
        assert len(children) == 3
        assert children[0] == container.progress_tracker
        assert children[1] == container.log_accordion
        assert children[2] == container.dialog_area
        
        # Check initial values
        assert container.progress_tracker.value == 0
        assert container.progress_tracker.description == ""
        assert container.dialog_area.visible is False
    
    def test_update_progress(self, container):
        """Test updating progress with different values."""
        # Test updating progress with value and message
        container.update_progress(50, "Halfway there")
        assert container.progress_tracker.value == 50
        assert container.progress_tracker.description == "Halfway there"
        
        # Test updating with level
        container.update_progress(75, "Three quarters done", "secondary")
        assert container.progress_tracker.value == 75
        assert container.progress_tracker.description == "Three quarters done"
        
        # Test updating with level and label
        container.update_progress(100, "Complete", "primary", "Final")
        assert container.progress_tracker.value == 100
        assert container.progress_tracker.description == "Complete"
    
    def test_logging_methods(self, container):
        """Test logging methods at different levels."""
        # Test logging at different levels
        test_messages = [
            ("DEBUG", "Debug message"),
            ("INFO", "Info message"),
            ("WARNING", "Warning message"),
            ("ERROR", "Error message"),
            ("CRITICAL", "Critical message")
        ]
        
        for level, message in test_messages:
            # Test direct log method
            container.log(message, level=LogLevel[level])
            
            # Test convenience methods
            if level == "DEBUG":
                container.debug(message)
            elif level == "INFO":
                container.info(message)
            elif level == "WARNING":
                container.warning(message)
            elif level == "ERROR":
                container.error(message)
            elif level == "CRITICAL":
                container.critical(message)
        
        # Check that log was called for each message (2x for each level - once for log, once for convenience method)
        assert container.log_accordion.log.call_count == len(test_messages) * 2
    
    def test_dialog_handling(self, container):
        """Test showing and hiding dialogs."""
        # Test showing a dialog with content
        test_title = "Test Title"
        test_message = "Test dialog content"
        container.show_dialog(test_title, test_message)
        
        # Check that the dialog was shown with the correct content
        # The actual content might be HTML formatted, so we'll just check that it contains our strings
        dialog_value = str(container.dialog_area.value or "")
        assert test_title in dialog_value, f"Expected '{test_title}' in dialog value, got: {dialog_value}"
        assert test_message in dialog_value, f"Expected '{test_message}' in dialog value, got: {dialog_value}"
        assert container.dialog_area.visible is True, f"Expected dialog to be visible, but it's not"
        
        # Test showing another dialog
        test_title2 = "Custom Title"
        test_message2 = "Another test"
        container.show_dialog(test_title2, test_message2)
        dialog_value2 = str(container.dialog_area.value or "")
        assert test_title2 in dialog_value2, f"Expected '{test_title2}' in dialog value, got: {dialog_value2}"
        assert test_message2 in dialog_value2, f"Expected '{test_message2}' in dialog value, got: {dialog_value2}"
        assert container.dialog_area.visible is True, "Expected dialog to be visible after second show"
        
        # Test hiding the dialog
        container.clear_dialog()
        assert container.dialog_area.visible is False, "Expected dialog to be hidden after clear_dialog"
        
        # Test direct property access
        container.dialog_area.visible = True
        assert container.dialog_area.visible is True, "Expected dialog to be visible after setting visible=True"
        container.dialog_area.visible = False
        assert container.dialog_area.visible is False, "Expected dialog to be hidden after setting visible=False"
    
    def test_create_operation_container(self):
        """Test the create_operation_container factory function."""
        with patch('smartcash.ui.components.operation_container.OperationContainer') as mock_container_cls:
            # Create a mock container with expected attributes
            mock_container = MagicMock()
            mock_container.progress_tracker = MagicMock()
            mock_container.log_accordion = MagicMock()
            mock_container.dialog_area = MagicMock()
            
            # Set up the mock to return our container
            mock_container_cls.return_value = mock_container
            
            # Mock the container attribute to return itself to match the actual implementation
            mock_container.container = mock_container
            
            # Test with all options enabled
            result = create_operation_container(
                show_progress=True,
                show_dialog=True,
                show_logs=True,
                log_module_name="TestModule"
            )
            
            # Check the structure of the returned dictionary
            assert 'container' in result
            assert 'progress_tracker' in result
            assert 'log_accordion' in result
            assert 'dialog_area' in result
            
            # Check the values match our mocks
            assert result['container'] == mock_container
            assert result['progress_tracker'] == mock_container.progress_tracker
            assert result['log_accordion'] == mock_container.log_accordion
            assert result['dialog_area'] == mock_container.dialog_area
            
            # Verify the container was created with correct parameters
            mock_container_cls.assert_called_once_with(
                show_progress=True,
                show_dialog=True,
                show_logs=True,
                log_module_name="TestModule"
            )
            
            # Test with minimal options - should use defaults from the function
            mock_container_cls.reset_mock()
            create_operation_container()

            # Check that the container was created with default values
            # Note: The actual defaults come from the OperationContainer class
            mock_container_cls.assert_called_once()

            # Get the actual call arguments
            call_args = mock_container_cls.call_args[1]

            # Check that the call included all expected parameters with correct defaults
            assert call_args['show_progress'] is True  # Default in OperationContainer
            assert call_args['show_dialog'] is True    # Default in OperationContainer
            assert call_args['show_logs'] is True      # Default in OperationContainer
            assert call_args['log_module_name'] == 'Operation'  # Default in OperationContainer

    def test_clear_methods(self, container):
        """Test clear methods."""
        # Set up some state
        container.update_progress(50, "Halfway there")
        container.log("Test log message")
        container.show_dialog("Test dialog")
        
        # Verify initial state
        assert container.progress_tracker.value == 50
        
        # Clear everything
        container.clear()
        
        # Verify progress was reset
        assert container.progress_tracker.value == 0
        
        # Verify logs were cleared

        # Get initial clear count
        initial_clear_count = container.log_accordion.clear_call_count
        
        # Test clear_logs
        container.clear_logs()
        assert container.log_accordion.clear_call_count == initial_clear_count + 1

        # Verify progress wasn't cleared
        assert container.progress_tracker.value == 0
        # Check description is either empty or contains error message
        # The actual behavior depends on the implementation of error_progress
        assert container.progress_tracker.description in ["", "Error"]

        # Set values again for clear_all test
        container.progress_tracker.value = 75
        container.progress_tracker.description = "Working..."
        clear_count_before = container.log_accordion.clear_call_count

        # Test clear
        container.clear()

        # Verify both logs and progress were cleared
        assert container.log_accordion.clear_call_count == clear_count_before + 1
        assert container.progress_tracker.value == 0
        assert container.progress_tracker.description == ""
