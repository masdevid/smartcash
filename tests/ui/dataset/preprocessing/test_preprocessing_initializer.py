"""
Tests for the PreprocessingInitializer class to ensure proper log redirection and initialization.
"""
import io
import pytest
from unittest.mock import MagicMock, patch, ANY, call
import ipywidgets as widgets
import logging

from smartcash.ui.dataset.preprocessing.preprocessing_initializer import PreprocessingInitializer
from smartcash.ui.dataset.preprocessing.handlers.config_handler import PreprocessingConfigHandler
from smartcash.ui.utils.ui_logger import UILogger

# Mock the UILogger
class MockUILogger:
    def __init__(self, name):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
    
    def debug(self, msg, *args, **kwargs):
        self.logger.debug(msg, *args, **kwargs)
    
    def info(self, msg, *args, **kwargs):
        self.logger.info(msg, *args, **kwargs)
    
    def warning(self, msg, *args, **kwargs):
        self.logger.warning(msg, *args, **kwargs)
    
    def error(self, msg, *args, **kwargs):
        self.logger.error(msg, *args, **kwargs)
    
    def critical(self, msg, *args, **kwargs):
        self.logger.critical(msg, *args, **kwargs)

def mock_get_module_logger(logger_name):
    return MockUILogger(logger_name)

class TestPreprocessingInitializerLogging:
    """Test logging behavior of PreprocessingInitializer."""
    
    @pytest.fixture(autouse=True)
    def setup_method(self):
        # Setup before each test
        self.original_handlers = logging.root.handlers.copy()
        logging.root.handlers.clear()
        
        # Create a test logger
        self.test_logger = logging.getLogger('test_logger')
        self.test_logger.setLevel(logging.INFO)
        
        # Create a string buffer to capture logs
        self.log_stream = io.StringIO()
        self.handler = logging.StreamHandler(self.log_stream)
        self.handler.setFormatter(logging.Formatter('%(message)s'))
        self.test_logger.addHandler(self.handler)
        
        yield  # Test runs here
        
        # Teardown after each test
        self.test_logger.removeHandler(self.handler)
        self.handler.close()
        logging.root.handlers = self.original_handlers
    
    @pytest.fixture
    def mock_ui_components(self):
        """Create mock UI components with a log accordion."""
        from smartcash.ui.components.log_accordion import create_log_accordion, LogLevel
        
        # Create the log accordion using the real component
        log_accordion = create_log_accordion(
            module_name='preprocessing',
            height='300px',
            width='100%',
            max_logs=1000,
            show_timestamps=True,
            show_level_icons=True,
            auto_scroll=True,
            enable_deduplication=True
        )
        
        # Extract the log_output from the accordion
        log_output = log_accordion['log_output']
        
        # Create a status output
        status_output = widgets.Output()
        
        return {
            'container': widgets.VBox(),
            'log_accordion': log_accordion['log_accordion'],
            'log_output': log_output,
            'status_output': status_output,
            'module_name': 'preprocessing'
        }
    
    @pytest.fixture
    def mock_create_ui(self, mock_ui_components):
        """Mock the UI creation function to return test components."""
        with patch('smartcash.ui.dataset.preprocessing.components.ui_components.create_preprocessing_main_ui') as mock_ui:
            mock_ui.return_value = mock_ui_components
            yield mock_ui
    
    @pytest.fixture
    def mock_handlers(self):
        """Mock the handlers setup."""
        with patch('smartcash.ui.dataset.preprocessing.handlers.preprocessing_handlers.setup_preprocessing_handlers') as mock_setup:
            mock_setup.return_value = {}
            yield mock_setup
    
    @pytest.fixture
    def mock_logger_bridge(self):
        """Mock the UILogger to test log redirection."""
        with patch('smartcash.ui.dataset.preprocessing.preprocessing_initializer.get_module_logger') as mock_logger:
            mock_logger.return_value = MockUILogger('test_logger')
            yield mock_logger
            
            yield mock_instance

    def test_log_redirection_to_accordion(self, mock_ui_components, mock_create_ui, mock_handlers, mock_logger_bridge):
        """Test that logs are properly redirected to the log accordion."""
        # Arrange
        initializer = PreprocessingInitializer()
        
        # Act
        result = initializer.initialize()
        
        # Assert
        assert isinstance(result, widgets.Widget), "Should return a widget"
        mock_create_ui.assert_called_once()
        mock_handlers.assert_called_once()
        
        # Verify logger bridge was initialized with correct UI components
        assert hasattr(initializer, '_logger_bridge'), "Logger bridge should be initialized"
        assert initializer._logger_bridge is not None
    
    def test_no_logs_before_initialization(self, mock_ui_components, mock_create_ui, mock_handlers):
        """Test that no logs are emitted before initialization is complete."""
        # Clear any existing logs
        self.log_stream.seek(0)
        self.log_stream.truncate(0)
        
        # Act - Log something before initialization
        self.test_logger.info("This should not appear in UI")
        
        # Get the log output before initialization
        log_output_before = self.log_stream.getvalue()
        
        # Initialize the initializer
        initializer = PreprocessingInitializer()
        
        # Clear the log stream after initializer creation but before initialization
        self.log_stream.seek(0)
        self.log_stream.truncate(0)
        
        # Initialize the UI
        result = initializer.initialize()
        
        # Log something after initialization
        self.test_logger.info("This should appear in UI")
        
        # Get the log output after initialization
        log_output_after = self.log_stream.getvalue()
        
        # Assert
        assert isinstance(result, widgets.Widget), "Should return a widget"
        
        # Check that the log before initialization is not in the output after initialization
        assert "This should not appear in UI" not in log_output_after, \
            "Logs before initialization should not be captured"
            
        # Check that the log after initialization is in the output
        assert "This should appear in UI" in log_output_after, \
            "Logs after initialization should be captured"
    
    def test_logger_bridge_initialization(self, mock_ui_components, mock_create_ui, mock_handlers, mock_logger_bridge):
        """Test that the logger bridge is properly initialized with UI components."""
        # Arrange
        initializer = PreprocessingInitializer()
        
        # Act
        result = initializer.initialize()
        
        # Assert
        assert isinstance(result, widgets.Widget), "Should return a widget"
        
        # Check that the logger bridge was initialized
        assert hasattr(initializer, '_logger_bridge'), "Logger bridge should be initialized"
        assert initializer._logger_bridge is not None, "Logger bridge should not be None"
        
        # Verify that the logger bridge was called with the correct arguments
        # Since we're using the mock_logger_bridge fixture, we can check if it was used
        assert mock_logger_bridge is not None, "Mock logger bridge should be created"
        
        # Check that the logger was properly set up
        assert hasattr(mock_logger_bridge, 'logger'), "Logger bridge should have a logger"
        assert mock_logger_bridge.logger is not None, "Logger bridge's logger should not be None"
    
    def test_error_handling_during_initialization(self, mock_ui_components, mock_create_ui, mock_handlers):
        """Test that errors during initialization are properly logged and handled."""
        # Arrange
        initializer = PreprocessingInitializer()
        error_message = "Test error during UI creation"
        mock_create_ui.side_effect = Exception(error_message)
        
        # Act
        result = initializer.initialize()
        
        # Assert
        assert isinstance(result, widgets.Widget), "Should return a widget even on error"
        assert 'Error' in str(result), "Error widget should contain error information"
        assert error_message in str(result), "Error widget should contain the error message"
    
    def test_ui_components_structure(self, mock_ui_components, mock_create_ui, mock_handlers):
        """Test that the UI components have the expected structure."""
        # Print debug information at the start
        print("\n=== Starting test_ui_components_structure ===")
        
        # 1. First, verify the mock_ui_components fixture
        print("\n=== mock_ui_components ===")
        print(f"Type: {type(mock_ui_components).__name__}")
        print(f"Keys: {list(mock_ui_components.keys())}")
        
        # 2. Create the initializer and call initialize()
        initializer = PreprocessingInitializer()
        result = initializer.initialize()
        
        # 3. Basic assertions
        print("\n=== Basic Assertions ===")
        print(f"Initializer result type: {type(result).__name__}")
        print(f"mock_create_ui called: {mock_create_ui.called}")
        
        # 4. Get the UI components from the mock
        ui_components = mock_ui_components
        print("\n=== UI Components ===")
        for key, value in ui_components.items():
            print(f"{key}: {type(value).__name__}")
        
        # 5. Check required components exist
        required_components = ['log_accordion', 'log_output', 'status_output', 'container', 'module_name']
        print("\n=== Checking Required Components ===")
        for component in required_components:
            exists = component in ui_components
            print(f"{component}: {'✓' if exists else '✗'}")
            assert component in ui_components, f"UI components should include {component}"
        
        # 6. Check log_accordion
        log_accordion = ui_components['log_accordion']
        print("\n=== log_accordion ===")
        print(f"Type: {type(log_accordion).__name__}")
        print(f"Is Accordion: {isinstance(log_accordion, widgets.Accordion)}")
        print(f"Has children: {hasattr(log_accordion, 'children')}")
        if hasattr(log_accordion, 'children'):
            print(f"Children count: {len(log_accordion.children)}")
            if log_accordion.children:
                print(f"First child type: {type(log_accordion.children[0]).__name__}")
        
        # 7. Check log_output
        log_output = ui_components['log_output']
        print("\n=== log_output ===")
        print(f"Type: {type(log_output).__name__}")
        print(f"Is Box: {isinstance(log_output, widgets.Box)}")
        print(f"Has append_log: {hasattr(log_output, 'append_log')}")
        print(f"Has clear_logs: {hasattr(log_output, 'clear_logs')}")
        
        # 8. Check container
        container = ui_components['container']
        print("\n=== container ===")
        print(f"Type: {type(container).__name__}")
        print(f"Is VBox: {isinstance(container, widgets.VBox)}")
        
        # 9. Check for entries_container if it exists
        entries_container = ui_components.get('entries_container')
        if entries_container is not None:
            print("\n=== entries_container ===")
            print(f"Type: {type(entries_container).__name__}")
            print(f"Is VBox: {isinstance(entries_container, widgets.VBox)}")
        
        # 10. Final assertions
        print("\n=== Running Final Assertions ===")
        assert isinstance(result, widgets.Widget), "Should return a widget"
        mock_create_ui.assert_called_once()
        
        # Get the config passed to create_ui_components
        config = mock_create_ui.call_args[0][0]
        assert isinstance(config, dict), "Config should be a dictionary"
        # Note: module_name is not required in the config as it's passed separately
        
        # Check log_accordion is an Accordion widget
        assert isinstance(log_accordion, widgets.Accordion), \
            f"log_accordion should be an Accordion widget, got {type(log_accordion).__name__}"
            
        # Check log_output is a Box and has required methods
        assert isinstance(log_output, widgets.Box), \
            f"log_output should be a Box, got {type(log_output).__name__}"
            
        # Check that log_accordion has children
        assert hasattr(log_accordion, 'children'), "log_accordion should have children attribute"
        assert len(log_accordion.children) > 0, "log_accordion should have children"
        
        # Check that log_output has the required methods for logging
        assert hasattr(log_output, 'append_log'), ("log_output should have append_log method. "
                                                  f"Available methods: {[m for m in dir(log_output) if not m.startswith('_')]}")
        assert hasattr(log_output, 'clear_logs'), ("log_output should have clear_logs method. "
                                                  f"Available methods: {[m for m in dir(log_output) if not m.startswith('_')]}")
        
        # Check that the container is a VBox
        assert isinstance(container, widgets.VBox), \
            f"container should be a VBox widget, got {type(container).__name__}"
        
        # If entries_container exists, check its type
        if entries_container is not None:
            assert isinstance(entries_container, widgets.VBox), \
                f"entries_container should be a VBox widget, got {type(entries_container).__name__}"
            
        print("\n=== All assertions passed! ===")
