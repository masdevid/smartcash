"""Integration tests for OperationContainer component.

Tests the functionality and integration of the OperationContainer component with
its child components and dependencies.
"""
import pytest
from unittest.mock import MagicMock, patch, PropertyMock
import ipywidgets as widgets

# Import the container we're testing
from smartcash.ui.components.operation_container import OperationContainer

# Fixtures
@pytest.fixture
def operation_container(optional_dependency_mock):
    """Create an OperationContainer instance for testing."""
    with patch('smartcash.ui.components.operation_container.ErrorHandler', 
              optional_dependency_mock['error_handler'].__class__):
        container = OperationContainer(
            component_name="test_operation",
            progress_levels='dual',
            show_progress=True,
            show_logs=True,
            show_dialog=True,
            log_module_name="TestModule"
        )
        # Add mock components
        container.progress_widget = MagicMock()
        container.progress_widget.bar_style = ''
        container.progress_widget.layout = MagicMock()
        container.progress_widget.description = ''
        container.progress_widget.value = 0.0
        
        container.log_widget = MagicMock()
        container.log_widget.logs = []
        container.log_widget.layout = MagicMock()
        
        container.dialog_widget = MagicMock()
        container.dialog_widget.visible = False
        container.dialog_widget.title = ''
        container.dialog_widget.description = ''
        container.dialog_widget.layout = MagicMock()
        
        container.progress_widgets = [container.progress_widget]
        
        return container

class TestOperationContainer:
    """Test suite for OperationContainer integration."""
    
    def test_initialization(self, operation_container):
        """Test basic initialization with parameters."""
        assert operation_container is not None
        assert operation_container.component_name == "test_operation"
        assert hasattr(operation_container, 'progress_widget')
        assert hasattr(operation_container, 'log_widget')
        assert hasattr(operation_container, 'dialog_widget')
    
    def test_progress_tracking(self, operation_container):
        """Test progress tracking functionality."""
        # Test initial progress
        assert operation_container.progress_widget.value == 0.0
        
        # Update progress
        operation_container.update_progress(0.5, "Halfway there")
        assert operation_container.progress_widget.value == 0.5
        assert operation_container.progress_widget.description == "Halfway there"
    
    @patch('smartcash.ui.components.operation_container.HTML')
    @patch('smartcash.ui.components.operation_container.Output')
    def test_logging(self, mock_output, mock_html, operation_container):
        """Test logging functionality."""
        # Setup mock output
        mock_output_instance = MagicMock()
        mock_output.return_value = mock_output_instance
        
        # Test info log
        operation_container.log("Test info message", level="INFO")
        
        # Verify output was used
        mock_output.assert_called_once()
        
        # Test error log
        operation_container.log_error("Test error message")
        assert mock_output_instance.append_stdout.called
    
    def test_dialog_management(self, operation_container):
        """Test dialog show/hide functionality."""
        # Test showing dialog
        operation_container.show_dialog("Test Title", "Test Message")
        assert operation_container.dialog_widget.title == "Test Title"
        assert operation_container.dialog_widget.description == "Test Message"
        assert operation_container.dialog_widget.visible is True
        
        # Test hiding dialog
        operation_container.hide_dialog()
        assert operation_container.dialog_widget.visible is False
    
    @patch('smartcash.ui.components.operation_container.VBox')
    @patch('smartcash.ui.components.operation_container.HTML')
    @patch('smartcash.ui.components.operation_container.Output')
    @patch('smartcash.ui.components.operation_container.FloatProgress')
    def test_ui_components_creation(self, mock_progress, mock_output, mock_html, mock_vbox, operation_container):
        """Test that all UI components are created correctly."""
        # Setup mocks
        mock_vbox.return_value = MagicMock()
        mock_progress.return_value = MagicMock()
        mock_output.return_value = MagicMock()
        mock_html.return_value = MagicMock()
        
        # Reinitialize to trigger UI creation
        container = OperationContainer("test_ui_creation")
        
        # Verify components were created
        mock_progress.assert_called()
        mock_output.assert_called()
        mock_html.assert_called()
        mock_vbox.assert_called()
    
    @patch('smartcash.ui.components.operation_container.HTML')
    def test_error_handling(self, mock_html, operation_container):
        """Test error handling and reporting."""
        # Setup mock
        mock_widget = MagicMock()
        mock_html.return_value = mock_widget
        
        # Test error reporting
        error = Exception("Test error")
        operation_container.report_error(error, "Test error context")
        
        # Verify error handling was called
        mock_html.assert_called()
    
    @patch('smartcash.ui.components.operation_container.FloatProgress')
    def test_progress_levels(self, mock_progress):
        """Test different progress level configurations."""
        # Setup mock
        mock_progress.return_value = MagicMock()
        
        # Test single progress level
        single = OperationContainer(progress_levels='single')
        assert len(single.progress_widgets) == 1
        
        # Test dual progress level
        dual = OperationContainer(progress_levels='dual')
        assert len(dual.progress_widgets) == 2
        
        # Test triple progress level
        triple = OperationContainer(progress_levels='triple')
        assert len(triple.progress_widgets) == 3
    
    def test_visibility_control(self, operation_container):
        """Test show/hide controls for different components."""
        # Test progress visibility
        operation_container.show_progress = False
        operation_container.show_progress = True  # Toggle to trigger setter
        
        # Test logs visibility
        operation_container.show_logs = False
        operation_container.show_logs = True  # Toggle to trigger setter
    
    def test_custom_styling(self, operation_container):
        """Test custom styling options."""
        # Apply custom styles
        operation_container.update_style(
            progress_bar_style='success',
            log_height='400px',
            dialog_width='600px'
        )
        
        # Verify styles were applied
        assert operation_container.progress_widget.bar_style == 'success'
        assert operation_container.log_widget.layout.height == '400px'
        assert operation_container.dialog_widget.layout.width == '600px'
