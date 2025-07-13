"""
Tests for preprocessing UI components and event handling.
"""

import pytest
from unittest.mock import MagicMock, patch, ANY, call, PropertyMock
import ipywidgets as widgets
from traitlets import HasTraits, Unicode, Instance, List, observe

# Create mock widget classes that properly implement traitlets
class MockWidget(HasTraits):
    """Base mock widget class that implements basic widget functionality."""
    description = Unicode('')
    button_style = Unicode('')
    disabled = False
    value = None
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.on_click = MagicMock()
    
    def observe(self, handler, names, type='change'):
        pass

class MockButton(MockWidget):
    """Mock button widget."""
    pass

class MockVBox(MockWidget):
    """Mock VBox widget."""
    children = List(trait=Instance(HasTraits))
    
    def __init__(self, children=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if children is not None:
            self.children = children

# Mock the necessary imports
with patch('ipywidgets.Button', MockButton), \
     patch('ipywidgets.VBox', MockVBox):
    from smartcash.ui.dataset.preprocess.components.preprocess_ui import create_preprocessing_main_ui

# Define test button configuration
TEST_BUTTON_CONFIG = {
    'preprocess': {'label': 'Preprocess', 'style': 'primary'},
    'check': {'label': 'Check', 'style': 'info'},
    'cleanup': {'label': 'Cleanup', 'style': 'warning'}
}

def create_mock_ui_components():
    """Helper to create a fresh set of mock UI components for each test."""
    # Create mock widgets
    mock_buttons = []
    for btn_id, config in TEST_BUTTON_CONFIG.items():
        btn = MockButton()
        btn.description = config['label']
        btn.button_style = config['style']
        btn.value = btn_id
        mock_buttons.append(btn)
    
    return {
        'main_container': MockVBox(),
        'header_container': MockVBox(),
        'form_container': MockVBox(),
        'action_container': MockVBox(children=mock_buttons),
        'operation_container': MockVBox(),
        'footer_container': MockVBox(),
        'log_output': MagicMock(),
        'progress_bar': MagicMock(),
        'status_display': MagicMock(),
        'progress_tracker': MagicMock(),
        'log_accordion': MagicMock(),
        'form_widgets': {
            'input_options': MagicMock(),
            'components': {'input_options': MagicMock()}
        },
        'info_box': MagicMock()
    }

def create_mock_operation_manager():
    """Helper to create a fresh mock operation manager for each test."""
    manager = MagicMock()
    manager.is_operation_in_progress = MagicMock(return_value=False)
    manager.execute_preprocess = MagicMock(return_value={
        'status': 'success',
        'message': 'Preprocessing completed'
    })
    manager.execute_check = MagicMock(return_value={
        'status': 'success',
        'message': 'Check completed'
    })
    manager.execute_cleanup = MagicMock(return_value={
        'status': 'success',
        'message': 'Cleanup completed'
    })
    return manager


class TestPreprocessUIComponents:
    """Test cases for preprocessing UI components."""
    
    @pytest.fixture
    def mock_ui_components(self):
        """Fixture providing mock UI components for testing."""
        return create_mock_ui_components()

    @pytest.fixture
    def mock_operation_manager(self):
        """Fixture providing a mock operation manager for testing."""
        return create_mock_operation_manager()
        
    @pytest.fixture(autouse=True)
    def patch_ipywidgets(self):
        """Automatically patch ipywidgets for all tests in this class."""
        with patch('ipywidgets.Button', MockButton), \
             patch('ipywidgets.VBox', MockVBox):
            yield

    @patch('smartcash.ui.dataset.preprocess.components.preprocess_ui._create_module_form_widgets')
    @patch('smartcash.ui.dataset.preprocess.components.preprocess_ui._create_module_info_box')
    def test_ui_initialization(self, mock_info_box, mock_form_widgets, mock_ui_components):
        """Test that the UI initializes with all required components."""
        # Setup mocks
        mock_form_widgets.return_value = {
            'components': {'input_options': MagicMock()},
            'input_options': MagicMock()
        }
        mock_info_box.return_value = MagicMock()
        
        # Create the UI
        ui = create_preprocessing_main_ui(components=mock_ui_components)
        
        # Verify the UI was created with all required components
        assert ui is not None
        assert 'main_container' in ui
        assert 'header_container' in ui
        assert 'form_container' in ui
        assert 'action_container' in ui
        assert 'operation_container' in ui
        assert 'footer_container' in ui
        
        # Verify form widgets and info box were created
        mock_form_widgets.assert_called_once()
        mock_info_box.assert_called_once()
    
    @patch('smartcash.ui.dataset.preprocess.components.preprocess_ui._create_module_form_widgets')
    @patch('smartcash.ui.dataset.preprocess.components.preprocess_ui._create_module_info_box')
    def test_button_click_handlers(self, mock_info_box, mock_form_widgets, mock_ui_components, mock_operation_manager):
        """Test that button click handlers are properly connected."""
        # Setup mocks
        mock_form_widgets.return_value = {
            'components': {'input_options': MagicMock()},
            'input_options': MagicMock()
        }
        mock_info_box.return_value = MagicMock()
        
        # Create the UI with operation manager
        ui = create_preprocessing_main_ui(
            components=mock_ui_components,
            operation_manager=mock_operation_manager
        )
        
        # Get the action container and its buttons
        action_container = mock_ui_components['action_container']
        assert len(action_container.children) == len(TEST_BUTTON_CONFIG)
        
        # Test each button
        for i, (btn_id, config) in enumerate(TEST_BUTTON_CONFIG.items()):
            # Get the button
            btn = action_container.children[i]
            
            # Verify button properties
            assert btn.description == config['label']
            assert btn.button_style == config['style']
            
            # Simulate button click
            click_handler = btn.on_click.call_args[0][0]
            click_handler(None)  # Pass a dummy event
            
            # Verify the corresponding operation was called
            if btn_id == 'preprocess':
                mock_operation_manager.execute_preprocess.assert_called_once()
                mock_operation_manager.execute_preprocess.reset_mock()
            elif btn_id == 'check':
                mock_operation_manager.execute_check.assert_called_once()
                mock_operation_manager.execute_check.reset_mock()
            elif btn_id == 'cleanup':
                mock_operation_manager.execute_cleanup.assert_called_once()
                mock_operation_manager.execute_cleanup.reset_mock()
    
    @patch('smartcash.ui.dataset.preprocess.components.preprocess_ui._create_module_form_widgets')
    @patch('smartcash.ui.dataset.preprocess.components.preprocess_ui._create_module_info_box')
    def test_concurrent_operation_handling(self, mock_info_box, mock_form_widgets, mock_ui_components, mock_operation_manager):
        """Test that concurrent operations are handled correctly."""
        # Setup mocks
        mock_form_widgets.return_value = {
            'components': {'input_options': MagicMock()},
            'input_options': MagicMock()
        }
        mock_info_box.return_value = MagicMock()
        
        # Configure operation manager to simulate an in-progress operation
        mock_operation_manager.is_operation_in_progress.return_value = True
        
        # Create the UI with operation manager
        ui = create_preprocessing_main_ui(
            components=mock_ui_components,
            operation_manager=mock_operation_manager
        )
        
        # Get the action container
        action_container = mock_ui_components['action_container']
        
        # Try to click all buttons
        for btn in action_container.children:
            # Simulate button click
            click_handler = btn.on_click.call_args[0][0]
            click_handler(None)  # Pass a dummy event
            
            # Verify the button was disabled during operation
            assert btn.disabled is True
            
            # Verify no operation was executed
            mock_operation_manager.execute_preprocess.assert_not_called()
            mock_operation_manager.execute_check.assert_not_called()
            mock_operation_manager.execute_cleanup.assert_not_called()
    
    @patch('smartcash.ui.dataset.preprocess.components.preprocess_ui._create_module_form_widgets')
    @patch('smartcash.ui.dataset.preprocess.components.preprocess_ui._create_module_info_box')
    def test_error_handling_in_operations(self, mock_info_box, mock_form_widgets, mock_ui_components, mock_operation_manager):
        """Test error handling during operations."""
        # Setup mocks
        mock_form_widgets.return_value = {
            'components': {'input_options': MagicMock()},
            'input_options': MagicMock()
        }
        mock_info_box.return_value = MagicMock()
        
        # Make operation raise an exception
        error_msg = "Test error: Something went wrong"
        mock_operation_manager.execute_preprocess.side_effect = Exception(error_msg)
        
        # Create the UI with operation manager
        ui = create_preprocessing_main_ui(
            components=mock_ui_components,
            operation_manager=mock_operation_manager
        )
        
        # Get the preprocess button
        preprocess_btn = next(
            btn for btn in mock_ui_components['action_container'].children 
            if btn.value == 'preprocess'
        )
        
        # Simulate button click
        click_handler = preprocess_btn.on_click.call_args[0][0]
        click_handler(None)  # Pass a dummy event
        
        # Verify the operation was called
        mock_operation_manager.execute_preprocess.assert_called_once()
        
        # Verify the error was logged
        # Note: The actual error logging would be verified through the log output mock
        assert mock_ui_components['log_output'].append_stdout.called or \
               mock_ui_components['log_output'].append_stderr.called
        
        # Check that the error message appears in the log
        error_logged = any(
            error_msg in str(call)
            for call in mock_ui_components['log_output'].append_stderr.call_args_list
        )
        assert error_logged, f"Error message '{error_msg}' was not logged"
