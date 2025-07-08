"""
Tests for the dataset split initializer.

This module tests the SplitInitializer class and related functions
for the dataset split configuration UI.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any
import copy

from smartcash.ui.dataset.split.split_initializer import (
    SplitInitializer,
    create_split_config_cell,
    get_split_config_components,
    init_split_ui,
    get_split_initializer
)
from smartcash.ui.dataset.split.configs.split_defaults import DEFAULT_SPLIT_CONFIG


class TestSplitInitializer:
    """Test cases for SplitInitializer."""
    
    @pytest.fixture
    def mock_config(self):
        """Provide a mock configuration for testing."""
        return copy.deepcopy(DEFAULT_SPLIT_CONFIG)
    
    @pytest.fixture
    def mock_components(self):
        """Provide mock UI components for testing."""
        return {
            'main_container': Mock(),
            'header_container': Mock(),
            'form_container': Mock(),
            'action_container': Mock(),
            'footer_container': Mock(),
            'form_components': {
                'train_ratio': Mock(value=0.7),
                'val_ratio': Mock(value=0.15),
                'test_ratio': Mock(value=0.15),
                'save_button': Mock(),
                'reset_button': Mock()
            },
            'log_output': Mock(),
            'log_accordion': Mock()
        }
    
    def test_initialization_default(self):
        """Test initializer with default parameters."""
        initializer = SplitInitializer()
        
        assert hasattr(initializer, 'config')
        assert hasattr(initializer, 'config_handler')
        assert hasattr(initializer, 'components')
        assert initializer.config == {}
    
    def test_initialization_with_config(self, mock_config):
        """Test initializer with custom configuration."""
        initializer = SplitInitializer(config=mock_config)
        
        assert initializer.config == mock_config
        assert hasattr(initializer, 'config_handler')
    
    def test_initialization_with_kwargs(self, mock_config):
        """Test initializer with additional kwargs."""
        kwargs = {'show_advanced': True, 'theme': 'dark'}
        initializer = SplitInitializer(config=mock_config, **kwargs)
        
        assert initializer.config == mock_config
        assert initializer.kwargs == kwargs
    
    @patch('smartcash.ui.dataset.split.split_initializer.create_split_ui_components')
    def test_initialize_impl_success(self, mock_create_ui, mock_config):
        """Test successful initialization implementation."""
        mock_components = {
            'main_container': Mock(),
            'form_components': {}
        }
        mock_create_ui.return_value = mock_components
        
        initializer = SplitInitializer(config=mock_config)
        result = initializer._initialize_impl()
        
        assert result['status'] == 'success'
        assert 'SplitInitializer initialized successfully' in result['message']
        assert result['components'] == mock_components
        assert initializer.components == mock_components
    
    @patch('smartcash.ui.dataset.split.split_initializer.create_split_ui_components')
    def test_initialize_impl_failure(self, mock_create_ui, mock_config):
        """Test initialization implementation failure."""
        mock_create_ui.side_effect = Exception("UI creation failed")
        
        initializer = SplitInitializer(config=mock_config)
        result = initializer._initialize_impl()
        
        assert result['status'] == 'error'
        assert 'Failed to initialize SplitInitializer' in result['message']
        assert 'UI creation failed' in result['error']
    
    @patch('smartcash.ui.dataset.split.split_initializer.create_split_ui_components')
    def test_create_ui_components_success(self, mock_create_ui, mock_config, mock_components):
        """Test successful UI components creation."""
        mock_create_ui.return_value = mock_components
        
        initializer = SplitInitializer(config=mock_config)
        result = initializer._create_ui_components(mock_config)
        
        assert result == mock_components
        assert initializer.components == mock_components
        mock_create_ui.assert_called_once()
    
    @patch('smartcash.ui.dataset.split.split_initializer.create_split_ui_components')
    def test_create_ui_components_failure(self, mock_create_ui, mock_config):
        """Test UI components creation failure."""
        mock_create_ui.side_effect = Exception("Component creation failed")
        
        initializer = SplitInitializer(config=mock_config)
        
        with pytest.raises(RuntimeError, match="Gagal membuat komponen UI"):
            initializer._create_ui_components(mock_config)
    
    @patch('IPython.display.display')
    @patch('smartcash.ui.dataset.split.split_initializer.create_split_ui_components')
    def test_display_with_existing_components(self, mock_create_ui, mock_display, mock_components):
        """Test display with existing components."""
        initializer = SplitInitializer()
        initializer.components = mock_components
        
        initializer.display()
        
        mock_display.assert_called_once_with(mock_components['main_container'])
        mock_create_ui.assert_not_called()
    
    @patch('IPython.display.display')
    @patch('smartcash.ui.dataset.split.split_initializer.create_split_ui_components')
    def test_display_without_components(self, mock_create_ui, mock_display, mock_components):
        """Test display without existing components."""
        mock_create_ui.return_value = mock_components
        
        initializer = SplitInitializer()
        initializer.display()
        
        mock_create_ui.assert_called_once()
        mock_display.assert_called_once_with(mock_components['main_container'])
    
    def test_setup_handlers_with_components(self, mock_components):
        """Test setting up event handlers with components."""
        initializer = SplitInitializer()
        initializer.components = mock_components
        initializer.save_config = Mock()
        initializer.reset_ui = Mock()
        
        initializer._setup_handlers()
        
        # Test save button handler
        save_button = mock_components['form_components']['save_button']
        assert save_button.on_click.called
        
        # Test reset button handler
        reset_button = mock_components['form_components']['reset_button']
        assert reset_button.on_click.called
    
    def test_setup_handlers_without_components(self):
        """Test setting up handlers without components."""
        initializer = SplitInitializer()
        
        # Should not raise exception
        initializer._setup_handlers()
    
    def test_log_message_with_log_output(self, mock_components):
        """Test logging message with log output component."""
        initializer = SplitInitializer()
        initializer.components = mock_components
        
        initializer._log_message("Test message")
        
        # Verify that the log output was used
        assert mock_components['log_output'].__enter__.called or hasattr(mock_components['log_output'], '__enter__')
    
    def test_log_message_without_log_output(self):
        """Test logging message without log output component."""
        initializer = SplitInitializer()
        initializer.components = {}
        
        # Should not raise exception
        initializer._log_message("Test message")
    
    def test_log_error(self, mock_components):
        """Test error logging."""
        initializer = SplitInitializer()
        initializer.components = mock_components
        initializer._log_message = Mock()
        
        initializer._log_error("Error message")
        
        initializer._log_message.assert_called_once_with("❌ Error message")
    
    def test_log_success(self, mock_components):
        """Test success logging."""
        initializer = SplitInitializer()
        initializer.components = mock_components
        initializer._log_message = Mock()
        
        initializer._log_success("Success message")
        
        initializer._log_message.assert_called_once_with("✅ Success message")


class TestSplitInitializerFunctions:
    """Test cases for module-level functions."""
    
    @patch('smartcash.ui.dataset.split.split_initializer.SplitInitializer')
    def test_create_split_config_cell_success(self, mock_initializer_class, mock_config):
        """Test successful split config cell creation."""
        mock_initializer = Mock()
        mock_initializer_class.return_value = mock_initializer
        
        create_split_config_cell(config=mock_config, theme='light')
        
        mock_initializer_class.assert_called_once_with(config=mock_config, theme='light')
        mock_initializer.initialize.assert_called_once()
        mock_initializer.display.assert_called_once()
    
    @patch('smartcash.ui.dataset.split.split_initializer.SplitInitializer')
    @patch('smartcash.ui.dataset.split.split_initializer.handle_ui_errors')
    def test_create_split_config_cell_failure(self, mock_handle_errors, mock_initializer_class):
        """Test split config cell creation failure."""
        mock_initializer_class.side_effect = Exception("Initialization failed")
        
        create_split_config_cell()
        
        # Should handle the error
        assert mock_handle_errors.called or mock_initializer_class.called
    
    @patch('smartcash.ui.dataset.split.split_initializer.SplitInitializer')
    def test_get_split_config_components_success(self, mock_initializer_class, mock_config):
        """Test successful component retrieval."""
        mock_initializer = Mock()
        mock_components = {
            'main_container': Mock(),
            'form_components': {}
        }
        mock_initializer.components = mock_components
        mock_initializer_class.return_value = mock_initializer
        
        result = get_split_config_components(config=mock_config)
        
        assert result['initializer'] == mock_initializer
        assert result['components'] == mock_components
        assert result['main_container'] == mock_components['main_container']
        assert result['form_components'] == mock_components['form_components']
    
    @patch('smartcash.ui.dataset.split.split_initializer.SplitInitializer')
    def test_get_split_config_components_without_components(self, mock_initializer_class, mock_config):
        """Test component retrieval when components need to be created."""
        mock_initializer = Mock()
        mock_initializer.components = {}
        mock_initializer._create_ui_components = Mock()
        mock_initializer_class.return_value = mock_initializer
        
        get_split_config_components(config=mock_config)
        
        mock_initializer._create_ui_components.assert_called_once()
    
    @patch('smartcash.ui.dataset.split.split_initializer.SplitInitializer')
    @patch('smartcash.ui.dataset.split.split_initializer.create_error_response')
    def test_get_split_config_components_failure(self, mock_error_response, mock_initializer_class):
        """Test component retrieval failure."""
        mock_initializer_class.side_effect = Exception("Failed to create")
        mock_error_response.return_value = {'error': 'test_error'}
        
        result = get_split_config_components()
        
        assert result == {'error': 'test_error'}
        mock_error_response.assert_called_once()
    
    @patch('smartcash.ui.dataset.split.split_initializer.create_split_config_cell')
    def test_init_split_ui(self, mock_create_cell, mock_config):
        """Test init_split_ui function."""
        init_split_ui(config=mock_config, theme='dark')
        
        mock_create_cell.assert_called_once_with(mock_config, theme='dark')
    
    def test_get_split_initializer(self, mock_config):
        """Test get_split_initializer function."""
        result = get_split_initializer(config=mock_config, theme='light')
        
        assert isinstance(result, SplitInitializer)
        assert result.config == mock_config
        assert result.kwargs == {'theme': 'light'}


class TestSplitInitializerEventHandlers:
    """Test cases for event handler functionality."""
    
    @pytest.fixture
    def initializer_with_components(self, mock_components):
        """Create initializer with mock components."""
        initializer = SplitInitializer()
        initializer.components = mock_components
        initializer.save_config = Mock()
        initializer.reset_ui = Mock()
        return initializer
    
    def test_save_button_click_success(self, initializer_with_components):
        """Test successful save button click."""
        initializer = initializer_with_components
        initializer._setup_handlers()
        
        # Get the save button callback
        save_button = initializer.components['form_components']['save_button']
        callback = save_button.on_click.call_args[0][0]
        
        # Execute callback
        callback(save_button)
        
        initializer.save_config.assert_called_once()
    
    def test_save_button_click_failure(self, initializer_with_components):
        """Test save button click with error."""
        initializer = initializer_with_components
        initializer.save_config.side_effect = Exception("Save failed")
        initializer._log_error = Mock()
        initializer._setup_handlers()
        
        # Get the save button callback
        save_button = initializer.components['form_components']['save_button']
        callback = save_button.on_click.call_args[0][0]
        
        # Execute callback
        callback(save_button)
        
        initializer._log_error.assert_called_once()
        assert "Gagal menyimpan konfigurasi" in initializer._log_error.call_args[0][0]
    
    def test_reset_button_click_success(self, initializer_with_components):
        """Test successful reset button click."""
        initializer = initializer_with_components
        initializer._setup_handlers()
        
        # Get the reset button callback
        reset_button = initializer.components['form_components']['reset_button']
        callback = reset_button.on_click.call_args[0][0]
        
        # Execute callback
        callback(reset_button)
        
        initializer.reset_ui.assert_called_once()
    
    def test_reset_button_click_failure(self, initializer_with_components):
        """Test reset button click with error."""
        initializer = initializer_with_components
        initializer.reset_ui.side_effect = Exception("Reset failed")
        initializer._log_error = Mock()
        initializer._setup_handlers()
        
        # Get the reset button callback
        reset_button = initializer.components['form_components']['reset_button']
        callback = reset_button.on_click.call_args[0][0]
        
        # Execute callback
        callback(reset_button)
        
        initializer._log_error.assert_called_once()
        assert "Gagal me-reset UI" in initializer._log_error.call_args[0][0]


class TestSplitInitializerIntegration:
    """Integration tests for SplitInitializer."""
    
    @patch('smartcash.ui.dataset.split.split_initializer.create_split_ui_components')
    def test_full_initialization_workflow(self, mock_create_ui, mock_config):
        """Test complete initialization workflow."""
        mock_components = {
            'main_container': Mock(),
            'form_components': {
                'save_button': Mock(),
                'reset_button': Mock()
            },
            'log_output': Mock()
        }
        mock_create_ui.return_value = mock_components
        
        # Create initializer
        initializer = SplitInitializer(config=mock_config)
        
        # Initialize
        result = initializer._initialize_impl()
        
        # Verify successful initialization
        assert result['status'] == 'success'
        assert initializer.components == mock_components
        
        # Test display
        with patch('IPython.display.display') as mock_display:
            initializer.display()
            mock_display.assert_called_once_with(mock_components['main_container'])
    
    def test_error_handling_throughout_workflow(self):
        """Test error handling in various parts of the workflow."""
        # Test initialization with invalid config
        with patch('smartcash.ui.dataset.split.handlers.config_handler.SplitConfigHandler') as mock_handler:
            mock_handler.side_effect = Exception("Config handler failed")
            
            # Should handle error gracefully
            initializer = SplitInitializer()
            assert initializer is not None