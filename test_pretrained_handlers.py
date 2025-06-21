"""
Test script for pretrained UI handlers and buttons
"""
import sys
import os
import pytest
from unittest.mock import MagicMock, patch, ANY
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

# Import the handlers we want to test
from smartcash.ui.pretrained.handlers.pretrained_handlers import (
    setup_pretrained_handlers,
    _setup_config_handlers,
    _setup_operation_handlers,
    _handle_download_sync,
    _handle_download_sync_with_confirmation,
    _execute_download_sync_with_progress
)

# Fixtures
@pytest.fixture
def ui_components() -> Dict[str, Any]:
    """Fixture providing mock UI components."""
    return {
        'save_button': MagicMock(),
        'reset_button': MagicMock(),
        'download_sync_button': MagicMock(),
        'log_output': MagicMock(),
        'confirmation_area': MagicMock(),
        'dialog_area': MagicMock(),
        'progress_tracker': MagicMock(),
        'status': MagicMock()
    }

@pytest.fixture
def config() -> Dict[str, Any]:
    """Fixture providing test configuration."""
    return {
        "pretrained_models": {
            "pretrained_type": "yolov5s",
            "models_dir": "/tmp/models",
            "drive_models_dir": "/content/drive/MyDrive/Models",
            "auto_download": True,
            "sync_drive": False
        }
    }

def test_setup_pretrained_handlers(ui_components, config):
    """Test setup_pretrained_handlers function."""
    # Mock the internal setup functions
    with patch('smartcash.ui.pretrained.handlers.pretrained_handlers._setup_config_handlers') as mock_setup_config, \
         patch('smartcash.ui.pretrained.handlers.pretrained_handlers._setup_operation_handlers') as mock_setup_ops:
        
        # Call the function
        result = setup_pretrained_handlers(ui_components, config)
        
        # Assertions
        assert result is not None
        assert 'config_handler' in result
        assert 'model_checker' in result
        assert 'model_downloader' in result
        assert 'model_syncer' in result
        assert 'handlers' in result
        
        # Check if setup functions were called
        mock_setup_config.assert_called_once_with(ui_components)
        mock_setup_ops.assert_called_once_with(ui_components)

def test_setup_config_handlers(ui_components):
    """Test _setup_config_handlers function."""
    # Call the function
    _setup_config_handlers(ui_components)
    
    # Assert that button click handlers were set up
    ui_components['save_button'].on_click.assert_called_once()
    ui_components['reset_button'].on_click.assert_called_once()

def test_setup_operation_handlers(ui_components):
    """Test _setup_operation_handlers function."""
    # Call the function
    _setup_operation_handlers(ui_components)
    
    # Assert that button click handler was set up
    ui_components['download_sync_button'].on_click.assert_called_once()

def test_handle_download_sync(ui_components):
    """Test _handle_download_sync function."""
    # Mock the confirmation handler
    with patch('smartcash.ui.pretrained.handlers.pretrained_handlers._handle_download_sync_with_confirmation') as mock_handler:
        # Call the function
        _handle_download_sync(ui_components)
        
        # Assert the confirmation handler was called
        mock_handler.assert_called_once_with(ui_components)

@patch('smartcash.ui.pretrained.handlers.pretrained_handlers._should_execute_download_sync')
@patch('smartcash.ui.pretrained.handlers.pretrained_handlers._execute_download_sync_with_progress')
def test_handle_download_sync_with_confirmation(mock_execute, mock_should_execute, ui_components):
    """Test _handle_download_sync_with_confirmation function."""
    # Test when confirmation is approved
    mock_should_execute.return_value = True
    _handle_download_sync_with_confirmation(ui_components)
    mock_execute.assert_called_once_with(ui_components)
    
    # Test when confirmation is denied
    mock_execute.reset_mock()
    mock_should_execute.return_value = False
    _handle_download_sync_with_confirmation(ui_components)
    mock_execute.assert_not_called()

@patch('smartcash.ui.pretrained.handlers.pretrained_handlers._process_model')
@patch('smartcash.ui.pretrained.handlers.pretrained_handlers._get_models_to_process')
def test_execute_download_sync_with_progress(mock_get_models, mock_process, ui_components, config):
    """Test _execute_download_sync_with_progress function."""
    # Setup mocks
    # Mock _get_models_to_process to return test models
    mock_get_models.return_value = ['yolov5s', 'yolov5m']
    
    # Setup test config and mock config handler
    test_config = {'pretrained_models': config['pretrained_models'].copy()}
    
    # Mock config handler
    mock_config_handler = MagicMock()
    mock_config_handler.extract_config.return_value = test_config
    
    # Setup UI components
    ui_components.update({
        'config_handler': mock_config_handler,
        'progress_tracker': MagicMock(),
        'output_area': MagicMock(),
        'dialog_area': MagicMock(),
        'download_sync_button': MagicMock(),
        'cancel_button': MagicMock(),
        'confirm_button': MagicMock()
    })
    
    # Call the function
    _execute_download_sync_with_progress(ui_components)
    
    # Verify config handler was called
    mock_config_handler.extract_config.assert_called_once_with(ui_components)
    
    # Verify _get_models_to_process was called with pretrained_config
    mock_get_models.assert_called_once_with(test_config['pretrained_models'])
    
    # Verify models were processed
    assert mock_process.call_count == 2  # Called for each model
    
    # Verify progress tracker was used
    progress_tracker = ui_components['progress_tracker']
    progress_tracker.start.assert_called_once()
    assert progress_tracker.update.call_count == 2  # One for each model

if __name__ == "__main__":
    pytest.main([__file__, '-v', '-s'])
