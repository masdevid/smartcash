"""
File: tests/ui/dataset/preprocessing/test_preprocessing_uimodule.py
Description: Unit tests for the PreprocessingUIModule.
"""
import pytest
from unittest.mock import MagicMock, patch, ANY

from smartcash.ui.dataset.preprocessing.preprocessing_uimodule import PreprocessingUIModule


@pytest.fixture
def mock_dependencies():
    """Patch all external dependencies for the UI module."""
    with (patch('smartcash.ui.dataset.preprocessing.preprocessing_uimodule.create_preprocessing_ui_components') as mock_create_ui,
          patch('smartcash.ui.dataset.preprocessing.preprocessing_uimodule.PreprocessingConfigHandler') as mock_config_handler,
          patch('smartcash.ui.dataset.preprocessing.preprocessing_uimodule.get_default_config') as mock_get_config,
          patch('smartcash.ui.dataset.preprocessing.preprocessing_uimodule.CheckOperationHandler') as mock_check_op,
          patch('smartcash.ui.dataset.preprocessing.preprocessing_uimodule.PreprocessOperationHandler') as mock_preprocess_op,
          patch('smartcash.ui.dataset.preprocessing.preprocessing_uimodule.CleanupOperationHandler') as mock_cleanup_op,
          patch('smartcash.ui.dataset.preprocessing.preprocessing_uimodule.get_preprocessed_data_stats') as mock_get_stats):
        
        # Setup mock UI components with proper hierarchy
        mock_main_container = MagicMock()
        mock_operation_container = MagicMock()
        mock_summary_container = MagicMock()

        # Simulate the parent relationship: operation_container's parent holds the summary_container
        mock_operation_container.parent = mock_main_container
        mock_main_container.operation_summary_container = mock_summary_container

        mock_ui_components = {
            'operation_summary_updater': MagicMock(),
            'operation_container': mock_operation_container
        }
        mock_create_ui.return_value = (mock_main_container, mock_ui_components)
        
        # Setup mock config handler
        mock_config_instance = MagicMock()
        mock_config_handler.return_value = mock_config_instance
        mock_config_instance.get_current_config.return_value = {'fake': 'config'}

        yield {
            'create_ui': mock_create_ui,
            'config_handler': mock_config_handler,
            'get_config': mock_get_config,
            'check_op': mock_check_op,
            'preprocess_op': mock_preprocess_op,
            'cleanup_op': mock_cleanup_op,
            'get_stats': mock_get_stats,
            'ui_components': mock_create_ui.return_value[1]
        }


def test_initialization(mock_dependencies):
    """Test that the module initializes correctly."""
    module = PreprocessingUIModule()
    mock_dependencies['create_ui'].assert_called_once()
    mock_dependencies['config_handler'].assert_called_once()
    assert module.get_ui() is not None
    # Check that handlers were connected
    assert mock_dependencies['create_ui'].return_value[0].action_buttons.check_button.on_click.called


def test_execute_check_operation(mock_dependencies):
    """Test that the check operation is executed correctly."""
    module = PreprocessingUIModule()
    module._execute_check_operation(MagicMock()) # Pass a mock button object

    mock_dependencies['check_op'].assert_called_once_with(
        ui_module=module,
        config={'fake': 'config'},
        callbacks={'on_success': module._update_operation_summary, 'on_failure': module._update_operation_summary}
    )
    mock_dependencies['check_op'].return_value.execute.assert_called_once()


@patch('smartcash.ui.dataset.preprocessing.preprocessing_uimodule.confirm_action_dialog')
def test_execute_cleanup_operation_confirmed(mock_confirm_dialog, mock_dependencies):
    """Test cleanup operation execution when user confirms."""
    mock_dependencies['get_stats'].return_value = (10, 20) # some files exist
    mock_confirm_dialog.return_value = True # User clicks 'yes'

    module = PreprocessingUIModule()
    module._execute_cleanup_operation(MagicMock())

    mock_confirm_dialog.assert_called_once()
    mock_dependencies['cleanup_op'].assert_called_once_with(
        ui_module=module,
        config={'fake': 'config'},
        callbacks={'on_success': module._update_operation_summary}
    )
    mock_dependencies['cleanup_op'].return_value.execute.assert_called_once()


@patch('smartcash.ui.dataset.preprocessing.preprocessing_uimodule.confirm_action_dialog')
def test_execute_cleanup_operation_not_confirmed(mock_confirm_dialog, mock_dependencies):
    """Test cleanup operation is not executed when user cancels."""
    mock_dependencies['get_stats'].return_value = (10, 20)
    mock_confirm_dialog.return_value = False # User clicks 'no'

    module = PreprocessingUIModule()
    module._execute_cleanup_operation(MagicMock())

    mock_confirm_dialog.assert_called_once()
    mock_dependencies['cleanup_op'].assert_not_called()


def test_update_operation_summary(mock_dependencies):
    """Test the _update_operation_summary method."""
    module = PreprocessingUIModule()
    summary_updater = mock_dependencies['ui_components']['operation_summary_updater']
    summary_container = mock_dependencies['ui_components']['operation_container'].parent.operation_summary_container

    # Initially hidden
    summary_container.layout.display = 'none'

    module._update_operation_summary("Test Summary")

    summary_updater.assert_called_once_with("Test Summary")
    assert summary_container.layout.display == 'block'
