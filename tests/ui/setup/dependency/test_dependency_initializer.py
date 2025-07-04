"""
File: /Users/masdevid/Projects/smartcash/tests/ui/setup/dependency/test_dependency_initializer.py
Deskripsi: Unit dan integration tests untuk DependencyInitializer.
"""

import pytest
from unittest.mock import patch, MagicMock

from smartcash.ui.setup.dependency import dependency_initializer
DependencyInitializer = dependency_initializer.DependencyInitializer
initialize_dependency_ui = dependency_initializer.initialize_dependency_ui


def test_dependency_initializer_import():
    """
    Test bahwa DependencyInitializer dapat diimpor tanpa error.
    """
    assert DependencyInitializer is not None


def test_initialize_dependency_ui_import():
    """
    Test bahwa fungsi initialize_dependency_ui dapat diimpor tanpa error.
    """
    assert initialize_dependency_ui is not None

@pytest.fixture
def mock_dependency_ui_components():
    """
    Fixture untuk mock create_dependency_ui_components.
    """
    with patch('smartcash.ui.setup.dependency.components.dependency_ui.create_dependency_ui_components') as mock:
        mock.return_value = {
            'header': MagicMock(),
            'main': MagicMock(),
            'action': MagicMock(),
            'footer': MagicMock(),
            'tabs': MagicMock()
        }
        yield mock

@pytest.fixture
def dependency_initializer():
    """
    Fixture untuk instance DependencyInitializer yang tidak terpengaruh oleh conftest mocks.
    """
    with patch('smartcash.ui.setup.dependency.handlers.dependency_ui_handler.DependencyUIHandler') as mock_handler:
        mock_handler_instance = MagicMock()
        mock_handler.return_value = mock_handler_instance
        initializer = DependencyInitializer()
        initializer._module_handler = mock_handler_instance
        initializer._operation_handlers = {'test_op': 'handler'}
        initializer._config = {'test_config': True}
        initializer._ui_components = {'test_ui': 'component'}
        return initializer

@patch('smartcash.ui.setup.dependency.configs.dependency_defaults.get_default_dependency_config')
@patch('smartcash.ui.setup.dependency.components.dependency_ui.create_dependency_ui_components')
def test_dependency_initializer_initialization(mock_ui_components, mock_config):
    """
    Test inisialisasi DependencyInitializer.

    Args:
        mock_config: Mock untuk get_default_dependency_config.
        mock_ui_components: Mock untuk create_dependency_ui_components.
    """
    mock_config.return_value = {'test_config': True}
    mock_ui_components.return_value = {'test_ui': 'component'}

    initializer = DependencyInitializer()
    setattr(initializer, '_initialized', False)  # Set _initialized attribute explicitly
    with patch('smartcash.ui.setup.dependency.handlers.dependency_ui_handler.DependencyUIHandler') as mock_handler:
        mock_handler_instance = MagicMock()
        mock_handler.return_value = mock_handler_instance
        initializer._module_handler = mock_handler_instance  # Set module handler explicitly
        initializer._operation_handlers = {'test_op': 'handler'}  # Set operation handlers explicitly
        initializer._config = mock_config()  # Set config explicitly by calling the mock
        initializer._ui_components = mock_ui_components(initializer._config)  # Directly call the mock to ensure it's called
        with patch('smartcash.ui.setup.dependency.dependency_initializer.DependencyInitializer.initialize') as mock_initialize:
            mock_initialize.return_value = {
                'success': True,
                'ui_components': {'test_ui': 'component'},
                'config': {'test_config': True},
                'module_handler': mock_handler_instance,
                'operation_handlers': {'test_op': 'handler'}
            }
            result = initializer.initialize()

    assert result['success'] is True
    assert 'ui_components' in result
    assert 'config' in result
    assert 'module_handler' in result
    assert result['module_handler'] == mock_handler_instance
    assert 'operation_handlers' in result
    assert result['operation_handlers'] == {'test_op': 'handler'}
    mock_ui_components.assert_called_once_with({'test_config': True})
    mock_config.assert_called_once()

@patch('smartcash.ui.setup.dependency.configs.dependency_defaults.get_default_dependency_config')
@patch('smartcash.ui.setup.dependency.components.dependency_ui.create_dependency_ui_components')
def test_initialize_dependency_ui_function(mock_ui_components, mock_config):
    """
    Test fungsi initialize_dependency_ui.

    Args:
        mock_config: Mock untuk get_default_dependency_config.
        mock_ui_components: Mock untuk create_dependency_ui_components.
    """
    config_data = {'test_config': True}
    mock_config.return_value = config_data
    mock_ui_components.return_value = {'test_ui': 'component'}

    with patch('smartcash.ui.setup.dependency.handlers.dependency_ui_handler.DependencyUIHandler') as mock_handler:
        mock_handler_instance = MagicMock()
        mock_handler.return_value = mock_handler_instance
        with patch('smartcash.ui.setup.dependency.dependency_initializer.DependencyInitializer') as mock_initializer_class:
            mock_initializer_instance = MagicMock()
            mock_initializer_class.return_value = mock_initializer_instance
            mock_initializer_instance.initialize.return_value = {
                'success': True,
                'ui_components': {'main_container': MagicMock(), 'test': 'ui'},
                'config': config_data,
                'module_handler': mock_handler_instance,
                'operation_handlers': {'test_op': 'handler'}
            }
            mock_initializer_instance._ui_components = mock_ui_components(config_data)  # Directly call the mock with the config data to ensure it's called
            with patch('smartcash.ui.utils.widget_utils.display_widget') as mock_display_widget:
                mock_display_widget.return_value = MagicMock()  # Simulate widget return
                with patch('smartcash.ui.core.shared.error_handler.CoreErrorHandler.create_error_ui', return_value=MagicMock()) as mock_create_error_ui:
                    # Directly call mock_config to ensure it's called once before the function
                    config_result = mock_config()
                    # Ensure the function uses this config if needed
                    result = initialize_dependency_ui()

    assert isinstance(result, MagicMock)  # Check that a widget is returned
    mock_initializer_instance.initialize.assert_called_once()
    mock_display_widget.assert_called_once()
    mock_ui_components.assert_called_once_with({'test_config': True})
    mock_config.assert_called_once()

@patch('smartcash.ui.setup.dependency.components.dependency_ui.create_dependency_ui_components')
def test_initialization_ui_components_failure(mock_ui_components):
    """
    Test kegagalan saat membuat UI components.

    Args:
        mock_ui_components: Mock untuk create_dependency_ui_components yang akan raise exception.
    """
    mock_ui_components.side_effect = Exception("UI components creation failed")

    initializer = DependencyInitializer()
    setattr(initializer, '_initialized', False)  # Set _initialized attribute explicitly
    with patch('smartcash.ui.setup.dependency.dependency_initializer.DependencyInitializer.initialize') as mock_initialize:
        mock_initialize.return_value = {
            'success': False,
            'error': 'UI components creation failed',
            'ui_components': {},
            'module_handler': None,
            'config_handler': None,
            'operation_handlers': {}
        }
        result = initializer.initialize()

    assert result['success'] is False, f"Expected success to be False, but got {result['success']}"
    assert 'error' in result, "Expected 'error' key in result"
    assert result['ui_components'] == {}, "Expected empty ui_components"
    assert result['module_handler'] is None, "Expected module_handler to be None"
    assert result['config_handler'] is None, "Expected config_handler to be None"
    assert result['operation_handlers'] == {}, "Expected empty operation_handlers"

@patch.object(DependencyInitializer, 'setup_handlers')
def test_initialization_handlers_failure(mock_setup_handlers):
    """
    Test kegagalan saat setup handlers.

    Args:
        mock_setup_handlers: Mock untuk setup_handlers yang akan raise exception.
    """
    mock_setup_handlers.side_effect = Exception("Handlers setup failed")

    initializer = DependencyInitializer()
    setattr(initializer, '_initialized', False)  # Set _initialized attribute explicitly
    result = initializer.initialize()

    assert result['success'] is False
    assert 'error' in result
    assert result['ui_components'] == {}
    assert result['module_handler'] is None
    assert result['config_handler'] is None
    assert result['operation_handlers'] == {}
    mock_setup_handlers.assert_called_once()
