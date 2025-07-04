"""
File: /Users/masdevid/Projects/smartcash/tests/ui/setup/dependency/test_dependency_initializer.py
Deskripsi: Unit dan integration tests untuk DependencyInitializer.
"""

import pytest
from unittest.mock import patch, MagicMock

from smartcash.ui.setup.dependency.dependency_initializer import DependencyInitializer, initialize_dependency_ui


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
def test_dependency_initializer_initialization(mock_config, mock_dependency_ui_components):
    """
    Test inisialisasi DependencyInitializer.

    Args:
        mock_config: Mock untuk get_default_dependency_config.
        mock_dependency_ui_components: Mock untuk UI components.
    """
    mock_config.return_value = {'test_config': True}
    mock_dependency_ui_components.return_value = {'test_ui': 'component'}

    initializer = DependencyInitializer()
    with patch('smartcash.ui.setup.dependency.handlers.dependency_ui_handler.DependencyUIHandler') as mock_handler:
        mock_handler_instance = MagicMock()
        mock_handler.return_value = mock_handler_instance
        with patch.object(initializer, 'setup_handlers', return_value=None) as mock_setup_handlers:
            with patch.object(initializer, 'setup_operation_handlers', return_value=None) as mock_setup_ops:
                with patch.object(initializer, 'get_default_config', return_value={'test_config': True}) as mock_get_config:
                    initializer._module_handler = mock_handler_instance  # Set module handler explicitly
                    initializer._operation_handlers = {'test_op': 'handler'}  # Set operation handlers explicitly
                    initializer._config = {'test_config': True}  # Set config explicitly
                    initializer._ui_components = {'test_ui': 'component'}  # Set UI components explicitly
                    result = initializer.initialize()

    assert result['success'] is True
    assert 'ui_components' in result
    assert 'config' in result
    assert 'module_handler' in result
    assert result['module_handler'] == mock_handler_instance
    assert 'operation_handlers' in result
    assert result['operation_handlers'] == {'test_op': 'handler'}
    mock_get_config.assert_called()
    mock_dependency_ui_components.assert_called_with({'test_config': True})

@patch('smartcash.ui.setup.dependency.configs.dependency_defaults.get_default_dependency_config')
def test_initialize_dependency_ui_function(mock_config, mock_dependency_ui_components):
    """
    Test fungsi initialize_dependency_ui.

    Args:
        mock_config: Mock untuk get_default_dependency_config.
        mock_dependency_ui_components: Mock untuk UI components.
    """
    mock_config.return_value = {'test_config': True}
    mock_dependency_ui_components.return_value = {'test_ui': 'component'}

    with patch('smartcash.ui.setup.dependency.handlers.dependency_ui_handler.DependencyUIHandler') as mock_handler:
        mock_handler_instance = MagicMock()
        mock_handler.return_value = mock_handler_instance
        with patch('smartcash.ui.setup.dependency.dependency_initializer.DependencyInitializer') as mock_initializer_class:
            mock_initializer_instance = MagicMock()
            mock_initializer_class.return_value = mock_initializer_instance
            mock_initializer_instance.initialize.return_value = {
                'success': True,
                'ui_components': {'test': 'ui'},
                'config': {'test_config': True},
                'module_handler': mock_handler_instance,
                'operation_handlers': {'test_op': 'handler'}
            }
            with patch('smartcash.ui.setup.dependency.dependency_initializer.DependencyInitializer.get_default_config', return_value={'test_config': True}):
                result = initialize_dependency_ui()

    assert 'success' in result
    assert result['success'] is True
    assert 'ui_components' in result
    assert 'module_handler' in result
    assert 'config_handler' in result
    assert 'operation_handlers' in result

@patch('smartcash.ui.setup.dependency.components.dependency_ui.create_dependency_ui_components')
def test_initialization_ui_components_failure(mock_ui_components):
    """
    Test kegagalan saat membuat UI components.

    Args:
        mock_ui_components: Mock untuk create_dependency_ui_components yang akan raise exception.
    """
    mock_ui_components.side_effect = Exception("UI components creation failed")

    initializer = DependencyInitializer()
    result = initializer.initialize()

    assert result['success'] is False
    assert 'error' in result
    assert result['ui_components'] == {}
    assert result['module_handler'] is None
    assert result['config_handler'] is None
    assert result['operation_handlers'] == {}

@patch.object(DependencyInitializer, 'setup_handlers')
def test_initialization_handlers_failure(mock_setup_handlers):
    """
    Test kegagalan saat setup handlers.

    Args:
        mock_setup_handlers: Mock untuk setup_handlers yang akan raise exception.
    """
    mock_setup_handlers.side_effect = Exception("Handlers setup failed")

    initializer = DependencyInitializer()
    result = initializer.initialize()

    assert result['success'] is False
    assert 'error' in result
    assert result['ui_components'] == {}
    assert result['module_handler'] is None
    assert result['config_handler'] is None
    assert result['operation_handlers'] == {}
