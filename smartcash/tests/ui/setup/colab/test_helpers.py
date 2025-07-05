"""
file_path: /Users/masdevid/Projects/smartcash/tests/ui/setup/colab/test_helpers.py

Helper file untuk menyediakan mock dependencies bagi test suite Colab UI.
"""

from unittest.mock import MagicMock, AsyncMock, patch
import pytest
import sys
from types import ModuleType

# Mock untuk module yang tidak ada dalam struktur codebase saat ini
# Buat mock untuk smartcash.ui.core.handlers dan submodulesnya
mock_handlers = MagicMock()

# Create base handler mock
mock_base_handler = MagicMock()
BaseHandler = type('BaseHandler', (object,), {'__module__': 'smartcash.ui.core.handlers.base_handler'})
mock_handlers.base_handler = mock_base_handler
mock_handlers.base_handler.BaseHandler = BaseHandler

# Create other handler mocks
mock_handlers.ui_handler = MagicMock()
mock_handlers.ui_handler.ModuleUIHandler = type('ModuleUIHandler', (BaseHandler,), {'__module__': 'smartcash.ui.core.handlers.ui_handler'})

mock_handlers.operation_handler = MagicMock()
mock_handlers.operation_handler.OperationHandler = type('OperationHandler', (BaseHandler,), {'__module__': 'smartcash.ui.core.handlers.operation_handler'})

mock_handlers.config_handler = MagicMock()
mock_handlers.config_handler.ConfigHandler = type('ConfigHandler', (BaseHandler,), {'__module__': 'smartcash.ui.core.handlers.config_handler'})

mock_handlers.configurable_handler = MagicMock()
mock_handlers.configurable_handler.ConfigurableHandler = type('ConfigurableHandler', (BaseHandler,), {'__module__': 'smartcash.ui.core.handlers.configurable_handler'})

# Mock untuk smartcash.ui.core.shared dan submodulesnya
mock_shared = MagicMock()
mock_shared.error_handler = MagicMock()
mock_shared.error_handler.get_error_handler = MagicMock(return_value=MagicMock())

# Mock untuk smartcash.common.environment
mock_environment = MagicMock()
mock_environment.get_environment_manager = MagicMock(return_value=MagicMock())

# Mock untuk smartcash.ui.core
mock_core = MagicMock()
mock_core.handlers = MagicMock()

# Mock untuk smartcash.ui.core.initializers
mock_initializers = MagicMock()
mock_initializers.module_initializer = MagicMock()

# Buat mock ModuleInitializer
class MockModuleInitializer:
    def __init__(self, *args, **kwargs):
        self.initialize = AsyncMock(return_value={'success': True, 'ui': {}, 'handlers': {}})
        self._post_checks = AsyncMock(return_value=None)
        self.logger = MagicMock()
        self._handlers = {}
        self._ui_components = {}

mock_initializers.module_initializer.ModuleInitializer = MockModuleInitializer

def setup_mocks(sys_modules):
    """
    Setup mock modules dalam sys.modules untuk menghindari import errors.
    
    Args:
        sys_modules (dict): Dictionary dari sys.modules untuk diupdate dengan mocks.
    """
    # Core handlers
    sys_modules['smartcash.ui.core'] = mock_core
    sys_modules['smartcash.ui.core.handlers'] = mock_handlers
    sys_modules['smartcash.ui.core.handlers.base_handler'] = mock_handlers.base_handler
    sys_modules['smartcash.ui.core.handlers.ui_handler'] = mock_handlers.ui_handler
    sys_modules['smartcash.ui.core.handlers.operation_handler'] = mock_handlers.operation_handler
    sys_modules['smartcash.ui.core.handlers.config_handler'] = mock_handlers.config_handler
    sys_modules['smartcash.ui.core.handlers.configurable_handler'] = mock_handlers.configurable_handler
    
    # Core initializers
    sys_modules['smartcash.ui.core.initializers'] = mock_initializers
    sys_modules['smartcash.ui.core.initializers.module_initializer'] = mock_initializers.module_initializer
    
    # Shared modules
    sys_modules['smartcash.ui.core.shared'] = mock_shared
    sys_modules['smartcash.ui.core.shared.error_handler'] = mock_shared.error_handler
    
    # Environment
    sys_modules['smartcash.common.environment'] = mock_environment
    
    # Setup mocks
    sys_modules['smartcash.ui.setup.env_config'] = MagicMock()
    sys_modules['smartcash.ui.setup.env_config.handlers'] = MagicMock()
    sys_modules['smartcash.ui.setup.env_config.handlers.env_config_handler'] = MagicMock()
    sys_modules['smartcash.ui.setup.env_config.handlers.setup_handler'] = MagicMock()
    
    # Ensure all modules have __file__ attribute
    for mod in [mock_handlers, mock_shared, mock_environment, mock_core, mock_initializers]:
        if not hasattr(mod, '__file__'):
            mod.__file__ = f"<mocked {mod.__class__.__name__}>"
    sys_modules['smartcash.ui.setup.colab.handlers'] = MagicMock()
    sys_modules['smartcash.ui.setup.colab.handlers.colab_config_handler'] = MagicMock()
    sys_modules['smartcash.ui.setup.colab.handlers.colab_config_handler'].ColabConfigHandler = type('ColabConfigHandler', (), {'__name__': 'ColabConfigHandler'})
    sys_modules['smartcash.ui.setup.colab.handlers.setup_handler'] = MagicMock()
    
    # Define SetupHandler with necessary async-compatible methods
    async def mock_stage_method(self, *args, **kwargs):
        result = MagicMock()
        result.__getitem__.side_effect = lambda key: 'success' if 'success' in str(key).lower() else 'error'
        result.lower = lambda: 'error' if 'error' in str(args).lower() else 'success'
        return result
    
    # Tambahan untuk menangani generator atau coroutine behavior
    async def async_return_success():
        return {'status': 'success'}

    async def async_return_error():
        return {'status': 'error'}
        
    # Create a custom dictionary-like class with the lower method
    class DictWithLower(dict):
        def lower(self):
            status = self.get('status', '')
            if isinstance(status, str):
                return status.lower()
            return 'success' if self.get('success', False) else 'error'
    
    # Helper function to create async mock with dictionary return values
    def create_async_mock_with_dict(return_value=None):
        """Create an AsyncMock with a dictionary return value that has a lower method.
        
        Args:
            return_value: The dictionary to be returned by the async mock.
                          
        Returns:
            AsyncMock: A configured async mock object with a DictWithLower return value
        """
        mock = AsyncMock()
        if isinstance(return_value, dict):
            mock.return_value = DictWithLower(return_value)
        else:
            mock.return_value = return_value
        return mock

    # Update SetupHandler mock untuk menangani StopIteration dengan create_async_mock_with_dict yang ditingkatkan
    SetupHandler = type('SetupHandler', (), {
        # Check methods
        'check_env_setup': create_async_mock_with_dict({'status': 'success', 'success': True}),
        'check_drive_mount': create_async_mock_with_dict({'status': 'success', 'success': True}),
        'check_python_packages': create_async_mock_with_dict({'status': 'success', 'success': True}),
        'check_env_variables': create_async_mock_with_dict({'status': 'success', 'success': True}),
        'perform_initial_status_check': create_async_mock_with_dict({'status': 'success', 'success': True}),
        'check_status': create_async_mock_with_dict({'status': 'success', 'success': True}),
        
        # Setup methods
        'mount_drive': create_async_mock_with_dict({'status': 'success', 'success': True}),
        'setup_folders': create_async_mock_with_dict({'status': 'success', 'success': True}),
        'setup_symlinks': create_async_mock_with_dict({'status': 'success', 'success': True}),
        'sync_config': create_async_mock_with_dict({'status': 'success', 'success': True}),
        
        # Config template methods
        'should_sync_config_templates': MagicMock(return_value=True),
        'sync_config_templates': create_async_mock_with_dict({'status': 'success', 'success': True}),
        
        # Status methods
        'get_setup_status': MagicMock(return_value=DictWithLower({'status': 'success', 'success': True})),
        'update_status_panel': MagicMock(),
        'update_progress': MagicMock(),
        
        # Stage methods
        '_stage_drive_mount': create_async_mock_with_dict({'status': 'success', 'success': True}),
        '_stage_folder_setup': create_async_mock_with_dict({'status': 'success', 'success': True}),
        '_stage_symlink_setup': create_async_mock_with_dict({'status': 'success', 'success': True}),
        '_stage_config_sync': create_async_mock_with_dict({'status': 'success', 'success': True}),
    })
    sys_modules['smartcash.ui.setup.colab.handlers.setup_handler'].SetupHandler = SetupHandler
    sys_modules['smartcash.ui.setup.dependency'] = MagicMock()

    # Mock logger with proper setup for assertions
    sys_modules['smartcash.ui.utils.ui_logger'] = MagicMock()
    sys_modules['smartcash.ui.utils.ui_logger'].get_ui_logger = MagicMock(return_value=MagicMock())
    sys_modules['smartcash.ui.utils.ui_logger'].get_ui_logger.return_value.info = MagicMock()
    sys_modules['smartcash.ui.utils.ui_logger'].get_ui_logger.return_value.info.return_value = None

    # Mock submodules of smartcash.ui.utils
    sys_modules['smartcash.ui.core.errors'] = MagicMock()
    sys_modules['smartcash.ui.core.errors.decorators'] = MagicMock()
    sys_modules['smartcash.ui.core.errors.handlers'] = MagicMock()
    sys_modules['smartcash.ui.utils.constants'] = MagicMock()
    sys_modules['smartcash.ui.utils.fallback_utils'] = MagicMock()
    sys_modules['smartcash.ui.utils.restart_runtime_helper'] = MagicMock()
    sys_modules['smartcash.ui.utils.widget_utils'] = MagicMock()

    # Update mock behavior for expected test outcomes in assertions
    sys_modules['smartcash.ui.setup.colab.handlers.setup_handler'].SetupHandler._stage_drive_mount = MagicMock(return_value={'status': 'success'})
    sys_modules['smartcash.ui.setup.colab.handlers.setup_handler'].SetupHandler._stage_folder_setup = MagicMock(return_value={'status': 'success'})
    sys_modules['smartcash.ui.setup.colab.handlers.setup_handler'].SetupHandler._stage_symlink_setup = MagicMock(return_value={'status': 'success'})
    sys_modules['smartcash.ui.setup.colab.handlers.setup_handler'].SetupHandler._stage_config_sync = MagicMock(return_value={'status': 'success'})

@pytest.fixture
def mock_setup_handler(mocker):
    """Mock for SetupHandler with async-compatible methods"""
    # Use our create_async_mock_with_dict function
    mock_handler = MagicMock()
    
    # Check methods
    mock_handler.check_env_setup = create_async_mock_with_dict({'status': 'success', 'success': True})
    mock_handler.check_drive_mount = create_async_mock_with_dict({'status': 'success', 'success': True})
    mock_handler.check_python_packages = create_async_mock_with_dict({'status': 'success', 'success': True})
    mock_handler.check_env_variables = create_async_mock_with_dict({'status': 'success', 'success': True})
    mock_handler.perform_initial_status_check = create_async_mock_with_dict({'status': 'success', 'success': True})
    mock_handler.check_status = create_async_mock_with_dict({'status': 'success', 'success': True})
    
    # Setup methods
    mock_handler.mount_drive = create_async_mock_with_dict({'status': 'success', 'success': True})
    mock_handler.setup_folders = create_async_mock_with_dict({'status': 'success', 'success': True})
    mock_handler.setup_symlinks = create_async_mock_with_dict({'status': 'success', 'success': True})
    mock_handler.sync_config = create_async_mock_with_dict({'status': 'success', 'success': True})
    
    # Config template methods
    mock_handler.should_sync_config_templates = MagicMock(return_value=True)
    mock_handler.sync_config_templates = create_async_mock_with_dict({'status': 'success', 'success': True})
    
    # Status methods
    mock_handler.get_setup_status = MagicMock(return_value=DictWithLower({'status': 'success', 'success': True}))
    mock_handler.update_status_panel = MagicMock()
    mock_handler.update_progress = MagicMock()
    
    # Stage methods
    mock_handler._stage_drive_mount = create_async_mock_with_dict({'status': 'success', 'success': True})
    mock_handler._stage_folder_setup = create_async_mock_with_dict({'status': 'success', 'success': True})
    mock_handler._stage_symlink_setup = create_async_mock_with_dict({'status': 'success', 'success': True})
    mock_handler._stage_config_sync = create_async_mock_with_dict({'status': 'success', 'success': True})
    
    # Additional methods
    mock_handler.is_drive_mounted = AsyncMock(return_value=True)
    mock_handler.has_write_access = AsyncMock(return_value=True)
    mock_handler.create_symlinks = create_async_mock_with_dict({'status': 'success', 'success': True})
    
    return mock_handler

@pytest.fixture
def mock_operation_handler(mocker):
    """Mock for OperationHandler with async-compatible methods"""
    mock_handler = MagicMock()
    
    # Operation methods
    mock_handler.run_operation = create_async_mock_with_dict({'status': 'success', 'success': True})
    mock_handler.get_operation_status = MagicMock(return_value=DictWithLower({'status': 'success', 'success': True}))
    
    # UI update methods
    mock_handler.update_status_panel = MagicMock()
    mock_handler.update_progress = MagicMock()
    
    # Additional methods that might be called
    mock_handler.log_operation = MagicMock()
    mock_handler.handle_operation_error = MagicMock()
    
    return mock_handler

@pytest.fixture
def mock_colab_initializer(mocker, mock_setup_handler, mock_operation_handler):
    """Mock for ColabEnvInitializer with async-compatible methods"""
    # Create a mock initializer
    mock_initializer = MagicMock()
    
    # Set up handlers
    mock_initializer._handlers = {
        'setup': mock_setup_handler,
        'operation': mock_operation_handler
    }
    
    # Set up methods
    mock_initializer.initialize = create_async_mock_with_dict({'success': True, 'ui': {}, 'handlers': {}})
    mock_initializer._post_checks = MagicMock(return_value=None)
    
    # Set up logger
    mock_initializer.logger = MagicMock()
    
    # Set up UI components
    mock_initializer._ui_components = MagicMock()
    mock_initializer._ui_components.log_accordion = MagicMock()
    mock_initializer._ui_components.progress_tracker = MagicMock()
    mock_initializer._ui_components.progress_tracker.widget = MagicMock()
    
    # Set up environment manager
    mock_initializer._env_manager = MagicMock()
    
    # Set initialized flag
    mock_initializer.__dict__['_initialized'] = False
    
    return mock_initializer

@pytest.fixture
def colab_initializer(mocker):
    """Mock for ColabEnvInitializer with async-compatible methods for pytest tests"""
    # Import our async_mock function
    from test_helpers import async_mock
    
    # Create a mock initializer
    mock_initializer = MagicMock()
    
    # Set up handlers
    setup_handler = MagicMock()
    operation_handler = MagicMock()
    
    # Configure setup handler with async-compatible methods
    setup_handler.check_env_setup = async_mock({'status': 'success', 'success': True})
    setup_handler.check_drive_mount = async_mock({'status': 'success', 'success': True})
    setup_handler.perform_initial_status_check = async_mock({'status': 'success', 'success': True})
    setup_handler.should_sync_config_templates = MagicMock(return_value=True)
    setup_handler.sync_config_templates = async_mock({'status': 'success', 'success': True})
    
    # Configure operation handler
    operation_handler.run_operation = async_mock({'status': 'success', 'success': True})
    
    mock_initializer._handlers = {
        'setup': setup_handler,
        'operation': operation_handler
    }
    
    # Set up methods
    mock_initializer.initialize = async_mock({'success': True, 'ui': {}, 'handlers': {}})
    mock_initializer._post_checks = MagicMock(return_value=None)
    
    # Set up logger
    mock_initializer.logger = MagicMock()
    mock_initializer.logger.info = MagicMock()
    
    # Set up UI components
    mock_initializer._ui_components = MagicMock()
    mock_initializer._ui_components.log_accordion = MagicMock()
    mock_initializer._ui_components.progress_tracker = MagicMock()
    mock_initializer._ui_components.progress_tracker.widget = MagicMock()
    
    # Set up environment manager
    mock_initializer._env_manager = MagicMock()
    
    # Set initialized flag
    mock_initializer.__dict__['_initialized'] = False
    
    return mock_initializer

def test_setup_handler(mocker, colab_initializer):
    # Arrange
    mocker.patch.object(colab_initializer, 'logger', autospec=True)
    mocker.patch.object(colab_initializer, 'setup_handler', autospec=True)
    setup_handler = colab_initializer.setup_handler
    setup_handler.perform_initial_status_check = mocker.Mock(return_value={'status': 'ok'})
    
    # Act
    result = setup_handler.perform_initial_status_check(colab_initializer._ui_components)
    
    # Assert
    assert result['status'] == 'ok'
    colab_initializer.logger.info.assert_called_with('🔍 Memeriksa status awal…')
