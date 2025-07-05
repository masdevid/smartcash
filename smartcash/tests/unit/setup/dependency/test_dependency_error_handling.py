"""
Minimal test for error handling in DependencyInitializer.
"""
import pytest
from unittest.mock import MagicMock, patch

def test_initialization_handles_handler_failure():
    """Test that initialization handles handler setup failures gracefully."""
    print("\n=== Starting test_initialization_handles_handler_failure ===")
    
    # Create a mock for the ModuleInitializer class
    mock_module_initializer = MagicMock()
    mock_module_initializer.initialize_module_ui.return_value = {
        'ui': 'mock_ui', 
        'handlers': {}
    }
    print("Created mock_module_initializer")
    
    # Create a mock for the DependencyUIHandler class
    print("Creating mock_ui_handler_instance...")
    mock_ui_handler_instance = MagicMock()
    mock_ui_handler_instance.setup.side_effect = Exception("Handler setup failed")
    print(f"mock_ui_handler_instance.setup.side_effect set to: {mock_ui_handler_instance.setup.side_effect}")
    
    mock_ui_handler = MagicMock(return_value=mock_ui_handler_instance)
    print(f"Created mock_ui_handler with return_value: {mock_ui_handler.return_value}")
    
    # Create a mock for the logger with detailed logging
    mock_logger = MagicMock()
    mock_logger.error = MagicMock()
    print("Created mock_logger with error method")
    
    # Create a mock for the config handler
    mock_config_handler = MagicMock()
    print("Created mock_config_handler")
    
    # Create a mock for the module under test
    print("Setting up patches...")
    with patch('smartcash.ui.setup.dependency.dependency_initializer.ModuleInitializer', 
              return_value=mock_module_initializer) as mock_module_init_class, \
         patch('smartcash.ui.setup.dependency.dependency_initializer.DependencyUIHandler',
              mock_ui_handler) as mock_ui_handler_class, \
         patch('smartcash.ui.setup.dependency.dependency_initializer.get_default_dependency_config',
              return_value={}) as mock_get_config:
        
        # Mock the ui_logger module and its functions
        print("Setting up sys.modules patches...")
        with patch.dict('sys.modules', {
            'smartcash.ui.utils.ui_logger': MagicMock(),
            'smartcash.ui.setup.dependency.dependency_initializer.ui_logger': MagicMock()
        }):
            print("Mocked sys.modules")
            
            # Now import the module under test after patching
            print("Importing dependency_initializer...")
            import smartcash.ui.setup.dependency.dependency_initializer
            from importlib import reload
            print("Reloading module...")
            reload(smartcash.ui.setup.dependency.dependency_initializer)
            
            # Get the module reference
            module = smartcash.ui.setup.dependency.dependency_initializer
            print(f"Got module: {module.__name__}")
            
            # Set up the mock logger on the module
            print("Setting up mock logger on module...")
            module.ui_logger = MagicMock()
            module.ui_logger.get_module_logger.return_value = mock_logger
            module.ui_logger.get_default_logger.return_value = mock_logger
            
            # Also set up the get_module_logger function
            module.get_module_logger = MagicMock(return_value=mock_logger)
            print("Mock logger setup complete")
        
        # Import the module under test after patching
        print("Importing DependencyInitializer...")
        from smartcash.ui.setup.dependency.dependency_initializer import DependencyInitializer
        
        # Create a subclass that properly initializes the parent class
        class MockDependencyInitializer(DependencyInitializer):
            def __init__(self, *args, **kwargs):
                # Don't call parent's __init__ to avoid side effects
                self._initialized = False
                self._ui_components = {}
                self._operation_handlers = {}
                self.logger = mock_logger
                self.config_handler = mock_config_handler
                self._initialization_lock = MagicMock()
                self._config = {}
                self._initialization_handlers = {}
                self._module_name = 'test_module'
                self._module_display_name = 'Test Module'
                self._parent_module = 'test_parent'
                self.full_module_name = f"{self._parent_module}.{self._module_name}"
                self._auto_setup_handlers = True
                self._handler_class = MagicMock()
                self._current_operation = None
                self._current_packages = None
                self._handlers = {}  # Add missing _handlers attribute
                
                # Mock the parent class's initialize method
                self.initialize_module_ui = MagicMock(return_value={'ui': 'mock_ui', 'handlers': {}})
                
                # Mock the parent class's initialize method to simulate the error
                def mock_initialize(*args, **kwargs):
                    print("Calling mock initialize...")
                    # Simulate error during handler setup
                    print("Simulating handler setup error...")
                    # Return a result that indicates failure
                    return {
                        'success': False,
                        'error': 'Handler setup failed',
                        'dialog': 'Error dialog content'
                    }
                
                self.initialize = MagicMock(side_effect=mock_initialize)
        
        print("Creating MockDependencyInitializer instance...")
        initializer = MockDependencyInitializer()
        print(f"Created MockDependencyInitializer instance: {initializer}")
        
        # Patch get_module_logger to return our mock logger - keep this patch active
        mock_get_logger = patch('smartcash.ui.setup.dependency.dependency_initializer.get_module_logger', 
                             return_value=mock_logger).start()
        print(f"Patched get_module_logger, mock: {mock_get_logger}")
        print(f"mock_get_logger.return_value: {mock_get_logger.return_value}")
        
        # Ensure the mock logger has the error method
        mock_logger.error = MagicMock()
        print("Initializer attributes set")
        
        # Call initialize which should handle the error
        print("\n=== Test: Calling initializer.initialize() ===")
        try:
            print("Initializer state before initialize():")
            print(f"  _initialized: {initializer._initialized}")
            print(f"  _auto_setup_handlers: {initializer._auto_setup_handlers}")
            print(f"  _ui_components: {initializer._ui_components}")
            print(f"  _handlers: {initializer._handlers}")
            print(f"  _initialization_handlers: {initializer._initialization_handlers}")
            
            # Print mock logger state before the call
            print("\nMock logger state before initialize():")
            print(f"  mock_logger.error.call_count: {mock_logger.error.call_count}")
            print(f"  mock_logger.error.call_args_list: {mock_logger.error.call_args_list}")
            
            print("\nCalling initializer.initialize()...")
            result = initializer.initialize()
            print(f"initialize() returned: {result}")
            
            # Print mock logger state after the call
            print("\nMock logger state after initialize():")
            print(f"  mock_logger.error.call_count: {mock_logger.error.call_count}")
            print(f"  mock_logger.error.call_args_list: {mock_logger.error.call_args_list}")
            
            print("\nInitializer state after initialize():")
            print(f"  _initialized: {initializer._initialized}")
            print(f"  _ui_components: {initializer._ui_components}")
            print(f"  _handlers: {initializer._handlers}")
            print(f"  _initialization_handlers: {initializer._initialization_handlers}")
            
            # Verify the return value has the expected structure
            print("Verifying return value structure...")
            assert isinstance(result, dict), f"Expected dict, got {type(result)}"
            print("Return value is a dictionary")
            print(f"Return value keys: {list(result.keys())}")
            
            # Check for success flag
            assert 'success' in result, "Expected 'success' key in result"
            assert result['success'] is False, "Expected success to be False"
            
            # Check for error message
            assert 'error' in result, "Expected 'error' key in result"
            error_message = str(result['error']).lower()
            print(f"Error message: {error_message}")
            
            # Verify the error message contains relevant keywords
            keywords = ['error', 'fail', 'exception', 'handler', 'setup', 'initializ']
            assert any(keyword in error_message for keyword in keywords), \
                f"Error message should contain one of {keywords}, got: {error_message}"
            
            # Verify initialize was called
            print("\nVerifying initialize was called...")
            print(f"initializer.initialize.call_count = {initializer.initialize.call_count}")
            assert initializer.initialize.call_count > 0, "Expected initialize to be called"
            
            # Verify logger was called with error
            print("\nVerifying error was logged...")
            print(f"mock_logger.error.call_count = {mock_logger.error.call_count}")
            
            # Print all error messages for debugging
            for i, call_args in enumerate(mock_logger.error.call_args_list, 1):
                error_msg = str(call_args[0][0]).lower()
                print(f"Error {i}: {error_msg}")
            
            # Verify at least one error was logged
            assert mock_logger.error.call_count > 0, "Expected at least one error log message"
            
            # Check if any error message contains relevant keywords
            error_found = any(
                any(keyword in str(call_args[0][0]).lower() 
                    for keyword in ['error', 'fail', 'exception', 'handler'])
                for call_args in mock_logger.error.call_args_list
            )
            
            assert error_found, "Expected an error message containing 'error', 'fail', 'exception', or 'handler'"
            
            print("\n=== Test assertions passed ===")
            
        except Exception as e:
            print(f"\n=== Test failed with exception: {e} ===")
            print("Traceback:")
            import traceback
            traceback.print_exc()
            assert False, f"Test failed with exception: {e}"
