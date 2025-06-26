"""
Tests for the ConfigCellInitializer class to ensure it returns display widgets.
"""
import logging
import sys
import unittest
from io import StringIO
from typing import Dict, Any, Optional, List, Type, TypeVar
from unittest.mock import MagicMock, patch, ANY, call, PropertyMock

import ipywidgets as widgets
from ipywidgets import VBox, HBox, HTML, Output, Button, Layout, Tab, Accordion

from smartcash.ui.initializers.config_cell_initializer import ConfigCellInitializer
from smartcash.ui.config_cell.handlers.config_handler import ConfigCellHandler
from smartcash.ui.config_cell.components.component_registry import ComponentRegistry

T = TypeVar('T', bound=ConfigCellHandler)

class TestConfigHandler(ConfigCellHandler):
    """Test implementation of ConfigCellHandler for testing."""
    
    def __init__(self, module_name='test_module'):
        super().__init__(module_name=module_name)
        self._is_initialized = False
    
    def initialize(self):
        self._is_initialized = True
        return self
    
    def get_config(self):
        return {'test': 'config'}
    
    def validate_config(self, config):
        return True, ""

class TestConfigCellInitializer(ConfigCellInitializer[TestConfigHandler]):
    """Test implementation of ConfigCellInitializer for testing."""
    
    def create_handler(self) -> TestConfigHandler:
        """Create a test handler."""
        return TestConfigHandler()
    
    def create_ui_components(self, config: Dict[str, Any]) -> dict:
        """Create test UI components."""
        return {
            'container': widgets.VBox(),
            'button': widgets.Button(description='Test Button')
        }
        
    def setup_handlers(self) -> None:
        """Set up event handlers."""
        pass

class TestConfigCellInitializerDisplay(unittest.TestCase):
    """Test that ConfigCellInitializer returns display widgets."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Patch the component registry
        self.registry_patcher = patch('smartcash.ui.config_cell.components.component_registry.component_registry')
        self.mock_registry = self.registry_patcher.start()
        self.mock_registry.get_component.return_value = None
        
        # Create test widgets with required widget attributes
        self.test_widget = MagicMock(spec=widgets.Widget)
        self.test_widget._ipython_display_ = MagicMock()
        self.test_widget._repr_mimebundle_ = MagicMock(return_value={'text/plain': 'test'})
        
        # Create a test handler
        self.mock_handler = MagicMock(spec=TestConfigHandler)
        self.mock_handler.get_config.return_value = {}
        self.mock_handler.initialize.return_value = self.mock_handler
        
        # Patch error handler
        self.error_widget = widgets.HTML("<div>Error</div>")
        self.error_patcher = patch('smartcash.ui.config_cell.handlers.error_handler.create_error_response')
        self.mock_error = self.error_patcher.start()
        self.mock_error.return_value = self.error_widget
        
        # Create a test initializer class that properly implements all abstract methods
        class TestInitializer(ConfigCellInitializer[TestConfigHandler]):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._handler_created = False
                
            def create_handler(self):
                self._handler_created = True
                return self.mock_handler
                
            def create_ui_components(self, config=None):
                return {'container': self.test_widget}
                
            def setup_handlers(self):
                pass
        
        self.TestInitializer = TestInitializer
        
        # Mock the ParentComponentManager
        self.mock_pcm_patcher = patch('smartcash.ui.initializers.config_cell_initializer.ParentComponentManager')
        self.mock_pcm_class = self.mock_pcm_patcher.start()
        self.mock_pcm = MagicMock()
        self.mock_pcm.container = self.test_widget
        self.mock_pcm.content_area = MagicMock()
        self.mock_pcm_class.return_value = self.mock_pcm
        
        # Mock UILoggerBridge
        self.mock_logger_bridge_patcher = patch('smartcash.ui.initializers.config_cell_initializer.UILoggerBridge')
        self.mock_logger_bridge = self.mock_logger_bridge_patcher.start()
        self.mock_logger = MagicMock()
        self.mock_logger_bridge_instance = MagicMock(logger=self.mock_logger)
        self.mock_logger_bridge.return_value = self.mock_logger_bridge_instance
        
        # Patch logging utilities
        self.mock_logging_utils_patcher = patch('smartcash.ui.initializers.config_cell_initializer.logging')
        self.mock_logging = self.mock_logging_utils_patcher.start()
        self.mock_logging.getLogger.return_value = MagicMock()
        
        # Patch component registry imports
        self.mock_component_registry_patcher = patch('smartcash.ui.initializers.config_cell_initializer.component_registry')
        self.mock_component_registry = self.mock_component_registry_patcher.start()
        self.mock_component_registry.get_component.return_value = None
        
        # Patch create_parent_component
        self.mock_create_parent_patcher = patch('smartcash.ui.initializers.config_cell_initializer.create_parent_component')
        self.mock_create_parent = self.mock_create_parent_patcher.start()
        self.mock_create_parent.return_value = MagicMock()
        
        # Patch restore_stdout
        self.mock_restore_stdout_patcher = patch('smartcash.ui.initializers.config_cell_initializer.restore_stdout')
        self.mock_restore_stdout = self.mock_restore_stdout_patcher.start()
        
        # Patch setup_output_suppression
        self.mock_setup_output_suppression_patcher = patch.object(
            ConfigCellInitializer, '_setup_output_suppression'
        )
        self.mock_setup_output_suppression = self.mock_setup_output_suppression_patcher.start()
        
        # Patch _restore_output
        self.mock_restore_output_patcher = patch.object(
            ConfigCellInitializer, '_restore_output'
        )
        self.mock_restore_output = self.mock_restore_output_patcher.start()
        
        # Patch _setup_logging
        self.mock_setup_logging_patcher = patch.object(
            ConfigCellInitializer, '_setup_logging'
        )
        self.mock_setup_logging = self.mock_setup_logging_patcher.start()
        
        # Patch _register_component
        self.mock_register_component_patcher = patch.object(
            ConfigCellInitializer, '_register_component'
        )
        self.mock_register_component = self.mock_register_component_patcher.start()
        
        # Patch _initialize_children
        self.mock_initialize_children_patcher = patch.object(
            ConfigCellInitializer, '_initialize_children'
        )
        self.mock_initialize_children = self.mock_initialize_children_patcher.start()
        
        # Patch _setup_ui_components
        self.mock_setup_ui_components_patcher = patch.object(
            ConfigCellInitializer, '_setup_ui_components'
        )
        self.mock_setup_ui_components = self.mock_setup_ui_components_patcher.start()
        
        # Patch create_handler to return our mock handler
        self.mock_create_handler_patcher = patch.object(
            self.TestInitializer, 'create_handler', return_value=self.mock_handler
        )
        self.mock_create_handler = self.mock_create_handler_patcher.start()
        
        # Patch create_ui_components to return our test widget
        self.mock_create_ui_components_patcher = patch.object(
            self.TestInitializer, 'create_ui_components', return_value={'container': self.test_widget}
        )
        self.mock_create_ui_components = self.mock_create_ui_components_patcher.start()
        
        # Patch setup_handlers
        self.mock_setup_handlers_patcher = patch.object(
            self.TestInitializer, 'setup_handlers'
        )
        self.mock_setup_handlers = self.mock_setup_handlers_patcher.start()
    
    def tearDown(self):
        """Clean up after each test method."""
        # Stop all patchers
        self.registry_patcher.stop()
        self.mock_pcm_patcher.stop()
        self.mock_logger_bridge_patcher.stop()
        self.mock_logging_utils_patcher.stop()
        self.mock_component_registry_patcher.stop()
        self.error_patcher.stop()
        self.mock_create_parent_patcher.stop()
        self.mock_restore_stdout_patcher.stop()
        self.mock_setup_output_suppression_patcher.stop()
        self.mock_restore_output_patcher.stop()
        self.mock_setup_logging_patcher.stop()
        self.mock_register_component_patcher.stop()
        self.mock_initialize_children_patcher.stop()
        self.mock_setup_ui_components_patcher.stop()
        self.mock_create_handler_patcher.stop()
        self.mock_create_ui_components_patcher.stop()
        self.mock_setup_handlers_patcher.stop()
    
    def test_initialize_returns_widgets(self):
        """Test that initialize() returns a valid ipywidgets.Widget container."""
        # Setup test data
        test_config = {"test": "config"}
        test_parent_id = "test_parent"
        test_component_id = "test_component"
        
        # Create mock widgets
        test_container = MagicMock(spec=widgets.VBox)
        test_content_area = MagicMock(spec=widgets.VBox)
        
        # Configure parent component manager mock
        self.mock_pcm.container = test_container
        self.mock_pcm.content_area = test_content_area
        self.mock_pcm_class.return_value = self.mock_pcm
        
        # Configure UI components mock
        ui_components = {
            'container': test_container,
            'content_area': test_content_area,
            'test_widget': MagicMock(spec=widgets.Widget)
        }
        self.mock_create_ui_components.return_value = ui_components
        
        # Configure test handler
        test_handler = TestConfigHandler()
        self.mock_create_handler.return_value = test_handler
        
        # Configure component registry mock
        mock_registry = MagicMock()
        mock_registry.get_component.return_value = None
        
        # Reset all mocks to ensure clean state
        self.mock_create_handler.reset_mock()
        self.mock_create_ui_components.reset_mock()
        self.mock_setup_handlers.reset_mock()
        self.mock_register_component.reset_mock()
        self.mock_setup_ui_components.reset_mock()
        self.mock_initialize_children.reset_mock()
        
        # Patch component registry
        with patch('smartcash.ui.config_cell.components.component_registry.component_registry', mock_registry):
            # Create initializer
            initializer = self.TestInitializer(
                config=test_config,
                parent_id=test_parent_id,
                component_id=test_component_id
            )
            
            # Verify initial state
            self.assertFalse(initializer._is_initialized, "Should not be initialized yet")
            
            # Execute test
            result = initializer.initialize()
            
            # Verify the result is a widget with display capabilities
            self.assertIsNotNone(result, "Should return a widget")
            self.assertTrue(hasattr(result, '_ipython_display_') or hasattr(result, 'value'),
                          "Widget should be displayable")
            
            # Verify the container widget is returned
            self.assertEqual(result, test_container, "Should return the container widget")
            
            # Verify initialization flow
            self.mock_create_handler.assert_called_once()
            self.mock_create_ui_components.assert_called_once_with(test_config)
            self.mock_setup_handlers.assert_called_once()
            
            # Verify component registration
            self.mock_register_component.assert_called_once()
            
            # Verify UI setup and children initialization
            self.mock_setup_ui_components.assert_called_once()
            self.mock_initialize_children.assert_called_once()
            
            # Verify state after initialization
            self.assertTrue(initializer._is_initialized, 
                          "Initializer should be marked as initialized")
            
            # Verify the handler was created with the correct config
            self.mock_create_handler.assert_called_once()
            
            # Verify UI components were created with the correct config
            self.mock_create_ui_components.assert_called_once_with(test_config)
            
            # Verify the container was set up correctly
            self.assertEqual(initializer.parent_component.container, test_container, 
                            "Container should be set on the parent component")
            self.assertEqual(initializer.parent_component.content_area, test_content_area,
                            "Content area should be set on the parent component")
            
            # Verify the handler was set on the initializer
            self.assertEqual(initializer.handler, test_handler,
                            "Handler should be set on the initializer")
            
            # Verify the component was registered with the correct ID
            mock_registry.register_component.assert_called_once()
            call_args = mock_registry.register_component.call_args[0]
            self.assertEqual(call_args[0], f"{test_parent_id}.{test_component_id}",
                            "Component ID should be registered with parent prefix")
            self.assertIn('container', call_args[1],
                        "UI components should be registered")
    
    def test_initialize_with_error_returns_error_widget(self):
        """Test that initialize() returns an error widget on failure."""
        # Setup test data
        error_msg = "Test error during initialization"
        test_config = {'test': 'config'}
        test_component_id = 'test_error_component'
        test_parent_id = 'test_parent_component'
        
        # Create a real error widget for testing
        error_widget = MagicMock(spec=widgets.HTML)
        
        # Configure error widget creation
        self.mock_error_widget.return_value = error_widget
        
        # Create a test handler that will cause an error
        test_handler = TestConfigHandler()
        
        # Configure mocks to simulate an error during handler setup
        self.mock_create_handler.return_value = test_handler
        self.mock_setup_handlers.side_effect = Exception(error_msg)
        
        # Create real widgets for testing
        test_container = MagicMock(spec=widgets.VBox)
        test_content_area = MagicMock(spec=widgets.VBox)
        
        # Configure parent component manager mock
        self.mock_pcm.container = test_container
        self.mock_pcm.content_area = test_content_area
        self.mock_pcm_class.return_value = self.mock_pcm
        
        # Configure UI components mock
        ui_components = {
            'container': test_container,
            'content_area': test_content_area
        }
        self.mock_create_ui_components.return_value = ui_components
        
        # Reset all mocks to ensure clean state
        self.mock_create_handler.reset_mock()
        self.mock_create_ui_components.reset_mock()
        self.mock_setup_handlers.reset_mock()
        self.mock_register_component.reset_mock()
        self.mock_setup_ui_components.reset_mock()
        self.mock_initialize_children.reset_mock()
        self.mock_logger.error.reset_mock()
        
        # Create initializer
        initializer = self.TestInitializer(
            config=test_config,
            parent_id=test_parent_id,
            component_id=test_component_id
        )
        
        # Verify initial state
        self.assertFalse(initializer._is_initialized, "Should not be initialized yet")
        
        # Execute test and verify the error is caught
        with self.assertLogs(level='ERROR') as log_context:
            result = initializer.initialize()
        
        # Verify the result is the error widget
        self.assertIsNotNone(result, "Should return a widget")
        self.assertEqual(result, error_widget, "Should return the error widget")
        
        # Verify error handling flow
        self.mock_create_handler.assert_called_once()
        self.mock_create_ui_components.assert_called_once_with(test_config)
        self.mock_setup_handlers.assert_called_once()
        
        # Verify error widget was created with the correct error message
        self.mock_error_widget.assert_called_once()
        error_widget_call_args = self.mock_error_widget.call_args[0][0]
        self.assertIn("Failed to initialize", error_widget_call_args)
        self.assertIn(error_msg, error_widget_call_args)
        
        # Verify the initializer is not marked as initialized
        self.assertFalse(initializer._is_initialized, 
                       "Should not be marked as initialized after error")
        
        # Verify the error was logged
        self.assertTrue(any(record.levelname == 'ERROR' for record in log_context.records),
                      "Error should be logged")
        error_log = '\n'.join(record.getMessage() for record in log_context.records)
        self.assertIn("Failed to initialize", error_log)
        self.assertIn(error_msg, error_log)
        
        # Verify no component registration occurred
        self.mock_register_component.assert_not_called()
        
        # Verify no children initialization occurred
        self.mock_initialize_children.assert_not_called()
        
        # Verify the error widget is displayable
        self.assertTrue(hasattr(result, '_ipython_display_') or hasattr(result, 'value'),
                      "Error widget should be displayable")
        
        # The initializer should be marked as initialized even after error
        self.assertTrue(initializer._is_initialized, 
                      "Initializer should be marked as initialized after error")
        
        # Verify cleanup was called on the logger bridge
        self.mock_logger_bridge.cleanup.assert_called_once()
    
    def test_initialize_with_parent_module(self):
        """Test initialization with a parent module."""
        # Setup test data
        test_config = {'test': 'config'}
        test_parent_id = 'test_parent'
        test_component_id = 'test_parent_module_component'
        test_parent_module = 'parent_module'
        
        # Create test widgets
        test_container = widgets.VBox()
        test_content_area = widgets.VBox()
        
        # Configure parent component manager mock
        self.mock_pcm.container = test_container
        self.mock_pcm.content_area = test_content_area
        self.mock_pcm_class.return_value = self.mock_pcm
        
        # Configure UI components mock
        self.mock_create_ui_components.return_value = {'container': test_container}
        
        # Create a test handler
        test_handler = TestConfigHandler()
        self.mock_create_handler.return_value = test_handler
        
        # Mock the component registry
        mock_parent_component = {
            'content_area': MagicMock(spec=widgets.VBox),
            'container': MagicMock(spec=widgets.VBox)
        }
        self.mock_component_registry.get_component.return_value = mock_parent_component
        
        # Reset mocks
        self.mock_register_component.reset_mock()
        
        # Create initializer with parent module
        initializer = self.TestInitializer(
            config=test_config,
            parent_id=test_parent_id,
            component_id=test_component_id,
            parent_module=test_parent_module
        )
        
        # Reset mock call counts after initialization
        self.mock_create_handler.reset_mock()
        self.mock_create_ui_components.reset_mock()
        
        # Ensure initial state is correct
        self.assertFalse(initializer._is_initialized, "Should not be initialized yet")
        
        # Call initialize
        result = initializer.initialize()
        
        # Verify the result is a widget with display capabilities
        self.assertIsNotNone(result, "Result should not be None")
        self.assertTrue(hasattr(result, '_ipython_display_'),
                      "Widget should have _ipython_display_ method")
        
        # Verify the container was set up correctly
        self.assertEqual(result, test_container, "Should return the container widget")
        
        # Verify parent module was set correctly
        self.assertEqual(initializer.parent_module, test_parent_module, 
                        "Parent module should be set correctly")
        
        # Verify component registration was called with correct parent ID
        self.mock_register_component.assert_called_once()
        
        # Verify the initializer is marked as initialized
        self.assertTrue(initializer._is_initialized, 
                      "Initializer should be marked as initialized")
        
        # Verify parent component manager was created with correct parameters
        self.mock_pcm_class.assert_called_once()
        
        # Verify component registration was called
        self.mock_register_component.assert_called_once()
        
        # Get the registration call arguments
        call_args = self.mock_register_component.call_args[0]
        
        # Verify the component was registered with the correct parameters
        self.assertEqual(call_args[0], initializer, "Initializer instance not passed correctly")
        self.assertEqual(call_args[1], test_component_id, "Component ID not passed correctly")
        self.assertEqual(call_args[2], test_parent_id, "Parent ID not passed correctly")
        self.assertEqual(call_args[3], test_parent_module, "Parent module not passed correctly")
        
        # Verify the initializer is marked as initialized
        self.assertTrue(initializer._is_initialized, 
                      "Initializer should be marked as initialized")
    
    def test_component_registration(self):
        """Test that components are properly registered with the component registry."""
        # Setup test data
        test_config = {'test': 'config'}
        test_parent_id = 'test_parent'
        test_component_id = 'test_component'
        test_parent_module = 'test_module'
        
        # Create test widgets with proper specs
        test_container = MagicMock(spec=widgets.VBox)
        test_content_area = MagicMock(spec=widgets.VBox)
        test_widget = MagicMock()
        
        # Configure UI components
        ui_components = {
            'container': test_container,
            'content_area': test_content_area,
            'test_widget': test_widget
        }
        
        # Create a test handler
        test_handler = TestConfigHandler()
        
        # Create a test initializer class
        class TestInitializer(ConfigCellInitializer[TestConfigHandler]):
            def create_handler(self):
                return test_handler
                
            def create_ui_components(self, config):
                return ui_components
                
            def setup_handlers(self):
                pass
        
        # Mock the component registry
        mock_registry = MagicMock()
        mock_registry.get_component.return_value = None
        
        # Patch the component registry
        with patch('smartcash.ui.initializers.config_cell_initializer.component_registry', mock_registry):
            # Create initializer
            initializer = TestInitializer(
                config=test_config,
                parent_id=test_parent_id,
                component_id=test_component_id,
                parent_module=test_parent_module
            )
            
            # Set up test data
            initializer.ui_components = ui_components
            initializer.container = test_container
            initializer.content_area = test_content_area
            
            # Call the method under test
            initializer._register_component()
            
            # Verify component registration
            mock_registry.register_component.assert_called_once_with(
                component_id=test_component_id,
                component={
                    'container': test_container,
                    'content_area': test_content_area,
                    'test_widget': test_widget,
                    'handler': test_handler,
                    'initializer': initializer
                },
                parent_id=test_parent_id,
                parent_module=test_parent_module
            )
            
            # Verify parent component lookup was attempted if parent_id is provided
            if test_parent_id:
                mock_registry.get_component.assert_called_once_with(test_parent_id)
            
            # Verify the component was marked as registered
            self.assertTrue(hasattr(initializer, '_component_registered'))
            self.assertTrue(initializer._component_registered)
        
        # Create a test handler
        test_handler = MagicMock()
        
        # Create a test initializer class
        class TestInitializer(ConfigCellInitializer[TestConfigHandler]):
            def create_handler(self):
                print("Creating test handler")
                return test_handler
                
            def create_ui_components(self, config):
                print(f"Creating UI components with config: {config}")
                return ui_components
                
            def setup_handlers(self):
                print("Setting up handlers")
        
        # Create a mock registry with proper specs
        print("Creating mock registry...")
        mock_registry = MagicMock(spec=ComponentRegistry)
        mock_registry.get_component.return_value = None
        
        # Save the original registry and replace with our mock
        print("Patching component_registry...")
        original_registry = config_cell_initializer.component_registry
        config_cell_initializer.component_registry = mock_registry
        
        try:
            # Create initializer with parent_id to test parent-child relationship
            print("Creating TestInitializer instance...")
            initializer = TestInitializer(
                config=test_config,
                parent_id=test_parent_id,
                component_id=test_component_id
            )
            
            # Set up test data
            print("Setting up test data...")
            initializer.ui_components = ui_components
            initializer.container = test_container
            initializer.content_area = test_content_area
            
            # Mock the parent component that would be returned from the registry
            print("Creating mock parent component...")
            mock_parent_component = {
                'content_area': MagicMock(spec=widgets.VBox),
                'container': MagicMock(spec=widgets.VBox)
            }
            mock_parent_component['content_area'].children = ()
            
            # Configure the mock registry to return our mock parent component
            print("Configuring mock registry to return parent component...")
            mock_registry.get_component.return_value = mock_parent_component
            
            # Add _ipython_display_ method to test_container to simulate IPython display
            test_container._ipython_display_ = MagicMock()
            
            # Call the method under test
            print("Calling _register_component()...")
            initializer._register_component()
            print("_register_component() completed")
            
            # Verify register_component was called with expected arguments
            expected_component_data = {
                'container': test_container,
                'content_area': test_content_area,
                'test_widget': test_widget
            }
            
            print("\n=== Verifying register_component calls ===")
            # Print all calls to the mock registry for debugging
            print("All calls to mock registry:")
            for call in mock_registry.method_calls:
                print(f"- {call}")
            
            # Check that register_component was called with the correct arguments
            print("\nChecking register_component call...")
            mock_registry.register_component.assert_called_once_with(
                component_id=full_component_id,
                component=expected_component_data,
                parent_id=test_parent_id
            )
            
            # Verify get_component was called with the parent_id
            print("\n=== Verifying get_component calls ===")
            mock_registry.get_component.assert_called_once_with(test_parent_id)
            
            # Verify the parent's content area was updated
            print("\n=== Verifying parent content area ===")
            if hasattr(mock_parent_component['content_area'], 'children'):
                print(f"Parent content area children: {mock_parent_component['content_area'].children}")
                self.assertEqual(len(mock_parent_component['content_area'].children), 1)
                self.assertIs(mock_parent_component['content_area'].children[0], test_container)
            
            print("\n✓ All assertions passed!")
            
        except Exception as e:
            print(f"\n!!! Test failed with error: {str(e)}")
            # Print traceback for more detailed error information
            import traceback
            traceback.print_exc()
            raise
            
        finally:
            # Restore the original registry
            print("\nRestoring original component_registry...")
            config_cell_initializer.component_registry = original_registry
            print("=== End of test_component_registration ===\n")

class TestLogAccordionIntegration(unittest.TestCase):
    """Test log redirection to log accordion component."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a real log accordion for testing
        from smartcash.ui.components.log_accordion import create_log_accordion
        self.log_components = create_log_accordion(module_name='Test Logs')
        
        # Set up mock parent component with log accordion
        self.mock_parent = MagicMock()
        self.mock_parent.get.return_value = {'ui_components': self.log_components}
        
        # Create a test initializer with mocked dependencies
        self.initializer = TestConfigCellInitializer(
            module_name='test_module',
            config_filename='test_config.yaml',
            component_id='test_component'
        )
        self.initializer.parent_component = self.mock_parent
        
        # Patch the logger bridge
        self.logger_bridge_patcher = patch('smartcash.ui.utils.logger_bridge.UILoggerBridge')
        self.mock_logger_bridge = self.logger_bridge_patcher.start()
        
        # Set up mock logger bridge instance with a real logger
        self.mock_bridge_instance = MagicMock()
        self.mock_bridge_instance.logger = logging.getLogger('test_logger')
        self.mock_bridge_instance.logger.handlers = []  # Clear any existing handlers
        self.mock_logger_bridge.return_value = self.mock_bridge_instance
        
        # Store original stdout/stderr and redirect to StringIO for testing
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        sys.stdout = StringIO()
        sys.stderr = StringIO()
        
    def tearDown(self):
        """Clean up after each test method."""
        self.logger_bridge_patcher.stop()
        if hasattr(self, 'initializer') and hasattr(self.initializer, 'cleanup'):
            self.initializer.cleanup()
        # Restore original stdout/stderr
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        # Clear any test handlers
        logging.getLogger().handlers = []
        
    def test_log_redirection_to_accordion(self):
        """Test that logs are properly redirected to the log accordion."""
        from smartcash.ui.utils.logger_bridge import UILoggerBridge
        
        # Setup test data
        test_message = f"TEST_LOG_{self.initializer.component_id}"
        
        # Create real log accordion components
        from smartcash.ui.components.log_accordion import create_log_accordion
        log_components = create_log_accordion(module_name='Test Logs')
        
        # Configure the mock parent to return our UI components
        self.mock_parent.get.return_value = {'ui_components': log_components}
        
        # Create a real logger bridge instance
        logger_bridge = UILoggerBridge(
            ui_components=log_components,
            logger_name='test_logger_bridge'
        )
        
        # Configure the mock to return our real logger bridge
        self.mock_logger_bridge.return_value = logger_bridge
        
        # Initialize the initializer to set up logging
        self.initializer.initialize()
        
        # Get the logger that was set up
        logger = logging.getLogger('test_logger_bridge')
        
        # Log a test message
        logger.info(test_message)
        
        # Verify the log was buffered (UI not ready yet)
        self.assertEqual(len(logger_bridge._log_buffer), 1)
        self.assertIn(test_message, logger_bridge._log_buffer[0])
        
        # Mark UI as ready to flush the buffer
        logger_bridge.set_ui_ready(True)
        
        # Get the log output widget's content
        log_output = log_components['log_output']
        log_content = log_output.outputs[0]['text']
        
        # Verify the log message appears in the output
        self.assertIn(test_message, log_content)
        
        # Log another message to verify direct logging (not buffered)
        test_message2 = f"DIRECT_LOG_{self.initializer.component_id}"
        logger.info(test_message2)
        
        # Get updated log content
        log_content = log_output.outputs[0]['text']
        
        # Verify the new log message appears in the output
        self.assertIn(test_message2, log_content)
        with patch('smartcash.ui.initializers.config_cell_initializer.UILoggerBridge') as mock_bridge_class:
            mock_bridge_class.return_value = mock_bridge
            
            # Call the method under test
            self.initializer._setup_logging()
            
            # Verify the bridge was created with correct params
            mock_bridge_class.assert_called_once()
            
            # Get the actual logger being used
            logger = self.initializer._logger
            
            # Test 1: Log before UI is ready (should buffer)
            logger.info(test_message)
            
            # Verify the log was buffered
            self.assertTrue(hasattr(mock_bridge, '_log_buffer'))
            self.assertEqual(len(mock_bridge._log_buffer), 1)
            self.assertEqual(mock_bridge._log_buffer[0][1], test_message)
            
            # Test 2: Mark UI as ready and verify logs are flushed
            mock_bridge._ui_ready = True
            
            # Add a new log message (should be processed immediately)
            test_message2 = f"{test_message}_ready"
            logger.info(test_message2)
            
            # Verify the log was processed (not buffered)
            self.assertEqual(len(mock_bridge._log_buffer), 1)  # Still only the first message
            
            # Verify the log output was called with our messages
            # Note: The actual call might be different based on UILoggerBridge implementation
            # We're checking that append_log was called with our messages
            if hasattr(mock_log_output, 'append_log'):
                # Check that append_log was called with our messages
                append_log_calls = [call[0][0] for call in mock_log_output.append_log.call_args_list]
                self.assertIn(test_message, str(append_log_calls))
                self.assertIn(test_message2, str(append_log_calls))
        
    def test_no_logs_before_ui_ready(self):
        """Test that no logs appear before UI is ready."""
        # Log a message before UI is ready
        test_message = "Early log message"
        self.initializer._logger.info(test_message)
        
        # Verify no output was written to stdout/stderr
        self.assertEqual(sys.stdout.getvalue(), '')
        self.assertEqual(sys.stderr.getvalue(), '')
        
        # Verify the log was buffered (if buffering is implemented)
        if hasattr(self.initializer._logger_bridge, 'buffer'):
            self.assertIn(test_message, self.initializer._logger_bridge.buffer)


class TestConfigCellInitializerLogging(unittest.TestCase):
    """Test logging functionality in ConfigCellInitializer."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Patch the component registry and logger bridge
        self.registry_patcher = patch('smartcash.ui.config_cell.components.component_registry.component_registry')
        self.mock_registry = self.registry_patcher.start()
        
        # Create a test initializer with mocks
        self.initializer = TestConfigCellInitializer(
            module_name='test_module',
            config_filename='test_config.yaml',
            component_id='test_component'
        )
        
        # Mock parent component and logger bridge
        self.mock_parent = MagicMock()
        self.mock_parent.container = MagicMock()
        self.mock_parent.content_area = MagicMock()
        self.initializer.parent_component = self.mock_parent
        
        # Mock logger bridge
        self.mock_logger_bridge = MagicMock()
        self.initializer._logger_bridge = self.mock_logger_bridge
        
        # Set up component registry mock
        self.mock_registry.get_component.return_value = None
        
    def tearDown(self):
        """Clean up after each test method."""
        self.registry_patcher.stop()
        if hasattr(self, 'initializer') and hasattr(self.initializer, 'cleanup'):
            self.initializer.cleanup()
    
    @patch('smartcash.ui.initializers.config_cell_initializer.logger')
    @patch('smartcash.ui.initializers.config_cell_initializer.UILoggerBridge')
    def test_setup_logging_initializes_logger_bridge(self, mock_ui_logger_bridge, mock_logger):
        """Test that _setup_logging initializes the logger bridge correctly."""
        # Arrange
        # Create a mock UILoggerBridge instance with a logger that has a name attribute
        mock_bridge_instance = MagicMock()
        mock_bridge_instance.logger = MagicMock()
        mock_bridge_instance.logger.name = 'smartcash.ui.test_component'
        mock_ui_logger_bridge.return_value = mock_bridge_instance
        
        # Create a mock parent component with a get method that returns some UI components
        mock_parent_components = {
            'ui_components': {
                'log_accordion': MagicMock(),
                'log_output': MagicMock()
            }
        }
        mock_parent_component = MagicMock()
        mock_parent_component.get.return_value = mock_parent_components
        
        # Set up the component registry to return our mock parent
        self.mock_registry.get_component.return_value = mock_parent_component
        
        # Set up the initializer with a parent_id to test parent component lookup
        self.initializer.parent_id = 'test_parent_id'
        self.initializer.component_id = 'test_component'
        
        # Reset any existing logger bridge
        self.initializer._logger_bridge = None
        
        # Act - First call with parent component
        self.initializer._setup_logging()
        
        # Assert - Verify UILoggerBridge was instantiated with correct parameters
        mock_ui_logger_bridge.assert_called_once()
        
        # Get the call arguments to UILoggerBridge
        call_args, call_kwargs = mock_ui_logger_bridge.call_args
        
        # Verify the logger name includes the component ID
        self.assertIn('test_component', call_kwargs['logger_name'])
        
        # Verify the logger was set on the initializer
        self.assertIsNotNone(self.initializer._logger)
        self.assertEqual(self.initializer._logger, mock_bridge_instance.logger)
        
        # Verify set_ui_ready was called on the bridge
        mock_bridge_instance.set_ui_ready.assert_called_once_with(True)
        
        # Verify the logger name includes the component ID
        self.assertIn('test_component', self.initializer._logger.name)
        
        # Reset mocks for the next test case
        mock_ui_logger_bridge.reset_mock()
        
        # Test fallback when parent component is not found
        self.mock_registry.get_component.return_value = None
        self.initializer._logger_bridge = None
        
        # Create a new mock bridge instance for the second test case
        mock_bridge_instance2 = MagicMock()
        mock_bridge_instance2.logger = MagicMock()
        mock_bridge_instance2.logger.name = 'smartcash.ui.test_component'
        mock_ui_logger_bridge.return_value = mock_bridge_instance2
        
        # Act - Second call with no parent component
        self.initializer._setup_logging()
        
        # Verify UILoggerBridge was called once more (total 2 times)
        self.assertEqual(mock_ui_logger_bridge.call_count, 1)
        
        # Test exception handling
        with patch.object(self.initializer, '_logger', side_effect=Exception('Test error')):
            self.initializer._logger_bridge = None
            # Should not raise an exception
            self.initializer._setup_logging()
            
        # Verify UILoggerBridge was called again (total 3 times)
        self.assertEqual(mock_ui_logger_bridge.call_count, 2)
        
        # Test exception handling
        with patch.object(self.initializer, '_logger', side_effect=Exception('Test error')):
            self.initializer._logger_bridge = None
            # Should not raise an exception
            self.initializer._setup_logging()
        
    @patch('smartcash.ui.initializers.config_cell_initializer.logger')
    def test_logging_to_ui_components(self, mock_logger):
        """Test that logs are properly directed to UI components."""
        # Skip this test if we can't import the required components
        try:
            from smartcash.ui.utils.logger_bridge import UILoggerBridge
            from smartcash.ui.config_cell.components.ui_components import create_log_accordion
        except ImportError:
            self.skipTest("Required UI components not available")
            
        # Create a real log_accordion for testing
        log_components = create_log_accordion(module_name='test_module')
        self.initializer.ui_components = log_components
        
        # Initialize the logger bridge with our test components
        with patch('smartcash.ui.initializers.config_cell_initializer.UILoggerBridge') as mock_bridge_cls:
            mock_bridge = MagicMock()
            mock_bridge_cls.return_value = mock_bridge

            # Call _setup_logging which should use our mock bridge
            self.initializer._setup_logging()

            # Verify the bridge was created with correct components
            mock_bridge_cls.assert_called_once()

    @patch('smartcash.ui.initializers.config_cell_initializer.sys')
    @patch('smartcash.ui.initializers.config_cell_initializer.logging')
    @patch('smartcash.ui.initializers.config_cell_initializer.component_registry')
    @patch('smartcash.ui.initializers.config_cell_initializer.super')
    def test_cleanup_releases_resources(self, mock_super, mock_registry, mock_logging, mock_sys):
        """Test that cleanup releases all resources."""
        # Setup test data
        test_component_id = 'test_cleanup_component'
        test_parent_id = 'test_parent_component'

        # Create test widgets
        test_container = MagicMock(spec=widgets.VBox)
        test_content_area = MagicMock(spec=widgets.VBox)

        # Configure parent component manager mock
        self.mock_pcm.container = test_container
        self.mock_pcm.content_area = test_content_area
        self.mock_pcm_class.return_value = self.mock_pcm

        # Configure UI components mock
        self.mock_create_ui_components.return_value = {'container': test_container}

        # Create a test handler with a cleanup method
        class TestHandlerWithCleanup(TestConfigHandler):
            def __init__(self):
                super().__init__()
                self.cleanup_called = False

            def cleanup(self):
                self.cleanup_called = True

        test_handler = TestHandlerWithCleanup()
        self.mock_create_handler.return_value = test_handler

        # Create initializer with test data
        initializer = self.TestInitializer(
            component_id=test_component_id,
            parent_id=test_parent_id
        )
        
        # Set up logger bridge
        mock_logger_bridge = MagicMock()
        initializer._logger_bridge = mock_logger_bridge
        initializer._logger = MagicMock()
        
        # Set up component registration state
        initializer._component_registered = True
        initializer._is_initialized = True
        initializer._component_id = test_component_id
        initializer.ui_components = {'container': test_container}
        initializer.handler = test_handler

        # Reset mocks before testing cleanup
        mock_registry.unregister_component.reset_mock()
        mock_logger_bridge.cleanup.reset_mock()
        
        # Mock the parent component cleanup if it exists
        mock_parent_component = MagicMock()
        mock_registry.get_component.return_value = mock_parent_component

        # Call cleanup
        initializer.cleanup()

        # Verify component was unregistered
        mock_registry.unregister_component.assert_called_once_with(test_component_id)
        
        # Verify logger bridge was cleaned up
        mock_logger_bridge.cleanup.assert_called_once()
        
        # Verify handler cleanup was called
        self.assertTrue(test_handler.cleanup_called, "Handler cleanup should be called")
        
        # Verify component registry was cleaned up
        mock_registry.unregister_component.assert_called_once_with(test_component_id)
        
        # Verify state was reset
        self.assertFalse(initializer._is_initialized, 
                        "Initializer should be marked as not initialized after cleanup")
        self.assertFalse(hasattr(initializer, '_component_registered'),
                        "Component registration flag should be removed")
        
        # Verify component ID was cleared
        self.assertIsNone(getattr(initializer, '_component_id', None), 
                         "Component ID should be cleared after cleanup")
        
        # Verify UI components were cleared
        self.assertIsNone(getattr(initializer, 'ui_components', None),
                         "UI components should be cleared")
        
        # Verify parent component reference was cleared
        self.assertIsNone(getattr(initializer, 'parent_component', None),
                         "Parent component reference should be cleared")
        
        # Verify handler reference was cleared
        self.assertIsNone(getattr(initializer, 'handler', None),
                         "Handler reference should be cleared")
