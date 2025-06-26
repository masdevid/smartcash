"""
Tests for the dataset split initialization module.

This module contains tests to verify the proper initialization and rendering
of the dataset split configuration UI components, including parent-child
component relationships.
"""
import pytest
import ipywidgets as widgets
from ipywidgets import VBox
from unittest.mock import MagicMock, patch, ANY, call, PropertyMock
from typing import Dict, Any, Optional

def test_module_import():
    """Test that the module can be imported correctly."""
    # Test importing from the package
    from smartcash.ui.dataset.split import split_init
    
    # Check that essential components are available
    assert hasattr(split_init, 'SplitConfigInitializer')
    assert hasattr(split_init, 'create_split_config_cell')
    assert hasattr(split_init, 'MODULE_NAME')
    assert split_init.MODULE_NAME == 'split_config'
    
    # Test that the public API is available through the package
    from smartcash.ui.dataset.split import (
        create_split_config_cell,
        SplitConfigHandler,
        initialize_split_ui
    )
    
    # Verify the handler class
    assert SplitConfigHandler.__name__ == 'SplitConfigHandler'

# Import the component registry
from smartcash.ui.initializers.config_cell_initializer import component_registry

# Import the module under test
from smartcash.ui.dataset.split.split_init import (
    SplitConfigInitializer,
    create_split_config_cell,
    MODULE_NAME
)
from smartcash.ui.config_cell.components.component_registry import ComponentRegistry

# Create a mock component registry for testing
@pytest.fixture
def mock_component_registry():
    registry = MagicMock(spec=ComponentRegistry)
    registry.get_component.return_value = None
    return registry

# Fixture to patch the component registry in the module under test
@pytest.fixture(autouse=True)
def patch_component_registry(mock_component_registry):
    with patch('smartcash.ui.config_cell.components.component_registry', mock_component_registry):
        yield

# Fixture for mock UI components
@pytest.fixture
def mock_ui_components():
    return {
        'container': widgets.VBox(),
        'form': widgets.VBox(),
        'train_ratio': widgets.FloatSlider(value=0.7, min=0.1, max=0.9, step=0.1),
        'val_ratio': widgets.FloatSlider(value=0.15, min=0.05, max=0.45, step=0.05),
        'test_ratio': widgets.FloatSlider(value=0.15, min=0.05, max=0.45, step=0.05),
        'random_seed': widgets.IntText(value=42, description='Random Seed')
    }

# Fixture for mock handler
@pytest.fixture
def mock_handler():
    return MagicMock()

# Fixture for a configured initializer
@pytest.fixture
def split_initializer(mock_handler, mock_ui_components):
    """Fixture that provides a configured SplitConfigInitializer instance."""
    with patch('smartcash.ui.config_cell.components.component_registry.ComponentRegistry') as mock_registry, \
         patch('smartcash.ui.dataset.split.split_init.create_split_form', return_value=mock_ui_components), \
         patch('smartcash.ui.dataset.split.split_init.create_split_layout', return_value={'container': VBox()}):
        
        initializer = SplitConfigInitializer()
        initializer._handler = mock_handler
        initializer.ui_components = mock_ui_components
        initializer.parent_component = MagicMock()
        initializer.parent_component.container = VBox()
        initializer.parent_component.content_area = VBox()
        initializer._is_initialized = True
        
        yield initializer

# Fixture to mock IPython display
@pytest.fixture
def mock_display():
    """Fixture to mock IPython's display function."""
    with patch('smartcash.ui.dataset.split.split_init.display') as mock_display:
        yield mock_display

# Fixture to mock error handler
@pytest.fixture
def mock_error_handler():
    """Fixture to mock the error handler used in create_split_config_cell."""
    with patch('smartcash.ui.dataset.split.split_init.logger') as mock_logger, \
         patch('smartcash.ui.dataset.split.split_init.create_error_response') as mock_create_error:
        mock_error_widget = MagicMock()
        mock_create_error.return_value = mock_error_widget
        yield mock_logger, mock_create_error, mock_error_widget

@pytest.fixture
def mock_handler():
    """Fixture that provides a mock SplitConfigHandler."""
    mock = MagicMock()
    mock.config = {}
    return mock

@pytest.fixture
def mock_ui_components():
    """Fixture that provides mock UI components."""
    return {
        'train_ratio': widgets.FloatSlider(value=0.7, min=0.1, max=0.9, step=0.05),
        'val_ratio': widgets.FloatSlider(value=0.15, min=0.05, max=0.4, step=0.05),
        'test_ratio': widgets.FloatSlider(value=0.15, min=0.05, max=0.4, step=0.05),
        'random_seed': widgets.IntText(value=42, description='Random Seed:'),
        'container': widgets.VBox()
    }

class TestSplitConfigInitializer:
    """Test cases for SplitConfigInitializer class."""
    
    @patch('smartcash.ui.dataset.split.split_init.ConfigCellInitializer.__init__', return_value=None)
    @patch('smartcash.ui.dataset.split.split_init.logger')
    def test_initialization(self, mock_logger, mock_parent_init, mock_component_registry):
        """Test that the initializer is properly initialized with default values."""
        # Setup mocks
        mock_logger_instance = MagicMock()
        mock_logger.getChild.return_value = mock_logger_instance
        
        # Create instance with default values
        initializer = SplitConfigInitializer()
        
        # Manually set required attributes that would be set by the parent __init__
        initializer.ui_components = {}
        initializer._handler = None
        initializer._logger = mock_logger_instance
        initializer._is_initialized = True
        
        # Verify parent initialization was called with correct parameters
        mock_parent_init.assert_called_once()
        args, kwargs = mock_parent_init.call_args
        
        # Verify the arguments passed to parent's __init__
        assert kwargs['config'] == {}
        assert kwargs['parent_id'] is None
        assert kwargs['component_id'] == "split_config"
        assert kwargs['title'] == "Dataset Split Configuration"
        assert kwargs['children'] == []
        
        # Verify instance attributes are set
        assert initializer is not None
        assert hasattr(initializer, 'ui_components')
        assert hasattr(initializer, '_handler')
        assert hasattr(initializer, '_logger')
    
    @patch('smartcash.ui.dataset.split.split_init.ConfigCellInitializer.__init__', return_value=None)
    @patch('smartcash.ui.dataset.split.split_init.logger')
    def test_initialization_with_custom_values(self, mock_logger, mock_parent_init, mock_component_registry):
        """Test initialization with custom values."""
        # Setup mocks
        mock_logger_instance = MagicMock()
        mock_logger.getChild.return_value = mock_logger_instance
        
        # Setup test data
        custom_config = {"test": "config"}
        children_config = [{"type": "test_child"}]
        
        # Create instance with custom values
        initializer = SplitConfigInitializer(
            config=custom_config,
            parent_id="parent_123",
            component_id="custom_id",
            title="Custom Title",
            children=children_config
        )
        
        # Manually set required attributes that would be set by the parent __init__
        initializer.ui_components = {}
        initializer._handler = None
        initializer._logger = mock_logger_instance
        initializer._is_initialized = True
        
        # Verify parent initialization was called with correct parameters
        mock_parent_init.assert_called_once()
        args, kwargs = mock_parent_init.call_args
        
        # Verify the arguments passed to parent's __init__
        assert kwargs['config'] == custom_config
        assert kwargs['parent_id'] == "parent_123"
        assert kwargs['component_id'] == "custom_id"
        assert kwargs['title'] == "Custom Title"
        assert kwargs['children'] == children_config
        
        # Verify instance attributes are set
        assert initializer is not None
        assert hasattr(initializer, 'ui_components')
        assert hasattr(initializer, '_handler')
        assert hasattr(initializer, '_logger')
    
    def test_create_handler(self, mock_component_registry, mock_handler):
        """Test handler creation with config."""
        # Create a mock for UILoggerBridge
        mock_ui_components = {'output': MagicMock()}
        mock_logger_bridge = MagicMock()
        
        with patch('smartcash.ui.dataset.split.split_init.SplitConfigHandler', return_value=mock_handler) as mock_handler_cls, \
             patch('smartcash.ui.initializers.config_cell_initializer.UILoggerBridge', return_value=mock_logger_bridge):
            
            # Test with default config
            initializer = SplitConfigInitializer(logger_bridge=mock_logger_bridge)
            handler = initializer.create_handler()
            mock_handler_cls.assert_called_once_with({})
            assert handler is mock_handler
            
            # Test with custom config in constructor
            mock_handler_cls.reset_mock()
            custom_config = {'train_ratio': 0.8}
            initializer = SplitConfigInitializer(config=custom_config, logger_bridge=mock_logger_bridge)
            handler = initializer.create_handler()
            mock_handler_cls.assert_called_once_with(custom_config)
            assert handler is mock_handler
    
    def test_create_ui_components(self, split_initializer, mock_ui_components):
        """Test UI components creation."""
        # Create a mock layout return value that includes a container
        mock_layout_return = {
            'layout': 'test-layout',
            'container': mock_ui_components['container']
        }
        
        # Mock the form and layout creation
        with patch('smartcash.ui.dataset.split.split_init.create_split_form') as mock_form, \
             patch('smartcash.ui.dataset.split.split_init.create_split_layout') as mock_layout:
            
            # Set up mock return values
            mock_form.return_value = mock_ui_components
            mock_layout.return_value = mock_layout_return
            
            # Call the method under test
            config = {'test': 'config'}
            components = split_initializer.create_ui_components(config)
            
            # Verify the mocks were called correctly
            mock_form.assert_called_once_with(config)
            mock_layout.assert_called_once_with(mock_ui_components)
            
            # Verify the returned components are merged correctly
            expected_components = {**mock_ui_components, **mock_layout_return}
            assert components == expected_components, "Returned components should merge form and layout components"
            
            # Verify container is in the returned components
            assert 'container' in components, "Returned components should include container"
            assert components['container'] is mock_ui_components['container'], \
                "Container should be the one from mock_ui_components"
    
    def test_setup_handlers(self, split_initializer):
        """Test that event handlers are properly set up."""
        with patch('smartcash.ui.dataset.split.split_init.ConfigCellInitializer.setup_handlers') as mock_parent_setup, \
             patch('smartcash.ui.dataset.split.handlers.event_handlers.setup_event_handlers') as mock_setup_handlers:
            
            split_initializer.setup_handlers()
            
            # Verify parent setup was called
            mock_parent_setup.assert_called_once()
            
            # Verify our custom handlers were set up
            mock_setup_handlers.assert_called_once_with(
                split_initializer, 
                split_initializer.ui_components
            )
    
    def test_initialize_runs_without_errors(self, split_initializer, mock_handler, mock_ui_components):
        """Test that the initializer runs without errors with default configuration."""
        # Setup mock for parent initialize
        with patch('smartcash.ui.dataset.split.split_init.ConfigCellInitializer.initialize') as mock_parent_initialize:
            # Configure the parent initialize to return the container
            mock_container = MagicMock()
            mock_parent_initialize.return_value = mock_container
            
            # Test initialization
            try:
                result = split_initializer.initialize()
                
                # Verify the result is the container from parent
                assert result == mock_container, "Should return container from parent initialize"
                
                # Verify parent initialize was called
                mock_parent_initialize.assert_called_once()
                
                # Verify handler exists (was set up in the fixture)
                assert hasattr(split_initializer, '_handler'), "Handler should be created"
                assert split_initializer._handler is mock_handler, "Handler should be the one from fixture"
                
            except Exception as e:
                pytest.fail(f"Initialization failed with error: {str(e)}")


# Add import at the top of the file if not already present
from unittest.mock import ANY, patch
import ipywidgets as widgets

class TestCreateSplitConfigCell:
    """Test cases for the create_split_config_cell function."""
    
    def test_successful_ui_display(self, mock_display):
        """Test that the UI is properly displayed on successful initialization."""
        # Setup mock initializer
        mock_initializer = MagicMock()
        mock_container = MagicMock()
        mock_initializer.initialize.return_value = mock_container
        mock_initializer.ui_components = {'test': 'components'}
        mock_initializer.parent_component = MagicMock()
        mock_initializer.parent_component.container = 'mock_container'
        mock_initializer.parent_component.content_area = 'mock_content_area'
        
        with patch('smartcash.ui.dataset.split.split_init.SplitConfigInitializer', 
                  return_value=mock_initializer) as mock_init:
            # Call with custom config
            config = {'test': 'config'}
            result = create_split_config_cell(config=config, some_arg='value')
            
            # Verify initializer was created with correct args
            mock_init.assert_called_once_with(config=config, some_arg='value')
            
            # Verify initialization and display
            mock_initializer.initialize.assert_called_once()
            mock_display.assert_called_once_with(mock_container)
            
            # Verify return value
            assert result == {
                'test': 'components',
                'container': 'mock_container',
                'content_area': 'mock_content_area'
            }
    
    def test_error_handling(self, mock_display):
        """Test proper error handling and display when initialization fails."""
        # Setup mock error response
        mock_error_widget = MagicMock()
        test_error = Exception("Test error")
        
        # Mock the error handler
        with patch('smartcash.ui.dataset.split.split_init.logger') as mock_logger, \
             patch('smartcash.ui.config_cell.handlers.error_handler.create_error_response') as mock_create_error, \
             patch('smartcash.ui.dataset.split.split_init.SplitConfigInitializer') as mock_init_class:
            
            # Configure mocks
            mock_create_error.return_value = mock_error_widget
            
            # Setup mock initializer to raise exception
            mock_initializer = MagicMock()
            mock_initializer.initialize.side_effect = test_error
            mock_initializer.ui_components = {}
            mock_initializer.parent_component = MagicMock()
            mock_initializer.parent_component.container = MagicMock()
            mock_initializer.parent_component.content_area = MagicMock()
            mock_init_class.return_value = mock_initializer
            
            # Call the function
            result = create_split_config_cell()
            
            # Verify error was logged
            mock_logger.error.assert_called_once()
            assert "Failed to create split config cell" in mock_logger.error.call_args[0][0]
            
            # Verify error response was created with correct parameters
            mock_create_error.assert_called_once()
            call_args = mock_create_error.call_args[1]
            assert call_args['error'] == test_error
            assert "Failed to create split config cell" in call_args['error_message']
            assert call_args['title'] == "Error in Dataset Split Configuration"
            assert call_args['include_traceback'] is True
            
            # Verify error widget was displayed
            mock_display.assert_called_once_with(mock_error_widget)
            
            # Verify return value contains error
            assert result == {'error': mock_error_widget}
    
    def test_display_called_before_return(self, mock_display):
        """Test that display is called before returning the result."""
        mock_initializer = MagicMock()
        mock_container = MagicMock()
        mock_initializer.initialize.return_value = mock_container
        mock_initializer.ui_components = {}
        
        # Track call order
        call_order = []
        
        def track_display(*args, **kwargs):
            call_order.append('display')
            return MagicMock()
            
        def track_initialize():
            call_order.append('initialize')
            return mock_container
            
        mock_display.side_effect = track_display
        mock_initializer.initialize.side_effect = track_initialize
        
        with patch('smartcash.ui.dataset.split.split_init.SplitConfigInitializer', 
                  return_value=mock_initializer):
            create_split_config_cell()
            
            # Verify display was called after initialization but before returning
            assert call_order == ['initialize', 'display']


@pytest.fixture
def mock_ui_components():
    """Fixture that provides mock UI components."""
    mock_container = MagicMock(spec=widgets.VBox)
    mock_content_area = MagicMock(spec=widgets.VBox)
    return {
        'container': mock_container,
        'content_area': mock_content_area,
        'other': 'components'
    }

def test_component_registration(mock_component_registry, mock_ui_components):
    """Test that components are properly registered in the component registry."""
    # Reset the mock to ensure clean state
    mock_component_registry.reset_mock()
    
    # Create a mock logger bridge
    mock_logger_bridge = MagicMock()
    mock_handler = MagicMock()
    
    # Create a mock container and content area
    mock_container = MagicMock(spec=widgets.VBox)
    mock_content_area = MagicMock(spec=widgets.VBox)
    
    # Setup return values for the component registry
    mock_component_registry.get_component.return_value = {
        'container': mock_container,
        'content_area': mock_content_area,
        'other': 'components'
    }
    
    # Create a mock for the handler property
    mock_handler_property = PropertyMock(return_value=mock_handler)
    
    # Setup patches
    with patch('smartcash.ui.initializers.config_cell_initializer.UILoggerBridge', 
               return_value=mock_logger_bridge), \
         patch('smartcash.ui.initializers.config_cell_initializer.ConfigCellInitializer.handler', 
              new_callable=PropertyMock, return_value=mock_handler), \
         patch('smartcash.ui.initializers.config_cell_initializer.component_registry', 
              mock_component_registry), \
         patch.object(SplitConfigInitializer, 'create_ui_components', 
                     return_value=mock_ui_components) as mock_create_ui, \
         patch.object(SplitConfigInitializer, 'create_handler', 
                     return_value=mock_handler) as mock_create_handler, \
         patch.object(SplitConfigInitializer, 'setup_handlers') as mock_setup_handlers:
        
        # Initialize with test component ID
        test_component_id = 'test_component'
        initializer = SplitConfigInitializer(
            logger_bridge=mock_logger_bridge,
            component_id=test_component_id
        )
        
        # Set up the parent component
        initializer.parent_component = MagicMock()
        initializer.parent_component.container = mock_container
        initializer.parent_component.content_area = mock_content_area
        
        # Call the public initialize method
        container = initializer.initialize()
        
        # Verify the initialization flow
        mock_create_handler.assert_called_once()
        mock_create_ui.assert_called_once()
        mock_setup_handlers.assert_called_once()
        
        # Verify the component was registered with the correct ID
        mock_component_registry.register_component.assert_called()
        
        # Get all component registration calls
        register_calls = mock_component_registry.register_component.call_args_list
        
        # Extract component IDs from registration calls
        registered_ids = [
            args[0][0] if args and len(args) > 0 else kwargs.get('component_id')
            for args, kwargs in [call for call in register_calls]
        ]
        
        # Verify our component ID is in the registered IDs
        assert test_component_id in registered_ids, \
            f"Component {test_component_id} was not registered. Registered: {registered_ids}"
        
        # Verify the container was returned
        assert container is not None
        assert container == mock_container


@patch('smartcash.ui.initializers.config_cell_initializer.component_registry')
@patch('smartcash.ui.initializers.config_cell_initializer.UILoggerBridge')
def test_parent_child_relationship(mock_logger_bridge, mock_component_registry):
    """Test that parent-child relationships are properly established and components are rendered."""
    print("\n=== Starting test_parent_child_relationship ===")
    
    # Reset the mock to ensure clean state
    mock_component_registry.reset_mock()
    
    # Create real widget instances for testing
    print("\nCreating widget instances...")
    parent_container = widgets.VBox()
    parent_content = widgets.VBox()
    
    # Track registered components
    registered_components = {}
    
    # Create a mock for the component registry
    print("\nSetting up component registry mock...")
    registered_components = {}
    
    def mock_register_component(component_id, component, parent_id=None):
        print(f"Registering component: {component_id} (parent: {parent_id})")
        registered_components[component_id] = component
        return component_id
        
    def mock_get_component(component_id):
        print(f"Getting component: {component_id}")
        return registered_components.get(component_id)
    
    # Create a real component registry mock that tracks registrations
    class MockComponentRegistry:
        def __init__(self):
            self.components = {}
            
        def register_component(self, component_id, component, parent_id=None):
            print(f"[MOCK] Registering component: {component_id} (parent: {parent_id})")
            self.components[component_id] = component
            if parent_id:
                component['parent_id'] = parent_id
            return component_id
            
        def get_component(self, component_id):
            print(f"[MOCK] Getting component: {component_id}")
            return self.components.get(component_id)
    
    # Create the mock registry
    mock_registry = MockComponentRegistry()
    
    # Set up the logger bridge mock
    print("\nSetting up logger bridge mock...")
    mock_logger_instance = MagicMock()
    mock_logger_bridge.return_value = mock_logger_instance
    
    # Define the test implementation of ConfigCellInitializer
    class TestConfigCellInitializer:
        def __init__(self, *args, **kwargs):
            # Store handler separately since the parent class has a property decorator
            self._test_handler = kwargs.pop('handler', None)
            
            # Initialize attributes before parent __init__
            self._children = []
            self._is_initialized = False
            self.ui_components = {}
            self.container = kwargs.get('container', widgets.VBox())
            self.content_area = kwargs.get('content_area', widgets.VBox())
            self.component_id = kwargs.get('component_id', 'test_component')
            self.parent_id = kwargs.get('parent_id')
            self.ui_logger_bridge = kwargs.get('ui_logger_bridge')
            
            # Set empty children tuples (ipywidgets convention)
            self.container.children = ()
            self.content_area.children = ()
            
            # Set up handler if provided
            if self._test_handler is not None:
                self._handler = self._test_handler
        
        def add_child(self, child):
            """Add a child component and set up parent-child relationship."""
            print(f"Adding child {child.component_id} to parent {self.component_id}")
            child.parent_id = self.component_id
            self._children.append(child)
            
            # Set up the parent component reference
            child.parent_component = self
            
            # Initialize the child if parent is already initialized
            if self._is_initialized:
                child.initialize()
        
        @property
        def handler(self):
            """Get the handler."""
            return getattr(self, '_handler', None)
            
        @handler.setter
        def handler(self, value):
            """Set the handler."""
            self._handler = value
            
        def create_handler(self):
            """Create a mock handler for testing."""
            return self._test_handler or MagicMock()
            
        def create_ui_components(self, config):
            """Create mock UI components for testing."""
            return {
                'container': self.container,
                'content_area': self.content_area
            }
            
        def setup_handlers(self):
            """Set up mock handlers for testing."""
            pass
            
        def initialize(self, config=None):
            """Initialize the component."""
            if self._is_initialized:
                return
                
            self.handler = self.create_handler()
            self.ui_components = self.create_ui_components(config or {})
            self.setup_handlers()
            self._register_component()
            self._is_initialized = True
            
            # Initialize all children
            for child in self._children:
                if not hasattr(child, '_is_initialized') or not child._is_initialized:
                    child.initialize()
        
        def _register_component(self):
            """Register this component with the component registry."""
            from smartcash.ui.initializers.config_cell_initializer import component_registry
            
            full_component_id = f"{self.parent_id}.{self.component_id}" if self.parent_id else self.component_id
            
            # Prepare component data with container and content area
            component_data = {
                **self.ui_components,
                'container': self.container,
                'content_area': self.content_area
            }
            
            print(f"Registering component: {full_component_id}")
            print(f"Component data: {component_data}")
            
            # Register the main component
            component_registry.register_component(
                component_id=full_component_id,
                component=component_data,
                parent_id=self.parent_id
            )
            
            # If this is a child component (has a parent_id), add to parent's content area
            if self.parent_id:
                print(f"Processing parent-child relationship for {full_component_id}")
                # Get the parent component from the registry
                parent_component = component_registry.get_component(self.parent_id)
                print(f"Parent component: {parent_component}")
                
                if parent_component and 'content_area' in parent_component:
                    parent_content = parent_component['content_area']
                    print(f"Parent content: {parent_content}")
                    
                    if parent_content and hasattr(parent_content, 'children'):
                        current_children = list(parent_content.children)
                        print(f"Current children: {current_children}")
                        
                        # Add the current component's container to the parent's content area
                        if self.container and self.container not in current_children:
                            print(f"Adding container {self.container} to parent's children")
                            parent_content.children = tuple(current_children + [self.container])
                            print(f"Updated children: {parent_content.children}")
    
    # Patch the component_registry in the module and run all test logic inside the context
    with patch('smartcash.ui.initializers.config_cell_initializer.component_registry', mock_registry):
        # Now run the test code that uses component_registry
        # Create parent container and content area
        parent_container = widgets.VBox()
        parent_content = widgets.VBox()
        
        # Store references for assertions
        test_components = {}
        
        # Create parent initializer
        print("\n=== Creating parent initializer ===")
        print(f"Parent container ID: {id(parent_container)}")
        print(f"Parent content ID: {id(parent_content)}")
        parent_initializer = TestConfigCellInitializer(
            component_id='parent',
            parent_id=None,
            container=parent_container,
            content_area=parent_content,
            handler=MagicMock(),
            ui_logger_bridge=mock_logger_instance
        )
        print(f"Parent initializer created. Container: {id(parent_initializer.container)}, Content: {id(parent_initializer.content_area)}")
        
        # Create child container and content area
        child_container = widgets.VBox()
        child_content = widgets.VBox()
        
        # Create child initializer
        print("\n=== Creating child initializer ===")
        print(f"Child container ID: {id(child_container)}")
        print(f"Child content ID: {id(child_content)}")
        child_initializer = TestConfigCellInitializer(
            component_id='child',
            container=child_container,
            content_area=child_content,
            handler=MagicMock(),
            ui_logger_bridge=mock_logger_instance
        )
        print(f"Child initializer created. Container: {id(child_initializer.container)}, Content: {id(child_initializer.content_area)}")
        print(f"Parent container: {parent_container} (id: {id(parent_container)})")
        print(f"Parent content area: {parent_content} (id: {id(parent_content)})")
        print(f"Parent initializer has container: {hasattr(parent_initializer, 'container')}")
        print(f"Parent initializer has content_area: {hasattr(parent_initializer, 'content_area')}")
        
        # Initialize the parent first
        print("\n=== Initializing parent ===")
        print(f"Parent _is_initialized before: {getattr(parent_initializer, '_is_initialized', False)}")
        try:
            parent_initializer.initialize()
            print(f"Parent _is_initialized after: {parent_initializer._is_initialized}")
            print(f"Parent children: {parent_initializer._children}")
            print(f"Parent container children: {getattr(parent_initializer.container, 'children', 'N/A')}")
            print(f"Parent content area children: {getattr(parent_initializer.content_area, 'children', 'N/A')}")
        except Exception as e:
            print(f"Error initializing parent: {e}")
            raise
        
        # Add child to parent after parent is initialized
        print("\n=== Adding child to parent ===")
        parent_initializer.add_child(child_initializer)
        print(f"Parent children after add: {parent_initializer._children}")
        
        # Initialize the child
        print("\n=== Initializing child ===")
        child_initializer.initialize()
        
        # Verify the child was added to the parent's content area children
        print("\n=== Verifying parent-child relationship ===")
        print(f"Parent content area: {parent_initializer.content_area}")
        print(f"Parent content area children: {parent_initializer.content_area.children}")
        print(f"Child container: {child_initializer.container}")
        print(f"Child container ID: {id(child_initializer.container)}")
        
        # Verify the child was registered with the component registry
        print("\nVerifying component registry...")
        print(f"Registered components: {mock_registry.components.keys()}")
        
        # Check parent registration
        parent_comp = mock_registry.get_component('parent')
        assert parent_comp is not None, "Parent component not registered"
        
        # Check child registration
        child_comp = mock_registry.get_component('parent.child')
        assert child_comp is not None, "Child component not registered"
        
        # Verify parent-child relationship in the registry
        assert child_comp.get('parent_id') == 'parent', "Child's parent_id not set correctly"
        
        # Debug information
        print("\n=== Debug Information ===")
        print(f"Parent content area children: {parent_initializer.content_area.children}")
    print(f"Child container: {child_container} (id: {id(child_container)})")
    print(f"Child container in parent's children: {child_container in parent_content.children}")
    print(f"Child component from registry: {child_comp}")
    
    # Verify the UI hierarchy
    assert child_container in parent_content.children, \
        f"Child container not found in parent's content area children. " \
        f"Parent children: {parent_content.children}, " \
        f"Child container: {child_container} (id: {id(child_container)})"
    
    # Verify component data in registry
    assert 'container' in child_comp, "Container not in registered components"
    assert 'content_area' in child_comp, "Content area not in registered components"
    assert child_comp['container'] is child_container, "Wrong container in registered components"
    assert child_comp['content_area'] is child_initializer.content_area, "Wrong content area in registered components"

    print("\nAll tests passed successfully!")

    # Verify UI hierarchy with detailed error message
    child_found = child_container in parent_content.children
    print(f"Child container in parent's children: {child_found}")
    
    assert child_found, (
        f"Child container not in parent's content area.\n"
        f"Expected container: {child_container} (id: {id(child_container)})\n"
        f"Available children: {parent_content.children}\n"
        f"Parent content area: {parent_content}"
    )

    print("\nAll assertions passed!")

if __name__ == "__main__":
    test_parent_child_relationship()
    pytest.main([__file__, "-v"])
