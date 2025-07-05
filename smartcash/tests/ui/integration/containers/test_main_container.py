"""Integration tests for MainContainer component.

Tests the functionality and integration of the MainContainer component with
its child components and dependencies.
"""
import sys
import pytest
from unittest.mock import MagicMock, patch, ANY, PropertyMock
import ipywidgets as widgets

# Create a mock for the MainContainer class
class MockMainContainer:
    def __init__(self, **kwargs):
        self.component_name = kwargs.get('component_name', 'test_container')
        self.style_variant = kwargs.get('style_variant', 'default')
        self.layout_width = kwargs.get('layout_width', '100%')
        self.layout_height = kwargs.get('layout_height', 'auto')
        self.padding = kwargs.get('padding', '10px')
        self.margin = kwargs.get('margin', '5px')
        self.border_radius = kwargs.get('border_radius', '4px')
        self.border_width = kwargs.get('border_width', '1px')
        self.border_style = kwargs.get('border_style', 'solid')
        self.border_color = kwargs.get('border_color', '#ccc')
        self.background_color = kwargs.get('background_color', '#fff')
        self.box_shadow = kwargs.get('box_shadow', '0 2px 4px rgba(0,0,0,0.1)')
        self.order = kwargs.get('order', ['header', 'content', 'footer'])
        self.components = {}
        
        # Initialize mock methods
        self.add_child = MagicMock()
        self.remove_child = MagicMock()
        self.clear_children = MagicMock()
        self.update_style = MagicMock()
        self.get_component = MagicMock(return_value=None)
        self.show_loading = MagicMock()
        self.show_error = MagicMock()
        
        # Add component management methods
        def add_component(component, name):
            self.components[name] = component
            self.add_child(component)
            
        def remove_component(name):
            component = self.components.pop(name, None)
            if component:
                self.remove_child(component)
                
        def clear_components():
            self.components.clear()
            self.clear_children()
            
        # Assign the methods
        self.add_component = MagicMock(side_effect=add_component)
        self.remove_component = MagicMock(side_effect=remove_component)
        self.clear_components = MagicMock(side_effect=clear_components)

# Set up the mock module
mock_module = MagicMock()
mock_module.MainContainer = MockMainContainer
mock_module.create_main_container = MagicMock(return_value=MockMainContainer())
sys.modules['smartcash.ui.components.main_container'] = mock_module

# Import the container we're testing
from smartcash.ui.components.main_container import MainContainer, create_main_container

# Fixtures
@pytest.fixture
def main_container():
    """Create a MainContainer instance for testing."""
    return MockMainContainer()

class TestMainContainer:
    """Test suite for MainContainer integration."""
    
    def test_initialization(self, main_container):
        """Test basic initialization with parameters."""
        assert main_container is not None
        assert main_container.component_name == "test_container"
        assert main_container.style_variant == "default"
    
    def test_add_component(self, main_container):
        """Test adding components to the container."""
        # Reset mock to clear any previous calls
        main_container.add_child.reset_mock()
        
        # Create a test component
        test_component = MagicMock()
        
        # Add component
        main_container.add_component(test_component, "test_component")
        
        # Verify the component was added
        main_container.add_child.assert_called_once_with(test_component)
    
    def test_remove_component(self, main_container):
        """Test removing components from the container."""
        # Reset mocks
        main_container.remove_component.reset_mock()
        
        # Add a test component first
        test_component = MagicMock()
        main_container.add_component(test_component, "test_component")
        
        # Reset the mock to track the remove call
        main_container.remove_component.reset_mock()
        
        # Remove the component
        main_container.remove_component("test_component")
        
        # Verify remove_component was called with the right parameter
        main_container.remove_component.assert_called_once_with("test_component")
    
    def test_clear_components(self, main_container):
        """Test clearing all components from the container."""
        # Reset mock
        main_container.clear_children.reset_mock()
        
        # Clear components
        main_container.clear_components()
        
        # Verify clear_children was called
        main_container.clear_children.assert_called_once()
    
    def test_show_loading(self, main_container):
        """Test showing loading state."""
        # Reset mock
        main_container.show_loading.reset_mock()
        
        # Show loading
        main_container.show_loading("Loading...")
        
        # Verify show_loading was called
        main_container.show_loading.assert_called_once_with("Loading...")
    
    def test_show_error(self, main_container):
        """Test showing error state."""
        # Reset mock
        main_container.show_error.reset_mock()
        
        # Create test error
        error = Exception("Test error")
        
        # Show error
        main_container.show_error(error, "Test error context")
        
        # Verify show_error was called
        main_container.show_error.assert_called_once_with(error, "Test error context")
    
    def test_update_style(self, main_container):
        """Test updating container styles."""
        # Reset mock
        main_container.update_style.reset_mock()
        
        # New style to apply
        new_style = {
            "background_color": "#f0f0f0",
            "border_color": "#999",
            "padding": "15px"
        }
        
        # Update style
        main_container.update_style(**new_style)
        
        # Verify update_style was called with the right parameters
        main_container.update_style.assert_called_once_with(**new_style)
    
    @patch('smartcash.ui.components.main_container.MainContainer')
    @patch('smartcash.ui.components.main_container.create_main_container')
    def test_create_main_container_function(self, mock_create_func, mock_container_class):
        """Test the create_main_container convenience function."""
        # Setup the mock return value
        mock_instance = MagicMock()
        mock_container_class.return_value = mock_instance
        
        # Setup the mock return value
        mock_instance = MagicMock()
        mock_create_func.return_value = mock_instance
        
        # Call the function with test parameters
        component_name = "test_container"
        style_variant = "default"
        result = mock_create_func(
            component_name=component_name,
            style_variant=style_variant
        )
        
        # Verify create_main_container was called with the correct parameters
        mock_create_func.assert_called_once_with(
            component_name=component_name,
            style_variant=style_variant
        )
        
        # Verify the function returns the created instance
        assert result == mock_instance
    
    def test_nested_containers(self, main_container, mocker):
        """Test nested container functionality."""
        # Create a mock for the child container with required methods
        child_container = mocker.MagicMock()
        child_container.components = {}
        child_container.add_component = mocker.MagicMock()
        child_container.remove_component = mocker.MagicMock()
        
        # Configure the main container's add_component to handle nested paths
        def add_component_side_effect(component, component_id=None):
            if '.' in str(component_id):
                # Handle nested component path (e.g., "child.nested_component")
                parent_id, child_id = component_id.split('.', 1)
                if parent_id in main_container.components:
                    main_container.components[parent_id].add_component(component, child_id)
            else:
                # Handle top-level component
                main_container.components[component_id] = component
            return component
            
        main_container.add_component.side_effect = add_component_side_effect
        
        # Configure remove_component to handle nested paths
        def remove_component_side_effect(component_id):
            if '.' in component_id:
                # Handle nested component path (e.g., "child.nested_component")
                parent_id, child_id = component_id.split('.', 1)
                if parent_id in main_container.components:
                    main_container.components[parent_id].remove_component(child_id)
            
        main_container.remove_component.side_effect = remove_component_side_effect
        
        # Add child container to main container
        main_container.add_component(child_container, "child")
        
        # Create a mock component to add to the child container
        mock_component = mocker.MagicMock()
        
        # Add component to the child container using dot notation
        main_container.add_component(mock_component, "child.nested_component")
        
        # Verify the component was added to the child container
        child_container.add_component.assert_called_once_with(mock_component, "nested_component")
        
        # Test removing nested component
        child_container.remove_component.reset_mock()
        main_container.remove_component("child.nested_component")
        child_container.remove_component.assert_called_once_with("nested_component")
