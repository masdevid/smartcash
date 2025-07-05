"""
Unit tests for main_container.py
"""
import unittest
from unittest.mock import MagicMock, patch
import ipywidgets as widgets

# Import the component to test
from smartcash.ui.components.main_container import MainContainer, create_main_container, ContainerConfig, ContainerType


class TestMainContainer(unittest.TestCase):
    """Test cases for the MainContainer class."""
    
    def setUp(self):
        """Set up the test environment."""
        # Create test widgets for each container type
        self.header_widget = widgets.HTML("<h1>Test Header</h1>")
        self.form_widget = widgets.VBox([widgets.Text(description="Test:")])
        self.action_widget = widgets.HBox([widgets.Button(description="Test Button")])
        self.operation_widget = widgets.VBox([widgets.Textarea(description="Log:")])
        self.footer_widget = widgets.HTML("<div>Test Footer</div>")
        self.custom_widget = widgets.Label("Custom Component")
        
        # Create a test container with all components
        self.container = MainContainer(
            header_container=self.header_widget,
            form_container=self.form_widget,
            action_container=self.action_widget,
            operation_container=self.operation_widget,
            footer_container=self.footer_widget
        )
    
    def tearDown(self):
        """Tear down the test environment."""
        pass
    
    def test_initialization_legacy(self):
        """Test initialization with legacy parameters."""
        # Test with all legacy parameters
        container = MainContainer(
            header_container=self.header_widget,
            form_container=self.form_widget,
            action_container=self.action_widget,
            operation_container=self.operation_widget,
            footer_container=self.footer_widget
        )
        
        # Check if all widgets are added
        self.assertIn(self.header_widget, container.container.children)
        self.assertIn(self.form_widget, container.container.children)
        self.assertIn(self.action_widget, container.container.children)
        self.assertIn(self.operation_widget, container.container.children)
        self.assertIn(self.footer_widget, container.container.children)
    
    def test_initialization_with_components(self):
        """Test initialization with the new components parameter."""
        components = [
            {'type': 'header', 'component': self.header_widget, 'order': 0},
            {'type': 'form', 'component': self.form_widget, 'order': 1},
            {'type': 'action', 'component': self.action_widget, 'order': 2},
            {'type': 'operation', 'component': self.operation_widget, 'order': 3},
            {'type': 'footer', 'component': self.footer_widget, 'order': 4},
            {'type': 'custom', 'component': self.custom_widget, 'order': 5, 'name': 'custom1'}
        ]
        
        container = MainContainer(components=components)
        
        # Check if all widgets are added in the correct order
        self.assertEqual(container.container.children[0], self.header_widget)
        self.assertEqual(container.container.children[1], self.form_widget)
        self.assertEqual(container.container.children[2], self.action_widget)
        self.assertEqual(container.container.children[3], self.operation_widget)
        self.assertEqual(container.container.children[4], self.footer_widget)
        self.assertEqual(container.container.children[5], self.custom_widget)
    
    def test_add_component(self):
        """Test adding a component to the container."""
        # Add a custom component
        custom_widget = widgets.Label("New Custom Component")
        component_name = self.container.add_component(custom_widget, 'custom', name='custom1')
        
        # Check if the component was added and the name is returned
        self.assertEqual(component_name, 'custom1')
        self.assertIn(custom_widget, self.container.container.children)
        self.assertEqual(self.container.get_component('custom1'), custom_widget)
    
    def test_remove_component(self):
        """Test removing a component from the container."""
        # First add a custom component
        custom_widget = widgets.Label("Custom to Remove")
        component_name = self.container.add_component(custom_widget, 'custom', name='to_remove')
        self.assertEqual(component_name, 'to_remove')
        
        # Then remove it
        self.container.remove_component('to_remove')
        
        # Check if the component was removed
        self.assertNotIn(custom_widget, self.container.container.children)
        self.assertIsNone(self.container.get_component('to_remove'))
    
    def test_show_hide_component(self):
        """Test showing and hiding a component."""
        # First add a custom component
        custom_widget = widgets.Label("Toggle Me")
        component_name = self.container.add_component(custom_widget, 'custom', name='toggle')
        self.assertEqual(component_name, 'toggle')
        
        # Hide the component
        self.container.hide_component('toggle')
        self.assertNotIn(custom_widget, self.container.container.children)
        
        # Show the component
        self.container.show_component('toggle')
        self.assertIn(custom_widget, self.container.container.children)


class TestCreateMainContainer(unittest.TestCase):
    """Test cases for the create_main_container function."""
    
    def test_create_main_container_legacy(self):
        """Test creating a main container with legacy parameters."""
        # Create test widgets
        header = widgets.HTML("<h1>Header</h1>")
        form = widgets.VBox([widgets.Text(description="Test:")])
        
        # Create container with legacy parameters
        container = create_main_container(
            header_container=header,
            form_container=form
        )
        
        # Check if container was created with the widgets
        self.assertIsInstance(container, MainContainer)
        self.assertIn(header, container.container.children)
        self.assertIn(form, container.container.children)
    
    def test_create_main_container_with_components(self):
        """Test creating a main container with the components parameter."""
        # Create test widgets
        header = widgets.HTML("<h1>Header</h1>")
        form = widgets.VBox([widgets.Text(description="Test:")])
        
        # Define components
        components = [
            {'type': 'header', 'component': header, 'order': 0},
            {'type': 'form', 'component': form, 'order': 1}
        ]
        
        # Create container with components
        container = create_main_container(components=components)
        
        # Check if container was created with the widgets in the correct order
        self.assertIsInstance(container, MainContainer)
        self.assertEqual(container.container.children[0], header)
        self.assertEqual(container.container.children[1], form)


if __name__ == '__main__':
    unittest.main()
