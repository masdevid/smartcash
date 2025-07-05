"""
Tests for the MainContainer component.
"""
import unittest
from unittest.mock import MagicMock, patch, call
import ipywidgets as widgets
from typing import List, Dict, Any

class TestMainContainer(unittest.TestCase):
    """Test cases for the MainContainer component."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Import here to avoid caching issues
        from smartcash.ui.components.main_container import MainContainer, ContainerConfig
        
        # Create mock widgets for each section
        self.mock_header = MagicMock(spec=widgets.Widget)
        self.mock_form = MagicMock(spec=widgets.Widget)
        self.mock_action = MagicMock(spec=widgets.Widget)
        self.mock_footer = MagicMock(spec=widgets.Widget)
        self.mock_operation = MagicMock(spec=widgets.Widget)
        self.mock_custom = MagicMock(spec=widgets.Widget)
        
        # Create a main container with all sections using legacy parameters
        self.legacy_container = MainContainer(
            header_container=self.mock_header,
            form_container=self.mock_form,
            action_container=self.mock_action,
            footer_container=self.mock_footer,
            operation_container=self.mock_operation,
            width='90%',
            max_width='1200px',
            margin='10px',
            padding='5px'
        )
        
        # Create components list for new style initialization
        self.components: List[ContainerConfig] = [
            {'type': 'header', 'component': self.mock_header, 'order': 0},
            {'type': 'form', 'component': self.mock_form, 'order': 1},
            {'type': 'action', 'component': self.mock_action, 'order': 2},
            {'type': 'operation', 'component': self.mock_operation, 'order': 3, 'visible': False},
            {'type': 'custom', 'component': self.mock_custom, 'name': 'custom1', 'order': 4}
        ]
        
        # Create a main container with the new components parameter
        self.new_container = MainContainer(
            components=self.components,
            width='95%',
            max_width='1400px',
            margin='15px',
            padding='10px'
        )
    
    def test_legacy_initialization(self):
        """Test that the container initializes with all sections using legacy parameters."""
        # Verify the container was created
        self.assertIsInstance(self.legacy_container.container, widgets.VBox)
        
        # Verify all sections are included and in correct order
        children = self.legacy_container.container.children
        self.assertEqual(len(children), 5)
        self.assertIs(children[0], self.mock_header)
        self.assertIs(children[1], self.mock_form)
        self.assertIs(children[2], self.mock_action)
        self.assertIs(children[3], self.mock_operation)
        self.assertIs(children[4], self.mock_footer)
        
        # Verify style was applied
        self.assertEqual(self.legacy_container.container.layout.width, '90%')
        self.assertEqual(self.legacy_container.container.layout.max_width, '1200px')
    
    def test_new_style_initialization(self):
        """Test that the container initializes with components list."""
        # Verify the container was created
        self.assertIsInstance(self.new_container.container, widgets.VBox)
        
        # Verify all components are included and in correct order
        children = self.new_container.container.children
        self.assertEqual(len(children), 5)
        
        # Verify visibility settings
        self.assertIn(self.mock_operation, children)  # Should be in children but hidden by layout
        
        # Verify style was applied
        self.assertEqual(self.new_container.container.layout.width, '95%')
        self.assertEqual(self.new_container.container.layout.max_width, '1400px')
    
    def test_component_visibility(self):
        """Test showing and hiding components."""
        # Initially hidden component should be in children but with display='none'
        op_container = next(
            c for c in self.new_container.container.children 
            if hasattr(c, 'children') and self.mock_operation in c.children
        )
        self.assertEqual(op_container.layout.display, 'none')
        
        # Show the operation component
        self.new_container.show_component('operation')
        self.assertEqual(op_container.layout.display, 'flex')
        
        # Hide it again
        self.new_container.hide_component('operation')
        self.assertEqual(op_container.layout.display, 'none')
    
    def test_custom_component_management(self):
        """Test adding and removing custom components."""
        # Add a new custom component
        new_custom = MagicMock(spec=widgets.Widget)
        self.new_container.add_component(
            component=new_custom,
            name='custom2',
            order=1,  # Should appear second
            visible=True
        )
        
        # Verify it was added
        self.assertIn(new_custom, self.new_container.container.children)
        
        # Remove a component
        self.new_container.remove_component('custom1')
        self.assertNotIn(self.mock_custom, self.new_container.container.children)
    
    def test_invalid_component_handling(self):
        """Test handling of invalid component configurations."""
        # Test with invalid component type
        with self.assertRaises(ValueError):
            self.new_container.add_component(
                component=MagicMock(),
                name='invalid',
                type='invalid_type',
                order=10
            )
        
        # Test with missing required fields
        with self.assertRaises(ValueError):
            self.new_container.add_component(
                component=MagicMock(),
                # Missing name
                order=10
            )
    
    def test_component_ordering(self):
        """Test that components maintain their specified order."""
        # Components should be in order: header(0), form(1), action(2), operation(3), custom(4)
        children = self.new_container.container.children
        self.assertIs(children[0], self.mock_header)
        self.assertIs(children[1], self.mock_form)
        self.assertIs(children[2], self.mock_action)
        
        # Add a component with order=1, should be inserted at position 1
        new_component = MagicMock(spec=widgets.Widget)
        self.new_container.add_component(
            component=new_component,
            name='inserted',
            order=1
        )
        
        # Verify new order: header(0), inserted(1), form(1), action(2), operation(3), custom(4)
        children = self.new_container.container.children
        self.assertIs(children[0], self.mock_header)
        self.assertIs(children[1], new_component)
        self.assertIs(children[2], self.mock_form)
        self.assertEqual(self.container.container.layout.width, '90%')
        self.assertEqual(self.container.container.layout.max_width, '1200px')
    
    def test_update_section(self):
        """Test updating a section of the container."""
        # Create a new mock widget
        new_header = MagicMock(spec=widgets.Widget)
        
        # Update the header section
        self.container.update_section('header', new_header)
        
        # Verify the section was updated
        self.assertEqual(self.container.get_section('header'), new_header)
        self.assertIn(new_header, self.container.container.children)
        self.assertNotIn(self.mock_header, self.container.container.children)
        
        # Verify invalid section name raises an error
        with self.assertRaises(ValueError):
            self.container.update_section('invalid_section', new_header)
    
    def test_get_section(self):
        """Test retrieving a section from the container."""
        # Test getting each section
        self.assertEqual(self.container.get_section('header'), self.mock_header)
        self.assertEqual(self.container.get_section('form'), self.mock_form)
        self.assertEqual(self.container.get_section('action'), self.mock_action)
        self.assertEqual(self.container.get_section('footer'), self.mock_footer)
        self.assertEqual(self.container.get_section('progress'), self.mock_progress)
        self.assertEqual(self.container.get_section('log'), self.mock_log)
        
        # Test getting a non-existent section
        self.assertIsNone(self.container.get_section('nonexistent'))
    
    def test_add_remove_class(self):
        """Test adding and removing CSS classes."""
        # Add a class
        self.container.add_class('test-class')
        self.assertIn('test-class', self.container.container._dom_classes)
        
        # Remove the class
        self.container.remove_class('test-class')
        self.assertNotIn('test-class', self.container.container._dom_classes)

    def test_empty_components_list(self):
        """Test that an empty components list raises an error."""
        with self.assertRaises(ValueError):
            MainContainer(components=[])

    @patch('ipywidgets.VBox')
    def test_style_application(self, mock_vbox):
        """Test that styles are correctly applied to the container."""
        # Mock the VBox to track style application
        mock_instance = MagicMock()
        mock_vbox.return_value = mock_instance
        
        # Create a container with custom styles
        custom_styles = {
            'border': '1px solid #ccc',
            'background_color': '#f5f5f5',
            'custom_style': 'value'
        }
        container = MainContainer(
            components=[],
            **custom_styles
        )
        
        # Verify styles were applied to the layout
        for key, value in custom_styles.items():
            self.assertEqual(getattr(container.container.layout, key), value)

    def test_component_retrieval(self):
        """Test getting components by name or type."""
        # Get by type
        header = self.new_container.get_component('header')
        self.assertIs(header, self.mock_header)
        
        # Get by name
        custom = self.new_container.get_component('custom1')
        self.assertIs(custom, self.mock_custom)
        
        # Non-existent component
        with self.assertRaises(ValueError):
            self.new_container.get_component('nonexistent')

    def test_component_update(self):
        """Test updating an existing component."""
        # Create a new version of the header
        new_header = MagicMock(spec=widgets.Widget)
        
        # Update the header component
        self.new_container.update_component('header', new_header)
        
        # Verify the component was updated
        self.assertIs(self.new_container.get_component('header'), new_header)
        self.assertIn(new_header, self.new_container.container.children)
        self.assertNotIn(self.mock_header, self.new_container.container.children)


class TestCreateMainContainer(unittest.TestCase):
    """Test cases for the create_main_container function."""
    
    def test_create_main_container_legacy(self):
        """Test creating a main container with legacy parameters."""
        from smartcash.ui.components.main_container import create_main_container
        
        # Create mock widgets
        mock_header = MagicMock(spec=widgets.Widget)
        mock_form = MagicMock(spec=widgets.Widget)
        mock_action = MagicMock(spec=widgets.Widget)
        mock_footer = MagicMock(spec=widgets.Widget)
        mock_operation = MagicMock(spec=widgets.Widget)
        
        # Create container using the factory function with legacy parameters
        container = create_main_container(
            header_container=mock_header,
            form_container=mock_form,
            action_container=mock_action,
            footer_container=mock_footer,
            operation_container=mock_operation,
            width='80%',
            max_width='1000px',
            margin='10px',
            padding='5px'
        )
        
        # Verify the container was created with the correct widgets
        children = container.container.children
        self.assertEqual(len(children), 5)
        self.assertIs(children[0], mock_header)
        self.assertIs(children[1], mock_form)
        self.assertIs(children[2], mock_action)
        self.assertIs(children[3], mock_operation)
        self.assertIs(children[4], mock_footer)
        
        # Verify style was applied
        self.assertEqual(container.container.layout.width, '80%')
        self.assertEqual(container.container.layout.max_width, '1000px')
    
    def test_create_main_container_with_components(self):
        """Test creating a main container with components list."""
        from smartcash.ui.components.main_container import create_main_container
        
        # Create mock widgets
        mock_header = MagicMock(spec=widgets.Widget)
        mock_form = MagicMock(spec=widgets.Widget)
        mock_action = MagicMock(spec=widgets.Widget)
        
        # Create container using the factory function with components list
        components = [
            {'type': 'header', 'component': mock_header, 'order': 0},
            {'type': 'form', 'component': mock_form, 'order': 1},
            {'type': 'action', 'component': mock_action, 'order': 2, 'visible': False}
        ]
        
        container = create_main_container(
            components=components,
            width='90%',
            max_width='1100px'
        )
        
        # Verify the container was created with the correct widgets
        children = container.container.children
        self.assertEqual(len(children), 3)
        self.assertIs(children[0], mock_header)
        self.assertIs(children[1], mock_form)
        
        # The action component should be in children but hidden
        self.assertIn(mock_action, children)
        
        # Verify style was applied
        self.assertEqual(container.container.layout.width, '90%')
        self.assertEqual(container.container.layout.max_width, '1100px')
    
    def test_create_main_container_validation(self):
        """Test validation of create_main_container parameters."""
        from smartcash.ui.components.main_container import create_main_container
        
        # Test with neither legacy params nor components
        with self.assertRaises(ValueError):
            create_main_container()
        
        # Test with both legacy params and components
        with self.assertRaises(ValueError):
            create_main_container(
                header_container=MagicMock(),
                components=[{'type': 'header', 'component': MagicMock()}]
            )
        
        # Test with invalid component config
        with self.assertRaises(ValueError):
            create_main_container(components=[{}])  # Missing required fields
        
        # Verify other sections are None
        self.assertIsNone(container.get_section('action'))
        self.assertIsNone(container.get_section('footer'))


if __name__ == "__main__":
    unittest.main()
