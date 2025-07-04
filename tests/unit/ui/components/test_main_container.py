"""
Tests for the MainContainer component.
"""
import unittest
from unittest.mock import MagicMock, patch
import ipywidgets as widgets

class TestMainContainer(unittest.TestCase):
    """Test cases for the MainContainer component."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Import here to avoid caching issues
        from smartcash.ui.components.main_container import MainContainer
        
        # Create mock widgets for each section
        self.mock_header = MagicMock(spec=widgets.Widget)
        self.mock_form = MagicMock(spec=widgets.Widget)
        self.mock_action = MagicMock(spec=widgets.Widget)
        self.mock_footer = MagicMock(spec=widgets.Widget)
        self.mock_progress = MagicMock(spec=widgets.Widget)
        self.mock_log = MagicMock(spec=widgets.Widget)
        
        # Create a main container with all sections
        self.container = MainContainer(
            header_container=self.mock_header,
            form_container=self.mock_form,
            action_container=self.mock_action,
            footer_container=self.mock_footer,
            progress_container=self.mock_progress,
            log_container=self.mock_log,
            width='90%',
            max_width='1200px'
        )
    
    def test_initialization(self):
        """Test that the container initializes with all sections."""
        # Verify the container was created
        self.assertIsInstance(self.container.container, widgets.VBox)
        
        # Verify all sections are included
        self.assertEqual(len(self.container.container.children), 6)
        self.assertIn(self.mock_header, self.container.container.children)
        self.assertIn(self.mock_form, self.container.container.children)
        self.assertIn(self.mock_action, self.container.container.children)
        self.assertIn(self.mock_progress, self.container.container.children)
        self.assertIn(self.mock_log, self.container.container.children)
        self.assertIn(self.mock_footer, self.container.container.children)
        
        # Verify style was applied
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


class TestCreateMainContainer(unittest.TestCase):
    """Test cases for the create_main_container function."""
    
    def test_create_main_container(self):
        """Test creating a main container with the factory function."""
        # Import the function directly to test
        from smartcash.ui.components.main_container import create_main_container
        
        # Create mock widgets
        mock_header = MagicMock(spec=widgets.Widget)
        mock_form = MagicMock(spec=widgets.Widget)
        
        # Create a main container with some sections
        container = create_main_container(
            header_container=mock_header,
            form_container=mock_form,
            width='80%',
            margin='10px'
        )
        
        # Verify the container was created
        self.assertIsInstance(container.container, widgets.VBox)
        
        # Verify the specified sections are included
        self.assertEqual(container.get_section('header'), mock_header)
        self.assertEqual(container.get_section('form'), mock_form)
        
        # Verify style was applied
        self.assertEqual(container.container.layout.width, '80%')
        self.assertEqual(container.container.layout.margin, '10px')
        
        # Verify other sections are None
        self.assertIsNone(container.get_section('action'))
        self.assertIsNone(container.get_section('footer'))


if __name__ == "__main__":
    unittest.main()
