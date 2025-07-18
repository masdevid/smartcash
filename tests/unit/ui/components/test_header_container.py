"""
Tests for the HeaderContainer component.
"""
import unittest
from unittest.mock import MagicMock, patch
import ipywidgets as widgets

from smartcash.ui.components.header_container import HeaderContainer

class TestHeaderContainer(unittest.TestCase):
    """Test cases for HeaderContainer component."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a test instance with minimal parameters
        self.header = HeaderContainer(
            title="Test Title",
            subtitle="Test Subtitle",
            icon="📝"
        )
    
    def test_initialization(self):
        """Test that the header initializes with correct properties."""
        self.assertEqual(self.header.title, "Test Title")
        self.assertEqual(self.header.subtitle, "Test Subtitle")
        self.assertEqual(self.header.icon, "📝")
        
        # Check that the container was created
        self.assertIsNotNone(self.header.container)
        self.assertIsInstance(self.header.container, widgets.VBox)
        
        # Check that the header was created
        self.assertIsNotNone(self.header.header)
    
    def test_update_title(self):
        """Test updating the header title, subtitle, and icon."""
        # Mock the _create_header and _create_container methods
        with patch.object(self.header, '_create_header') as mock_create_header, \
             patch.object(self.header, '_create_container') as mock_create_container:
            
            # Update the title
            self.header.update_title("New Title", "New Subtitle", "🔔")
            
            # Check that the properties were updated
            self.assertEqual(self.header.title, "New Title")
            self.assertEqual(self.header.subtitle, "New Subtitle")
            self.assertEqual(self.header.icon, "🔔")
            
            # Check that the methods were called
            mock_create_header.assert_called_once()
            mock_create_container.assert_called_once()
    
    def test_add_remove_class(self):
        """Test adding and removing CSS classes."""
        # Add a class
        self.header.add_class("test-class")
        self.assertIn("test-class", self.header.container._dom_classes)
        
        # Remove the class
        self.header.remove_class("test-class")
        self.assertNotIn("test-class", self.header.container._dom_classes)


class TestCreateHeaderContainer(unittest.TestCase):
    """Test cases for create_header_container function."""
    
    def test_create_header_container(self):
        """Test creating a header container with the factory function."""
        # Import the function directly to test
        from smartcash.ui.components.header_container import create_header_container
        
        # Create a header container
        header = create_header_container(
            title="Test Title",
            subtitle="Test Subtitle",
            icon="📝"
        )
        
        # Check that the header was created with the correct properties
        self.assertEqual(header.title, "Test Title")
        self.assertEqual(header.subtitle, "Test Subtitle")
        self.assertEqual(header.icon, "📝")
        
        # Check that the container was created
        self.assertIsNotNone(header.container)
        self.assertIsInstance(header.container, widgets.VBox)


if __name__ == "__main__":
    unittest.main()
