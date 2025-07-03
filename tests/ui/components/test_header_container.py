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
            status_message="Initial status",
            status_type="info"
        )
    
    def test_initialization(self):
        """Test that the header initializes with correct properties."""
        self.assertEqual(self.header.title, "Test Title")
        self.assertEqual(self.header.subtitle, "Test Subtitle")
        self.assertEqual(self.header.status_message, "Initial status")
        self.assertEqual(self.header.status_type, "info")
        self.assertTrue(self.header.show_status_panel)
        
        # Check that the container was created
        self.assertIsNotNone(self.header.container)
        self.assertIsInstance(self.header.container, widgets.VBox)
        
        # Check that the header and status panel were created
        self.assertIsNotNone(self.header.header)
        self.assertIsNotNone(self.header.status_panel)
    
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
    
    def test_update_status(self):
        """Test updating the status message and type."""
        # Create a mock for the status panel
        mock_status_panel = MagicMock()
        mock_status_panel.layout = MagicMock()
        self.header.status_panel = mock_status_panel
        
        # Update the status
        self.header.update_status("New status", "success", False)
        
        # Check that the properties were updated
        self.assertEqual(self.header.status_message, "New status")
        self.assertEqual(self.header.status_type, "success")
        self.assertFalse(self.header.show_status_panel)
        
        # Check that the status panel visibility was updated
        self.assertEqual(mock_status_panel.layout.display, 'none')
    
    def test_show_status(self):
        """Test showing and hiding the status panel."""
        # Create a mock for the status panel
        mock_status_panel = MagicMock()
        mock_status_panel.layout = MagicMock(display='block')
        self.header.status_panel = mock_status_panel
        
        # Initially visible
        self.assertEqual(mock_status_panel.layout.display, 'block')
        
        # Hide the status panel
        self.header.show_status(False)
        self.assertEqual(mock_status_panel.layout.display, 'none')
        self.assertFalse(self.header.show_status_panel)
        
        # Show the status panel
        self.header.show_status(True)
        self.assertEqual(mock_status_panel.layout.display, 'block')
        self.assertTrue(self.header.show_status_panel)
    
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
            status_message="Initial status",
            status_type="info"
        )
        
        # Check that the header was created with the correct properties
        self.assertEqual(header.title, "Test Title")
        self.assertEqual(header.subtitle, "Test Subtitle")
        self.assertEqual(header.status_message, "Initial status")
        self.assertEqual(header.status_type, "info")
        
        # Check that the container was created
        self.assertIsNotNone(header.container)
        self.assertIsInstance(header.container, widgets.VBox)


if __name__ == "__main__":
    unittest.main()
