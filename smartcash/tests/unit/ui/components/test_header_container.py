"""
Tests for the HeaderContainer component.
"""
import unittest
import pytest
from unittest.mock import patch, MagicMock, call
import ipywidgets as widgets

from smartcash.ui.components import HeaderContainer
from smartcash.ui.components.header_container import create_header_container
from smartcash.tests.test_helpers import (
    assert_has_trait,
    assert_widget_visible,
    assert_has_class,
    assert_widget_children,
    mock_widget,
    mock_vbox,
    mock_button,
    mock_text,
    MockWidget,
    MockButton,
    patch_display,
    patch_widget,
    mock_ui_components
)

class TestHeaderContainer(unittest.TestCase):
    """Test cases for HeaderContainer component."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Set up common test data
        self.test_title = "Test Title"
        self.test_subtitle = "Test Subtitle"
        self.test_status = "Initial status"
        self.test_status_type = "info"
        
        # Patch display to prevent actual display during tests
        self.display_patch = patch_display()
        self.display_patch.start()
        
        # Create a test instance with minimal parameters
        self.header = HeaderContainer(
            title=self.test_title,
            subtitle=self.test_subtitle,
            status_message=self.test_status,
            status_type=self.test_status_type
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.display_patch.stop()
    
    def test_initialization(self):
        """Test that the header container initializes with the correct default values."""
        # Test default values
        self.assertEqual(self.header.title, self.test_title)
        self.assertEqual(self.header.subtitle, self.test_subtitle)
        self.assertEqual(self.header.status_message, self.test_status)
        self.assertEqual(self.header.status_type, self.test_status_type)
        self.assertTrue(self.header.show_status_panel)
        
        # Test container was created
        self.assertIsNotNone(self.header.container)
        self.assertIsInstance(self.header.container, widgets.VBox)
        
        # Test children
        self.assertEqual(len(self.header.container.children), 2)
        self.assertEqual(self.header.container.children[0], self.header.header)
        self.assertEqual(self.header.container.children[1], self.header.status_panel)
        
        # Test default styles
        self.assertEqual(self.header.container.layout.margin_bottom, '16px')
        self.assertEqual(self.header.container.layout.padding_bottom, '8px')
        self.assertEqual(self.header.container.layout.border_bottom, '1px solid #e0e0e0')
    
    def test_initialization_with_custom_styles(self):
        """Test initialization with custom style options."""
        custom_styles = {
            'margin_bottom': '20px',
            'padding_bottom': '10px',
            'border_bottom': '2px solid red',
            'custom_style': 'value'
        }
        
        header = HeaderContainer(
            title="Test",
            **custom_styles
        )
        
        # Test custom styles
        self.assertEqual(header.container.layout.margin_bottom, '20px')
        self.assertEqual(header.container.layout.padding_bottom, '10px')
        self.assertEqual(header.container.layout.border_bottom, '2px solid red')
        
        # Test custom style is in the style dict but not applied to layout
        self.assertIn('custom_style', header.style)
        self.assertFalse(hasattr(header.container.layout, 'custom_style'))
    
    def test_initialization_without_status_panel(self):
        """Test initialization with status panel hidden."""
        header = HeaderContainer(
            title="Test",
            show_status_panel=False
        )
        
        self.assertFalse(header.show_status_panel)
        self.assertEqual(header.status_panel.layout.display, 'none')
    
    def test_update_status(self):
        """Test updating the status message and type."""
        with patch('smartcash.ui.components.header_container.update_status_panel') as mock_update:
            # Test updating status
            self.header.update_status("New status", "success")
            self.assertEqual(self.header.status_message, "New status")
            self.assertEqual(self.header.status_type, "success")
            self.assertTrue(self.header.show_status_panel)
            
            # Verify update_status_panel was called
            mock_update.assert_called_with(
                self.header.status_panel,
                "New status",
                "success"
            )
            
            # Test hiding the status panel
            self.header.update_status("Hidden status", "info", False)
            self.assertEqual(self.header.status_message, "Hidden status")
            self.assertFalse(self.header.show_status_panel)
            self.assertEqual(self.header.status_panel.layout.display, 'none')
    
    def test_update_status_invalid_type(self):
        """Test updating status with an invalid status type."""
        with self.assertRaises(ValueError):
            self.header.update_status("Test", "invalid_type")
    
    def test_update_title(self):
        """Test updating the header title, subtitle, and icon."""
        with patch('smartcash.ui.components.header_container.create_header') as mock_create_header:
            # Setup the mock to return a new header
            new_header = MagicMock()
            mock_create_header.return_value = new_header
            
            # Test updating just the title
            self.header.update_title("New Title")
            self.assertEqual(self.header.title, "New Title")
            self.assertEqual(self.header.subtitle, self.test_subtitle)  # Should remain unchanged
            
            # Verify create_header was called with correct args
            mock_create_header.assert_called_with(
                title="New Title",
                description=self.test_subtitle,
                icon=""
            )
            
            # Test updating subtitle and icon
            self.header.update_title("New Title", "New Subtitle", "🚀")
            self.assertEqual(self.header.title, "New Title")
            self.assertEqual(self.header.subtitle, "New Subtitle")
            self.assertEqual(self.header.icon, "🚀")
            
            # Verify container was updated
            self.assertIn(new_header, self.header.container.children)
    
    def test_update_title_with_none_values(self):
        """Test updating title with None values for optional parameters."""
        # Set initial values
        self.header.title = "Initial Title"
        self.header.subtitle = "Initial Subtitle"
        self.header.icon = "⭐"
        
        # Update with None values
        self.header.update_title("New Title", None, None)
        
        # Should keep previous values for None parameters
        self.assertEqual(self.header.title, "New Title")
        self.assertEqual(self.header.subtitle, "Initial Subtitle")
        self.assertEqual(self.header.icon, "⭐")
        self.assertIsNotNone(self.header.status_panel)
    
    def test_show_status(self):
        """Test showing and hiding the status panel."""
        # Test hiding
        self.header.show_status(False)
        self.assertFalse(self.header.show_status_panel)
        self.assertEqual(self.header.status_panel.layout.display, 'none')
        
        # Test showing
        self.header.show_status(True)
        self.assertTrue(self.header.show_status_panel)
        self.assertEqual(self.header.status_panel.layout.display, 'block')
    
    def test_add_remove_class(self):
        """Test adding and removing CSS classes."""
        # Test adding a class
        self.header.add_class('test-class')
        self.assertIn('test-class', self.header.container._dom_classes)
        
        # Test removing the class
        self.header.remove_class('test-class')
        self.assertNotIn('test-class', self.header.container._dom_classes)
    
    def test_add_remove_class_with_existing_classes(self):
        """Test adding/removing classes when other classes exist."""
        # Add initial class
        self.header.container._dom_classes = ['existing-class']
        
        # Add new class
        self.header.add_class('new-class')
        self.assertIn('new-class', self.header.container._dom_classes)
        self.assertIn('existing-class', self.header.container._dom_classes)
        
        # Remove one class
        self.header.remove_class('new-class')
        self.assertNotIn('new-class', self.header.container._dom_classes)
        self.assertIn('existing-class', self.header.container._dom_classes)
    
    def test_show_status_panel(self):
        """Test showing and hiding the status panel using property."""
        # Test hiding the panel
        self.header.show_status_panel = False
        self.assertFalse(self.header.show_status_panel)
        self.assertEqual(self.header.status_panel.layout.display, 'none')
        
        # Test showing the panel
        self.header.show_status_panel = True
        self.assertTrue(self.header.show_status_panel)
        self.assertEqual(self.header.status_panel.layout.display, 'block')
    
    def test_create_header_container_factory(self):
        """Test the create_header_container factory function."""
        with patch('smartcash.ui.components.header_container.HeaderContainer') as mock_class:
            # Setup mock return value
            mock_instance = MagicMock()
            mock_class.return_value = mock_instance
            
            # Call the factory function
            result = create_header_container(
                title="Test Title",
                subtitle="Test Subtitle",
                icon="⭐",
                status_message="Test Status",
                status_type="success",
                show_status_panel=True,
                custom_style="value"
            )
            
            # Verify HeaderContainer was instantiated with correct args
            mock_class.assert_called_once_with(
                title="Test Title",
                subtitle="Test Subtitle",
                icon="⭐",
                status_message="Test Status",
                status_type="success",
                show_status_panel=True,
                custom_style="value"
            )
            
            # Verify the result is our mock instance
            self.assertIs(result, mock_instance)


class TestCreateHeaderContainer(unittest.TestCase):
    """Test cases for create_header_container function."""
    
    def setUp(self):
        """Set up test data and patches."""
        self.test_title = "Test Title"
        self.test_subtitle = "Test Subtitle"
        self.test_status = "Test Status"
        self.test_status_type = "info"
        
        # Create a real HeaderContainer instance for testing
        self.header = HeaderContainer(
            title=self.test_title,
            subtitle=self.test_subtitle,
            status_message=self.test_status,
            status_type=self.test_status_type
        )
        
        # Patch the HeaderContainer class to return our test instance
        self.header_patch = patch(
            'smartcash.ui.components.header_container.HeaderContainer',
            return_value=self.header
        )
        self.mock_header_class = self.header_patch.start()
        
        # Patch display
        self.display_patch = patch_display()
        self.mock_display = self.display_patch.start()
    
    def tearDown(self):
        """Clean up patches."""
        self.header_patch.stop()
        self.display_patch.stop()
    
    def test_create_header_container(self):
        """Test creating a header container with the factory function."""
        # Call the factory function
        result = create_header_container(
            title=self.test_title,
            subtitle=self.test_subtitle,
            status_message=self.test_status,
            status_type=self.test_status_type
        )
        
        # Verify the HeaderContainer was instantiated with correct parameters
        self.mock_header_class.assert_called_once_with(
            title=self.test_title,
            subtitle=self.test_subtitle,
            status_message=self.test_status,
            status_type=self.test_status_type,
            show_logo=True
        )
        
        # Verify the result is our test instance
        self.assertIs(result, self.header)
        
        # Verify display was called with the header
        self.mock_display.assert_called_once()


if __name__ == "__main__":
    unittest.main()
