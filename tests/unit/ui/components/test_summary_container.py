"""
Unit tests for summary_container.py
"""
import unittest
from unittest.mock import patch, MagicMock, ANY
import ipywidgets as widgets

# Import the component to test
from smartcash.ui.components.summary_container import SummaryContainer, create_summary_container


class TestSummaryContainer(unittest.TestCase):
    """Test cases for the SummaryContainer class."""
    
    def setUp(self):
        """Set up the test environment."""
        # Patch the error handler to avoid actual error handling during tests
        self.error_handler_patcher = patch('smartcash.ui.components.base_component.ErrorContext')
        self.mock_error_handler = self.error_handler_patcher.start()
        self.mock_error_handler.return_value.__enter__.return_value = None
        
        # Create a test instance of SummaryContainer
        self.container = SummaryContainer(
            component_name="test_summary",
            theme="primary",
            title="Test Container",
            icon="📊"
        )
        
        # Initialize the container
        self.container.initialize()
        
    def tearDown(self):
        """Tear down the test environment."""
        self.error_handler_patcher.stop()
    
    def tearDown(self):
        """Tear down the test environment."""
        pass
    
    def test_initialization(self):
        """Test SummaryContainer initialization."""
        # Check if container is created
        self.assertIn('container', self.container._ui_components)
        self.assertIsInstance(self.container._ui_components['container'], widgets.Box)
        
        # Check if content area is created
        self.assertIn('content', self.container._ui_components)
        self.assertIsInstance(self.container._ui_components['content'], widgets.HTML)
        
        # Check if title is set correctly
        self.assertIn('title', self.container._ui_components)
        title_widget = self.container._ui_components['title']
        self.assertIsInstance(title_widget, widgets.HTML)
        self.assertIn('Test Container', title_widget.value)
        self.assertIn('📊', title_widget.value)
        
        # Check theme
        self.assertEqual(self.container._theme, "primary")
        self.assertEqual(self.container._theme_style['text_color'], "#084298")
    
    def test_set_content(self):
        """Test setting content."""
        test_content = "<p>This is a test content</p>"
        self.container.set_content(test_content)
        
        # Check if content was set
        self.assertEqual(self.container._ui_components['content'].value, test_content)
    
    def test_set_theme(self):
        """Test changing the theme."""
        # Change to success theme
        self.container.set_theme("success")
        
        # Check if theme was updated
        self.assertEqual(self.container._theme, "success")
        self.assertEqual(self.container._theme_style['text_color'], "#0f5132")
        
        # Check if container style was updated
        container = self.container._ui_components['container']
        self.assertEqual(container.layout.background, self.container._theme_style["gradient"])
        self.assertEqual(container.layout.border, self.container._theme_style["border"])
        
        # Test with invalid theme (should fall back to default)
        self.container.set_theme("invalid_theme")
        self.assertEqual(self.container._theme, "default")
    
    @patch('smartcash.ui.components.summary_container.SummaryContainer.set_theme')
    def test_show_message(self, mock_set_theme):
        """Test showing a message with different types."""
        # Test info message
        self.container.show_message("Info Title", "This is an info message", "info")
        
        # Verify set_theme was called with the correct theme
        mock_set_theme.assert_called_with("info")
        
        # Verify content was set correctly
        content = self.container._ui_components['content'].value
        self.assertIn("Info Title", content)
        self.assertIn("This is an info message", content)
        self.assertIn("ℹ️", content)
        
        # Reset mock for next test
        mock_set_theme.reset_mock()
        
        # Test success message
        self.container.show_message("Success!", "Operation completed", "success")
        mock_set_theme.assert_called_with("success")
        
        # Test with custom icon
        self.container.show_message("Custom", "With custom icon", "warning", icon="⚠️ Custom")
        content = self.container._ui_components['content'].value
        self.assertIn("⚠️ Custom", content)
    
    def test_set_html(self):
        """Test setting HTML content with optional theme."""
        test_html = "<div><h3>Custom HTML</h3><p>With custom content</p></div>"
        
        # Set HTML without changing theme
        self.container.set_html(test_html)
        self.assertEqual(self.container._ui_components['content'].value, test_html)
        
        # Set HTML with theme change
        self.container.set_html(test_html, "danger")
        self.assertEqual(self.container._theme, "danger")
        self.assertEqual(self.container._ui_components['content'].value, test_html)


class TestCreateSummaryContainer(unittest.TestCase):
    """Test cases for the create_summary_container function."""
    
    def test_create_summary_container(self):
        """Test creating a summary container with the factory function."""
        # Create container with factory function
        container = create_summary_container(
            theme="warning",
            title="Factory Container",
            icon="⚙️"
        )
        
        # Check if container was created
        self.assertIsInstance(container, SummaryContainer)
        self.assertEqual(container._theme, "warning")
        
        # Check if title and icon are set
        self.assertEqual(container._title, "Factory Container")
        self.assertEqual(container._icon, "⚙️")


if __name__ == '__main__':
    unittest.main()
