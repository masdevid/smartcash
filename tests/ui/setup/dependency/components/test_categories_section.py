"""
Test cases for the categories section component in dependency management UI.
"""
import unittest
from unittest.mock import MagicMock, patch, ANY

class TestCategoriesSection(unittest.TestCase):
    """Test cases for the categories section component."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Mock the get_default_dependency_config function
        self.default_config_patch = patch(
            'smartcash.ui.setup.dependency.handlers.defaults.get_default_dependency_config',
            return_value={'categories': []}
        )
        self.mock_get_default_config = self.default_config_patch.start()
        
        # Create test data
        self.test_config = {
            'categories': [
                {
                    'name': 'Test Category',
                    'description': 'Test Description',
                    'icon': '📦',
                    'packages': [
                        {
                            'name': 'test-package',
                            'description': 'Test package',
                            'key': 'test-pkg',
                            'pip_name': 'test-package',
                            'required': True,
                            'installed': False,
                            'version': '1.0.0',
                            'latest_version': '1.0.0',
                            'update_available': False,
                            'dependencies': []
                        }
                    ]
                }
            ]
        }
        
        # Patch the widget constructors
        self.checkbox_patch = patch('ipywidgets.Checkbox')
        self.vbox_patch = patch('ipywidgets.VBox')
        self.html_patch = patch('ipywidgets.HTML')
        
        # Start the patches
        self.mock_checkbox = self.checkbox_patch.start()
        self.mock_vbox = self.vbox_patch.start()
        self.mock_html = self.html_patch.start()
        
        # Import the function after patching
        from smartcash.ui.setup.dependency.components.ui_components import _create_categories_section
        self.create_categories_section = _create_categories_section
    
    def tearDown(self):
        """Tear down test fixtures after each test method."""
        self.default_config_patch.stop()
        self.checkbox_patch.stop()
        self.vbox_patch.stop()
        self.html_patch.stop()
    
    def test_create_categories_section_with_config(self):
        """Test creating categories section with provided config."""
        # Setup mock return values
        mock_vbox = MagicMock()
        self.mock_vbox.return_value = mock_vbox
        
        # Call the function with test config
        result = self.create_categories_section(self.test_config)
        
        # Verify the result is a VBox
        self.assertIsInstance(result, MagicMock)  # Mocked VBox
        
        # Verify HTML was created for the category header
        self.assertTrue(self.mock_html.called)
        
        # Verify checkbox was created for the package
        self.assertTrue(self.mock_checkbox.called)
        
        # Verify VBox was called
        self.assertTrue(self.mock_vbox.called)
        
        # Verify the result is the mock VBox
        self.assertEqual(result, mock_vbox)
    
    def test_create_categories_section_without_config(self):
        """Test creating categories section without config (uses default)."""
        # Setup mock return values
        mock_vbox = MagicMock()
        self.mock_vbox.return_value = mock_vbox
        
        # Setup default config mock
        default_config = {
            'categories': [
                {
                    'name': 'Default Category',
                    'description': 'Default Description',
                    'icon': '📦',
                    'packages': [
                        {
                            'name': 'default-package',
                            'description': 'Default package',
                            'key': 'default-pkg',
                            'pip_name': 'default-package',
                            'required': True,
                            'installed': False,
                            'version': '1.0.0',
                            'latest_version': '1.0.0',
                            'update_available': False,
                            'dependencies': []
                        }
                    ]
                }
            ]
        }
        self.mock_get_default_config.return_value = default_config
        
        # Call the function without config
        result = self.create_categories_section({})
        
        # Verify the result is a VBox
        self.assertIsInstance(result, MagicMock)  # Mocked VBox
        
        # Verify default config was used
        self.mock_get_default_config.assert_called_once()
        
        # Verify VBox was called
        self.assertTrue(self.mock_vbox.called)
        
        # Verify the result is the mock VBox
        self.assertEqual(result, mock_vbox)
    
    def test_create_categories_section_empty(self):
        """Test creating categories section with empty categories."""
        # Setup mock return values
        mock_html = MagicMock()
        self.mock_html.return_value = mock_html
        
        # Call the function with empty categories
        result = self.create_categories_section({'categories': []})
        
        # Verify the result is an HTML widget
        self.assertIsInstance(result, MagicMock)  # Mocked HTML
        
        # Verify HTML was created for the empty message
        self.assertTrue(self.mock_html.called)
        
        # Check if the correct message is in the HTML call
        html_call_args = self.mock_html.call_args[0][0]
        self.assertIn("Tidak ada kategori package yang tersedia", html_call_args)
        
        # Verify the result is the mock HTML
        self.assertEqual(result, mock_html)

if __name__ == '__main__':
    unittest.main()
