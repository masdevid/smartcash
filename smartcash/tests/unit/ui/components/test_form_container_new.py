"""
Simplified tests for the FormContainer component.
"""
import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Add the project root to the Python path for direct imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../..')))

# Import the form_container module
from smartcash.ui.components import form_container

# Import the components we need
create_form_container = form_container.create_form_container
LayoutType = form_container.LayoutType
FormItem = form_container.FormItem

# Import widgets for testing
import ipywidgets as widgets

class TestFormItem(unittest.TestCase):
    """Test cases for FormItem class."""
    
    def test_form_item_initialization(self):
        """Test FormItem initialization with default values."""
        widget = MagicMock()
        form_item = FormItem(widget)
        
        self.assertEqual(form_item.widget, widget)
        self.assertEqual(form_item.width, '100%')
        self.assertEqual(form_item.height, 'auto')
        self.assertEqual(form_item.align_items, 'stretch')
        self.assertEqual(form_item.align_self, 'stretch')


class TestFormContainer(unittest.TestCase):
    """Test cases for FormContainer component."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock ipywidgets
        self.mock_widget = MagicMock()
        self.mock_widget.layout = MagicMock()
        
        # Patch ipywidgets
        self.ipywidgets_patcher = patch('ipywidgets.VBox')
        self.mock_vbox = self.ipywidgets_patcher.start()
        self.mock_vbox.return_value = MagicMock()
        self.mock_vbox.return_value.layout = MagicMock()
        
        # Patch Layout
        self.layout_patcher = patch('ipywidgets.Layout')
        self.mock_layout = self.layout_patcher.start()
    
    def tearDown(self):
        """Tear down test fixtures."""
        self.ipywidgets_patcher.stop()
        self.layout_patcher.stop()
    
    def test_create_form_container_basic(self):
        """Test basic form container creation."""
        # Create a basic form container
        form = create_form_container()
        
        # Verify the form was created with default layout
        self.mock_layout.assert_called()
        self.mock_vbox.assert_called()
        
        # Verify the form has the expected structure
        self.assertIn('container', form)
        self.assertIn('add_item', form)
        self.assertIn('set_layout', form)


if __name__ == '__main__':
    unittest.main()
