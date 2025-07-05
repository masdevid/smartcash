"""
Tests for the FormContainer component.

These tests focus on the behavior of the form container using real ipywidgets.
"""
import unittest
import sys
import os

# Add the project root to the Python path for direct imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../..')))

# Import ipywidgets
import ipywidgets as widgets

# Import the form_container module
from smartcash.ui.components.form_container import create_form_container, LayoutType, FormItem

class TestFormItem(unittest.TestCase):
    """Test cases for the FormItem class."""
    
    def test_form_item_initialization(self):
        """Test FormItem initialization with default and custom values."""
        # Create a real widget
        widget = widgets.Button(description='Test')
        
        # Test with default values
        form_item = FormItem(widget)
        self.assertEqual(form_item.widget, widget)
        self.assertIsNone(form_item.width)
        self.assertIsNone(form_item.height)
        self.assertEqual(form_item.align_items, 'stretch')
        
        # Test with custom values
        form_item = FormItem(
            widget=widget,
            width='100%',
            height='50px',
            flex=1,
            grid_area='header',
            justify_content='center',
            align_items='center'
        )
        self.assertEqual(form_item.width, '100%')
        self.assertEqual(form_item.height, '50px')
        self.assertEqual(form_item.flex, 1)
        self.assertEqual(form_item.grid_area, 'header')
        self.assertEqual(form_item.justify_content, 'center')
        self.assertEqual(form_item.align_items, 'center')


class TestFormContainerBehavior(unittest.TestCase):
    """Test cases for FormContainer's behavior."""
    
    def test_form_container_creation(self):
        """Test that create_form_container returns the expected structure."""
        # Create a form container
        form = create_form_container()
        
        # Verify the returned object is a dictionary with the expected keys
        self.assertIsInstance(form, dict)
        expected_keys = ['container', 'form_container', 'add_item', 'set_layout']
        for key in expected_keys:
            self.assertIn(key, form)
        
        # Verify the methods are callable
        self.assertTrue(callable(form['add_item']))
        self.assertTrue(callable(form['set_layout']))
    
    def test_add_item_behavior(self):
        """Test that add_item works with real widgets."""
        # Create a form container
        form = create_form_container()
        
        # Create a real widget
        button = widgets.Button(description='Test Button')
        
        # Test adding a widget directly
        try:
            form['add_item'](button)
            success = True
        except Exception as e:
            print(f"Error in add_item: {e}")
            success = False
        
        self.assertTrue(success, "add_item should accept a widget without raising an exception")
    
    def test_add_form_item_behavior(self):
        """Test that add_item works with FormItem objects."""
        # Create a form container
        form = create_form_container()
        
        # Create a real widget wrapped in a FormItem
        button = widgets.Button(description='Test Button')
        form_item = FormItem(button, width='100%', height='50px')
        
        # Test adding a FormItem
        try:
            form['add_item'](form_item)
            success = True
        except Exception as e:
            print(f"Error adding FormItem: {e}")
            success = False
        
        self.assertTrue(success, "add_item should accept a FormItem without raising an exception")
    
    def test_set_layout_behavior(self):
        """Test that set_layout works with different layout types."""
        # Create a form container
        form = create_form_container()
        
        # Test setting layout to ROW
        try:
            form['set_layout'](LayoutType.ROW)
            success = True
        except Exception as e:
            print(f"Error setting ROW layout: {e}")
            success = False
        self.assertTrue(success, "Should be able to set layout to ROW")
        
        # Test setting layout with a string
        try:
            form['set_layout']('GRID')
            success = True
        except Exception as e:
            print(f"Error setting GRID layout: {e}")
            success = False
        self.assertTrue(success, "Should be able to set layout using a string")

if __name__ == '__main__':
    unittest.main()
