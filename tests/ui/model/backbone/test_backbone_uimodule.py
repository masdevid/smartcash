"""
File: tests/ui/model/backbone/test_backbone_uimodule.py
Tests for backbone UI module functionality.
"""
import unittest
from unittest.mock import MagicMock, patch, ANY, PropertyMock
from IPython.display import display

# Mock the display function to capture UI output
class MockDisplay:
    def __init__(self):
        self.displayed = []
    
    def __call__(self, obj):
        self.displayed.append(obj)
        return obj

class TestBackboneUIModule(unittest.TestCase):
    """Test cases for BackboneUIModule and related functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Patch necessary modules and functions
        self.display_patcher = patch('IPython.display.display', new_callable=MockDisplay)
        self.mock_display = self.display_patcher.start()
        
        # Patch get_ipython
        self.ipython_patcher = patch('IPython.get_ipython')
        self.mock_get_ipython = self.ipython_patcher.start()
        
        # Patch clear_output
        self.clear_output_patcher = patch('IPython.display.clear_output')
        self.mock_clear_output = self.clear_output_patcher.start()
        
        # Import the module under test after patching
        from smartcash.ui.model.backbone.backbone_uimodule import (
            BackboneUIModule,
            initialize_backbone_ui,
            get_backbone_components,
            _backbone_uimodule_instance
        )
        
        self.module = BackboneUIModule
        self.initialize_fn = initialize_backbone_ui
        self.get_components_fn = get_backbone_components
        self.instance_var = _backbone_uimodule_instance
        
        # Create a mock UI components dictionary
        self.mock_ui_components = {
            'main_container': MagicMock(),
            'config_panel': MagicMock()
        }
        
        # Configure the mock UI components
        self.mock_ui_components['main_container'].show.return_value = "mock_widget"
    
    def tearDown(self):
        """Clean up after tests."""
        self.display_patcher.stop()
        self.ipython_patcher.stop()
        self.clear_output_patcher.stop()
    
    def test_initialize_backbone_ui_display_true(self):
        """Test initialize_backbone_ui with display=True."""
        # Mock the module and its methods
        with patch('smartcash.ui.model.backbone.backbone_uimodule.get_backbone_uimodule') as mock_get_module:
            # Create a mock module instance
            mock_module = MagicMock()
            mock_module.get_ui_components.return_value = self.mock_ui_components
            mock_module.get_backbone_status.return_value = {'status': 'ready'}
            mock_get_module.return_value = mock_module
            
            # Mock the get_ipython return value
            mock_ipython = MagicMock()
            self.mock_get_ipython.return_value = mock_ipython
            
            # Call the function with display=True
            result = self.initialize_fn(display=True)
            
            # Verify the module was retrieved
            mock_get_module.assert_called_once()
            
            # Verify the UI components were retrieved
            mock_module.get_ui_components.assert_called_once()
            
            # Verify clear_output was called
            self.mock_clear_output.assert_called_once_with(wait=True)
            
            # Verify the main container's show() method was called
            self.mock_ui_components['main_container'].show.assert_called_once()
            
            # Verify the result is None when display=True
            self.assertIsNone(result)
            
            # Verify display was called with the widget
            self.assertEqual(len(self.mock_display.displayed), 1)
            self.assertEqual(self.mock_display.displayed[0], "mock_widget")
    
    def test_initialize_backbone_ui_display_false(self):
        """Test initialize_backbone_ui with display=False."""
        # Mock the module and its methods
        with patch('smartcash.ui.model.backbone.backbone_uimodule.get_backbone_uimodule') as mock_get_module:
            # Create a mock module instance
            mock_module = MagicMock()
            mock_module.get_ui_components.return_value = self.mock_ui_components
            mock_module.get_backbone_status.return_value = {'status': 'ready'}
            mock_get_module.return_value = mock_module
            
            # Call the function with display=False
            result = self.initialize_fn(display=False)
            
            # Verify the module was retrieved with default kwargs
            mock_get_module.assert_called_once()
            
            # Verify the UI components were retrieved
            mock_module.get_ui_components.assert_called_once()
            
            # Verify clear_output was not called
            self.mock_clear_output.assert_not_called()
            
            # Verify the main container's show() method was not called
            self.mock_ui_components['main_container'].show.assert_not_called()
            
            # Verify the result contains the expected keys
            self.assertIsInstance(result, dict)
            self.assertIn('success', result)
            self.assertIn('module', result)
            self.assertIn('ui_components', result)
            self.assertIn('status', result)
            self.assertTrue(result['success'])
    
    def test_initialize_backbone_ui_error_handling(self):
        """Test error handling in initialize_backbone_ui."""
        # Mock get_backbone_uimodule to raise an exception
        with patch('smartcash.ui.model.backbone.backbone_uimodule.get_backbone_uimodule') as mock_get_module:
            mock_get_module.side_effect = Exception("Test error")
            
            # Call the function
            result = self.initialize_fn(display=False)
            
            # Verify the result indicates failure
            self.assertIsInstance(result, dict)
            self.assertIn('success', result)
            self.assertFalse(result['success'])
            self.assertIn('error', result)
            self.assertEqual(result['error'], 'Test error')
            self.assertEqual(result['ui_components'], {})
            self.assertEqual(result['status'], {})
    
    def test_get_backbone_components(self):
        """Test get_backbone_components function."""
        # Mock get_backbone_uimodule
        with patch('smartcash.ui.model.backbone.backbone_uimodule.get_backbone_uimodule') as mock_get_module:
            # Create a mock module instance
            mock_module = MagicMock()
            mock_module.get_ui_components.return_value = self.mock_ui_components
            mock_get_module.return_value = mock_module
            
            # Call the function
            result = self.get_components_fn()
            
            # Verify the module was retrieved with auto_initialize=False
            mock_get_module.assert_called_once_with(auto_initialize=False)
            
            # Verify the UI components were retrieved
            mock_module.get_ui_components.assert_called_once()
            
            # Verify the result matches the mock UI components
            self.assertEqual(result, self.mock_ui_components)
    
    def test_get_backbone_components_error_handling(self):
        """Test error handling in get_backbone_components."""
        # Mock get_backbone_uimodule to raise an exception
        with patch('smartcash.ui.model.backbone.backbone_uimodule.get_backbone_uimodule') as mock_get_module:
            mock_get_module.side_effect = Exception("Test error")
            
            # Call the function
            result = self.get_components_fn()
            
            # Verify an empty dict is returned on error
            self.assertEqual(result, {})


class TestCell32Backbone(unittest.TestCase):
    """Test cases for cell_3_2_backbone.py."""
    
    def test_cell_imports_and_runs(self):
        """Test that the cell imports and runs without errors."""
        # Patch the initialize_backbone_ui function
        with patch('smartcash.ui.model.backbone.backbone_uimodule.initialize_backbone_ui') as mock_init:
            # Import the cell module
            import smartcash.ui.cells.cell_3_2_backbone as cell
            
            # Verify initialize_backbone_ui was called with display=True
            mock_init.assert_called_once_with(display=True)
            
            # Verify the module has the expected attributes
            self.assertTrue(hasattr(cell, 'initialize_backbone_ui'))


if __name__ == '__main__':
    unittest.main()
