"""
Test cases for the BackboneUIFactory class.

This module contains unit tests for the BackboneUIFactory class to ensure
proper functionality of the singleton pattern, caching, and error handling.
"""

"""
Test cases for the BackboneUIFactory class.

This module contains unit tests for the BackboneUIFactory class to ensure
proper functionality of the singleton pattern, caching, and error handling.
"""

import sys
import os
import unittest
from unittest.mock import MagicMock, patch

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../..')))

from smartcash.ui.model.backbone.backbone_ui_factory import BackboneUIFactory, create_backbone_display

class TestBackboneUIFactory(unittest.TestCase):
    """Test cases for BackboneUIFactory."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Reset the singleton instance before each test
        BackboneUIFactory.reset_cache()
        
        # Create a mock module for testing
        self.mock_module = MagicMock()
        self.mock_module.initialize.return_value = True
        self.mock_module._is_initialized = True
        
    def test_singleton_pattern(self):
        """Test that only one instance of BackboneUIFactory exists."""
        instance1 = BackboneUIFactory()
        instance2 = BackboneUIFactory()
        self.assertIs(instance1, instance2)
        
    @patch('smartcash.ui.model.backbone.backbone_ui_factory.BackboneUIModule')
    def test_create_backbone_module_caching(self, mock_module_class):
        """Test that create_backbone_module caches instances correctly."""
        # Configure the mock
        mock_module = MagicMock()
        mock_module.initialize.return_value = True
        mock_module._is_initialized = True
        mock_module_class.return_value = mock_module
        
        # First call - should create a new instance
        config1 = {"param1": "value1"}
        result1 = BackboneUIFactory.create_backbone_module(config1)
        
        # Second call with same config - should return cached instance
        result2 = BackboneUIFactory.create_backbone_module(config1)
        
        # Verify only one module was created and it was cached
        mock_module_class.assert_called_once()
        self.assertIs(result1, result2)
        
    @patch('smartcash.ui.model.backbone.backbone_ui_factory.BackboneUIModule')
    def test_force_refresh(self, mock_module_class):
        """Test that force_refresh creates a new instance."""
        # Configure the mock
        mock_module1 = MagicMock()
        mock_module1.initialize.return_value = True
        mock_module1._is_initialized = True
        
        mock_module2 = MagicMock()
        mock_module2.initialize.return_value = True
        mock_module2._is_initialized = True
        
        mock_module_class.side_effect = [mock_module1, mock_module2]
        
        # First call - creates first instance
        config = {"param1": "value1"}
        result1 = BackboneUIFactory.create_backbone_module(config)
        
        # Second call with force_refresh=True - should create new instance
        result2 = BackboneUIFactory.create_backbone_module(config, force_refresh=True)
        
        # Verify two different instances were created
        self.assertIsNot(result1, result2)
        self.assertEqual(mock_module_class.call_count, 2)
        
    @patch('smartcash.ui.model.backbone.backbone_ui_factory.get_module_logger')
    @patch('smartcash.ui.model.backbone.backbone_ui_factory.BackboneUIModule')
    def test_error_handling(self, mock_module_class, mock_get_logger):
        """Test error handling in create_backbone_module."""
        # Configure the mock to raise an exception
        mock_module = MagicMock()
        mock_module.initialize.return_value = False  # Simulate failed initialization
        mock_module_class.return_value = mock_module
        
        # Mock the logger
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        # Test that an exception is raised when initialization fails
        with self.assertRaises(RuntimeError):
            BackboneUIFactory.create_backbone_module({})
            
        # Verify the cache was invalidated
        self.assertFalse(BackboneUIFactory._cache_valid)
        
    def test_create_backbone_display(self):
        """Test that create_backbone_display returns a callable."""
        display_fn = create_backbone_display()
        self.assertTrue(callable(display_fn))
        
        # Test that the display function calls create_and_display_backbone
        with patch('smartcash.ui.model.backbone.backbone_ui_factory.BackboneUIFactory.create_and_display_backbone') as mock_display:
            display_fn()
            mock_display.assert_called_once()

if __name__ == '__main__':
    unittest.main()
