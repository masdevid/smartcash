"""
Tests for configuration handling in environment setup handlers.
"""
import unittest
from unittest.mock import MagicMock, patch, mock_open

from smartcash.ui.setup.env_config.handlers.base_config_mixin import BaseConfigMixin
from smartcash.ui.handlers.base_handler import BaseHandler


class TestBaseConfigMixin(unittest.TestCase):
    """Test cases for BaseConfigMixin functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        class TestHandler(BaseHandler, BaseConfigMixin):
            DEFAULT_CONFIG = {'test_key': 'default_value'}
            
            def __init__(self, config_handler=None, **kwargs):
                super().__init__(module_name='test', parent_module='test', **kwargs)
                BaseConfigMixin.__init__(self, config_handler=config_handler, **kwargs)
        
        self.TestHandler = TestHandler
        self.config_handler = MagicMock()
        self.config_handler.get_handler_config.return_value = {'test_key': 'custom_value'}
    
    def test_init_without_config_handler(self):
        """Test initialization without a config handler."""
        handler = self.TestHandler()
        self.assertEqual(handler.config, {'test_key': 'default_value'})
        self.assertIsNone(handler._config_handler)
    
    def test_init_with_config_handler(self):
        """Test initialization with a config handler."""
        handler = self.TestHandler(config_handler=self.config_handler)
        self.config_handler.get_handler_config.assert_called_once_with('test', {'test_key': 'default_value'})
        self.assertEqual(handler.config, {'test_key': 'custom_value'})
    
    def test_get_config_value(self):
        """Test getting a config value."""
        handler = self.TestHandler(config_handler=self.config_handler)
        self.assertEqual(handler.get_config_value('test_key'), 'custom_value')
        self.assertEqual(handler.get_config_value('nonexistent', 'default'), 'default')
    
    def test_set_config_value(self):
        """Test setting a config value."""
        handler = self.TestHandler(config_handler=self.config_handler)
        handler.set_config_value('test_key', 'new_value')
        self.assertEqual(handler.get_config_value('test_key'), 'new_value')
        self.config_handler.update_handler_config.assert_called_once_with('test', {'test_key': 'new_value'})
    
    def test_reset_config(self):
        """Test resetting config to defaults."""
        handler = self.TestHandler(config_handler=self.config_handler)
        handler.set_config_value('test_key', 'modified')
        handler.reset_config()
        self.assertEqual(handler.config, {'test_key': 'default_value'})
        self.config_handler.reset_handler_config.assert_called_once_with('test', {'test_key': 'default_value'})


class TestConfigHandlerIntegration(unittest.TestCase):
    """Integration tests for config handler with BaseConfigMixin."""
    
    def setUp(self):
        """Set up test fixtures."""
        from smartcash.ui.setup.env_config.handlers.config_handler import ConfigHandler
        
        self.config_handler = ConfigHandler(persistence_enabled=False)
        self.config_handler.update_handler_config = MagicMock()
        self.config_handler.reset_handler_config = MagicMock()
    
    def test_config_handler_initialization(self):
        """Test ConfigHandler initialization with BaseConfigMixin."""
        self.assertIn('config_dir', self.config_handler.config)
        self.assertIn('repo_config_dir', self.config_handler.config)
        self.assertTrue(isinstance(self.config_handler.config, dict))
    
    def test_config_handler_defaults(self):
        """Test ConfigHandler default values."""
        self.assertEqual(self.config_handler.get_config_value('auto_sync', True), True)
        self.assertEqual(self.config_handler.get_config_value('max_retries', 3), 3)
    
    def test_config_handler_update(self):
        """Test updating ConfigHandler config."""
        self.config_handler.set_config_value('auto_sync', False)
        self.assertEqual(self.config_handler.get_config_value('auto_sync'), False)
        self.config_handler.update_handler_config.assert_called_once()


if __name__ == '__main__':
    unittest.main()
