"""
File: smartcash/ui/dataset/split/tests/test_config_handlers.py
Deskripsi: Test suite untuk config handlers
"""

import unittest
from unittest.mock import MagicMock, patch
import ipywidgets as widgets
from smartcash.ui.dataset.split.handlers.config_handlers import (
    load_config,
    save_config,
    load_default_config,
    update_config_from_ui,
    update_ui_from_config,
    get_config_manager_instance
)
from smartcash.ui.dataset.split.components.split_components import create_split_ui

class TestConfigHandlers(unittest.TestCase):
    def setUp(self):
        """Setup test environment"""
        self.ui_components = create_split_ui()
        self.config = load_default_config()
    
    def test_load_config(self):
        """Test loading konfigurasi"""
        with patch('smartcash.ui.dataset.split.handlers.config_handlers.get_config_manager_instance') as mock_get_config_manager_instance:
            mock_manager = MagicMock()
            mock_manager.get_module_config.return_value = self.config
            mock_get_config_manager_instance.return_value = mock_manager
            
            config = load_config()
            self.assertIsInstance(config, dict)
            self.assertIn('data', config)
            self.assertIn('split', config['data'])
    
    def test_load_default_config(self):
        """Test loading default konfigurasi"""
        default_config = load_default_config()
        self.assertIsInstance(default_config, dict)
        self.assertIn('data', default_config)
        self.assertIn('split', default_config['data'])
        self.assertEqual(default_config['data']['split']['train'], 0.7)
        self.assertEqual(default_config['data']['split']['val'], 0.15)
        self.assertEqual(default_config['data']['split']['test'], 0.15)
    
    def test_save_config(self):
        """Test saving konfigurasi"""
        with patch('smartcash.ui.dataset.split.handlers.config_handlers.get_config_manager_instance') as mock_get_config_manager_instance:
            mock_manager = MagicMock()
            mock_manager.save_module_config.return_value = True
            mock_get_config_manager_instance.return_value = mock_manager
            
            success = save_config(self.config)
            self.assertTrue(success)
    
    def test_update_config_from_ui(self):
        """Test update konfigurasi dari UI"""
        # Set nilai UI
        self.ui_components['train_slider'].value = 0.8
        self.ui_components['val_slider'].value = 0.1
        self.ui_components['test_slider'].value = 0.1
        
        # Update konfigurasi
        updated_config = update_config_from_ui(self.config, self.ui_components)
        
        # Verifikasi hasil
        self.assertEqual(updated_config['data']['split']['train'], 0.8)
        self.assertEqual(updated_config['data']['split']['val'], 0.1)
        self.assertEqual(updated_config['data']['split']['test'], 0.1)
    
    def test_update_ui_from_config(self):
        """Test update UI dari konfigurasi"""
        # Update konfigurasi
        self.config['data']['split']['train'] = 0.8
        self.config['data']['split']['val'] = 0.1
        self.config['data']['split']['test'] = 0.1
        
        # Update UI
        update_ui_from_config(self.ui_components, self.config)
        
        # Verifikasi hasil
        self.assertEqual(self.ui_components['train_slider'].value, 0.8)
        self.assertEqual(self.ui_components['val_slider'].value, 0.1)
        self.assertEqual(self.ui_components['test_slider'].value, 0.1)

if __name__ == '__main__':
    unittest.main()
