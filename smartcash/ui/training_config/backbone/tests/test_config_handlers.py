"""
File: smartcash/ui/training_config/backbone/tests/test_config_handlers.py
Deskripsi: Test untuk config_handlers backbone
"""

import unittest
from unittest.mock import MagicMock, patch
import ipywidgets as widgets

from smartcash.ui.training_config.backbone.handlers.config_handlers import (
    get_default_backbone_config,
    load_config,
    save_config
)

class TestBackboneConfigHandlers(unittest.TestCase):
    """
    Test untuk config_handlers backbone
    """
    
    def setUp(self):
        """
        Setup test
        """
        # Create mock UI components
        self.mock_ui_components = {
            'model_type_dropdown': MagicMock(value='efficient_basic'),
            'backbone_dropdown': MagicMock(value='efficientnet_b4'),
            'use_attention_checkbox': MagicMock(value=True),
            'use_residual_checkbox': MagicMock(value=True),
            'use_ciou_checkbox': MagicMock(value=False),
            'info_panel': MagicMock(),
            'status_panel': MagicMock()
        }
        
        # Create mock config
        self.mock_config = {
            'model': {
                'backbone': 'efficientnet_b4',
                'model_type': 'efficient_basic',
                'use_attention': True,
                'use_residual': True,
                'use_ciou': False
            }
        }
    
    def test_get_default_backbone_config(self):
        """
        Test get_default_backbone_config
        """
        config = get_default_backbone_config()
        self.assertIn('model', config)
        self.assertIn('backbone', config['model'])
        self.assertIn('model_type', config['model'])
        self.assertIn('use_attention', config['model'])
        self.assertIn('use_residual', config['model'])
        self.assertIn('use_ciou', config['model'])
    
    @patch('smartcash.ui.training_config.backbone.handlers.config_handlers.get_config_manager')
    def test_load_config_success(self, mock_get_config):
        """
        Test load_config dengan skenario sukses
        """
        # Setup mocks
        mock_config_manager = MagicMock()
        mock_config_manager.get_module_config.return_value = self.mock_config
        mock_get_config.return_value = mock_config_manager
        
        # Call function
        config = load_config()
        
        # Verify calls
        mock_get_config.assert_called_once()
        mock_config_manager.get_module_config.assert_called_once_with('model')
        self.assertEqual(config, self.mock_config)
    
    @patch('smartcash.ui.training_config.backbone.handlers.config_handlers.get_config_manager')
    @patch('smartcash.ui.training_config.backbone.handlers.config_handlers.get_default_backbone_config')
    def test_load_config_not_found(self, mock_get_default, mock_get_config):
        """
        Test load_config saat konfigurasi tidak ditemukan
        """
        # Setup mocks
        mock_config_manager = MagicMock()
        mock_config_manager.get_module_config.return_value = None
        mock_get_config.return_value = mock_config_manager
        mock_get_default.return_value = self.mock_config
        
        # Call function
        config = load_config()
        
        # Verify calls
        mock_get_config.assert_called_once()
        mock_config_manager.get_module_config.assert_called_once_with('model')
        mock_get_default.assert_called_once()
        self.assertEqual(config, self.mock_config)
    
    @patch('smartcash.ui.training_config.backbone.handlers.config_handlers.get_config_manager')
    @patch('smartcash.ui.training_config.backbone.handlers.sync_logger.update_sync_status_only')
    def test_save_config_success(self, mock_update_status, mock_get_config):
        """
        Test save_config dengan skenario sukses
        """
        # Setup mocks
        mock_config_manager = MagicMock()
        mock_config_manager.save_module_config.return_value = True
        mock_config_manager.get_module_config.return_value = self.mock_config
        mock_get_config.return_value = mock_config_manager
        
        # Mock is_colab_environment dan sync_with_drive
        with patch('smartcash.ui.training_config.backbone.handlers.drive_handlers.is_colab_environment', return_value=False):
            # Call function
            result = save_config(self.mock_config, self.mock_ui_components)
            
            # Verify calls
            mock_get_config.assert_called_once()
            mock_config_manager.save_module_config.assert_called_once_with('model', self.mock_config)
            mock_update_status.assert_any_call(self.mock_ui_components, "Konfigurasi backbone berhasil disimpan", 'success')
            self.assertEqual(result, self.mock_config)
    
    @patch('smartcash.ui.training_config.backbone.handlers.config_handlers.get_config_manager')
    @patch('smartcash.ui.training_config.backbone.handlers.sync_logger.update_sync_status_only')
    def test_save_config_failure(self, mock_update_status, mock_get_config):
        """
        Test save_config dengan skenario gagal
        """
        # Setup mocks
        mock_config_manager = MagicMock()
        mock_config_manager.save_module_config.return_value = False
        mock_get_config.return_value = mock_config_manager
        
        # Call function
        result = save_config(self.mock_config, self.mock_ui_components)
        
        # Verify calls
        mock_get_config.assert_called_once()
        mock_config_manager.save_module_config.assert_called_once_with('model', self.mock_config)
        mock_update_status.assert_any_call(self.mock_ui_components, "Gagal menyimpan konfigurasi backbone", 'error')
        self.assertEqual(result, self.mock_config)
    
    @patch('smartcash.ui.training_config.backbone.handlers.config_handlers.get_config_manager')
    @patch('smartcash.ui.training_config.backbone.handlers.drive_handlers.is_colab_environment')
    @patch('smartcash.ui.training_config.backbone.handlers.drive_handlers.sync_with_drive')
    @patch('smartcash.ui.training_config.backbone.handlers.sync_logger.update_sync_status_only')
    def test_save_config_with_sync(self, mock_update_status, mock_sync, mock_is_colab, mock_get_config):
        """
        Test save_config dengan sinkronisasi
        """
        # Setup mocks
        mock_config_manager = MagicMock()
        mock_config_manager.save_module_config.return_value = True
        mock_config_manager.get_module_config.return_value = self.mock_config
        mock_get_config.return_value = mock_config_manager
        
        mock_is_colab.return_value = True
        mock_sync.return_value = self.mock_config
        
        # Call function
        result = save_config(self.mock_config, self.mock_ui_components)
        
        # Verify calls
        mock_get_config.assert_called_once()
        mock_config_manager.save_module_config.assert_called_once_with('model', self.mock_config)
        mock_is_colab.assert_called_once()
        mock_sync.assert_called_once_with(self.mock_config, self.mock_ui_components)
        mock_update_status.assert_any_call(self.mock_ui_components, "Konfigurasi berhasil disimpan dan disinkronkan", 'success')
        self.assertEqual(result, self.mock_config)

if __name__ == '__main__':
    unittest.main() 