"""
File: smartcash/ui/pretrained_model/tests/test_pretrained_init.py
Deskripsi: Unit test untuk inisialisasi modul pretrained_model
"""

import unittest
from unittest.mock import patch, MagicMock
import pytest
from smartcash.ui.pretrained_model.pretrained_init import PretrainedInit, initialize_pretrained_model_ui
from smartcash.ui.pretrained_model.components import create_pretrained_main_ui

class TestPretrainedInit(unittest.TestCase):
    def setUp(self):
        """Setup test environment"""
        self.pretrained_init = PretrainedInit()
        self.mock_config = {
            'pretrained_models': {
                'models_dir': '/content/models',
                'drive_models_dir': '/content/drive/MyDrive/SmartCash/models',
                'models': {
                    'yolov5': {'url': 'https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5s.pt'},
                    'efficientnet_b4': {'url': 'https://huggingface.co/timm/efficientnet_b4.ra2_in1k/resolve/main/pytorch_model.bin'}
                }
            }
        }
        self.mock_env = MagicMock()
        
    def test_get_default_config(self):
        """Test pengambilan konfigurasi default"""
        # Buat mock untuk config_handler_class
        with patch.object(self.pretrained_init, 'config_handler_class', new=MagicMock()) as mock_config_handler:
            mock_handler_instance = MagicMock()
            mock_handler_instance.get_default_config.return_value = {'default': 'config'}
            mock_config_handler.return_value = mock_handler_instance
            
            config = self.pretrained_init._get_default_config()
            self.assertEqual(config, {'default': 'config'})
        
    def test_get_critical_components(self):
        """Test komponen kritis yang diperlukan"""
        critical_components = self.pretrained_init._get_critical_components()
        self.assertIn('ui', critical_components)
        self.assertIn('download_sync_button', critical_components)
        self.assertIn('log_output', critical_components)
        self.assertIn('progress_tracker', critical_components)
        
    @patch('smartcash.ui.pretrained_model.components.create_pretrained_main_ui')
    def test_create_ui_components_success(self, mock_create_ui):
        """Test pembuatan UI components berhasil"""
        mock_ui = {'ui': MagicMock(), 'progress_tracker': MagicMock()}
        mock_create_ui.return_value = mock_ui
        
        ui_components = self.pretrained_init._create_ui_components(self.mock_config)
        
        self.assertIn('ui', ui_components)
        self.assertIn('progress_tracker', ui_components)
        self.assertTrue(ui_components['pretrained_model_initialized'])
        self.assertEqual(ui_components['module_name'], 'pretrained_model')
        
    @patch('smartcash.ui.pretrained_model.components.create_pretrained_main_ui')
    def test_create_ui_components_failure(self, mock_create_ui):
        """Test penanganan error saat pembuatan UI"""
        mock_create_ui.side_effect = Exception("Test error")
        
        with self.assertRaises(Exception):
            self.pretrained_init._create_ui_components(self.mock_config)
            
    @patch('smartcash.ui.pretrained_model.handlers.pretrained_handlers.setup_pretrained_handlers')
    def test_setup_module_handlers(self, mock_setup_handlers):
        """Test setup handlers modul"""
        mock_ui = {'ui': MagicMock()}
        mock_setup_handlers.return_value = {'ui': MagicMock(), 'handlers': 'installed'}
        
        result = self.pretrained_init._setup_module_handlers(mock_ui, self.mock_config)
        
        self.assertIn('handlers', result)
        mock_setup_handlers.assert_called_once_with(mock_ui, self.mock_config, None)
        
    @patch('smartcash.ui.pretrained_model.handlers.pretrained_handlers.setup_pretrained_handlers')
    def test_setup_module_handlers_error(self, mock_setup_handlers):
        """Test penanganan error saat setup handlers"""
        mock_ui = {'ui': MagicMock()}
        mock_setup_handlers.side_effect = Exception("Handler error")
        
        result = self.pretrained_init._setup_module_handlers(mock_ui, self.mock_config)
        
        self.assertEqual(result, mock_ui)  # Should return original UI components on error
        
    @patch.object(PretrainedInit, 'initialize')
    def test_public_api(self, mock_initialize):
        """Test public API initialize_pretrained_model_ui"""
        mock_initialize.return_value = {'success': True}
        
        result = initialize_pretrained_model_ui(env=self.mock_env, config=self.mock_config)
        
        self.assertTrue(result['success'])
        mock_initialize.assert_called_once_with(env=self.mock_env, config=self.mock_config)

    # Test tambahan untuk menutupi semua skenario

if __name__ == '__main__':
    unittest.main()
