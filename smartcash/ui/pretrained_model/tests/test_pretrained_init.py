"""
File: smartcash/ui/pretrained_model/tests/test_pretrained_init.py
Deskripsi: Unit test untuk inisialisasi modul pretrained_model
"""

import unittest
import warnings
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
    @patch('smartcash.ui.pretrained_model.pretrained_init.PretrainedInit._clear_existing_widgets')
    @patch('smartcash.common.progress_tracker.create_dual_progress_tracker')
    def test_create_ui_components_success(self, mock_create_tracker, mock_clear_widgets, mock_create_ui):
        """Test pembuatan UI components berhasil"""
        # Setup mock tracker
        mock_tracker = MagicMock()
        mock_create_tracker.return_value = mock_tracker
        
        # Mock return value dari create_pretrained_main_ui
        mock_ui = MagicMock()
        mock_components = {
            'ui': mock_ui,
            'progress_tracker': mock_tracker,
            'log_output': MagicMock(),
            'models_dir_input': MagicMock(),
            'drive_models_dir_input': MagicMock(),
            'yolov5_url_input': MagicMock(),
            'efficientnet_url_input': MagicMock(),
            'module_name': 'pretrained_model',
            'auto_check_enabled': True
        }
        mock_create_ui.return_value = mock_components
        
        # Mock logger untuk menangkap pesan log
        with patch.object(self.pretrained_init, 'logger') as mock_logger:
            # Panggil method yang diuji
            with warnings.catch_warnings():
                # Abaikan peringatan dari ipywidgets
                warnings.simplefilter("ignore", category=DeprecationWarning)
                ui_components = self.pretrained_init._create_ui_components(self.mock_config)
            
            # Verifikasi log dipanggil dengan benar
            mock_logger.info.assert_called_with("✅ Berhasil membuat UI components")
        
        # Verifikasi hasil
        self.assertIn('ui', ui_components)
        self.assertIn('progress_tracker', ui_components)
        self.assertTrue(ui_components['pretrained_model_initialized'])
        self.assertEqual(ui_components['module_name'], 'pretrained_model')
        
        # Verifikasi mock dipanggil dengan benar
        mock_clear_widgets.assert_called_once()
        mock_create_ui.assert_called_once_with(self.mock_config)
        mock_create_tracker.assert_called_once()
        
    @patch('smartcash.ui.pretrained_model.components.create_pretrained_main_ui')
    @patch('smartcash.ui.pretrained_model.pretrained_init.PretrainedInit._clear_existing_widgets')
    @patch('smartcash.common.progress_tracker.create_dual_progress_tracker')
    def test_create_ui_components_failure(self, mock_create_tracker, mock_clear_widgets, mock_create_ui):
        """Test penanganan error saat pembuatan UI"""
        # Setup mock tracker
        mock_tracker = MagicMock()
        mock_create_tracker.return_value = mock_tracker
        
        # Setup mock untuk melempar exception
        test_error = Exception("Test error")
        mock_create_ui.side_effect = test_error
        
        # Mock logger untuk menangkap pesan log
        with patch.object(self.pretrained_init, 'logger') as mock_logger:
            # Panggil method yang diuji dan pastikan exception dilempar
            with self.assertRaises(Exception) as context:
                with warnings.catch_warnings():
                    # Abaikan peringatan dari ipywidgets
                    warnings.simplefilter("ignore", category=DeprecationWarning)
                    self.pretrained_init._create_ui_components(self.mock_config)
            
            # Verifikasi pesan error log
            mock_logger.error.assert_any_call("💥 Gagal membuat UI components: Test error")
            mock_logger.error.assert_any_call("💥 Kesalahan tidak terduga: Test error")
            
            # Verifikasi mock dipanggil dengan benar
            mock_clear_widgets.assert_called_once()
            mock_create_ui.assert_called_once_with(self.mock_config)
            mock_create_tracker.assert_not_called()
            
    @patch('smartcash.ui.pretrained_model.handlers.pretrained_handlers.setup_pretrained_handlers')
    def test_setup_module_handlers(self, mock_setup_handlers):
        """Test setup handlers modul"""
        # Setup mock
        mock_ui = {'ui': MagicMock()}
        expected_handlers = {'ui': MagicMock(), 'handlers': 'installed'}
        mock_setup_handlers.return_value = expected_handlers
        
        # Panggil method yang diuji
        with patch.object(self.pretrained_init, 'logger') as mock_logger:
            result = self.pretrained_init._setup_module_handlers(mock_ui, self.mock_config)
            
            # Verifikasi log
            mock_logger.info.assert_called_with("✅ Berhasil setup handlers modul")
        
        # Verifikasi hasil
        self.assertEqual(result, expected_handlers)
        mock_setup_handlers.assert_called_once_with(mock_ui, self.mock_config, None)
        
    @patch('smartcash.ui.pretrained_model.handlers.pretrained_handlers.setup_pretrained_handlers')
    def test_setup_module_handlers_error(self, mock_setup_handlers):
        """Test penanganan error saat setup handlers"""
        # Setup mock untuk melempar exception
        mock_ui = {'ui': MagicMock()}
        test_error = Exception("Handler error")
        mock_setup_handlers.side_effect = test_error
        
        # Panggil method yang diuji
        with patch.object(self.pretrained_init, 'logger') as mock_logger:
            result = self.pretrained_init._setup_module_handlers(mock_ui, self.mock_config)
            
            # Verifikasi log error
            mock_logger.error.assert_called_with("💥 Gagal setup handlers: Handler error")
        
        # Verifikasi hasil (harus mengembalikan UI asli saat error)
        self.assertEqual(result, mock_ui)
        mock_setup_handlers.assert_called_once_with(mock_ui, self.mock_config, None)
        
    @patch.object(PretrainedInit, 'initialize')
    def test_public_api(self, mock_initialize):
        """Test public API initialize_pretrained_model_ui"""
        # Setup mock
        expected_result = {
            'success': True,
            'ui': MagicMock(),
            'progress_tracker': MagicMock()
        }
        mock_initialize.return_value = expected_result
        
        # Panggil fungsi public API
        result = initialize_pretrained_model_ui(env=self.mock_env, config=self.mock_config)
        
        # Verifikasi hasil
        self.assertEqual(result, expected_result)
        mock_initialize.assert_called_once_with(env=self.mock_env, config=self.mock_config)
        
        # Test dengan parameter opsional
        result = initialize_pretrained_model_ui()
        self.assertEqual(result, expected_result)
        self.assertEqual(mock_initialize.call_count, 2)
        mock_initialize.assert_called_with(env=None, config=None)

    # Test tambahan untuk menutupi semua skenario

if __name__ == '__main__':
    unittest.main()
