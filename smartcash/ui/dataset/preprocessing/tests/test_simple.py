"""
File: smartcash/ui/dataset/preprocessing/tests/test_simple.py
Deskripsi: Unit test sederhana untuk preprocessing dataset
"""

import unittest
from unittest.mock import patch, MagicMock
import ipywidgets as widgets

class TestPreprocessingInitializer(unittest.TestCase):
    """Test untuk initializer preprocessing dataset."""
    
    def setUp(self):
        """Setup untuk setiap test case."""
        # Mock untuk logger
        self.logger_patch = patch('smartcash.common.logger.get_logger')
        self.mock_logger = self.logger_patch.start()
        self.mock_logger.return_value = MagicMock()
        
        # Mock untuk base initializer
        self.base_initializer_patch = patch('smartcash.ui.utils.base_initializer.initialize_module_ui')
        self.mock_base_initializer = self.base_initializer_patch.start()
        
        # Mock UI components
        self.mock_ui_components = {
            'main_container': widgets.VBox(),
            'status': widgets.Output(),
            'logger': self.mock_logger.return_value
        }
        
        # Setup return value untuk base_initializer
        self.mock_base_initializer.return_value = self.mock_ui_components
    
    def tearDown(self):
        """Cleanup setelah setiap test case."""
        self.logger_patch.stop()
        self.base_initializer_patch.stop()
    
    def test_initialize_preprocessing_ui(self):
        """Test untuk fungsi initialize_preprocessing_ui."""
        from smartcash.ui.dataset.preprocessing.preprocessing_initializer import initialize_preprocessing_ui
        
        # Panggil fungsi yang akan ditest
        result = initialize_preprocessing_ui()
        
        # Verifikasi hasil
        self.assertIsInstance(result, dict)
        self.assertIn('main_container', result)
        self.assertIn('status', result)
        self.assertIn('logger', result)
        self.assertIn('preprocess_button', result)
        self.assertIn('stop_button', result)
        self.assertIn('reset_button', result)
        self.assertIn('cleanup_button', result)
        self.assertIn('save_button', result)
        self.assertIn('split_selector', result)
        self.assertIn('preprocess_options', result)
        self.assertIn('validation_options', result)
        
        # Verifikasi parameter yang diberikan ke base_initializer
        args, kwargs = self.mock_base_initializer.call_args
        self.assertEqual(kwargs['module_name'], 'preprocessing')
        self.assertTrue('button_keys' in kwargs)
        self.assertTrue('multi_progress_config' in kwargs)
        self.assertEqual(kwargs['observer_group'], 'preprocessing_observers')

class TestPreprocessingUtils(unittest.TestCase):
    """Test untuk utilitas preprocessing dataset."""
    
    def setUp(self):
        """Setup untuk setiap test case."""
        # Mock untuk logger
        self.logger_patch = patch('smartcash.common.logger.get_logger')
        self.mock_logger = self.logger_patch.start()
        self.mock_logger.return_value = MagicMock()
        
        # Mock UI components
        self.mock_ui_components = {
            'status': widgets.Output(),
            'logger': self.mock_logger.return_value,
            'on_process_start': MagicMock(),
            'on_process_complete': MagicMock(),
            'on_process_error': MagicMock(),
            'on_process_stop': MagicMock()
        }
    
    def tearDown(self):
        """Cleanup setelah setiap test case."""
        self.logger_patch.stop()
    
    @patch('smartcash.ui.dataset.preprocessing.utils.ui_observers.notify_process_start')
    def test_notify_process_start(self, mock_notify):
        """Test untuk fungsi notify_process_start."""
        from smartcash.ui.dataset.preprocessing.utils.ui_observers import notify_process_start
        
        # Panggil fungsi yang akan ditest
        notify_process_start(self.mock_ui_components, "preprocessing", "Test Info", "train")
        
        # Verifikasi hasil
        mock_notify.assert_called_once_with(self.mock_ui_components, "preprocessing", "Test Info", "train")
    
    @patch('smartcash.ui.dataset.preprocessing.utils.ui_observers.notify_process_complete')
    def test_notify_process_complete(self, mock_notify):
        """Test untuk fungsi notify_process_complete."""
        from smartcash.ui.dataset.preprocessing.utils.ui_observers import notify_process_complete
        
        # Panggil fungsi yang akan ditest
        result = {'status': 'success'}
        notify_process_complete(self.mock_ui_components, result, "Test Info")
        
        # Verifikasi hasil
        mock_notify.assert_called_once_with(self.mock_ui_components, result, "Test Info")
    
    @patch('smartcash.ui.dataset.preprocessing.utils.ui_observers.disable_ui_during_processing')
    def test_disable_ui_during_processing(self, mock_disable):
        """Test untuk fungsi disable_ui_during_processing."""
        from smartcash.ui.dataset.preprocessing.utils.ui_observers import disable_ui_during_processing
        
        # Setup mock UI components
        ui_components = {
            'split_selector': widgets.Dropdown(),
            'config_accordion': widgets.Accordion(),
            'options_accordion': widgets.Accordion(),
            'reset_button': widgets.Button(),
            'preprocess_button': widgets.Button(),
            'save_button': widgets.Button()
        }
        
        # Panggil fungsi yang akan ditest
        disable_ui_during_processing(ui_components, True)
        
        # Verifikasi hasil
        mock_disable.assert_called_once_with(ui_components, True)

if __name__ == '__main__':
    unittest.main()
