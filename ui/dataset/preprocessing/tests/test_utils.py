"""
File: smartcash/ui/dataset/preprocessing/tests/test_utils.py
Deskripsi: Unit test untuk utilitas preprocessing dataset
"""

import unittest
from unittest.mock import patch, MagicMock, call
import ipywidgets as widgets
from IPython.display import display

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
            'on_process_stop': MagicMock(),
            'split_selector': widgets.Dropdown(
                options=['All Splits', 'Train Only', 'Validation Only', 'Test Only'],
                value='All Splits'
            ),
            'config_accordion': widgets.Accordion(),
            'options_accordion': widgets.Accordion(),
            'reset_button': widgets.Button(description='Reset'),
            'preprocess_button': widgets.Button(description='Preprocess'),
            'save_button': widgets.Button(description='Save'),
            'update_status_panel': MagicMock()
        }
    
    def tearDown(self):
        """Cleanup setelah setiap test case."""
        self.logger_patch.stop()
    
    def test_notify_process_start(self):
        """Test untuk fungsi notify_process_start."""
        from smartcash.ui.dataset.preprocessing.utils.ui_observers import notify_process_start
        
        # Panggil fungsi yang akan ditest
        notify_process_start(self.mock_ui_components, "preprocessing", "Train Split", "train")
        
        # Verifikasi hasil
        self.mock_logger.return_value.info.assert_called_once()
        self.mock_ui_components['on_process_start'].assert_called_once_with("preprocessing", {
            'split': "train",
            'display_info': "Train Split"
        })
    
    def test_notify_process_complete(self):
        """Test untuk fungsi notify_process_complete."""
        from smartcash.ui.dataset.preprocessing.utils.ui_observers import notify_process_complete
        
        # Panggil fungsi yang akan ditest
        result = {'status': 'success'}
        notify_process_complete(self.mock_ui_components, result, "Train Split")
        
        # Verifikasi hasil
        self.mock_logger.return_value.info.assert_called_once()
        self.mock_ui_components['on_process_complete'].assert_called_once_with("preprocessing", result)
    
    def test_notify_process_error(self):
        """Test untuk fungsi notify_process_error."""
        from smartcash.ui.dataset.preprocessing.utils.ui_observers import notify_process_error
        
        # Panggil fungsi yang akan ditest
        notify_process_error(self.mock_ui_components, "Test error")
        
        # Verifikasi hasil
        self.mock_logger.return_value.error.assert_called_once()
        self.mock_ui_components['on_process_error'].assert_called_once_with("preprocessing", "Test error")
    
    def test_notify_process_stop(self):
        """Test untuk fungsi notify_process_stop."""
        from smartcash.ui.dataset.preprocessing.utils.ui_observers import notify_process_stop
        
        # Panggil fungsi yang akan ditest
        notify_process_stop(self.mock_ui_components, "Train Split")
        
        # Verifikasi hasil
        self.mock_logger.return_value.warning.assert_called_once()
        self.mock_ui_components['on_process_stop'].assert_called_once_with("preprocessing", {
            'display_info': "Train Split"
        })
    
    def test_disable_ui_during_processing(self):
        """Test untuk fungsi disable_ui_during_processing."""
        from smartcash.ui.dataset.preprocessing.utils.ui_observers import disable_ui_during_processing
        
        # Panggil fungsi yang akan ditest
        disable_ui_during_processing(self.mock_ui_components, True)
        
        # Verifikasi hasil
        self.assertTrue(self.mock_ui_components['split_selector'].disabled)
        self.assertTrue(self.mock_ui_components['config_accordion'].disabled)
        self.assertTrue(self.mock_ui_components['options_accordion'].disabled)
        self.assertTrue(self.mock_ui_components['reset_button'].disabled)
        self.assertTrue(self.mock_ui_components['preprocess_button'].disabled)
        self.assertTrue(self.mock_ui_components['save_button'].disabled)
        
        # Test enable
        disable_ui_during_processing(self.mock_ui_components, False)
        
        # Verifikasi hasil
        self.assertFalse(self.mock_ui_components['split_selector'].disabled)
        self.assertFalse(self.mock_ui_components['config_accordion'].disabled)
        self.assertFalse(self.mock_ui_components['options_accordion'].disabled)
        self.assertFalse(self.mock_ui_components['reset_button'].disabled)
        self.assertFalse(self.mock_ui_components['preprocess_button'].disabled)
        self.assertFalse(self.mock_ui_components['save_button'].disabled)

class TestNotificationManager(unittest.TestCase):
    """Test untuk notification manager preprocessing dataset."""
    
    def setUp(self):
        """Setup untuk setiap test case."""
        # Mock untuk logger
        self.logger_patch = patch('smartcash.common.logger.get_logger')
        self.mock_logger = self.logger_patch.start()
        self.mock_logger.return_value = MagicMock()
        
        # Mock untuk display
        self.display_patch = patch('IPython.display.display')
        self.mock_display = self.display_patch.start()
        
        # Mock untuk create_status_indicator
        self.status_indicator_patch = patch('smartcash.ui.utils.alert_utils.create_status_indicator')
        self.mock_status_indicator = self.status_indicator_patch.start()
        
        # Mock UI components
        self.mock_ui_components = {
            'status': widgets.Output(),
            'logger': self.mock_logger.return_value,
            'update_status_panel': MagicMock(),
            'update_progress': MagicMock()
        }
    
    def tearDown(self):
        """Cleanup setelah setiap test case."""
        self.logger_patch.stop()
        self.display_patch.stop()
        self.status_indicator_patch.stop()
    
    def test_notification_manager_update_status(self):
        """Test untuk metode update_status dari PreprocessingNotificationManager."""
        from smartcash.ui.dataset.preprocessing.utils.notification_manager import PreprocessingNotificationManager
        
        # Buat instance notification manager
        manager = PreprocessingNotificationManager(self.mock_ui_components)
        
        # Panggil metode yang akan ditest
        with patch('IPython.display.display') as mock_display_local:
            manager.update_status("success", "Test message")
            
            # Verifikasi hasil
            mock_display_local.assert_called_once()
        
        self.mock_status_indicator.assert_called_once_with("success", "Test message")
        self.mock_ui_components['update_status_panel'].assert_called_once_with(self.mock_ui_components, "success", "Test message")
        self.mock_logger.return_value.info.assert_called_once()
    
    def test_notification_manager_notify_progress(self):
        """Test untuk metode notify_progress dari PreprocessingNotificationManager."""
        from smartcash.ui.dataset.preprocessing.utils.notification_manager import PreprocessingNotificationManager
        
        # Buat instance notification manager
        manager = PreprocessingNotificationManager(self.mock_ui_components)
        
        # Panggil metode yang akan ditest
        manager.notify_progress(1, 5, "Test progress")
        
        # Verifikasi hasil
        self.mock_ui_components['update_progress'].assert_called_once_with(1, 5, "Test progress")
        self.mock_logger.return_value.info.assert_called_once()
    
    def test_get_notification_manager(self):
        """Test untuk fungsi get_notification_manager."""
        from smartcash.ui.dataset.preprocessing.utils.notification_manager import get_notification_manager, PreprocessingNotificationManager
        
        # Panggil fungsi yang akan ditest
        manager = get_notification_manager(self.mock_ui_components)
        
        # Verifikasi hasil
        self.assertIsInstance(manager, PreprocessingNotificationManager)
        self.assertEqual(manager, self.mock_ui_components['notification_manager'])
        
        # Panggil lagi untuk mendapatkan instance yang sama
        manager2 = get_notification_manager(self.mock_ui_components)
        
        # Verifikasi hasil
        self.assertEqual(manager, manager2)

if __name__ == '__main__':
    unittest.main()
