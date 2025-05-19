"""
File: smartcash/ui/dataset/augmentation/tests/test_simple.py
Deskripsi: Unit test sederhana untuk augmentasi dataset
"""

import unittest
from unittest.mock import patch, MagicMock
import ipywidgets as widgets

class TestAugmentationInitializer(unittest.TestCase):
    """Test untuk initializer augmentasi dataset."""
    
    def setUp(self):
        """Setup untuk setiap test case."""
        # Mock untuk logger
        self.logger_patch = patch('smartcash.common.logger.get_logger')
        self.mock_logger = self.logger_patch.start()
        self.mock_logger.return_value = MagicMock()
        
        # Mock untuk setup_handlers
        self.setup_handlers_patch = patch('smartcash.ui.dataset.augmentation.handlers.setup_handlers.setup_augmentation_handlers')
        self.mock_setup_handlers = self.setup_handlers_patch.start()
        
        # Mock untuk IPython.display
        self.display_patch = patch('IPython.display.display')
        self.mock_display = self.display_patch.start()
        
        # Mock UI components
        self.mock_ui_components = {
            'augmentation_container': widgets.VBox(),
            'status_panel': widgets.HTML(),
            'logger': self.mock_logger.return_value
        }
        
        # Setup return value untuk setup_handlers
        self.mock_setup_handlers.return_value = self.mock_ui_components
    
    def tearDown(self):
        """Cleanup setelah setiap test case."""
        self.logger_patch.stop()
        self.setup_handlers_patch.stop()
        self.display_patch.stop()
    
    def test_initialize_augmentation(self):
        """Test untuk fungsi initialize_augmentation."""
        from smartcash.ui.dataset.augmentation.augmentation_initializer import initialize_augmentation
        
        # Panggil fungsi yang akan ditest
        result = initialize_augmentation(self.mock_ui_components)
        
        # Verifikasi hasil
        self.assertEqual(result, self.mock_ui_components)
        self.mock_setup_handlers.assert_called_once()
        
        # Verifikasi bahwa logger ditambahkan ke ui_components
        self.assertIn('logger', result)
        self.assertEqual(result['logger'], self.mock_logger.return_value)

class TestAugmentationUtils(unittest.TestCase):
    """Test untuk utilitas augmentasi dataset."""
    
    def setUp(self):
        """Setup untuk setiap test case."""
        # Mock untuk logger
        self.logger_patch = patch('smartcash.common.logger.get_logger')
        self.mock_logger = self.logger_patch.start()
        self.mock_logger.return_value = MagicMock()
        
        # Mock UI components
        self.mock_ui_components = {
            'status_panel': widgets.HTML(),
            'logger': self.mock_logger.return_value,
            'update_status_panel': MagicMock(),
            'progress_bar': widgets.IntProgress(),
            'progress_label': widgets.Label()
        }
    
    def tearDown(self):
        """Cleanup setelah setiap test case."""
        self.logger_patch.stop()
    
    def test_notification_manager(self):
        """Test untuk NotificationManager class."""
        from smartcash.ui.dataset.augmentation.utils.notification_manager import NotificationManager
        
        # Buat instance NotificationManager
        notification_manager = NotificationManager(self.mock_ui_components)
        
        # Test update_status method
        notification_manager.update_status("info", "Test Message")
        self.mock_ui_components['update_status_panel'].assert_called_once()
        
        # Reset mock
        self.mock_ui_components['update_status_panel'].reset_mock()
        
        # Test notify_process_start method
        notification_manager.notify_process_start("augmentation", "Test Info", "train")
        self.mock_ui_components['update_status_panel'].assert_called_once()
        
        # Reset mock
        self.mock_ui_components['update_status_panel'].reset_mock()
        
        # Test notify_process_complete method
        result = {'status': 'success'}
        notification_manager.notify_process_complete(result, "Test Info")
        self.mock_ui_components['update_status_panel'].assert_called_once()
        
        # Reset mock
        self.mock_ui_components['update_status_panel'].reset_mock()
        
        # Test notify_process_error method
        notification_manager.notify_process_error("Test Error")
        self.mock_ui_components['update_status_panel'].assert_called_once()
    
    def test_get_notification_manager(self):
        """Test untuk fungsi get_notification_manager."""
        from smartcash.ui.dataset.augmentation.utils.notification_manager import get_notification_manager
        
        # Panggil fungsi yang akan ditest
        notification_manager = get_notification_manager(self.mock_ui_components)
        
        # Verifikasi hasil
        self.assertIn('notification_manager', self.mock_ui_components)
        self.assertEqual(notification_manager, self.mock_ui_components['notification_manager'])
        
        # Panggil lagi untuk verifikasi reuse instance
        notification_manager2 = get_notification_manager(self.mock_ui_components)
        self.assertEqual(notification_manager, notification_manager2)

if __name__ == '__main__':
    unittest.main()
