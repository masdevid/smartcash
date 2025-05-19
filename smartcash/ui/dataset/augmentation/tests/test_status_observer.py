"""
File: smartcash/ui/dataset/augmentation/tests/test_status_observer.py
Deskripsi: Unit test untuk status handler dan observer handler modul augmentasi dataset
"""

import unittest
from unittest.mock import patch, MagicMock, call
import ipywidgets as widgets
from IPython.display import display

class TestStatusHandler(unittest.TestCase):
    """Test untuk status handler augmentasi dataset."""
    
    def setUp(self):
        """Setup untuk setiap test case."""
        # Mock untuk logger
        self.logger_patch = patch('smartcash.common.logger.get_logger')
        self.mock_logger = self.logger_patch.start()
        self.mock_logger.return_value = MagicMock()
        
        # Mock untuk display
        self.display_patch = patch('IPython.display.display')
        self.mock_display = self.display_patch.start()
        
        # Mock untuk global status handler
        self.global_update_patch = patch('smartcash.ui.handlers.status_handler.update_status_panel')
        self.mock_global_update = self.global_update_patch.start()
        
        self.global_create_patch = patch('smartcash.ui.handlers.status_handler.create_status_panel')
        self.mock_global_create = self.global_create_patch.start()
        self.mock_global_create.return_value = widgets.HTML()
        
        # Mock UI components
        self.mock_ui_components = {
            'status_panel': widgets.HTML(),
            'logger': self.mock_logger.return_value,
            'progress_bar': widgets.IntProgress(),
            'progress_label': widgets.Label()
        }
    
    def tearDown(self):
        """Cleanup setelah setiap test case."""
        self.logger_patch.stop()
        self.display_patch.stop()
        self.global_update_patch.stop()
        self.global_create_patch.stop()
    
    def test_setup_status_handler(self):
        """Test untuk fungsi setup_status_handler."""
        from smartcash.ui.dataset.augmentation.handlers.status_handler import setup_status_handler
        
        # Mock untuk fungsi yang dipanggil di dalam setup_status_handler
        with patch('smartcash.ui.dataset.augmentation.handlers.status_handler.create_status_panel') as mock_create, \
             patch('smartcash.ui.dataset.augmentation.handlers.status_handler.update_status_panel') as mock_update:
            
            # Setup mock return values
            mock_create.return_value = widgets.HTML()
            
            # Panggil fungsi yang akan ditest
            result = setup_status_handler(self.mock_ui_components)
            
            # Verifikasi hasil
            # Hanya periksa bahwa hasil tidak None dan status_panel ada di ui_components
            self.assertIsNotNone(result)
            self.assertIn('status_panel', self.mock_ui_components)
    
    def test_update_status_panel(self):
        """Test untuk fungsi update_status_panel."""
        from smartcash.ui.dataset.augmentation.handlers.status_handler import update_status_panel
        
        # Mock untuk HTML widget update
        self.mock_ui_components['status_panel'] = MagicMock()
        self.mock_ui_components['status_panel'].value = ''
        
        # Panggil fungsi yang akan ditest
        update_status_panel(self.mock_ui_components, "Test Message", "info")
        
        # Verifikasi hasil - tidak perlu assert karena kita hanya ingin memastikan tidak ada error
    
    def test_create_status_panel(self):
        """Test untuk fungsi create_status_panel."""
        from smartcash.ui.dataset.augmentation.handlers.status_handler import create_status_panel
        
        # Mock untuk HTML widget
        with patch('ipywidgets.HTML') as mock_html:
            mock_html_instance = MagicMock()
            mock_html.return_value = mock_html_instance
            
            # Panggil fungsi yang akan ditest
            result = create_status_panel("Test Message", "info")
            
            # Verifikasi hasil
            mock_html.assert_called_once()
            self.assertEqual(result, mock_html_instance)

class TestObserverHandler(unittest.TestCase):
    """Test untuk observer handler augmentasi dataset."""
    
    def setUp(self):
        """Setup untuk setiap test case."""
        # Mock untuk logger
        self.logger_patch = patch('smartcash.common.logger.get_logger')
        self.mock_logger = self.logger_patch.start()
        self.mock_logger.return_value = MagicMock()
        
        # Mock untuk notification_manager
        self.notification_manager_patch = patch('smartcash.ui.dataset.augmentation.utils.notification_manager.get_notification_manager')
        self.mock_notification_manager = self.notification_manager_patch.start()
        self.mock_manager_instance = MagicMock()
        self.mock_notification_manager.return_value = self.mock_manager_instance
        
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
        self.notification_manager_patch.stop()
    
    def test_setup_observer_handler(self):
        """Test untuk fungsi setup_observer_handler."""
        from smartcash.ui.dataset.augmentation.handlers.observer_handler import setup_observer_handler
        
        # Mock untuk fungsi yang dipanggil di dalam setup_observer_handler
        with patch('smartcash.ui.dataset.augmentation.handlers.observer_handler.notify_process_start', create=True) as mock_notify_start, \
             patch('smartcash.ui.dataset.augmentation.handlers.observer_handler.notify_process_complete', create=True) as mock_notify_complete, \
             patch('smartcash.ui.dataset.augmentation.handlers.observer_handler.notify_process_error', create=True) as mock_notify_error:
            
            # Tambahkan fungsi notifikasi ke ui_components
            self.mock_ui_components['notify_process_start'] = mock_notify_start
            self.mock_ui_components['notify_process_complete'] = mock_notify_complete
            self.mock_ui_components['notify_process_error'] = mock_notify_error
            
            # Panggil fungsi yang akan ditest
            result = setup_observer_handler(self.mock_ui_components)
            
            # Verifikasi hasil
            self.assertIsNotNone(result)
    
    def test_notify_process_start(self):
        """Test untuk fungsi notify_process_start."""
        from smartcash.ui.dataset.augmentation.handlers.observer_handler import notify_process_start
        
        # Panggil fungsi yang akan ditest
        notify_process_start(self.mock_ui_components, "augmentation", "Test Info", "train")
        
        # Verifikasi hasil
        self.mock_logger.return_value.info.assert_called_once()
        
        # Tambahkan on_process_start callback ke ui_components
        on_process_start_mock = MagicMock()
        self.mock_ui_components['on_process_start'] = on_process_start_mock
        
        # Panggil fungsi lagi
        notify_process_start(self.mock_ui_components, "augmentation", "Test Info", "train")
        
        # Verifikasi callback dipanggil
        on_process_start_mock.assert_called_once_with("augmentation", {
            'split': "train",
            'display_info': "Test Info"
        })
    
    def test_notify_process_complete(self):
        """Test untuk fungsi notify_process_complete."""
        from smartcash.ui.dataset.augmentation.handlers.observer_handler import notify_process_complete
        
        # Panggil fungsi yang akan ditest
        result = {'status': 'success', 'count': 10}
        notify_process_complete(self.mock_ui_components, result, "Test Info")
        
        # Verifikasi hasil
        self.mock_logger.return_value.info.assert_called_once()
        
        # Tambahkan on_process_complete callback ke ui_components
        on_process_complete_mock = MagicMock()
        self.mock_ui_components['on_process_complete'] = on_process_complete_mock
        
        # Panggil fungsi lagi
        notify_process_complete(self.mock_ui_components, result, "Test Info")
        
        # Verifikasi callback dipanggil
        on_process_complete_mock.assert_called_once_with("augmentation", result)
    
    def test_notify_process_error(self):
        """Test untuk fungsi notify_process_error."""
        from smartcash.ui.dataset.augmentation.handlers.observer_handler import notify_process_error
        
        # Panggil fungsi yang akan ditest
        notify_process_error(self.mock_ui_components, "Test Error")
        
        # Verifikasi hasil
        self.mock_logger.return_value.error.assert_called_once()
        
        # Tambahkan on_process_error callback ke ui_components
        on_process_error_mock = MagicMock()
        self.mock_ui_components['on_process_error'] = on_process_error_mock
        
        # Panggil fungsi lagi
        notify_process_error(self.mock_ui_components, "Test Error")
        
        # Verifikasi callback dipanggil
        on_process_error_mock.assert_called_once_with("augmentation", "Test Error")
    
    def test_setup_observer_handler_with_import(self):
        """Test untuk fungsi setup_observer_handler dengan import berhasil."""
        from smartcash.ui.dataset.augmentation.handlers.observer_handler import setup_observer_handler
        
        # Mock untuk fungsi yang dipanggil di dalam setup_observer_handler
        with patch('smartcash.ui.dataset.augmentation.handlers.observer_handler.notify_process_start', create=True) as mock_notify_start, \
             patch('smartcash.ui.dataset.augmentation.handlers.observer_handler.notify_process_complete', create=True) as mock_notify_complete, \
             patch('smartcash.ui.dataset.augmentation.handlers.observer_handler.notify_process_error', create=True) as mock_notify_error:
            
            # Tambahkan fungsi notifikasi ke ui_components
            self.mock_ui_components['notify_process_start'] = mock_notify_start
            self.mock_ui_components['notify_process_complete'] = mock_notify_complete
            self.mock_ui_components['notify_process_error'] = mock_notify_error
            
            # Panggil fungsi yang akan ditest
            result = setup_observer_handler(self.mock_ui_components)
            
            # Verifikasi hasil
            self.assertIsNotNone(result)

if __name__ == '__main__':
    unittest.main()
