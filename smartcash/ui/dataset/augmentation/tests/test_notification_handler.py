"""
File: smartcash/ui/dataset/augmentation/tests/test_notification_handler.py
Deskripsi: Test untuk notification handler augmentasi dataset
"""

import unittest
from unittest.mock import MagicMock, patch
from typing import Dict, Any

from smartcash.ui.dataset.augmentation.handlers.notification_handler import (
    notify_process_start,
    notify_process_stop,
    notify_process_complete
)

class TestNotificationHandler(unittest.TestCase):
    """Test untuk notification handler augmentasi dataset"""
    
    def setUp(self):
        """Setup untuk setiap test case"""
        # Mock UI components
        self.ui_components = {
            'logger': MagicMock()
        }
        
        # Mock untuk observer.notify
        self.patcher = patch('smartcash.components.observer.notify')
        self.mock_notify = self.patcher.start()
    
    def tearDown(self):
        """Cleanup setelah setiap test case"""
        self.patcher.stop()
    
    def test_notify_process_start(self):
        """Test notifikasi proses augmentasi dimulai"""
        # Panggil fungsi
        notify_process_start(self.ui_components)
        
        # Verifikasi hasil
        self.mock_notify.assert_called_once_with('augmentation_started', {'ui_components': self.ui_components})
    
    def test_notify_process_start_exception(self):
        """Test notifikasi proses augmentasi dimulai dengan exception"""
        # Setup exception
        self.mock_notify.side_effect = Exception("Test exception")
        
        # Panggil fungsi
        notify_process_start(self.ui_components)
        
        # Verifikasi hasil
        self.ui_components['logger'].debug.assert_called_once()
        self.assertTrue("Gagal mengirim notifikasi augmentasi_started" in self.ui_components['logger'].debug.call_args[0][0])
    
    def test_notify_process_stop(self):
        """Test notifikasi proses augmentasi dihentikan"""
        # Panggil fungsi
        notify_process_stop(self.ui_components)
        
        # Verifikasi hasil
        self.mock_notify.assert_called_once_with('augmentation_stopped', {'ui_components': self.ui_components})
    
    def test_notify_process_stop_exception(self):
        """Test notifikasi proses augmentasi dihentikan dengan exception"""
        # Setup exception
        self.mock_notify.side_effect = Exception("Test exception")
        
        # Panggil fungsi
        notify_process_stop(self.ui_components)
        
        # Verifikasi hasil
        self.ui_components['logger'].debug.assert_called_once()
        self.assertTrue("Gagal mengirim notifikasi augmentasi_stopped" in self.ui_components['logger'].debug.call_args[0][0])
    
    def test_notify_process_complete(self):
        """Test notifikasi proses augmentasi selesai"""
        # Setup result
        result = {'total_images': 100, 'augmented_images': 200}
        
        # Panggil fungsi
        notify_process_complete(self.ui_components, result)
        
        # Verifikasi hasil
        self.mock_notify.assert_called_once_with('augmentation_completed', {
            'ui_components': self.ui_components,
            'result': result
        })
    
    def test_notify_process_complete_no_result(self):
        """Test notifikasi proses augmentasi selesai tanpa result"""
        # Panggil fungsi
        notify_process_complete(self.ui_components)
        
        # Verifikasi hasil
        self.mock_notify.assert_called_once_with('augmentation_completed', {
            'ui_components': self.ui_components,
            'result': None
        })
    
    def test_notify_process_complete_exception(self):
        """Test notifikasi proses augmentasi selesai dengan exception"""
        # Setup exception
        self.mock_notify.side_effect = Exception("Test exception")
        
        # Panggil fungsi
        notify_process_complete(self.ui_components)
        
        # Verifikasi hasil
        self.ui_components['logger'].debug.assert_called_once()
        self.assertTrue("Gagal mengirim notifikasi augmentation_completed" in self.ui_components['logger'].debug.call_args[0][0] or
                       "Gagal mengirim notifikasi augmentasi_completed" in self.ui_components['logger'].debug.call_args[0][0])

if __name__ == '__main__':
    unittest.main()
