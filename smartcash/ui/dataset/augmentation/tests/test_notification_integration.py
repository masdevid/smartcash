"""
File: smartcash/ui/dataset/augmentation/tests/test_notification_integration.py
Deskripsi: Test integrasi notifikasi dan observer untuk augmentasi dataset
"""

import unittest
from unittest.mock import MagicMock, patch
import sys

class TestNotificationIntegration(unittest.TestCase):
    """Test integrasi notifikasi dan observer untuk augmentasi dataset"""
    
    def setUp(self):
        """Setup untuk setiap test case"""
        # Mock modules untuk menghindari import error
        sys.modules['smartcash.ui.utils.constants'] = MagicMock()
        sys.modules['smartcash.ui.utils.alert_utils'] = MagicMock()
        sys.modules['smartcash.common.logger'] = MagicMock()
        sys.modules['smartcash.components.observer'] = MagicMock()
        
        # Mock untuk logger
        self.mock_logger = MagicMock()
        
        # Mock untuk UI components
        self.ui_components = {
            'logger': self.mock_logger,
            'status': MagicMock(),
            'status_panel': MagicMock(),
            'progress_bar': MagicMock(),
            'overall_label': MagicMock(),
            'current_progress': MagicMock(),
            'step_label': MagicMock(),
            'on_process_start': MagicMock(),
            'on_process_complete': MagicMock(),
            'on_process_error': MagicMock(),
            'on_process_stop': MagicMock()
        }
        
        # Mock untuk observer.notify
        self.patcher1 = patch('smartcash.components.observer.notify')
        self.mock_notify = self.patcher1.start()
    
    def tearDown(self):
        """Cleanup setelah setiap test case"""
        self.patcher1.stop()
    
    def test_notify_process_start(self):
        """Test notifikasi proses augmentasi dimulai"""
        from smartcash.ui.dataset.augmentation.handlers.notification_handler import notify_process_start
        
        # Panggil fungsi
        notify_process_start(self.ui_components, "augmentasi", "split train", "train")
        
        # Verifikasi callback dipanggil
        self.ui_components['on_process_start'].assert_called_once_with("augmentation", {
            'split': 'train',
            'display_info': 'split train'
        })
        
        # Verifikasi observer.notify dipanggil
        self.mock_notify.assert_called_once_with('augmentation_started', {
            'ui_components': self.ui_components,
            'split': 'train',
            'display_info': 'split train'
        })
        
        # Verifikasi logger dipanggil
        self.mock_logger.info.assert_called_once()
    
    def test_notify_process_complete(self):
        """Test notifikasi proses augmentasi selesai"""
        from smartcash.ui.dataset.augmentation.handlers.notification_handler import notify_process_complete
        
        # Hasil augmentasi
        result = {
            'status': 'success',
            'generated': 100,
            'split': 'train'
        }
        
        # Panggil fungsi
        notify_process_complete(self.ui_components, result, "split train")
        
        # Verifikasi callback dipanggil
        self.ui_components['on_process_complete'].assert_called_once_with("augmentation", result)
        
        # Verifikasi observer.notify dipanggil
        self.mock_notify.assert_called_once_with('augmentation_completed', {
            'ui_components': self.ui_components,
            'result': result,
            'display_info': 'split train'
        })
        
        # Verifikasi logger dipanggil
        self.mock_logger.info.assert_called_once()
    
    def test_notify_process_error(self):
        """Test notifikasi proses augmentasi error"""
        from smartcash.ui.dataset.augmentation.handlers.notification_handler import notify_process_error
        
        # Panggil fungsi
        notify_process_error(self.ui_components, "Error saat augmentasi")
        
        # Verifikasi callback dipanggil
        self.ui_components['on_process_error'].assert_called_once_with("augmentation", "Error saat augmentasi")
        
        # Verifikasi observer.notify dipanggil
        self.mock_notify.assert_called_once_with('augmentation_error', {
            'ui_components': self.ui_components,
            'error': 'Error saat augmentasi'
        })
        
        # Verifikasi logger dipanggil
        self.mock_logger.error.assert_called_once()
    
    def test_notify_process_stop(self):
        """Test notifikasi proses augmentasi dihentikan"""
        from smartcash.ui.dataset.augmentation.handlers.notification_handler import notify_process_stop
        
        # Panggil fungsi
        notify_process_stop(self.ui_components, "split train")
        
        # Verifikasi callback dipanggil
        self.ui_components['on_process_stop'].assert_called_once_with("augmentation", {
            'display_info': 'split train'
        })
        
        # Verifikasi observer.notify dipanggil
        self.mock_notify.assert_called_once_with('augmentation_stopped', {
            'ui_components': self.ui_components,
            'display_info': 'split train'
        })
        
        # Verifikasi logger dipanggil
        self.mock_logger.info.assert_called_once()

if __name__ == '__main__':
    unittest.main()
