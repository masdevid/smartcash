"""
File: smartcash/ui/dataset/download/tests/test_notification_mapping.py
Deskripsi: Test suite untuk mapping notifikasi antara service dan UI download dataset
"""

import unittest
from unittest.mock import MagicMock, patch, call
import sys
import os
from typing import Dict, Any
from ipywidgets import Layout

# Tambahkan path ke sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))

from smartcash.ui.dataset.download.utils.notification_manager import (
    notify_log, notify_progress, DownloadUIEvents
)
from smartcash.dataset.services.downloader.notification_utils import (
    notify_service_event, notify_download
)
from smartcash.components.observer import ObserverManager, EventTopics
from smartcash.ui.dataset.download.utils.ui_observers import register_ui_observers

class TestNotificationMapping(unittest.TestCase):
    """Test suite untuk mapping notifikasi antara service dan UI download dataset"""
    
    def setUp(self):
        """Setup test environment"""
        # Mock UI components
        self.ui_components = {
            'log_output': MagicMock(),
            'progress_bar': MagicMock(),
            'overall_label': MagicMock(),
            'step_label': MagicMock(),
            'current_progress': MagicMock(),
            'progress_container': MagicMock(layout=MagicMock()),
            'log_accordion': MagicMock(selected_index=0),
            'download_running': False
        }
        
        # Setup mock untuk progress bar dan labels
        for key in ['progress_bar', 'overall_label', 'step_label', 'current_progress']:
            self.ui_components[key].layout = Layout()
            self.ui_components[key].value = 0
            self.ui_components[key].description = ""
        
        # Mock service
        self.mock_service = MagicMock()
        
    @patch('smartcash.ui.dataset.download.utils.notification_manager.notify')
    def test_log_notification_mapping(self, mock_notify):
        """Test mapping notifikasi log antara service dan UI"""
        # Kirim notifikasi log langsung
        notify_log(
            sender=self.mock_service,
            message="Info log message",
            level="info"
        )
        
        # Verifikasi notifikasi dikirim dengan benar
        mock_notify.assert_called_with(
            DownloadUIEvents.LOG_INFO,
            self.mock_service,
            message="Info log message",
            level="info"
        )
    
    @patch('smartcash.ui.dataset.download.utils.notification_manager.notify')
    def test_progress_notification_mapping(self, mock_notify):
        """Test mapping notifikasi progress antara service dan UI"""
        # Kirim notifikasi progress langsung
        notify_progress(
            sender=self.mock_service,
            event_type="update",
            progress=50,
            total=100,
            message="Download progress 50%"
        )
        
        # Verifikasi notifikasi dikirim dengan benar
        mock_notify.assert_called_with(
            DownloadUIEvents.PROGRESS_UPDATE,
            self.mock_service,
            progress=50,
            total=100,
            message="Download progress 50%",
            percentage=50
        )
    
    @patch('smartcash.ui.dataset.download.utils.notification_manager.notify')
    def test_step_progress_notification_mapping(self, mock_notify):
        """Test mapping notifikasi step progress antara service dan UI"""
        # Kirim notifikasi step progress langsung
        notify_progress(
            sender=self.mock_service,
            event_type="update",
            progress=50,
            total=100,
            message="Step 2 progress",
            step="metadata",
            current_step=2,
            total_steps=5
        )
        
        # Verifikasi notifikasi step progress dikirim dengan benar
        mock_notify.assert_any_call(
            DownloadUIEvents.STEP_PROGRESS_UPDATE,
            self.mock_service,
            step="metadata",
            step_message="Step 2 progress",
            step_progress=50,
            step_total=100,
            current_step=2,
            total_steps=5
        )
    
    @patch('smartcash.components.observer.notify')
    def test_service_to_ui_notification_conversion(self, mock_notify):
        """Test konversi notifikasi dari service ke format UI"""
        # Kirim notifikasi dari service
        notify_service_event(
            "download", 
            "start", 
            self.mock_service, 
            None,  # Tidak gunakan observer_manager
            message="Memulai download dataset"
        )
        
        # Verifikasi konversi notifikasi
        mock_notify.assert_called_with(
            EventTopics.DOWNLOAD_START, 
            self.mock_service, 
            message="Memulai download dataset"
        )
    
    def test_ui_notification_handler_response(self):
        """Test respons handler notifikasi UI terhadap notifikasi yang diterima"""
        # Daftarkan observer asli
        observer_manager = register_ui_observers(self.ui_components)
        
        # Tambahkan observer manager ke ui_components
        self.ui_components['observer_manager'] = observer_manager
        
        # Dapatkan log observer dari observer_manager
        log_observer = None
        for observer in observer_manager._observers:
            if observer.name == "log_observer":
                log_observer = observer
                break
        
        self.assertIsNotNone(log_observer, "Log observer tidak ditemukan")
        
        # Panggil callback log observer langsung
        log_observer.callback(
            DownloadUIEvents.LOG_INFO,
            self,
            message="Test log message",
            level="info"
        )
        
        # Verifikasi UI diupdate
        self.ui_components['log_output'].append_stdout.assert_called()
        
        # Reset mock
        self.ui_components['log_output'].reset_mock()
        
        # Dapatkan progress observer dari observer_manager
        progress_observer = None
        for observer in observer_manager._observers:
            if observer.name == "progress_observer":
                progress_observer = observer
                break
        
        self.assertIsNotNone(progress_observer, "Progress observer tidak ditemukan")
        
        # Panggil callback progress observer langsung
        progress_observer.callback(
            DownloadUIEvents.PROGRESS_UPDATE,
            self,
            progress=75,
            total=100,
            message="Test progress message"
        )
        
        # Verifikasi progress bar diupdate
        self.assertEqual(self.ui_components['progress_bar'].value, 75)
        self.assertEqual(self.ui_components['progress_bar'].description, "Progress: 75%")

if __name__ == '__main__':
    unittest.main() 