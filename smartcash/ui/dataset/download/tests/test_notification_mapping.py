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
from smartcash.components.observer import EventTopics
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
    
    @patch('smartcash.dataset.services.downloader.notification_utils.notify_event')
    def test_service_to_ui_notification_conversion(self, mock_notify_event):
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
        mock_notify_event.assert_called_with(
            self.mock_service, 
            'DOWNLOAD_START', 
            None,  # observer_manager
            message="Memulai download dataset",
            progress=0,
            total_steps=5,
            current_step=1
        )
    
    def test_ui_observers_creation(self):
        """Test bahwa observer UI berhasil dibuat"""
        # Daftarkan observer asli
        observer_manager = register_ui_observers(self.ui_components)
        
        # Verifikasi bahwa observer manager berhasil dibuat
        self.assertIsNotNone(observer_manager)
        
        # Verifikasi bahwa observer manager memiliki atribut yang diharapkan
        self.assertTrue(hasattr(observer_manager, 'auto_register'))
        self.assertTrue(observer_manager.auto_register)

if __name__ == '__main__':
    unittest.main() 