"""
File: smartcash/tests/ui/test_download_integration.py
Deskripsi: Test untuk integrasi antara UI download button dan download service
"""

import unittest
from unittest.mock import MagicMock, patch
import sys
import os
from pathlib import Path

# Tambahkan root project ke sys.path untuk import
sys.path.append(str(Path(__file__).parent.parent.parent))

from smartcash.ui.dataset.downloader.handlers.download_handler import _execute_download
from smartcash.dataset.downloader import get_downloader_instance


class MockDownloader:
    """Mock downloader untuk testing"""
    
    def __init__(self, success=True):
        self.success = success
        self.progress_callback = None
        self.download_called = False
    
    def set_progress_callback(self, callback):
        """Set progress callback"""
        self.progress_callback = callback
    
    def download_dataset(self):
        """Mock download dataset"""
        self.download_called = True
        
        # Simulasi progress updates jika callback diset
        if self.progress_callback:
            self.progress_callback("download", 0, 100, "Memulai download")
            self.progress_callback("download", 50, 100, "Setengah jalan")
            self.progress_callback("download", 100, 100, "Selesai")
        
        if self.success:
            return {
                'status': 'success',
                'stats': {
                    'total_images': 1000
                }
            }
        else:
            return {
                'status': 'error',
                'message': 'Simulasi error download'
            }


class TestDownloadIntegration(unittest.TestCase):
    """Test integrasi antara UI download button dan download service"""
    
    def setUp(self):
        """Setup untuk setiap test case"""
        # Mock UI components
        self.ui_components = {
            'progress_tracker': MagicMock(),
            'logger': MagicMock(),
            'download_button': MagicMock(),
            'config_handler': MagicMock()
        }
        
        # Mock config
        self.config = {
            'data': {
                'roboflow': {
                    'workspace': 'smartcash-wo2us',
                    'project': 'rupiah-emisi-2022',
                    'version': '3',
                    'api_key': 'test-api-key'
                }
            },
            'download': {
                'validate_download': True,
                'backup_existing': False
            }
        }
        
        # Mock logger
        self.logger = MagicMock()
    
    @patch('smartcash.ui.dataset.downloader.handlers.download_handler.get_downloader_instance')
    def test_download_success(self, mock_get_downloader):
        """Test download berhasil"""
        # Setup mock downloader
        mock_downloader = MockDownloader(success=True)
        mock_get_downloader.return_value = mock_downloader
        
        # Execute download
        _execute_download(self.ui_components, self.config, self.logger)
        
        # Verifikasi downloader dipanggil dengan benar
        mock_get_downloader.assert_called_once()
        self.assertTrue(mock_downloader.download_called)
        
        # Verifikasi progress tracker diupdate
        self.ui_components['progress_tracker'].show.assert_called_once()
        self.ui_components['progress_tracker'].complete.assert_called_once()
        
        # Verifikasi logger dipanggil
        self.logger.info.assert_called()
        self.logger.success.assert_called()
    
    @patch('smartcash.ui.dataset.downloader.handlers.download_handler.get_downloader_instance')
    def test_download_failure(self, mock_get_downloader):
        """Test download gagal"""
        # Setup mock downloader
        mock_downloader = MockDownloader(success=False)
        mock_get_downloader.return_value = mock_downloader
        
        # Execute download
        _execute_download(self.ui_components, self.config, self.logger)
        
        # Verifikasi downloader dipanggil dengan benar
        mock_get_downloader.assert_called_once()
        self.assertTrue(mock_downloader.download_called)
        
        # Verifikasi progress tracker diupdate
        self.ui_components['progress_tracker'].show.assert_called_once()
        self.ui_components['progress_tracker'].error.assert_called_once()
        
        # Verifikasi logger dipanggil
        self.logger.info.assert_called()
        self.logger.error.assert_called()
    
    @patch('smartcash.ui.dataset.downloader.handlers.download_handler.get_downloader_instance')
    def test_download_service_creation_error(self, mock_get_downloader):
        """Test error saat membuat download service"""
        # Setup mock downloader untuk mengembalikan None
        mock_get_downloader.return_value = None
        
        # Execute download
        _execute_download(self.ui_components, self.config, self.logger)
        
        # Verifikasi progress tracker diupdate
        self.ui_components['progress_tracker'].error.assert_called_once()
        
        # Verifikasi logger dipanggil
        self.logger.info.assert_called()
    
    @patch('smartcash.ui.dataset.downloader.handlers.download_handler.get_downloader_instance')
    def test_download_service_exception(self, mock_get_downloader):
        """Test exception saat membuat download service"""
        # Setup mock downloader untuk raise exception
        mock_get_downloader.side_effect = Exception("Test exception")
        
        # Execute download
        _execute_download(self.ui_components, self.config, self.logger)
        
        # Verifikasi progress tracker diupdate
        self.ui_components['progress_tracker'].error.assert_called_once()
        
        # Verifikasi logger dipanggil
        self.logger.info.assert_called()
        self.logger.error.assert_called()
    
    @patch('smartcash.ui.dataset.downloader.handlers.download_handler.get_downloader_instance')
    def test_progress_callback(self, mock_get_downloader):
        """Test progress callback integration"""
        # Setup mock downloader
        mock_downloader = MockDownloader(success=True)
        mock_get_downloader.return_value = mock_downloader
        
        # Execute download
        _execute_download(self.ui_components, self.config, self.logger)
        
        # Verifikasi progress callback diset
        self.assertIsNotNone(mock_downloader.progress_callback)
        
        # Verifikasi progress tracker diupdate
        self.assertEqual(self.ui_components['progress_tracker'].update_progress.call_count, 3)


if __name__ == '__main__':
    unittest.main()
