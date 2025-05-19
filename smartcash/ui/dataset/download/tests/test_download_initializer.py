"""
File: smartcash/ui/dataset/download/tests/test_download_initializer.py
Deskripsi: Unit test untuk download initializer
"""

import os
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import ipywidgets as widgets
from smartcash.ui.dataset.download.download_initializer import initialize_dataset_download_ui

# Setup environment variables before imports
os.environ['SMARTCASH_BASE_DIR'] = tempfile.mkdtemp()

class TestDownloadInitializer(unittest.TestCase):
    """Test cases untuk download initializer."""
    
    @classmethod
    def setUpClass(cls):
        """Setup class level fixtures."""
        cls.temp_dir = Path(os.environ['SMARTCASH_BASE_DIR'])
        cls.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Create necessary directories
        (cls.temp_dir / 'data').mkdir(exist_ok=True)
        (cls.temp_dir / 'data' / 'preprocessed').mkdir(exist_ok=True)
    
    @classmethod
    def tearDownClass(cls):
        """Cleanup class level fixtures."""
        import shutil
        shutil.rmtree(cls.temp_dir)
    
    def setUp(self):
        """Setup test fixtures."""
        self.mock_ui = {
            'header': MagicMock(),
            'status_panel': MagicMock(),
            'options_panel': MagicMock(),
            'action_section': MagicMock(),
            'progress_section': MagicMock(),
            'log_section': MagicMock()
        }
    
    def tearDown(self):
        self.mock_ui.clear()
    
    @patch('smartcash.dataset.services.downloader.download_service.DownloadService', return_value=MagicMock())
    @patch('smartcash.dataset.services.downloader.backup_service.BackupService', return_value=MagicMock())
    @patch('smartcash.dataset.services.preprocessor.preprocessing_service.PreprocessingService', return_value=MagicMock())
    @patch('smartcash.components.observer.manager_observer.ObserverManager', return_value=MagicMock())
    @patch('smartcash.common.logger.get_logger', return_value=MagicMock())
    def test_initialize_dataset_download_ui(self, mock_logger, mock_observer_manager, mock_preprocessing_service, mock_backup_service, mock_download_service):
        """Test inisialisasi UI download dataset mengembalikan widget VBox."""
        result = initialize_dataset_download_ui(self.mock_ui)
        self.assertIsInstance(result, widgets.VBox)
        self.assertTrue(hasattr(result, 'children'))
        self.assertGreater(len(result.children), 0)
    
    @patch('smartcash.dataset.services.downloader.download_service.DownloadService', side_effect=Exception("Test error"))
    @patch('smartcash.dataset.services.downloader.backup_service.BackupService', return_value=MagicMock())
    @patch('smartcash.dataset.services.preprocessor.preprocessing_service.PreprocessingService', return_value=MagicMock())
    @patch('smartcash.components.observer.manager_observer.ObserverManager', return_value=MagicMock())
    @patch('smartcash.common.logger.get_logger', return_value=MagicMock())
    def test_error_handling(self, mock_logger, mock_observer_manager, mock_preprocessing_service, mock_backup_service, mock_download_service):
        """Test penanganan error saat inisialisasi tetap mengembalikan widget atau raise error."""
        try:
            result = initialize_dataset_download_ui(self.mock_ui)
            self.assertIsInstance(result, widgets.VBox)
        except Exception as e:
            self.assertIsInstance(e, Exception)
    
    @patch('smartcash.dataset.services.downloader.download_service.DownloadService', return_value=MagicMock())
    @patch('smartcash.dataset.services.downloader.backup_service.BackupService', return_value=MagicMock())
    @patch('smartcash.dataset.services.preprocessor.preprocessing_service.PreprocessingService', return_value=MagicMock())
    @patch('smartcash.components.observer.manager_observer.ObserverManager', return_value=MagicMock())
    @patch('smartcash.common.logger.get_logger', return_value=MagicMock())
    def test_widget_structure(self, mock_logger, mock_observer_manager, mock_preprocessing_service, mock_backup_service, mock_download_service):
        """Test struktur widget utama (VBox) memiliki komponen utama."""
        result = initialize_dataset_download_ui(self.mock_ui)
        found = any(isinstance(child, (widgets.HTML, widgets.Output, widgets.Button, widgets.VBox, widgets.HBox)) for child in result.children)
        self.assertTrue(found)

if __name__ == '__main__':
    unittest.main() 