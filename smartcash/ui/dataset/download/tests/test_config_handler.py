"""
File: smartcash/ui/dataset/download/tests/test_config_handler.py
Deskripsi: Test untuk config_handler pada modul download dataset
"""

import unittest
import os
import tempfile
import shutil
import yaml
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
import ipywidgets as widgets

from smartcash.ui.dataset.download.handlers.config_handler import (
    get_config_from_ui,
    update_config_from_ui,
    get_download_config,
    update_ui_from_config,
    get_dataset_manager,
    get_download_service
)

class TestDownloadConfigHandler(unittest.TestCase):
    """Test suite untuk config_handler pada modul download dataset."""
    
    def setUp(self):
        """Setup sebelum setiap test case."""
        # Setup temporary directory
        self.temp_dir = Path("temp_test_config")
        self.temp_dir.mkdir(exist_ok=True)
        
        # Setup test config
        self.config = {
            'download': {
                'source': 'roboflow',
                'output_dir': 'data/downloads',
                'backup_before_download': True,
                'backup_dir': 'data/downloads_backup',
                'roboflow': {
                    'workspace': 'smartcash-wo2us',
                    'project': 'rupiah-emisi-2022',
                    'version': '3',
                    'api_key': 'test-api-key'
                }
            }
        }
        
        # Setup patches
        self.dataset_manager_patch = patch('smartcash.ui.dataset.download.handlers.config_handler.get_dataset_manager')
        self.download_service_patch = patch('smartcash.ui.dataset.download.handlers.config_handler.get_download_service')
        
        # Start patches
        self.dataset_manager_mock = self.dataset_manager_patch.start()
        self.download_service_mock = self.download_service_patch.start()
        
        # Buat mock UI components
        self.ui_components = self._create_mock_ui_components()
        
        # Setup logger mock
        self.logger_mock = MagicMock()
        self.ui_components['logger'] = self.logger_mock
        
        # Set output_dir to match test expectation
        self.ui_components['output_dir'].value = 'data/test'
    
    def tearDown(self):
        """Cleanup setelah setiap test case."""
        # Stop patches
        self.dataset_manager_patch.stop()
        self.download_service_patch.stop()
        
        # Hapus temporary directory
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def _create_mock_ui_components(self):
        """Buat mock UI components untuk testing."""
        # Buat mock untuk semua komponen UI yang diperlukan
        ui_components = {
            'download_button': MagicMock(spec=widgets.Button),
            'check_button': MagicMock(spec=widgets.Button),
            'reset_button': MagicMock(spec=widgets.Button),
            'save_button': MagicMock(spec=widgets.Button),
            'source_dropdown': MagicMock(spec=widgets.Dropdown, value='roboflow'),
            'backup_checkbox': MagicMock(spec=widgets.Checkbox, value=True),
            'backup_dir': MagicMock(spec=widgets.Text, value='data/backup'),
            'cleanup_button': MagicMock(spec=widgets.Button),
            'progress_bar': MagicMock(spec=widgets.FloatProgress),
            'overall_label': MagicMock(spec=widgets.HTML),
            'step_label': MagicMock(spec=widgets.HTML),
            'status_panel': MagicMock(spec=widgets.HTML),
            'log_output': MagicMock(spec=widgets.Output),
            'summary_container': MagicMock(spec=widgets.Output),
            'confirmation_area': MagicMock(spec=widgets.Output),
            'progress_container': MagicMock(spec=widgets.VBox),
            'workspace': MagicMock(spec=widgets.Text, value='test-workspace'),
            'project': MagicMock(spec=widgets.Text, value='test-project'),
            'version': MagicMock(spec=widgets.Text, value='1'),
            'api_key': MagicMock(spec=widgets.Text, value='test-api-key'),
            'output_dir': MagicMock(spec=widgets.Text, value='data/test'),
            'validate_dataset': MagicMock(spec=widgets.Checkbox, value=True),
            'reset_progress_bar': MagicMock(),
            'dataset_stats': {
                'total_images': 100,
                'total_labels': 90,
                'classes': {'0': 50, '1': 40}
            },
            'download_timestamp': '2025-05-19T12:14:39+07:00',
            'download_running': False
        }
        
        # Tambahkan layout ke komponen yang membutuhkannya
        for key in ['progress_bar', 'overall_label', 'step_label', 'progress_container', 
                   'summary_container', 'confirmation_area']:
            ui_components[key].layout = MagicMock()
            ui_components[key].layout.visibility = 'visible'
            ui_components[key].layout.display = 'block'
        
        return ui_components
    
    @patch('smartcash.ui.dataset.download.handlers.config_handler.get_config_manager')
    def test_get_config_from_ui(self, mock_get_config_manager):
        """Test get_config_from_ui."""
        # Setup mock
        mock_config_manager = MagicMock()
        mock_config_manager.get_config.return_value = self.config
        mock_get_config_manager.return_value = mock_config_manager
        
        # Panggil fungsi
        result = get_config_from_ui(self.ui_components)
        
        # Verifikasi hasil
        self.assertIn('download', result)
        self.assertEqual(result['download']['source'], 'roboflow')
        self.assertEqual(result['download']['output_dir'], 'data/test')
        self.assertEqual(result['download']['roboflow']['workspace'], 'test-workspace')
    
    @patch('smartcash.ui.dataset.download.handlers.config_handler.get_config_manager')
    def test_update_config_from_ui(self, mock_get_config_manager):
        """Test update_config_from_ui."""
        # Setup mock
        mock_config_manager = MagicMock()
        mock_config_manager.get_config.return_value = self.config
        mock_get_config_manager.return_value = mock_config_manager
        
        # Panggil fungsi
        result = update_config_from_ui(self.ui_components)
        
        # Verifikasi hasil
        self.assertIn('download', result)
        mock_config_manager.update_config.assert_called_once()
    
    @patch('smartcash.ui.dataset.download.handlers.config_handler.get_config_manager')
    def test_get_download_config(self, mock_get_config_manager):
        """Test get_download_config."""
        # Setup mock
        mock_config_manager = MagicMock()
        mock_config_manager.get_config.return_value = self.config
        mock_get_config_manager.return_value = mock_config_manager
        
        # Panggil fungsi
        result = get_download_config(self.ui_components)
        
        # Verifikasi hasil
        self.assertIn('source', result)
        self.assertIn('output_dir', result)
        self.assertIn('roboflow', result)
    
    def test_update_ui_from_config(self):
        """Test update_ui_from_config."""
        # Panggil fungsi
        update_ui_from_config(self.ui_components, self.config['download'])
        
        # Verifikasi hasil
        self.assertEqual(self.ui_components['source_dropdown'].value, 'roboflow')
        self.assertEqual(self.ui_components['output_dir'].value, 'data/downloads')
        self.assertEqual(self.ui_components['workspace'].value, 'smartcash-wo2us')

if __name__ == '__main__':
    unittest.main()
