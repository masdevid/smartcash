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
    load_default_config,
    load_config,
    save_config,
    get_config_manager_instance,
    save_config_with_manager,
    update_config_from_ui,
    update_ui_from_config
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
            'data': {
                'download': {
                    'source': 'roboflow',
                    'output_dir': 'data/downloads',
                    'backup_before_download': True,
                    'backup_dir': 'data/downloads_backup'
                },
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
    
    def test_load_default_config(self):
        """Test load_default_config."""
        config = load_default_config()
        
        # Verifikasi struktur config
        self.assertIn('data', config)
        self.assertIn('download', config['data'])
        self.assertIn('roboflow', config['data'])
        
        # Verifikasi nilai default
        self.assertEqual(config['data']['download']['source'], 'roboflow')
        self.assertEqual(config['data']['download']['output_dir'], 'data/downloads')
        self.assertEqual(config['data']['roboflow']['workspace'], 'smartcash-wo2us')
    
    @patch('os.makedirs')
    @patch('builtins.open', new_callable=mock_open)
    @patch('yaml.dump')
    def test_save_config(self, mock_yaml_dump, mock_file_open, mock_makedirs):
        """Test save_config."""
        # Mock logger
        logger = MagicMock()
        
        # Panggil fungsi
        result = save_config(self.config, logger)
        
        # Verifikasi hasil
        self.assertTrue(result)
        mock_makedirs.assert_called_once()
        # Verifikasi bahwa file config dibuka untuk menulis
        mock_file_open.assert_any_call(Path('configs/dataset_config.yaml'), 'w')
        mock_yaml_dump.assert_called_once()
        logger.info.assert_called()
    
    @patch('smartcash.ui.dataset.download.handlers.config_handler.get_config_manager_instance')
    @patch('smartcash.ui.dataset.download.handlers.config_handler.load_default_config')
    def test_load_config_file_not_exists(self, mock_load_default, mock_get_config_manager_instance):
        """Test load_config ketika file tidak ada."""
        # Setup mock
        mock_config_manager = MagicMock()
        mock_config_manager.get_module_config.return_value = None
        mock_get_config_manager_instance.return_value = mock_config_manager
        mock_load_default.return_value = self.config
        
        # Panggil fungsi
        result = load_config()
        
        # Verifikasi hasil
        self.assertEqual(result, self.config)
        mock_load_default.assert_called_once()
    
    @patch('smartcash.ui.dataset.download.handlers.config_handler.get_config_manager_instance')
    @patch('smartcash.ui.dataset.download.handlers.config_handler.load_default_config')
    def test_load_config_file_exists(self, mock_load_default, mock_get_config_manager_instance):
        """Test load_config ketika file ada."""
        # Setup mock
        mock_config_manager = MagicMock()
        mock_config_manager.get_module_config.return_value = self.config
        mock_get_config_manager_instance.return_value = mock_config_manager
        mock_load_default.return_value = self.config
        
        # Panggil fungsi
        result = load_config()
        
        # Verifikasi hasil
        self.assertEqual(result, self.config)
        mock_config_manager.get_module_config.assert_called_once_with('dataset_download')
    
    def test_update_config_from_ui(self):
        """Test update_config_from_ui."""
        # Panggil fungsi
        result = update_config_from_ui({}, self.ui_components)
        
        # Verifikasi hasil
        self.assertIn('data', result)
        self.assertIn('download', result['data'])
        self.assertIn('roboflow', result['data'])
        
        # Verifikasi nilai dari UI
        self.assertEqual(result['data']['download']['output_dir'], 'data/test')
        self.assertEqual(result['data']['roboflow']['workspace'], 'test-workspace')
        self.assertEqual(result['data']['roboflow']['project'], 'test-project')
        self.assertEqual(result['data']['roboflow']['version'], '1')
        self.assertEqual(result['data']['roboflow']['api_key'], 'test-api-key')
    
    def test_update_ui_from_config(self):
        """Test update_ui_from_config."""
        # Panggil fungsi
        update_ui_from_config(self.config, self.ui_components)
        
        # Verifikasi UI components diupdate
        self.ui_components['workspace'].value = self.config['data']['roboflow']['workspace']
        self.ui_components['project'].value = self.config['data']['roboflow']['project']
        self.ui_components['version'].value = self.config['data']['roboflow']['version']
        self.ui_components['api_key'].value = self.config['data']['roboflow']['api_key']
        self.ui_components['output_dir'].value = self.config['data']['download']['output_dir']
    
    @patch('smartcash.ui.dataset.download.handlers.config_handler.get_config_manager_instance')
    @patch('smartcash.ui.dataset.download.handlers.config_handler.save_config')
    def test_save_config_with_manager_fallback(self, mock_save_config, mock_get_manager):
        """Test save_config_with_manager dengan fallback."""
        # Setup mock
        mock_get_manager.return_value = None
        mock_save_config.return_value = True
        
        # Panggil fungsi
        result = save_config_with_manager(self.config, self.ui_components, self.ui_components['logger'])
        
        # Verifikasi hasil
        self.assertTrue(result)
        mock_save_config.assert_called_once()
    
    @patch('smartcash.ui.dataset.download.handlers.config_handler.get_config_manager_instance')
    def test_save_config_with_manager(self, mock_get_manager):
        """Test save_config_with_manager dengan ConfigManager."""
        # Setup mock
        mock_manager = MagicMock()
        mock_manager.save_module_config.return_value = True
        mock_get_manager.return_value = mock_manager
        
        # Panggil fungsi
        result = save_config_with_manager(self.config, self.ui_components, self.ui_components['logger'])
        
        # Verifikasi hasil
        self.assertTrue(result)
        mock_manager.register_ui_components.assert_called_once()
        mock_manager.save_module_config.assert_called_once()

if __name__ == '__main__':
    unittest.main()
