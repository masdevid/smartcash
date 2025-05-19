"""
File: smartcash/ui/dataset/download/tests/test_config_handler.py
Deskripsi: Test untuk config_handler pada modul download dataset
"""

import unittest
import os
import tempfile
import yaml
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

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
    """Test untuk config_handler pada modul download dataset."""
    
    def setUp(self):
        """Setup untuk setiap test."""
        # Buat mock untuk UI components
        self.ui_components = {
            'workspace': MagicMock(value='test-workspace'),
            'project': MagicMock(value='test-project'),
            'version': MagicMock(value='1'),
            'api_key': MagicMock(value='test-api-key'),
            'output_dir': MagicMock(value='test-output-dir'),
            'validate_dataset': MagicMock(value=True),
            'backup_checkbox': MagicMock(value=True),
            'backup_dir': MagicMock(value='test-backup-dir'),
            'source_dropdown': MagicMock(value='roboflow'),
            'logger': MagicMock()
        }
        
        # Buat mock untuk config
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
        mock_file_open.assert_called_once()
        mock_yaml_dump.assert_called_once()
        logger.info.assert_called()
    
    @patch('smartcash.ui.dataset.download.handlers.config_handler.load_default_config')
    @patch('pathlib.Path.exists')
    @patch('builtins.open', new_callable=mock_open, read_data='{}')
    @patch('yaml.safe_load')
    def test_load_config_file_not_exists(self, mock_yaml_load, mock_file_open, mock_path_exists, mock_load_default):
        """Test load_config ketika file tidak ada."""
        # Setup mock
        mock_path_exists.return_value = False
        mock_load_default.return_value = self.config
        
        # Panggil fungsi
        result = load_config()
        
        # Verifikasi hasil
        self.assertEqual(result, self.config)
        mock_load_default.assert_called_once()
    
    @patch('smartcash.ui.dataset.download.handlers.config_handler.load_default_config')
    @patch('pathlib.Path.exists')
    @patch('builtins.open', new_callable=mock_open, read_data='{}')
    @patch('yaml.safe_load')
    def test_load_config_file_exists(self, mock_yaml_load, mock_file_open, mock_path_exists, mock_load_default):
        """Test load_config ketika file ada."""
        # Setup mock
        mock_path_exists.return_value = True
        mock_yaml_load.return_value = self.config
        
        # Panggil fungsi
        result = load_config()
        
        # Verifikasi hasil
        self.assertEqual(result, self.config)
        # Verifikasi bahwa file config dibuka (tidak memeriksa jumlah panggilan karena ada panggilan untuk log)
        mock_file_open.assert_any_call(Path('config/dataset_config.yaml'), 'r')
        mock_yaml_load.assert_called_once()
    
    def test_update_config_from_ui(self):
        """Test update_config_from_ui."""
        # Panggil fungsi
        result = update_config_from_ui({}, self.ui_components)
        
        # Verifikasi hasil
        self.assertIn('data', result)
        self.assertIn('download', result['data'])
        self.assertIn('roboflow', result['data'])
        
        # Verifikasi nilai dari UI
        self.assertEqual(result['data']['download']['output_dir'], 'test-output-dir')
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
