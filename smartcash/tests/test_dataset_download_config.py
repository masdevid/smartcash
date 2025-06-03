"""
File: smartcash/tests/test_dataset_download_config.py
Deskripsi: Test untuk memverifikasi kesesuaian struktur config dataset download dengan dataset_config.yaml
"""

import os
import unittest
import yaml
from typing import Dict, Any


class TestDatasetDownloadConfig(unittest.TestCase):
    """Test untuk memverifikasi kesesuaian struktur config dataset download dengan dataset_config.yaml"""

    def setUp(self):
        """Setup test dengan memuat file konfigurasi"""
        # Path ke file konfigurasi
        self.base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.config_path = os.path.join(self.base_path, 'configs', 'dataset_config.yaml')
        
        # Memuat konfigurasi dari file YAML
        with open(self.config_path, 'r') as f:
            self.yaml_config = yaml.safe_load(f)
        
        # Mock UI components untuk testing
        self.mock_ui_components = {
            'workspace': type('', (), {'value': 'smartcash-wo2us'})(),
            'project': type('', (), {'value': 'rupiah-emisi-2022'})(),
            'version': type('', (), {'value': '3'})(),
            'output_dir': type('', (), {'value': 'data/downloads'})(),
            'backup_dir': type('', (), {'value': 'data/backup'})(),
            'backup_checkbox': type('', (), {'value': False})(),
            'logger': type('', (), {'info': lambda x: None, 'warning': lambda x: None})(),
            'status_panel': type('', (), {'value': ''})(),
            'drive_info': type('', (), {'value': ''})()
        }

    def test_yaml_config_structure(self):
        """Test struktur dasar dataset_config.yaml untuk bagian download"""
        # Verifikasi struktur dasar
        self.assertIn('data', self.yaml_config)
        
        # Verifikasi sub-struktur data
        data_config = self.yaml_config['data']
        self.assertIn('source', data_config)
        self.assertIn('roboflow', data_config)
        self.assertIn('split_ratios', data_config)
        self.assertIn('stratified_split', data_config)
        self.assertIn('random_seed', data_config)
        self.assertIn('validation', data_config)
        
        # Verifikasi sub-struktur roboflow
        roboflow_config = data_config['roboflow']
        self.assertIn('api_key', roboflow_config)
        self.assertIn('workspace', roboflow_config)
        self.assertIn('project', roboflow_config)
        self.assertIn('version', roboflow_config)
        
        # Verifikasi sub-struktur split_ratios
        split_ratios = data_config['split_ratios']
        self.assertIn('train', split_ratios)
        self.assertIn('valid', split_ratios)
        self.assertIn('test', split_ratios)
        
        # Verifikasi sub-struktur validation
        validation_config = data_config['validation']
        self.assertIn('enabled', validation_config)
        self.assertIn('fix_issues', validation_config)
        self.assertIn('move_invalid', validation_config)
        self.assertIn('invalid_dir', validation_config)
        self.assertIn('visualize_issues', validation_config)
        
        # Verifikasi struktur dataset
        self.assertIn('dataset', self.yaml_config)
        dataset_config = self.yaml_config['dataset']
        
        # Verifikasi sub-struktur backup
        self.assertIn('backup', dataset_config)
        backup_config = dataset_config['backup']
        self.assertIn('enabled', backup_config)
        self.assertIn('dir', backup_config)
        self.assertIn('count', backup_config)
        self.assertIn('auto', backup_config)
        
        # Verifikasi sub-struktur export
        self.assertIn('export', dataset_config)
        export_config = dataset_config['export']
        self.assertIn('enabled', export_config)
        self.assertIn('formats', export_config)
        self.assertIn('dir', export_config)
        
        # Verifikasi sub-struktur import
        self.assertIn('import', dataset_config)
        import_config = dataset_config['import']
        self.assertIn('allowed_formats', import_config)
        self.assertIn('temp_dir', import_config)
        
        # Verifikasi struktur cache
        self.assertIn('cache', self.yaml_config)
        cache_config = self.yaml_config['cache']
        self.assertIn('enabled', cache_config)
        self.assertIn('dir', cache_config)
        self.assertIn('max_size_gb', cache_config)
        self.assertIn('ttl_hours', cache_config)
        self.assertIn('auto_cleanup', cache_config)

    def test_config_handlers_setup(self):
        """Test setup config handlers dengan dataset_config.yaml"""
        from smartcash.ui.dataset.download.handlers.config_handlers import setup_config_handlers
        
        # Test setup config handlers
        config = {
            'workspace': 'smartcash-wo2us',
            'project': 'rupiah-emisi-2022',
            'version': '3',
            'output_dir': 'data/downloads',
            'backup_dir': 'data/backup',
            'backup_before_download': False
        }
        
        # Setup config handlers
        updated_ui = setup_config_handlers(self.mock_ui_components, config)
        
        # Verifikasi UI components diupdate dengan benar
        self.assertEqual(updated_ui['workspace'].value, 'smartcash-wo2us')
        self.assertEqual(updated_ui['project'].value, 'rupiah-emisi-2022')
        self.assertEqual(updated_ui['version'].value, '3')
        
        # Verifikasi defaults disimpan
        self.assertIn('_defaults', updated_ui)
        defaults = updated_ui['_defaults']
        self.assertEqual(defaults['workspace'], 'smartcash-wo2us')
        self.assertEqual(defaults['project'], 'rupiah-emisi-2022')
        self.assertEqual(defaults['version'], '3')
        
    def test_config_handlers_functions(self):
        """Test fungsi helper di config_handlers.py"""
        from smartcash.ui.dataset.download.handlers.config_handlers import _merge_configs, _create_smart_defaults
        
        # Test merge configs
        base_config = {
            'workspace': 'default-workspace',
            'project': 'default-project',
            'version': '1'
        }
        
        saved_config = {
            'workspace': 'saved-workspace',
            'project': 'saved-project',
            'version': '2'
        }
        
        paths = {
            'downloads': 'data/downloads-test',
            'backup': 'data/backup-test'
        }
        
        merged = _merge_configs(base_config, saved_config, paths)
        
        # Verifikasi hasil merge
        self.assertEqual(merged['workspace'], 'saved-workspace')
        self.assertEqual(merged['project'], 'saved-project')
        self.assertEqual(merged['version'], '2')
        self.assertEqual(merged['output_dir'], 'data/downloads-test')
        self.assertEqual(merged['backup_dir'], 'data/backup-test')
        
        # Test create smart defaults
        api_key = 'test-api-key'
        defaults = _create_smart_defaults(paths, api_key)
        
        # Verifikasi defaults
        self.assertEqual(defaults['workspace'], 'smartcash-wo2us')
        self.assertEqual(defaults['project'], 'rupiah-emisi-2022')
        self.assertEqual(defaults['version'], '3')
        self.assertEqual(defaults['api_key'], 'test-api-key')
        self.assertEqual(defaults['output_dir'], 'data/downloads-test')
        self.assertEqual(defaults['backup_dir'], 'data/backup-test')
