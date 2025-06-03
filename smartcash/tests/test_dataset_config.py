"""
File: smartcash/tests/test_dataset_config.py
Deskripsi: Test untuk memverifikasi kesesuaian struktur config dataset dengan dataset_config.yaml
"""

import os
import unittest
import yaml
from typing import Dict, Any

from smartcash.ui.dataset.split.handlers.defaults import get_default_split_config


class TestDatasetConfig(unittest.TestCase):
    """Test untuk memverifikasi kesesuaian struktur config dataset dengan dataset_config.yaml"""

    def setUp(self):
        """Setup test dengan memuat file konfigurasi"""
        # Path ke file konfigurasi
        self.base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.config_path = os.path.join(self.base_path, 'configs', 'dataset_config.yaml')
        
        # Memuat konfigurasi dari file YAML
        with open(self.config_path, 'r') as f:
            self.yaml_config = yaml.safe_load(f)
        
        # Memuat default config dari handler
        self.default_split_config = get_default_split_config()
        
        # Mock UI components untuk testing
        self.mock_ui_components = {
            'train_ratio': type('', (), {'value': 0.7})(),
            'valid_ratio': type('', (), {'value': 0.15})(),
            'test_ratio': type('', (), {'value': 0.15})(),
            'stratified_split': type('', (), {'value': True})(),
            'random_seed': type('', (), {'value': 42})(),
            'backup_before_split': type('', (), {'value': True})()
        }

    def test_yaml_config_structure(self):
        """Test struktur dasar dataset_config.yaml"""
        # Verifikasi struktur dasar
        self.assertIn('data', self.yaml_config)
        self.assertIn('dataset', self.yaml_config)
        self.assertIn('cache', self.yaml_config)
        
        # Verifikasi sub-struktur data
        data_config = self.yaml_config['data']
        self.assertIn('source', data_config)
        self.assertIn('roboflow', data_config)
        self.assertIn('split_ratios', data_config)
        self.assertIn('stratified_split', data_config)
        self.assertIn('random_seed', data_config)
        self.assertIn('validation', data_config)
        
        # Verifikasi sub-struktur dataset
        dataset_config = self.yaml_config['dataset']
        self.assertIn('backup', dataset_config)
        self.assertIn('export', dataset_config)
        self.assertIn('import', dataset_config)
        
        # Verifikasi sub-struktur cache
        cache_config = self.yaml_config['cache']
        self.assertIn('enabled', cache_config)
        self.assertIn('dir', cache_config)
        self.assertIn('max_size_gb', cache_config)
        self.assertIn('ttl_hours', cache_config)
        self.assertIn('auto_cleanup', cache_config)

    def test_default_split_config_consistency(self):
        """Test konsistensi default split config dengan dataset_config.yaml"""
        # Verifikasi struktur default split config
        self.assertIn('data', self.default_split_config)
        self.assertIn('split_ratios', self.default_split_config['data'])
        self.assertIn('stratified_split', self.default_split_config['data'])
        self.assertIn('random_seed', self.default_split_config['data'])
        
        # Verifikasi nilai default split config
        split_ratios = self.default_split_config['data']['split_ratios']
        self.assertEqual(split_ratios['train'], self.yaml_config['data']['split_ratios']['train'])
        self.assertEqual(split_ratios['valid'], self.yaml_config['data']['split_ratios']['valid'])
        self.assertEqual(split_ratios['test'], self.yaml_config['data']['split_ratios']['test'])
        
        self.assertEqual(
            self.default_split_config['data']['stratified_split'], 
            self.yaml_config['data']['stratified_split']
        )
        self.assertEqual(
            self.default_split_config['data']['random_seed'], 
            self.yaml_config['data']['random_seed']
        )

    def test_split_config_types(self):
        """Test tipe data dalam split config"""
        # Verifikasi tipe data dalam split config
        split_ratios = self.yaml_config['data']['split_ratios']
        self.assertIsInstance(split_ratios['train'], float)
        self.assertIsInstance(split_ratios['valid'], float)
        self.assertIsInstance(split_ratios['test'], float)
        
        self.assertIsInstance(self.yaml_config['data']['stratified_split'], bool)
        self.assertIsInstance(self.yaml_config['data']['random_seed'], int)
        
        # Verifikasi total split ratios = 1.0
        total = split_ratios['train'] + split_ratios['valid'] + split_ratios['test']
        self.assertAlmostEqual(total, 1.0, places=2)
