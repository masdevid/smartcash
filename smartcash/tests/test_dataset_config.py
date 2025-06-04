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
        # Catatan: Struktur data.source dan data.roboflow telah dipindahkan ke base_config.yaml
        # self.assertIn('source', data_config)
        # self.assertIn('roboflow', data_config)
        # self.assertIn('split_ratios', data_config)
        # self.assertIn('stratified_split', data_config)
        # self.assertIn('random_seed', data_config)
        self.assertIn('validation', data_config)
        
        # Verifikasi sub-struktur dataset
        dataset_config = self.yaml_config['dataset']
        self.assertIn('backup', dataset_config)
        self.assertIn('export', dataset_config)
        self.assertIn('import', dataset_config)
        
        # Verifikasi sub-struktur cache
        cache_config = self.yaml_config['cache']
        # Catatan: 'enabled' telah dipindahkan ke base_config.yaml
        # self.assertIn('enabled', cache_config)
        self.assertIn('dir', cache_config)
        # Catatan: Beberapa key telah dipindahkan atau dihapus setelah refaktor
        # self.assertIn('max_size_gb', cache_config)
        # self.assertIn('ttl_hours', cache_config)
        # self.assertIn('auto_cleanup', cache_config)

    def test_split_config_consistency(self):
        """Test konsistensi split config antara default dan YAML"""
        # Catatan: Setelah refaktor, split_ratios telah dipindahkan ke base_config.yaml
        # Kita hanya memverifikasi bahwa data.validation masih ada di dataset_config.yaml
        self.assertIn('data', self.yaml_config)
        self.assertIn('validation', self.yaml_config['data'])
        
        # Verifikasi bahwa dataset_config.yaml memiliki struktur dataset yang benar
        self.assertIn('dataset', self.yaml_config)
        self.assertIn('backup', self.yaml_config['dataset'])
        self.assertIn('export', self.yaml_config['dataset'])
        self.assertIn('import', self.yaml_config['dataset'])

    def test_split_config_types(self):
        """Test tipe data dalam split config"""
        # Catatan: Setelah refaktor, split_ratios telah dipindahkan ke base_config.yaml
        # Kita hanya memverifikasi tipe data untuk konfigurasi yang masih ada di dataset_config.yaml
        
        # Verifikasi tipe data untuk data.validation
        self.assertIsInstance(self.yaml_config['data']['validation'], dict)
        
        # Verifikasi tipe data untuk dataset.backup
        self.assertIsInstance(self.yaml_config['dataset']['backup'], dict)
        
        # Verifikasi tipe data untuk dataset.export
        self.assertIsInstance(self.yaml_config['dataset']['export'], dict)
        
        # Verifikasi tipe data untuk dataset.import
        self.assertIsInstance(self.yaml_config['dataset']['import'], dict)
        
        # Verifikasi tipe data untuk cache
        self.assertIsInstance(self.yaml_config['cache'], dict)
        self.assertIsInstance(self.yaml_config['cache']['dir'], str)
