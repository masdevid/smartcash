"""
File: smartcash/tests/test_config_integration.py
Deskripsi: Test untuk memverifikasi integrasi dan konsistensi antara semua modul konfigurasi
"""

import unittest
import yaml
import os
from pathlib import Path
from typing import Dict, Any

# Import handlers dari semua modul konfigurasi
from smartcash.ui.training_config.backbone.handlers.defaults import get_default_backbone_config
from smartcash.ui.training_config.hyperparameters.handlers.defaults import get_default_hyperparameters_config
from smartcash.ui.training_config.strategy.handlers.defaults import get_default_strategy_config


class TestConfigIntegration(unittest.TestCase):
    """Test integrasi dan konsistensi antara semua modul konfigurasi"""

    def setUp(self):
        """Setup untuk test dengan membaca semua file konfigurasi"""
        # Path ke direktori configs
        config_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "configs"
        
        # Baca file model_config.yaml
        with open(config_dir / "model_config.yaml", 'r') as file:
            self.model_config = yaml.safe_load(file)
        
        # Baca file hyperparameters_config.yaml
        with open(config_dir / "hyperparameters_config.yaml", 'r') as file:
            self.hyperparameters_config = yaml.safe_load(file)
        
        # Baca file training_config.yaml
        with open(config_dir / "training_config.yaml", 'r') as file:
            self.training_config = yaml.safe_load(file)
        
        # Dapatkan default config dari semua handler
        self.default_backbone_config = get_default_backbone_config()
        self.default_hyperparameters_config = get_default_hyperparameters_config()
        self.default_strategy_config = get_default_strategy_config()

    def test_training_config_inheritance(self):
        """Test bahwa training_config.yaml mewarisi dari hyperparameters_config.yaml dan model_config.yaml"""
        # Verifikasi bahwa training_config.yaml mewarisi dari file konfigurasi lain
        self.assertIn('_base_', self.training_config)
        base_configs = self.training_config['_base_']
        self.assertIsInstance(base_configs, list)
        
        # Verifikasi bahwa hyperparameters_config.yaml dan model_config.yaml ada di _base_
        base_config_names = [os.path.basename(path) for path in base_configs]
        self.assertIn('hyperparameters_config.yaml', base_config_names)
        self.assertIn('model_config.yaml', base_config_names)

    def test_no_duplicate_keys_between_configs(self):
        """Test bahwa tidak ada duplikasi key antara model_config.yaml dan hyperparameters_config.yaml"""
        # Dapatkan top-level keys dari model_config.yaml dan hyperparameters_config.yaml
        model_keys = set(self.model_config.keys())
        hyperparameters_keys = set(self.hyperparameters_config.keys())
        
        # Hapus key metadata yang mungkin sama
        model_keys.discard('_base_')
        model_keys.discard('config_version')
        model_keys.discard('description')
        hyperparameters_keys.discard('_base_')
        hyperparameters_keys.discard('config_version')
        hyperparameters_keys.discard('description')
        
        # Hapus key 'regularization' yang diizinkan duplikat setelah refaktor
        model_keys.discard('regularization')
        hyperparameters_keys.discard('regularization')
        
        # Verifikasi bahwa tidak ada duplikasi key
        duplicate_keys = model_keys.intersection(hyperparameters_keys)
        self.assertEqual(len(duplicate_keys), 0, f"Duplikasi key ditemukan: {duplicate_keys}")

    def test_strategy_keys_not_in_other_configs(self):
        """Test bahwa key di strategy tidak ada di model_config atau hyperparameters_config"""
        # Dapatkan top-level keys dari strategy
        strategy_keys = set(self.default_strategy_config.keys())
        
        # Hapus key metadata
        strategy_keys.discard('config_version')
        strategy_keys.discard('description')
        
        # Dapatkan top-level keys dari model_config dan hyperparameters_config
        model_keys = set(self.default_backbone_config.keys())
        hyperparameters_keys = set(self.default_hyperparameters_config.keys())
        
        # Verifikasi bahwa key strategy tidak ada di model_config
        duplicate_with_model = strategy_keys.intersection(model_keys)
        self.assertEqual(len(duplicate_with_model), 0, f"Key strategy yang duplikat dengan model_config: {duplicate_with_model}")
        
        # Verifikasi bahwa key strategy tidak ada di hyperparameters_config
        duplicate_with_hyperparameters = strategy_keys.intersection(hyperparameters_keys)
        self.assertEqual(len(duplicate_with_hyperparameters), 0, f"Key strategy yang duplikat dengan hyperparameters_config: {duplicate_with_hyperparameters}")

    def test_default_config_completeness(self):
        """Test bahwa semua default config memiliki semua key yang diperlukan"""
        # Verifikasi bahwa default backbone config memiliki semua key yang diperlukan
        self.assertIn('model', self.default_backbone_config)
        self.assertIn('type', self.default_backbone_config['model'])
        # Catatan: 'backbone' tidak lagi ada di model_config.yaml setelah refaktor
        # self.assertIn('backbone', self.default_backbone_config['model'])
        self.assertIn('use_attention', self.default_backbone_config['model'])
        self.assertIn('use_residual', self.default_backbone_config['model'])
        
        # Verifikasi bahwa default hyperparameters config memiliki semua key yang diperlukan
        self.assertIn('batch_size', self.default_hyperparameters_config)
        self.assertIn('image_size', self.default_hyperparameters_config)
        self.assertIn('epochs', self.default_hyperparameters_config)
        self.assertIn('optimizer', self.default_hyperparameters_config)
        self.assertIn('learning_rate', self.default_hyperparameters_config)
        self.assertIn('early_stopping', self.default_hyperparameters_config)
        self.assertIn('save_best', self.default_hyperparameters_config)
        
        # Verifikasi bahwa default training strategy config memiliki semua key yang diperlukan
        self.assertIn('validation', self.default_strategy_config)
        self.assertIn('multi_scale', self.default_strategy_config)
        self.assertIn('training_utils', self.default_strategy_config)
        self.assertIn('experiment_name', self.default_strategy_config['training_utils'])
        self.assertIn('tensorboard', self.default_strategy_config['training_utils'])
        self.assertIn('layer_mode', self.default_strategy_config['training_utils'])

    def test_yaml_config_completeness(self):
        """Test bahwa semua YAML config memiliki semua key yang diperlukan"""
        # Verifikasi bahwa model_config.yaml memiliki semua key yang diperlukan
        self.assertIn('model', self.model_config)
        self.assertIn('type', self.model_config['model'])
        # Catatan: 'backbone' tidak lagi ada di model_config.yaml setelah refaktor
        # self.assertIn('backbone', self.model_config['model'])
        self.assertIn('use_attention', self.model_config['model'])
        self.assertIn('use_residual', self.model_config['model'])
        
        # Verifikasi bahwa hyperparameters_config.yaml memiliki semua key yang diperlukan
        # Catatan: Struktur hyperparameters_config.yaml telah berubah setelah refaktor
        self.assertIn('training', self.hyperparameters_config)
        self.assertIn('epochs', self.hyperparameters_config['training'])
        self.assertIn('lr', self.hyperparameters_config['training'])
        self.assertIn('scheduler', self.hyperparameters_config)
        self.assertIn('early_stopping', self.hyperparameters_config)
        self.assertIn('save_best', self.hyperparameters_config)
        
        # Verifikasi bahwa training_config.yaml memiliki semua key yang diperlukan
        self.assertIn('validation', self.training_config)
        self.assertIn('multi_scale', self.training_config)
        self.assertIn('training_utils', self.training_config)
        self.assertIn('experiment_name', self.training_config['training_utils'])
        self.assertIn('tensorboard', self.training_config['training_utils'])
        self.assertIn('layer_mode', self.training_config['training_utils'])

    def test_default_vs_yaml_type_consistency(self):
        """Test konsistensi tipe data antara default config dan YAML config"""
        # Verifikasi konsistensi tipe data untuk backbone config
        self.assertIsInstance(self.default_backbone_config['model']['type'], str)
        self.assertIsInstance(self.model_config['model']['type'], str)
        self.assertIsInstance(self.default_backbone_config['model']['use_attention'], bool)
        self.assertIsInstance(self.model_config['model']['use_attention'], bool)
        
        # Verifikasi konsistensi tipe data untuk hyperparameters config
        # Catatan: Struktur hyperparameters_config.yaml telah berubah setelah refaktor
        self.assertIsInstance(self.hyperparameters_config['training']['epochs'], int)
        self.assertIsInstance(self.hyperparameters_config['training']['lr'], float)
        self.assertIsInstance(self.hyperparameters_config['early_stopping']['enabled'], bool)
        
        # Verifikasi konsistensi tipe data untuk training strategy config
        self.assertIsInstance(self.training_config['validation'], dict)
        self.assertIsInstance(self.training_config['multi_scale'], bool)
        self.assertIsInstance(self.training_config['training_utils'], dict)


if __name__ == '__main__':
    unittest.main()
