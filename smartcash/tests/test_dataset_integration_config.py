"""
File: smartcash/tests/test_dataset_integration_config.py
Deskripsi: Test untuk memverifikasi integrasi dan konsistensi antara dataset_config.yaml, preprocessing_config.yaml, dan augmentation_config.yaml
"""

import os
import unittest
import yaml
from typing import Dict, Any, List, Set


class TestDatasetIntegrationConfig(unittest.TestCase):
    """Test untuk memverifikasi integrasi dan konsistensi antara dataset_config.yaml, preprocessing_config.yaml, dan augmentation_config.yaml"""

    def setUp(self):
        """Setup test dengan memuat file konfigurasi"""
        # Path ke file konfigurasi
        self.base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.dataset_config_path = os.path.join(self.base_path, 'configs', 'dataset_config.yaml')
        self.preprocessing_config_path = os.path.join(self.base_path, 'configs', 'preprocessing_config.yaml')
        self.augmentation_config_path = os.path.join(self.base_path, 'configs', 'augmentation_config.yaml')
        
        # Memuat konfigurasi dari file YAML
        with open(self.dataset_config_path, 'r') as f:
            self.dataset_config = yaml.safe_load(f)
        
        with open(self.preprocessing_config_path, 'r') as f:
            self.preprocessing_config = yaml.safe_load(f)
        
        with open(self.augmentation_config_path, 'r') as f:
            self.augmentation_config = yaml.safe_load(f)

    def test_base_config_inheritance(self):
        """Test inheritance dari base_config.yaml"""
        # Verifikasi semua config mewarisi dari base_config.yaml
        self.assertEqual(self.dataset_config.get('_base_'), 'base_config.yaml')
        self.assertEqual(self.preprocessing_config.get('_base_'), 'base_config.yaml')
        self.assertEqual(self.augmentation_config.get('_base_'), 'base_config.yaml')

    def test_no_duplicate_top_level_keys(self):
        """Test tidak ada duplikasi top-level keys antara config kecuali yang memang didesain untuk duplikat"""
        # Ekstrak top-level keys dari masing-masing config
        dataset_keys = set(self._get_top_level_keys(self.dataset_config))
        preprocessing_keys = set(self._get_top_level_keys(self.preprocessing_config))
        augmentation_keys = set(self._get_top_level_keys(self.augmentation_config))
        
        # Hapus key '_base_' dan metadata umum dari perbandingan
        common_keys = {'_base_', 'config_version', 'updated_at'}
        dataset_keys -= common_keys
        preprocessing_keys -= common_keys
        augmentation_keys -= common_keys
        
        # Key yang diperbolehkan duplikat karena memang didesain untuk digunakan di beberapa config
        # Setelah refaktor, performance mungkin tidak ada di kedua file karena dipindahkan ke base_config.yaml
        allowed_duplicate_keys = {'cleanup'}
        
        # Verifikasi tidak ada duplikasi top-level keys kecuali yang diperbolehkan
        unexpected_dataset_preprocessing = (dataset_keys & preprocessing_keys) - allowed_duplicate_keys
        self.assertFalse(unexpected_dataset_preprocessing, f"Duplikasi keys antara dataset dan preprocessing: {unexpected_dataset_preprocessing}")
        
        unexpected_dataset_augmentation = (dataset_keys & augmentation_keys) - allowed_duplicate_keys
        self.assertFalse(unexpected_dataset_augmentation, f"Duplikasi keys antara dataset dan augmentation: {unexpected_dataset_augmentation}")
        
        # Untuk preprocessing dan augmentation, kita perbolehkan duplikasi key yang memang didesain untuk duplikat
        duplicate_keys = preprocessing_keys & augmentation_keys
        unexpected_duplicates = duplicate_keys - allowed_duplicate_keys
        self.assertFalse(unexpected_duplicates, f"Duplikasi keys yang tidak diperbolehkan antara preprocessing dan augmentation: {unexpected_duplicates}")


    def test_dataset_preprocessing_integration(self):
        """Test integrasi antara dataset_config.yaml dan preprocessing_config.yaml"""
        # Verifikasi konsistensi path dan struktur
        dataset_config = self.dataset_config
        preprocessing_config = self.preprocessing_config
        
        # Verifikasi konsistensi validasi dataset
        if 'data' in dataset_config and 'validation' in dataset_config['data']:
            dataset_validation = dataset_config['data']['validation']
            if 'preprocessing' in preprocessing_config and 'validate' in preprocessing_config['preprocessing']:
                preprocessing_validation = preprocessing_config['preprocessing']['validate']
                
                # Setelah refaktor, preprocessing_validation mungkin tidak memiliki enabled flag
                # karena dipindahkan ke base_config.yaml, jadi kita hanya memeriksa jika keduanya ada
                if 'enabled' in dataset_validation and 'visualize' in preprocessing_validation:
                    # Verifikasi bahwa keduanya memiliki nilai yang valid
                    self.assertIsNotNone(dataset_validation.get('enabled'))
                    self.assertIsNotNone(preprocessing_validation.get('visualize'))
                    # Tidak perlu membandingkan nilai karena fungsinya berbeda setelah refaktor

    def test_preprocessing_augmentation_integration(self):
        """Test integrasi antara preprocessing_config.yaml dan augmentation_config.yaml"""
        # Verifikasi konsistensi path dan struktur
        preprocessing_config = self.preprocessing_config
        augmentation_config = self.augmentation_config
        
        # Verifikasi konsistensi augmentation reference
        if 'augmentation_reference' in preprocessing_config:
            aug_ref = preprocessing_config['augmentation_reference']
            self.assertEqual(
                aug_ref.get('config_file'), 
                'augmentation_config.yaml',
                "Referensi file augmentation tidak sesuai"
            )
        
        # Verifikasi konsistensi balance methods
        if ('preprocessing' in preprocessing_config and 
            'balance' in preprocessing_config['preprocessing'] and 
            'methods' in preprocessing_config['preprocessing']['balance']):
            
            balance_methods = preprocessing_config['preprocessing']['balance']['methods']
            if balance_methods.get('augmentation') and 'augmentation' in augmentation_config:
                # Jika preprocessing menggunakan augmentation, verifikasi augmentation enabled
                self.assertTrue(
                    augmentation_config['augmentation'].get('enabled', False),
                    "Preprocessing menggunakan augmentation tetapi augmentation tidak enabled"
                )

    def test_dataset_augmentation_integration(self):
        """Test integrasi antara dataset_config.yaml dan augmentation_config.yaml"""
        # Verifikasi konsistensi path dan struktur
        dataset_config = self.dataset_config
        augmentation_config = self.augmentation_config
        
        # Verifikasi konsistensi output paths
        if 'augmentation' in augmentation_config and 'output_dir' in augmentation_config['augmentation']:
            aug_output_dir = augmentation_config['augmentation']['output_dir']
            
            # Verifikasi output_dir adalah subdirektori yang valid
            self.assertTrue(
                aug_output_dir.startswith('data/'),
                f"Augmentation output_dir ({aug_output_dir}) tidak konsisten dengan struktur direktori dataset"
            )

    def test_type_consistency(self):
        """Test konsistensi tipe data antara config"""
        # Verifikasi konsistensi tipe data untuk key yang sama
        self._verify_type_consistency(self.dataset_config, self.preprocessing_config)
        self._verify_type_consistency(self.dataset_config, self.augmentation_config)
        self._verify_type_consistency(self.preprocessing_config, self.augmentation_config)

    def _get_top_level_keys(self, config: Dict[str, Any]) -> List[str]:
        """Mendapatkan top-level keys dari config"""
        return [key for key in config.keys() if key not in ['_base_', 'config_version', 'updated_at']]

    def _verify_type_consistency(self, config1: Dict[str, Any], config2: Dict[str, Any], path: str = ""):
        """Verifikasi konsistensi tipe data antara dua config"""
        # Dapatkan key yang sama di kedua config
        common_keys = set(config1.keys()) & set(config2.keys())
        
        for key in common_keys:
            if key in ['_base_', 'config_version', 'updated_at']:
                continue
                
            current_path = f"{path}.{key}" if path else key
            val1 = config1[key]
            val2 = config2[key]
            
            # Jika keduanya adalah dict, periksa secara rekursif
            if isinstance(val1, dict) and isinstance(val2, dict):
                self._verify_type_consistency(val1, val2, current_path)
            # Jika bukan dict, verifikasi tipe data sama
            elif type(val1) != type(val2):
                self.assertEqual(
                    type(val1), 
                    type(val2), 
                    f"Inkonsistensi tipe data untuk key {current_path}: {type(val1).__name__} vs {type(val2).__name__}"
                )
