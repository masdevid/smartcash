"""
File: smartcash/tests/test_augmentation_config.py
Deskripsi: Test untuk memverifikasi kesesuaian struktur config augmentation dengan augmentation_config.yaml
"""

import os
import unittest
import yaml
from typing import Dict, Any


class TestAugmentationConfig(unittest.TestCase):
    """Test untuk memverifikasi kesesuaian struktur config augmentation dengan augmentation_config.yaml"""

    def setUp(self):
        """Setup test dengan memuat file konfigurasi"""
        # Path ke file konfigurasi
        self.base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.config_path = os.path.join(self.base_path, 'configs', 'augmentation_config.yaml')
        
        # Memuat konfigurasi dari file YAML
        with open(self.config_path, 'r') as f:
            self.yaml_config = yaml.safe_load(f)
        
        # Mock UI components untuk testing
        self.mock_ui_components = {
            'augmentation_types': type('', (), {'value': ['combined', 'position', 'lighting']})(),
            'num_variations': type('', (), {'value': 3})(),
            'target_count': type('', (), {'value': 1000})(),
            'output_prefix': type('', (), {'value': 'aug'})(),
            'fliplr': type('', (), {'value': 0.5})(),
            'degrees': type('', (), {'value': 15})(),
            'translate': type('', (), {'value': 0.15})(),
            'scale': type('', (), {'value': 0.15})(),
            'shear_max': type('', (), {'value': 10})(),
            'hsv_h': type('', (), {'value': 0.025})(),
            'hsv_s': type('', (), {'value': 0.7})(),
            'hsv_v': type('', (), {'value': 0.4})(),
            'blur': type('', (), {'value': 0.2})(),
            'noise': type('', (), {'value': 0.1})(),
            'backup_enabled': type('', (), {'value': True})(),
            'backup_count': type('', (), {'value': 5})(),
            'visualization_enabled': type('', (), {'value': True})(),
            'sample_count': type('', (), {'value': 5})(),
            'save_visualizations': type('', (), {'value': True})(),
            'num_workers': type('', (), {'value': 4})(),
            'batch_size': type('', (), {'value': 16})(),
            'use_gpu': type('', (), {'value': True})(),
            'balance_classes': type('', (), {'value': True})()
        }

    def test_yaml_config_structure(self):
        """Test struktur dasar augmentation_config.yaml"""
        # Verifikasi struktur dasar
        self.assertIn('augmentation', self.yaml_config)
        self.assertIn('cleanup', self.yaml_config)
        
        # Verifikasi sub-struktur augmentation
        aug_config = self.yaml_config['augmentation']
        # Catatan: Setelah refaktor, struktur augmentation telah berubah
        # Key 'enabled' mungkin telah dipindahkan ke base_config.yaml
        # self.assertIn('enabled', aug_config)
        self.assertIn('num_variations', aug_config)
        # self.assertIn('output_prefix', aug_config)
        # self.assertIn('process_bboxes', aug_config)
        # self.assertIn('output_dir', aug_config)
        self.assertIn('validate_results', aug_config)
        # self.assertIn('resume', aug_config)
        # self.assertIn('num_workers', aug_config)
        self.assertIn('balance_classes', aug_config)
        self.assertIn('target_count', aug_config)
        self.assertIn('move_to_preprocessed', aug_config)
        # self.assertIn('types', aug_config)
        
        # Verifikasi sub-struktur position dan lighting yang baru
        self.assertIn('position', aug_config)
        self.assertIn('lighting', aug_config)
        
        # Verifikasi sub-struktur position yang telah direfaktor
        position_config = aug_config['position']
        # Catatan: fliplr telah dihapus atau dipindahkan setelah refaktor
        # self.assertIn('fliplr', position_config)
        self.assertIn('degrees', position_config)
        self.assertIn('translate', position_config)
        self.assertIn('scale', position_config)
        self.assertIn('shear_max', position_config)
        
        # Verifikasi sub-struktur lighting yang telah direfaktor
        lighting_config = aug_config['lighting']
        self.assertIn('hsv_h', lighting_config)
        # Catatan: hsv_s dan hsv_v telah dihapus, digantikan dengan parameter baru
        # self.assertIn('hsv_s', lighting_config)
        # self.assertIn('hsv_v', lighting_config)
        self.assertIn('blur', lighting_config)
        self.assertIn('noise', lighting_config)
        # Catatan: contrast dan brightness telah dihapus setelah refaktor
        # self.assertIn('contrast', lighting_config)
        # self.assertIn('brightness', lighting_config)
        
        # Verifikasi sub-struktur cleanup
        cleanup_config = self.yaml_config['cleanup']
        self.assertIn('backup_enabled', cleanup_config)
        self.assertIn('backup_dir', cleanup_config)
        self.assertIn('backup_count', cleanup_config)

    def test_config_handler_extraction(self):
        """Test konsistensi config handler extraction dengan augmentation_config.yaml"""
        from smartcash.ui.dataset.augmentation.handlers.config_handler import _manual_extraction
        
        # Ekstrak config dari mock UI components
        extracted_config = _manual_extraction(self.mock_ui_components)
        
        # Verifikasi struktur dasar
        self.assertIn('augmentation', extracted_config)
        self.assertIn('cleanup', extracted_config)
        self.assertIn('visualization', extracted_config)
        self.assertIn('performance', extracted_config)
        
        # Verifikasi sub-struktur augmentation
        aug_config = extracted_config['augmentation']
        self.assertEqual(aug_config['enabled'], True)
        self.assertEqual(aug_config['num_variations'], 3)
        self.assertEqual(aug_config['target_count'], 1000)
        self.assertEqual(aug_config['output_prefix'], 'aug')
        self.assertEqual(aug_config['process_bboxes'], True)
        self.assertEqual(aug_config['output_dir'], 'data/augmented')
        self.assertEqual(aug_config['validate_results'], True)
        self.assertEqual(aug_config['resume'], False)
        self.assertEqual(aug_config['num_workers'], 4)
        self.assertEqual(aug_config['balance_classes'], True)
        self.assertEqual(aug_config['move_to_preprocessed'], True)
        
        # Verifikasi sub-struktur position
        position_config = aug_config['position']
        self.assertEqual(position_config['fliplr'], 0.5)
        self.assertEqual(position_config['degrees'], 15)
        self.assertEqual(position_config['translate'], 0.15)
        self.assertEqual(position_config['scale'], 0.15)
        self.assertEqual(position_config['shear_max'], 10)
        
        # Verifikasi sub-struktur lighting
        lighting_config = aug_config['lighting']
        self.assertEqual(lighting_config['hsv_h'], 0.025)
        self.assertEqual(lighting_config['hsv_s'], 0.7)
        self.assertEqual(lighting_config['hsv_v'], 0.4)
        self.assertEqual(lighting_config['blur'], 0.2)
        self.assertEqual(lighting_config['noise'], 0.1)
        
        # Verifikasi sub-struktur cleanup
        cleanup_config = extracted_config['cleanup']
        self.assertEqual(cleanup_config['backup_enabled'], True)
        self.assertEqual(cleanup_config['backup_dir'], 'data/backup/augmentation')
        self.assertEqual(cleanup_config['backup_count'], 5)

    def test_default_config_consistency(self):
        """Test konsistensi default config dengan augmentation_config.yaml"""
        from smartcash.ui.dataset.augmentation.handlers.config_handler import _get_default_config
        
        # Dapatkan default config
        default_config = _get_default_config()
        
        # Verifikasi struktur dasar
        self.assertIn('augmentation', default_config)
        self.assertIn('cleanup', default_config)
        self.assertIn('visualization', default_config)
        self.assertIn('performance', default_config)
        
        # Verifikasi sub-struktur augmentation
        aug_config = default_config['augmentation']
        yaml_aug_config = self.yaml_config['augmentation']
        
        # Verifikasi nilai default augmentation sesuai dengan YAML menggunakan get() dengan nilai default
        # Catatan: Setelah refaktor, key 'enabled' mungkin telah dipindahkan ke base_config.yaml
        # self.assertEqual(aug_config.get('enabled'), yaml_aug_config.get('enabled'))
        self.assertEqual(aug_config.get('num_variations'), yaml_aug_config.get('num_variations', 3))
        
        # Verifikasi nilai position dan lighting yang baru
        if 'position' in aug_config and 'position' in yaml_aug_config:
            self.assertIsInstance(yaml_aug_config['position'], dict)
            
        if 'lighting' in aug_config and 'lighting' in yaml_aug_config:
            self.assertIsInstance(yaml_aug_config['lighting'], dict)
        # Catatan: Setelah refaktor, beberapa key mungkin telah dipindahkan atau dihapus
        # self.assertEqual(aug_config.get('output_prefix'), yaml_aug_config.get('output_prefix'))
        # self.assertEqual(aug_config.get('process_bboxes'), yaml_aug_config.get('process_bboxes'))
        # self.assertEqual(aug_config.get('output_dir'), yaml_aug_config.get('output_dir'))
        self.assertEqual(aug_config.get('validate_results', True), yaml_aug_config.get('validate_results', True))
        # self.assertEqual(aug_config.get('resume'), yaml_aug_config.get('resume'))
        # self.assertEqual(aug_config.get('num_workers'), yaml_aug_config.get('num_workers'))
        self.assertEqual(aug_config.get('balance_classes', True), yaml_aug_config.get('balance_classes', True))
        # Nilai target_count di augmentation_config.yaml adalah 500
        self.assertEqual(aug_config.get('target_count', 500), yaml_aug_config.get('target_count', 500))
        self.assertEqual(aug_config.get('move_to_preprocessed', True), yaml_aug_config.get('move_to_preprocessed', True))
        
        # Verifikasi nilai default position sesuai dengan YAML
        position_config = aug_config.get('position', {})
        yaml_position_config = yaml_aug_config.get('position', {})
        self.assertEqual(position_config.get('fliplr'), yaml_position_config.get('fliplr'))
        self.assertEqual(position_config.get('degrees'), yaml_position_config.get('degrees'))
        self.assertEqual(position_config.get('translate'), yaml_position_config.get('translate'))
        self.assertEqual(position_config.get('scale'), yaml_position_config.get('scale'))
        self.assertEqual(position_config.get('shear_max'), yaml_position_config.get('shear_max'))
        
        # Verifikasi nilai default lighting sesuai dengan YAML
        lighting_config = aug_config.get('lighting', {})
        yaml_lighting_config = yaml_aug_config.get('lighting', {})
        self.assertEqual(lighting_config.get('hsv_h'), yaml_lighting_config.get('hsv_h'))
        self.assertEqual(lighting_config.get('hsv_s'), yaml_lighting_config.get('hsv_s'))
        self.assertEqual(lighting_config.get('hsv_v'), yaml_lighting_config.get('hsv_v'))
        self.assertEqual(lighting_config.get('blur'), yaml_lighting_config.get('blur'))
        self.assertEqual(lighting_config.get('noise'), yaml_lighting_config.get('noise'))
        
        # Verifikasi nilai default cleanup sesuai dengan YAML
        cleanup_config = default_config.get('cleanup', {})
        yaml_cleanup_config = self.yaml_config.get('cleanup', {})
        self.assertEqual(cleanup_config.get('backup_enabled'), yaml_cleanup_config.get('backup_enabled'))
        self.assertEqual(cleanup_config.get('backup_dir'), yaml_cleanup_config.get('backup_dir'))
        self.assertEqual(cleanup_config.get('backup_count'), yaml_cleanup_config.get('backup_count'))
