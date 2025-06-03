"""
File: smartcash/tests/test_preprocessing_config.py
Deskripsi: Test untuk memverifikasi kesesuaian struktur config preprocessing dengan preprocessing_config.yaml
"""

import os
import unittest
import yaml
from typing import Dict, Any


class TestPreprocessingConfig(unittest.TestCase):
    """Test untuk memverifikasi kesesuaian struktur config preprocessing dengan preprocessing_config.yaml"""

    def setUp(self):
        """Setup test dengan memuat file konfigurasi"""
        # Path ke file konfigurasi
        self.base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.config_path = os.path.join(self.base_path, 'configs', 'preprocessing_config.yaml')
        
        # Memuat konfigurasi dari file YAML
        with open(self.config_path, 'r') as f:
            self.yaml_config = yaml.safe_load(f)
        
        # Mock UI components untuk testing
        self.mock_ui_components = {
            'output_dir': type('', (), {'value': 'data/preprocessed'})(),
            'save_visualizations': type('', (), {'value': True})(),
            'vis_dir': type('', (), {'value': 'visualizations/preprocessing'})(),
            'sample_size': type('', (), {'value': 500})(),
            'validate_enabled': type('', (), {'value': True})(),
            'fix_issues': type('', (), {'value': True})(),
            'move_invalid': type('', (), {'value': True})(),
            'visualize': type('', (), {'value': True})(),
            'check_image_quality': type('', (), {'value': True})(),
            'check_labels': type('', (), {'value': True})(),
            'check_coordinates': type('', (), {'value': True})(),
            'normalization_enabled': type('', (), {'value': True})(),
            'normalization_dropdown': type('', (), {'value': 'minmax'})(),
            'resolution_dropdown': type('', (), {'value': '640x640'})(),
            'preserve_aspect_ratio': type('', (), {'value': True})(),
            'normalize_pixel_values': type('', (), {'value': True})(),
            'analysis_enabled': type('', (), {'value': True})(),
            'class_balance': type('', (), {'value': True})(),
            'image_size_distribution': type('', (), {'value': True})(),
            'bbox_statistics': type('', (), {'value': True})(),
            'layer_balance': type('', (), {'value': True})(),
            'balance_enabled': type('', (), {'value': False})(),
            'target_distribution': type('', (), {'value': 'auto'})(),
            'undersampling': type('', (), {'value': False})(),
            'oversampling': type('', (), {'value': True})(),
            'augmentation': type('', (), {'value': True})(),
            'min_samples_per_class': type('', (), {'value': 100})(),
            'max_samples_per_class': type('', (), {'value': 1000})(),
            'use_augmentation_for_preprocessing': type('', (), {'value': True})(),
            'preprocessing_variations': type('', (), {'value': 3})(),
            'backup_dir': type('', (), {'value': 'data/backup/preprocessing'})(),
            'backup_enabled': type('', (), {'value': True})(),
            'auto_cleanup_preprocessed': type('', (), {'value': False})()
        }

    def test_yaml_config_structure(self):
        """Test struktur dasar preprocessing_config.yaml"""
        # Verifikasi struktur dasar
        self.assertIn('preprocessing', self.yaml_config)
        
        # Verifikasi sub-struktur preprocessing
        preprocessing_config = self.yaml_config['preprocessing']
        self.assertIn('output_dir', preprocessing_config)
        self.assertIn('save_visualizations', preprocessing_config)
        self.assertIn('vis_dir', preprocessing_config)
        self.assertIn('sample_size', preprocessing_config)
        self.assertIn('validate', preprocessing_config)
        self.assertIn('normalization', preprocessing_config)
        self.assertIn('analysis', preprocessing_config)
        self.assertIn('balance', preprocessing_config)
        
        # Verifikasi sub-struktur validate
        validate_config = preprocessing_config['validate']
        self.assertIn('enabled', validate_config)
        self.assertIn('fix_issues', validate_config)
        self.assertIn('move_invalid', validate_config)
        self.assertIn('visualize', validate_config)
        self.assertIn('check_image_quality', validate_config)
        self.assertIn('check_labels', validate_config)
        self.assertIn('check_coordinates', validate_config)
        
        # Verifikasi sub-struktur normalization
        normalization_config = preprocessing_config['normalization']
        self.assertIn('enabled', normalization_config)
        self.assertIn('method', normalization_config)
        self.assertIn('target_size', normalization_config)
        self.assertIn('preserve_aspect_ratio', normalization_config)
        self.assertIn('normalize_pixel_values', normalization_config)
        self.assertIn('pixel_range', normalization_config)
        
        # Verifikasi sub-struktur analysis
        analysis_config = preprocessing_config['analysis']
        self.assertIn('enabled', analysis_config)
        self.assertIn('class_balance', analysis_config)
        self.assertIn('image_size_distribution', analysis_config)
        self.assertIn('bbox_statistics', analysis_config)
        self.assertIn('layer_balance', analysis_config)
        
        # Verifikasi sub-struktur balance
        balance_config = preprocessing_config['balance']
        self.assertIn('enabled', balance_config)
        self.assertIn('target_distribution', balance_config)
        self.assertIn('methods', balance_config)
        self.assertIn('min_samples_per_class', balance_config)
        self.assertIn('max_samples_per_class', balance_config)

    def test_config_extractor_consistency(self):
        """Test konsistensi config extractor dengan preprocessing_config.yaml"""
        from smartcash.ui.dataset.preprocessing.handlers.config_extractor import extract_preprocessing_config
        
        # Ekstrak config dari mock UI components
        extracted_config = extract_preprocessing_config(self.mock_ui_components)
        
        # Verifikasi struktur dasar
        self.assertIn('preprocessing', extracted_config)
        
        # Verifikasi sub-struktur preprocessing
        preprocessing_config = extracted_config['preprocessing']
        self.assertEqual(preprocessing_config['output_dir'], 'data/preprocessed')
        self.assertEqual(preprocessing_config['save_visualizations'], True)
        self.assertEqual(preprocessing_config['vis_dir'], 'visualizations/preprocessing')
        self.assertEqual(preprocessing_config['sample_size'], 500)
        
        # Verifikasi sub-struktur validate
        validate_config = preprocessing_config['validate']
        self.assertEqual(validate_config['enabled'], True)
        self.assertEqual(validate_config['fix_issues'], True)
        self.assertEqual(validate_config['move_invalid'], True)
        self.assertEqual(validate_config['visualize'], True)
        self.assertEqual(validate_config['check_image_quality'], True)
        self.assertEqual(validate_config['check_labels'], True)
        self.assertEqual(validate_config['check_coordinates'], True)
        
        # Verifikasi sub-struktur normalization
        normalization_config = preprocessing_config['normalization']
        self.assertEqual(normalization_config['enabled'], True)
        self.assertEqual(normalization_config['method'], 'minmax')
        self.assertEqual(normalization_config['target_size'], [640, 640])
        self.assertEqual(normalization_config['preserve_aspect_ratio'], True)
        self.assertEqual(normalization_config['normalize_pixel_values'], True)
        self.assertEqual(normalization_config['pixel_range'], [0, 1])
        
        # Verifikasi sub-struktur analysis
        analysis_config = preprocessing_config['analysis']
        self.assertEqual(analysis_config['enabled'], True)
        self.assertEqual(analysis_config['class_balance'], True)
        self.assertEqual(analysis_config['image_size_distribution'], True)
        self.assertEqual(analysis_config['bbox_statistics'], True)
        self.assertEqual(analysis_config['layer_balance'], True)
        
        # Verifikasi sub-struktur balance
        balance_config = preprocessing_config['balance']
        self.assertEqual(balance_config['enabled'], False)
        self.assertEqual(balance_config['target_distribution'], 'auto')
        self.assertEqual(balance_config['methods']['undersampling'], False)
        self.assertEqual(balance_config['methods']['oversampling'], True)
        self.assertEqual(balance_config['methods']['augmentation'], True)
        self.assertEqual(balance_config['min_samples_per_class'], 100)
        self.assertEqual(balance_config['max_samples_per_class'], 1000)

    def test_config_updater_consistency(self):
        """Test konsistensi config updater dengan preprocessing_config.yaml"""
        from smartcash.ui.dataset.preprocessing.handlers.config_updater import update_preprocessing_ui
        
        # Buat mock UI components baru untuk testing update
        mock_ui_components = {key: type('', (), {'value': None})() for key in self.mock_ui_components.keys()}
        
        # Buat config untuk update
        test_config = {
            'preprocessing': {
                'output_dir': 'data/test_preprocessed',
                'save_visualizations': False,
                'vis_dir': 'test/visualizations',
                'sample_size': 200,
                'validate': {
                    'enabled': False,
                    'fix_issues': False,
                    'move_invalid': False,
                    'visualize': False,
                    'check_image_quality': False,
                    'check_labels': False,
                    'check_coordinates': False
                },
                'normalization': {
                    'enabled': False,
                    'method': 'zscore',
                    'target_size': [320, 320],
                    'preserve_aspect_ratio': False,
                    'normalize_pixel_values': False,
                    'pixel_range': [0, 255]
                },
                'analysis': {
                    'enabled': False,
                    'class_balance': False,
                    'image_size_distribution': False,
                    'bbox_statistics': False,
                    'layer_balance': False
                },
                'balance': {
                    'enabled': True,
                    'target_distribution': 'equal',
                    'methods': {
                        'undersampling': True,
                        'oversampling': False,
                        'augmentation': False
                    },
                    'min_samples_per_class': 50,
                    'max_samples_per_class': 500
                }
            },
            'augmentation_reference': {
                'use_for_preprocessing': False,
                'preprocessing_variations': 1
            },
            'cleanup': {
                'backup_dir': 'test/backup',
                'backup_enabled': False,
                'auto_cleanup_preprocessed': True
            },
            'performance': {
                'num_workers': 4,
                'batch_size': 16,
                'use_gpu': False,
                'compression_level': 80,
                'max_memory_usage_gb': 2.0,
                'use_mixed_precision': False
            }
        }
        
        # Update UI components dengan config
        update_preprocessing_ui(mock_ui_components, test_config)
        
        # Verifikasi update UI components
        self.assertEqual(mock_ui_components['output_dir'].value, 'data/test_preprocessed')
        self.assertEqual(mock_ui_components['save_visualizations'].value, False)
        self.assertEqual(mock_ui_components['vis_dir'].value, 'test/visualizations')
        self.assertEqual(mock_ui_components['sample_size'].value, 200)
        
        self.assertEqual(mock_ui_components['validate_enabled'].value, False)
        self.assertEqual(mock_ui_components['fix_issues'].value, False)
        self.assertEqual(mock_ui_components['move_invalid'].value, False)
        self.assertEqual(mock_ui_components['visualize'].value, False)
        self.assertEqual(mock_ui_components['check_image_quality'].value, False)
        self.assertEqual(mock_ui_components['check_labels'].value, False)
        self.assertEqual(mock_ui_components['check_coordinates'].value, False)
        
        self.assertEqual(mock_ui_components['normalization_enabled'].value, False)
        self.assertEqual(mock_ui_components['normalization_dropdown'].value, 'zscore')
        self.assertEqual(mock_ui_components['resolution_dropdown'].value, '320x320')
        self.assertEqual(mock_ui_components['preserve_aspect_ratio'].value, False)
        self.assertEqual(mock_ui_components['normalize_pixel_values'].value, False)
        
        self.assertEqual(mock_ui_components['analysis_enabled'].value, False)
        self.assertEqual(mock_ui_components['class_balance'].value, False)
        self.assertEqual(mock_ui_components['image_size_distribution'].value, False)
        self.assertEqual(mock_ui_components['bbox_statistics'].value, False)
        self.assertEqual(mock_ui_components['layer_balance'].value, False)
        
        self.assertEqual(mock_ui_components['balance_enabled'].value, True)
        self.assertEqual(mock_ui_components['target_distribution'].value, 'equal')
        self.assertEqual(mock_ui_components['undersampling'].value, True)
        self.assertEqual(mock_ui_components['oversampling'].value, False)
        self.assertEqual(mock_ui_components['augmentation'].value, False)
        self.assertEqual(mock_ui_components['min_samples_per_class'].value, 50)
        self.assertEqual(mock_ui_components['max_samples_per_class'].value, 500)
        
        self.assertEqual(mock_ui_components['use_augmentation_for_preprocessing'].value, False)
        self.assertEqual(mock_ui_components['preprocessing_variations'].value, 1)
        
        self.assertEqual(mock_ui_components['backup_dir'].value, 'test/backup')
        self.assertEqual(mock_ui_components['backup_enabled'].value, False)
        self.assertEqual(mock_ui_components['auto_cleanup_preprocessed'].value, True)
