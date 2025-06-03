"""
File: smartcash/tests/test_backbone_config.py
Deskripsi: Test untuk memverifikasi kesesuaian struktur config backbone dengan model_config.yaml
"""

import unittest
import yaml
import os
from pathlib import Path
from typing import Dict, Any

from smartcash.ui.training_config.backbone.handlers.defaults import get_default_backbone_config
from smartcash.ui.training_config.backbone.handlers.ui_extractor import extract_backbone_config
from smartcash.ui.training_config.backbone.handlers.ui_updater import update_backbone_ui


class TestBackboneConfig(unittest.TestCase):
    """Test kesesuaian struktur config backbone dengan model_config.yaml"""

    def setUp(self):
        """Setup untuk test dengan membaca file model_config.yaml"""
        # Path ke file model_config.yaml
        config_path = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "configs" / "model_config.yaml"
        
        # Baca file model_config.yaml
        with open(config_path, 'r') as file:
            self.yaml_config = yaml.safe_load(file)
        
        # Dapatkan default config dari handler
        self.default_config = get_default_backbone_config()
        
        # Mock UI components untuk testing
        self.mock_ui_components = {
            'backbone_dropdown': type('obj', (object,), {'value': 'efficientnet_b4'}),
            'model_type_dropdown': type('obj', (object,), {'value': 'efficient_basic'}),
            'use_attention_checkbox': type('obj', (object,), {'value': True}),
            'use_residual_checkbox': type('obj', (object,), {'value': True}),
            'use_ciou_checkbox': type('obj', (object,), {'value': False}),
            '_suppress_backbone_change': False,
            '_suppress_optimization_change': False
        }
        
        # Extract config dari mock UI components
        self.extracted_config = extract_backbone_config(self.mock_ui_components)

    def test_default_config_structure(self):
        """Test struktur default config sesuai dengan model_config.yaml"""
        # Verifikasi bahwa semua key yang diperlukan ada di default config
        self.assertIn('model', self.default_config)
        
        # Verifikasi struktur model di default config
        model_config = self.default_config['model']
        self.assertIn('type', model_config)
        self.assertIn('backbone', model_config)
        self.assertIn('backbone_pretrained', model_config)
        self.assertIn('backbone_weights', model_config)
        self.assertIn('backbone_freeze', model_config)
        self.assertIn('backbone_unfreeze_epoch', model_config)
        self.assertIn('input_size', model_config)
        self.assertIn('confidence', model_config)
        self.assertIn('iou_threshold', model_config)
        self.assertIn('max_detections', model_config)
        self.assertIn('transfer_learning', model_config)
        self.assertIn('pretrained', model_config)
        self.assertIn('pretrained_weights', model_config)
        self.assertIn('anchors', model_config)
        self.assertIn('strides', model_config)
        self.assertIn('workers', model_config)
        self.assertIn('depth_multiple', model_config)
        self.assertIn('width_multiple', model_config)
        self.assertIn('use_efficient_blocks', model_config)
        self.assertIn('use_adaptive_anchors', model_config)
        self.assertIn('quantization', model_config)
        self.assertIn('quantization_aware_training', model_config)
        self.assertIn('fp16_training', model_config)
        self.assertIn('use_attention', model_config)
        self.assertIn('use_residual', model_config)
        self.assertIn('use_ciou', model_config)
        
        # Verifikasi tipe data untuk beberapa field penting
        self.assertIsInstance(model_config['type'], str)
        self.assertIsInstance(model_config['backbone'], str)
        self.assertIsInstance(model_config['backbone_pretrained'], bool)
        self.assertIsInstance(model_config['input_size'], list)
        self.assertEqual(len(model_config['input_size']), 2)
        self.assertIsInstance(model_config['anchors'], list)
        self.assertIsInstance(model_config['strides'], list)

    def test_yaml_config_structure(self):
        """Test struktur YAML config sesuai dengan yang diharapkan"""
        # Verifikasi bahwa semua key yang diperlukan ada di YAML config
        self.assertIn('model', self.yaml_config)
        
        # Verifikasi struktur model di YAML config
        model_config = self.yaml_config['model']
        self.assertIn('type', model_config)
        self.assertIn('backbone', model_config)
        self.assertIn('backbone_pretrained', model_config)
        self.assertIn('backbone_weights', model_config)
        self.assertIn('backbone_freeze', model_config)
        self.assertIn('backbone_unfreeze_epoch', model_config)
        self.assertIn('input_size', model_config)
        self.assertIn('confidence', model_config)
        self.assertIn('iou_threshold', model_config)
        self.assertIn('max_detections', model_config)
        self.assertIn('transfer_learning', model_config)
        self.assertIn('pretrained', model_config)
        self.assertIn('pretrained_weights', model_config)
        self.assertIn('anchors', model_config)
        self.assertIn('strides', model_config)
        self.assertIn('workers', model_config)
        self.assertIn('depth_multiple', model_config)
        self.assertIn('width_multiple', model_config)
        self.assertIn('use_efficient_blocks', model_config)
        self.assertIn('use_adaptive_anchors', model_config)
        self.assertIn('quantization', model_config)
        self.assertIn('quantization_aware_training', model_config)
        self.assertIn('fp16_training', model_config)
        self.assertIn('use_attention', model_config)
        self.assertIn('use_residual', model_config)
        self.assertIn('use_ciou', model_config)

    def test_extracted_config_structure(self):
        """Test struktur extracted config sesuai dengan model_config.yaml"""
        # Verifikasi bahwa semua key yang diperlukan ada di extracted config
        self.assertIn('model', self.extracted_config)
        
        # Verifikasi struktur model di extracted config
        model_config = self.extracted_config['model']
        self.assertIn('type', model_config)
        self.assertIn('backbone', model_config)
        self.assertIn('backbone_pretrained', model_config)
        self.assertIn('backbone_weights', model_config)
        self.assertIn('backbone_freeze', model_config)
        self.assertIn('backbone_unfreeze_epoch', model_config)
        self.assertIn('input_size', model_config)
        self.assertIn('confidence', model_config)
        self.assertIn('iou_threshold', model_config)
        self.assertIn('max_detections', model_config)
        self.assertIn('transfer_learning', model_config)
        self.assertIn('pretrained', model_config)
        self.assertIn('pretrained_weights', model_config)
        self.assertIn('anchors', model_config)
        self.assertIn('strides', model_config)
        self.assertIn('workers', model_config)
        self.assertIn('depth_multiple', model_config)
        self.assertIn('width_multiple', model_config)
        self.assertIn('use_efficient_blocks', model_config)
        self.assertIn('use_adaptive_anchors', model_config)
        self.assertIn('quantization', model_config)
        self.assertIn('quantization_aware_training', model_config)
        self.assertIn('fp16_training', model_config)
        self.assertIn('use_attention', model_config)
        self.assertIn('use_residual', model_config)
        self.assertIn('use_ciou', model_config)
        
        # Verifikasi nilai yang diambil dari UI components
        self.assertEqual(model_config['backbone'], 'efficientnet_b4')
        self.assertEqual(model_config['type'], 'efficient_basic')
        self.assertEqual(model_config['use_attention'], True)
        self.assertEqual(model_config['use_residual'], True)
        self.assertEqual(model_config['use_ciou'], False)

    def test_ui_updater(self):
        """Test UI updater dapat mengupdate UI components dengan benar"""
        # Buat mock UI components baru untuk testing
        mock_ui_for_update = {
            'backbone_dropdown': type('obj', (object,), {'value': 'cspdarknet_s'}),
            'model_type_dropdown': type('obj', (object,), {'value': 'yolov5s'}),
            'use_attention_checkbox': type('obj', (object,), {'value': False}),
            'use_residual_checkbox': type('obj', (object,), {'value': False}),
            'use_ciou_checkbox': type('obj', (object,), {'value': True}),
            '_suppress_backbone_change': False,
            '_suppress_optimization_change': False
        }
        
        # Buat config untuk update
        update_config = {
            'model': {
                'backbone': 'efficientnet_b4',
                'model_type': 'efficient_optimized',  # Sesuai dengan default di ui_updater.py
                'use_attention': True,
                'use_residual': True,
                'use_ciou': False
            }
        }
        
        # Update UI components
        update_backbone_ui(mock_ui_for_update, update_config)
        
        # Verifikasi bahwa UI components telah diupdate dengan benar
        self.assertEqual(mock_ui_for_update['backbone_dropdown'].value, 'efficientnet_b4')
        self.assertEqual(mock_ui_for_update['model_type_dropdown'].value, 'efficient_optimized')  # Sesuai dengan default di ui_updater.py
        self.assertEqual(mock_ui_for_update['use_attention_checkbox'].value, True)
        self.assertEqual(mock_ui_for_update['use_residual_checkbox'].value, True)
        self.assertEqual(mock_ui_for_update['use_ciou_checkbox'].value, False)

    def test_default_vs_yaml_config(self):
        """Test kesesuaian default config dengan YAML config"""
        # Dapatkan key dari model di default config dan YAML config
        default_model_keys = set(self.default_config['model'].keys())
        yaml_model_keys = set(self.yaml_config['model'].keys())
        
        # Verifikasi bahwa semua key di default config ada di YAML config
        missing_in_yaml = default_model_keys - yaml_model_keys
        self.assertEqual(len(missing_in_yaml), 0, f"Key berikut tidak ada di YAML config: {missing_in_yaml}")
        
        # Verifikasi bahwa semua key di YAML config ada di default config
        missing_in_default = yaml_model_keys - default_model_keys
        self.assertEqual(len(missing_in_default), 0, f"Key berikut tidak ada di default config: {missing_in_default}")


if __name__ == '__main__':
    unittest.main()
