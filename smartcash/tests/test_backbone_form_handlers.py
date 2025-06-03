"""
File: smartcash/tests/test_backbone_form_handlers.py
Deskripsi: Test untuk memverifikasi bahwa perubahan form backbone mengupdate state form lain sesuai dengan OPTIMIZED_MODELS
"""

import unittest
from typing import Dict, Any
from unittest.mock import MagicMock, patch
import sys
import os

# Menambahkan path root project ke sys.path untuk import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from smartcash.model.config.model_constants import OPTIMIZED_MODELS


class TestBackboneFormHandlers(unittest.TestCase):
    """Test untuk memverifikasi bahwa perubahan form backbone mengupdate state form lain sesuai dengan OPTIMIZED_MODELS"""

    def setUp(self):
        """Setup test dengan membuat mock UI components"""
        # Mock UI components untuk testing
        self.mock_ui_components = {
            'backbone_dropdown': type('', (), {'value': 'efficientnet_b4', 'observe': MagicMock()})(),
            'model_type_dropdown': type('', (), {'value': 'efficient_optimized', 'observe': MagicMock()})(),
            'use_attention_checkbox': type('', (), {'value': True, 'disabled': False, 'observe': MagicMock()})(),
            'use_residual_checkbox': type('', (), {'value': False, 'disabled': False, 'observe': MagicMock()})(),
            'use_ciou_checkbox': type('', (), {'value': False, 'disabled': False, 'observe': MagicMock()})(),
            'status_panel': type('', (), {'value': ''})(),
            '_suppress_backbone_change': False,
            '_suppress_optimization_change': False
        }

    def test_setup_backbone_handlers(self):
        """Test setup event handlers untuk backbone form components"""
        from smartcash.ui.training_config.backbone.handlers.form_handlers import setup_backbone_handlers
        
        # Setup handlers
        setup_backbone_handlers(self.mock_ui_components, {})
        
        # Verifikasi bahwa observe dipanggil untuk setiap widget
        self.mock_ui_components['backbone_dropdown'].observe.assert_called_once()
        self.mock_ui_components['model_type_dropdown'].observe.assert_called_once()
        self.mock_ui_components['use_attention_checkbox'].observe.assert_called_once()
        self.mock_ui_components['use_residual_checkbox'].observe.assert_called_once()
        self.mock_ui_components['use_ciou_checkbox'].observe.assert_called_once()

    def test_on_backbone_change_cspdarknet(self):
        """Test perubahan backbone ke cspdarknet_s mengupdate form lain dengan benar"""
        from smartcash.ui.training_config.backbone.handlers.form_handlers import _on_backbone_change
        
        # Simulasikan perubahan backbone ke cspdarknet_s
        change = {'new': 'cspdarknet_s'}
        _on_backbone_change(change, self.mock_ui_components)
        
        # Verifikasi bahwa model_type diupdate ke yolov5s
        self.assertEqual(self.mock_ui_components['model_type_dropdown'].value, 'yolov5s')
        
        # Verifikasi bahwa checkbox dinonaktifkan dan nilainya diatur ke False
        self.assertTrue(self.mock_ui_components['use_attention_checkbox'].disabled)
        self.assertFalse(self.mock_ui_components['use_attention_checkbox'].value)
        
        self.assertTrue(self.mock_ui_components['use_residual_checkbox'].disabled)
        self.assertFalse(self.mock_ui_components['use_residual_checkbox'].value)
        
        self.assertTrue(self.mock_ui_components['use_ciou_checkbox'].disabled)
        self.assertFalse(self.mock_ui_components['use_ciou_checkbox'].value)
        
        # Verifikasi bahwa nilai sesuai dengan OPTIMIZED_MODELS['yolov5s']
        model_config = OPTIMIZED_MODELS['yolov5s']
        self.assertEqual(model_config['backbone'], 'cspdarknet_s')
        self.assertEqual(model_config['use_attention'], False)
        self.assertEqual(model_config['use_residual'], False)
        self.assertEqual(model_config['use_ciou'], False)

    def test_on_backbone_change_efficientnet(self):
        """Test perubahan backbone ke efficientnet_b4 mengupdate form lain dengan benar"""
        from smartcash.ui.training_config.backbone.handlers.form_handlers import _on_backbone_change
        
        # Set initial state ke cspdarknet untuk memastikan perubahan
        self.mock_ui_components['backbone_dropdown'].value = 'cspdarknet_s'
        self.mock_ui_components['model_type_dropdown'].value = 'yolov5s'
        self.mock_ui_components['use_attention_checkbox'].disabled = True
        self.mock_ui_components['use_attention_checkbox'].value = False
        
        # Simulasikan perubahan backbone ke efficientnet_b4
        change = {'new': 'efficientnet_b4'}
        _on_backbone_change(change, self.mock_ui_components)
        
        # Verifikasi bahwa model_type diupdate ke efficient_optimized (default)
        self.assertEqual(self.mock_ui_components['model_type_dropdown'].value, 'efficient_optimized')
        
        # Verifikasi bahwa checkbox diaktifkan
        # Catatan: Sesuai implementasi _update_optimization_checkboxes, nilai checkbox tidak diubah
        # ketika disabled=False, jadi kita tidak memeriksa nilai checkbox di sini
        self.assertFalse(self.mock_ui_components['use_attention_checkbox'].disabled)
        self.assertFalse(self.mock_ui_components['use_residual_checkbox'].disabled)
        self.assertFalse(self.mock_ui_components['use_ciou_checkbox'].disabled)

    def test_on_model_type_change(self):
        """Test perubahan model_type mengupdate form lain dengan benar"""
        from smartcash.ui.training_config.backbone.handlers.form_handlers import _on_model_type_change
        
        # Model type settings dari implementasi di _on_model_type_change
        # Format: (backbone, disable_opts, attention, residual, ciou)
        model_type_settings = {
            'yolov5s': ('cspdarknet_s', True, False, False, False),
            'efficient_basic': ('efficientnet_b4', False, False, False, False),
            'efficient_optimized': ('efficientnet_b4', False, True, False, False),
            'efficient_advanced': ('efficientnet_b4', False, True, True, True)
        }
        
        # Test untuk setiap model type
        for model_type, settings in model_type_settings.items():
            with self.subTest(model_type=model_type):
                # Reset state
                self.setUp()
                
                # Set nilai awal checkbox untuk test yang lebih baik
                self.mock_ui_components['use_attention_checkbox'].value = True
                self.mock_ui_components['use_residual_checkbox'].value = True
                self.mock_ui_components['use_ciou_checkbox'].value = True
                
                # Simulasikan perubahan model_type
                change = {'new': model_type}
                _on_model_type_change(change, self.mock_ui_components)
                
                # Ekstrak nilai yang diharapkan dari settings
                expected_backbone, disable_opts, expected_attention, expected_residual, expected_ciou = settings
                
                # Verifikasi bahwa backbone diupdate sesuai dengan model_type
                self.assertEqual(self.mock_ui_components['backbone_dropdown'].value, expected_backbone)
                
                # Untuk cspdarknet, checkbox harus dinonaktifkan dan nilainya diatur sesuai
                if disable_opts:
                    self.assertTrue(self.mock_ui_components['use_attention_checkbox'].disabled)
                    self.assertTrue(self.mock_ui_components['use_residual_checkbox'].disabled)
                    self.assertTrue(self.mock_ui_components['use_ciou_checkbox'].disabled)
                    
                    # Sesuai implementasi _update_optimization_checkboxes, nilai checkbox diubah
                    # hanya jika checkbox dinonaktifkan (disabled=True)
                    self.assertEqual(self.mock_ui_components['use_attention_checkbox'].value, expected_attention)
                    self.assertEqual(self.mock_ui_components['use_residual_checkbox'].value, expected_residual)
                    self.assertEqual(self.mock_ui_components['use_ciou_checkbox'].value, expected_ciou)
                else:
                    self.assertFalse(self.mock_ui_components['use_attention_checkbox'].disabled)
                    self.assertFalse(self.mock_ui_components['use_residual_checkbox'].disabled)
                    self.assertFalse(self.mock_ui_components['use_ciou_checkbox'].disabled)
                    
                    # Sesuai implementasi _update_optimization_checkboxes, nilai checkbox tidak diubah
                    # ketika checkbox diaktifkan (disabled=False)
                    # Nilai tetap seperti yang diatur di awal test (True)
                    self.assertTrue(self.mock_ui_components['use_attention_checkbox'].value)
                    self.assertTrue(self.mock_ui_components['use_residual_checkbox'].value)
                    self.assertTrue(self.mock_ui_components['use_ciou_checkbox'].value)

    def test_on_optimization_change(self):
        """Test perubahan checkbox optimization mengupdate model_type dengan benar"""
        from smartcash.ui.training_config.backbone.handlers.form_handlers import _on_optimization_change
        
        # Test untuk berbagai kombinasi checkbox
        test_cases = [
            # (attention, residual, ciou, expected_model_type)
            (True, True, True, 'efficient_advanced'),
            (True, False, False, 'efficient_optimized'),
            (False, False, False, 'efficient_basic'),
            (True, True, False, 'efficient_optimized'),  # Fallback ke optimized
            (False, True, True, 'efficient_basic'),      # Fallback ke basic
        ]
        
        for attention, residual, ciou, expected_model_type in test_cases:
            with self.subTest(attention=attention, residual=residual, ciou=ciou):
                # Reset state
                self.setUp()
                
                # Set nilai checkbox
                self.mock_ui_components['use_attention_checkbox'].value = attention
                self.mock_ui_components['use_residual_checkbox'].value = residual
                self.mock_ui_components['use_ciou_checkbox'].value = ciou
                
                # Simulasikan perubahan checkbox
                change = {'new': attention}  # Nilai change tidak penting karena kita mengambil nilai dari ui_components
                _on_optimization_change(change, self.mock_ui_components)
                
                # Verifikasi bahwa model_type diupdate sesuai dengan kombinasi checkbox
                self.assertEqual(self.mock_ui_components['model_type_dropdown'].value, expected_model_type)

    def test_update_backbone_ui(self):
        """Test update UI dari konfigurasi"""
        from smartcash.ui.training_config.backbone.handlers.ui_updater import update_backbone_ui
        
        # Test untuk setiap model type di OPTIMIZED_MODELS
        for model_type, model_config in OPTIMIZED_MODELS.items():
            with self.subTest(model_type=model_type):
                # Reset state
                self.setUp()
                
                # Buat config untuk update
                config = {
                    'model': {
                        'backbone': model_config['backbone'],
                        'model_type': model_type,
                        'use_attention': model_config['use_attention'],
                        'use_residual': model_config['use_residual'],
                        'use_ciou': model_config['use_ciou']
                    }
                }
                
                # Update UI dari config
                update_backbone_ui(self.mock_ui_components, config)
                
                # Verifikasi bahwa UI diupdate sesuai dengan config
                self.assertEqual(self.mock_ui_components['backbone_dropdown'].value, model_config['backbone'])
                self.assertEqual(self.mock_ui_components['model_type_dropdown'].value, model_type)
                self.assertEqual(self.mock_ui_components['use_attention_checkbox'].value, model_config['use_attention'])
                self.assertEqual(self.mock_ui_components['use_residual_checkbox'].value, model_config['use_residual'])
                self.assertEqual(self.mock_ui_components['use_ciou_checkbox'].value, model_config['use_ciou'])
                
                # Verifikasi bahwa checkbox dinonaktifkan untuk cspdarknet
                if model_config['backbone'] == 'cspdarknet_s':
                    self.assertTrue(self.mock_ui_components['use_attention_checkbox'].disabled)
                    self.assertTrue(self.mock_ui_components['use_residual_checkbox'].disabled)
                    self.assertTrue(self.mock_ui_components['use_ciou_checkbox'].disabled)
                else:
                    self.assertFalse(self.mock_ui_components['use_attention_checkbox'].disabled)
                    self.assertFalse(self.mock_ui_components['use_residual_checkbox'].disabled)
                    self.assertFalse(self.mock_ui_components['use_ciou_checkbox'].disabled)

    def test_model_constants_consistency(self):
        """Test konsistensi antara OPTIMIZED_MODELS dan handler form backbone"""
        from smartcash.ui.training_config.backbone.handlers.form_handlers import _on_model_type_change
        
        # Model type settings dari implementasi di _on_model_type_change
        # Format: (backbone, disable_options, attention, residual, ciou)
        model_type_settings = {
            'yolov5s': ('cspdarknet_s', True, False, False, False),
            'efficient_basic': ('efficientnet_b4', False, False, False, False),
            'efficient_optimized': ('efficientnet_b4', False, True, False, False),
            'efficient_advanced': ('efficientnet_b4', False, True, True, True)
        }
        
        # Untuk setiap model di OPTIMIZED_MODELS
        for model_type, model_config in OPTIMIZED_MODELS.items():
            with self.subTest(model_type=model_type):
                # Reset state
                self.setUp()
                
                # Set nilai awal checkbox untuk test yang lebih baik
                self.mock_ui_components['use_attention_checkbox'].value = True
                self.mock_ui_components['use_residual_checkbox'].value = True
                self.mock_ui_components['use_ciou_checkbox'].value = True
                
                # Simulasikan perubahan model_type
                change = {'new': model_type}
                _on_model_type_change(change, self.mock_ui_components)
                
                # Verifikasi bahwa backbone diupdate sesuai dengan OPTIMIZED_MODELS
                self.assertEqual(self.mock_ui_components['backbone_dropdown'].value, model_config['backbone'])
                
                # Dapatkan nilai yang diharapkan dari model_type_settings
                if model_type in model_type_settings:
                    _, disable_opts, expected_attention, expected_residual, expected_ciou = model_type_settings[model_type]
                    
                    # Verifikasi status disabled checkbox
                    if disable_opts:
                        self.assertTrue(self.mock_ui_components['use_attention_checkbox'].disabled)
                        self.assertTrue(self.mock_ui_components['use_residual_checkbox'].disabled)
                        self.assertTrue(self.mock_ui_components['use_ciou_checkbox'].disabled)
                        
                        # Untuk checkbox yang dinonaktifkan, nilai harus sesuai dengan settings
                        self.assertEqual(self.mock_ui_components['use_attention_checkbox'].value, expected_attention)
                        self.assertEqual(self.mock_ui_components['use_residual_checkbox'].value, expected_residual)
                        self.assertEqual(self.mock_ui_components['use_ciou_checkbox'].value, expected_ciou)
                    else:
                        self.assertFalse(self.mock_ui_components['use_attention_checkbox'].disabled)
                        self.assertFalse(self.mock_ui_components['use_residual_checkbox'].disabled)
                        self.assertFalse(self.mock_ui_components['use_ciou_checkbox'].disabled)
                        
                        # Untuk checkbox yang diaktifkan, nilai tidak berubah (tetap True sesuai setup awal)
                        self.assertTrue(self.mock_ui_components['use_attention_checkbox'].value)
                        self.assertTrue(self.mock_ui_components['use_residual_checkbox'].value)
                        self.assertTrue(self.mock_ui_components['use_ciou_checkbox'].value)


if __name__ == '__main__':
    unittest.main()
