"""
File: smartcash/ui/training_config/backbone/tests/test_backbone_config.py
Deskripsi: Tes untuk memastikan konfigurasi backbone dapat disimpan dan dimuat dengan benar
"""

import unittest
import os
import tempfile
import yaml
from unittest.mock import patch, MagicMock

from smartcash.common.config.manager import ConfigManager
from smartcash.ui.training_config.backbone.handlers.config_handlers import (
    update_config_from_ui,
    update_ui_from_config,
    get_default_backbone_config
)

class Dummy:
    def __init__(self, value=None):
        self.value = value

class TestBackboneConfig(unittest.TestCase):
    """Tes untuk konfigurasi backbone"""
    
    def setUp(self):
        """Setup untuk tes"""
        # Buat mock untuk ui_components
        self.ui_components = {
            'enabled_checkbox': Dummy(True),
            'backbone_dropdown': Dummy('efficientnet_b4'),
            'pretrained_checkbox': Dummy(True),
            'freeze_backbone_checkbox': Dummy(False),
            'freeze_bn_checkbox': Dummy(True),
            'dropout_slider': Dummy(0.2),
            'activation_dropdown': Dummy('relu'),
            'normalization_dropdown': Dummy('batch'),
            'bn_momentum_slider': Dummy(0.1),
            'weights_path': Dummy(None),
            'strict_weights_checkbox': Dummy(True),
            'info_panel': Dummy(''),
            'status_panel': Dummy(''),
        }
        
        # Buat temporary file untuk konfigurasi
        self.temp_config_file = tempfile.NamedTemporaryFile(delete=False, suffix='.yaml')
        self.temp_config_file.close()
        
        # Buat konfigurasi awal
        self.initial_config = get_default_backbone_config()
        
        # Tulis konfigurasi awal ke file
        with open(self.temp_config_file.name, 'w') as f:
            yaml.dump(self.initial_config, f)
    
    def tearDown(self):
        """Cleanup setelah tes"""
        # Hapus temporary file
        os.unlink(self.temp_config_file.name)
    
    @patch('smartcash.common.config.manager.get_config_manager')
    def test_update_config_from_ui(self, mock_get_instance):
        """Tes untuk update_config_from_ui"""
        # Setup mock
        mock_config_manager = MagicMock()
        mock_config_manager.get_module_config.return_value = self.initial_config
        mock_get_instance.return_value = mock_config_manager
        
        # Panggil fungsi yang dites
        updated_config = update_config_from_ui(self.ui_components)
        
        # Verifikasi hasil
        self.assertEqual(updated_config['backbone']['enabled'], True)
        self.assertEqual(updated_config['backbone']['type'], 'efficientnet_b4')
        self.assertEqual(updated_config['backbone']['pretrained'], True)
        self.assertEqual(updated_config['backbone']['freeze_backbone'], False)
        self.assertEqual(updated_config['backbone']['freeze_bn'], True)
        self.assertEqual(updated_config['backbone']['dropout'], 0.2)
        self.assertEqual(updated_config['backbone']['activation'], 'relu')
        self.assertEqual(updated_config['backbone']['normalization']['type'], 'batch')
        self.assertEqual(updated_config['backbone']['normalization']['momentum'], 0.1)
        self.assertEqual(updated_config['backbone']['weights']['path'], None)
        self.assertEqual(updated_config['backbone']['weights']['strict'], True)
    
    @patch('smartcash.common.config.manager.get_config_manager')
    def test_update_ui_from_config(self, mock_get_instance):
        """Tes untuk update_ui_from_config"""
        # Setup mock (not used for config in this test)
        mock_config_manager = MagicMock()
        mock_get_instance.return_value = mock_config_manager
        
        # Panggil fungsi yang dites dengan config langsung
        update_ui_from_config(self.ui_components, {
            'backbone': {
                'enabled': False,
                'type': 'resnet50',
                'pretrained': False,
                'freeze_backbone': True,
                'freeze_bn': False,
                'dropout': 0.3,
                'activation': 'swish',
                'normalization': {
                    'type': 'instance',
                    'momentum': 0.2
                },
                'weights': {
                    'path': '/path/to/weights',
                    'strict': False
                }
            }
        })
        
        # Verifikasi hasil
        self.assertEqual(self.ui_components['enabled_checkbox'].value, False)
        self.assertEqual(self.ui_components['backbone_dropdown'].value, 'resnet50')
        self.assertEqual(self.ui_components['pretrained_checkbox'].value, False)
        self.assertEqual(self.ui_components['freeze_backbone_checkbox'].value, True)
        self.assertEqual(self.ui_components['freeze_bn_checkbox'].value, False)
        self.assertEqual(self.ui_components['dropout_slider'].value, 0.3)
        self.assertEqual(self.ui_components['activation_dropdown'].value, 'swish')
        self.assertEqual(self.ui_components['normalization_dropdown'].value, 'instance')
        self.assertEqual(self.ui_components['bn_momentum_slider'].value, 0.2)
        self.assertEqual(self.ui_components['weights_path'].value, '/path/to/weights')
        self.assertEqual(self.ui_components['strict_weights_checkbox'].value, False)
    
    @patch('smartcash.common.config.manager.get_config_manager')
    def test_invalid_values(self, mock_get_instance):
        """Tes untuk nilai yang tidak valid"""
        # Setup mock dengan nilai yang tidak valid
        mock_config_manager = MagicMock()
        mock_config_manager.get_module_config.return_value = {
            'backbone': {
                'enabled': 'not_a_boolean',
                'type': 'invalid_type',
                'pretrained': 'not_a_boolean',
                'freeze_backbone': 'not_a_boolean',
                'freeze_bn': 'not_a_boolean',
                'dropout': 'not_a_number',
                'activation': 'invalid_activation',
                'normalization': {
                    'type': 'invalid_norm',
                    'momentum': 'not_a_number'
                },
                'weights': {
                    'path': 123,  # should be string or None
                    'strict': 'not_a_boolean'
                }
            }
        }
        mock_get_instance.return_value = mock_config_manager
        
        # Panggil fungsi yang dites
        update_ui_from_config(self.ui_components)
        
        # Verifikasi bahwa nilai default digunakan
        default_config = get_default_backbone_config()
        self.assertEqual(self.ui_components['backbone_dropdown'].value, default_config['backbone']['type'])
        self.assertEqual(self.ui_components['activation_dropdown'].value, default_config['backbone']['activation'])
        self.assertEqual(self.ui_components['normalization_dropdown'].value, default_config['backbone']['normalization']['type'])

if __name__ == '__main__':
    unittest.main()
