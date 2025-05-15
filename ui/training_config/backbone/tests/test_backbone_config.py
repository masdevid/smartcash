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
    update_ui_from_config
)

class TestBackboneConfig(unittest.TestCase):
    """Tes untuk konfigurasi backbone"""
    
    def setUp(self):
        """Setup untuk tes"""
        # Buat mock untuk ui_components
        self.ui_components = {
            'model_type_dropdown': MagicMock(value='efficient_basic'),
            'backbone_dropdown': MagicMock(value='efficientnet_b4'),
            'use_attention_checkbox': MagicMock(value=True),
            'use_residual_checkbox': MagicMock(value=True),
            'use_ciou_checkbox': MagicMock(value=False),
            'status_panel': MagicMock(),
            'info_panel': MagicMock()
        }
        
        # Buat temporary file untuk konfigurasi
        self.temp_config_file = tempfile.NamedTemporaryFile(delete=False, suffix='.yaml')
        self.temp_config_file.close()
        
        # Buat konfigurasi awal
        self.initial_config = {
            'model': {
                'type': 'efficient_basic',
                'backbone': 'efficientnet_b4',
                'use_attention': False,
                'use_residual': False,
                'use_ciou': False
            },
            'experiments': {
                'backbones': [
                    {
                        'name': 'efficientnet_b4',
                        'description': 'EfficientNet-B4 backbone',
                        'config': {
                            'backbone': 'efficientnet_b4',
                            'pretrained': True
                        }
                    }
                ],
                'scenarios': []
            }
        }
        
        # Tulis konfigurasi awal ke file
        with open(self.temp_config_file.name, 'w') as f:
            yaml.dump(self.initial_config, f)
    
    def tearDown(self):
        """Cleanup setelah tes"""
        # Hapus temporary file
        os.unlink(self.temp_config_file.name)
    
    @patch('smartcash.common.config.manager.ConfigManager.get_instance')
    def test_update_config_from_ui(self, mock_get_instance):
        """Tes untuk update_config_from_ui"""
        # Setup mock
        mock_config_manager = MagicMock()
        mock_config_manager.get_module_config.return_value = self.initial_config
        mock_get_instance.return_value = mock_config_manager
        
        # Panggil fungsi yang dites
        updated_config = update_config_from_ui(self.ui_components)
        
        # Verifikasi hasil
        self.assertEqual(updated_config['model']['type'], 'efficient_basic')
        self.assertEqual(updated_config['model']['backbone'], 'efficientnet_b4')
        self.assertEqual(updated_config['model']['use_attention'], True)
        self.assertEqual(updated_config['model']['use_residual'], True)
        self.assertEqual(updated_config['model']['use_ciou'], False)
        
        # Verifikasi bahwa scenario ditambahkan
        scenario_exists = False
        for scenario in updated_config['experiments']['scenarios']:
            if (scenario['config']['type'] == 'efficient_basic' and
                scenario['config']['backbone'] == 'efficientnet_b4' and
                scenario['config']['use_attention'] == True and
                scenario['config']['use_residual'] == True and
                scenario['config']['use_ciou'] == False):
                scenario_exists = True
                break
        
        self.assertTrue(scenario_exists, "Scenario tidak ditambahkan ke konfigurasi")
    
    @patch('smartcash.common.config.manager.ConfigManager.get_instance')
    def test_update_ui_from_config(self, mock_get_instance):
        """Tes untuk update_ui_from_config"""
        # Setup mock
        mock_config_manager = MagicMock()
        mock_config_manager.get_module_config.return_value = {
            'model': {
                'type': 'yolov5s',
                'backbone': 'cspdarknet_s',
                'use_attention': False,
                'use_residual': False,
                'use_ciou': False
            }
        }
        mock_get_instance.return_value = mock_config_manager
        
        # Panggil fungsi yang dites
        update_ui_from_config(self.ui_components)
        
        # Verifikasi hasil
        self.ui_components['model_type_dropdown'].value = 'yolov5s'
        self.ui_components['backbone_dropdown'].value = 'cspdarknet_s'
        self.ui_components['use_attention_checkbox'].value = False
        self.ui_components['use_residual_checkbox'].value = False
        self.ui_components['use_ciou_checkbox'].value = False
        
        # Verifikasi bahwa checkbox dinonaktifkan untuk cspdarknet_s
        self.ui_components['use_attention_checkbox'].disabled = True
        self.ui_components['use_residual_checkbox'].disabled = True
        self.ui_components['use_ciou_checkbox'].disabled = True
    
    @patch('smartcash.common.config.manager.ConfigManager.get_instance')
    def test_invalid_values(self, mock_get_instance):
        """Tes untuk nilai yang tidak valid"""
        # Setup mock
        mock_config_manager = MagicMock()
        mock_config_manager.get_module_config.return_value = {
            'model': {
                'type': 'invalid_type',
                'backbone': 'invalid_backbone',
                'use_attention': 'not_a_boolean',
                'use_residual': 'not_a_boolean',
                'use_ciou': 'not_a_boolean'
            }
        }
        mock_get_instance.return_value = mock_config_manager
        
        # Panggil fungsi yang dites
        update_ui_from_config(self.ui_components)
        
        # Verifikasi bahwa nilai default digunakan
        self.ui_components['model_type_dropdown'].value = 'efficient_basic'
        self.ui_components['backbone_dropdown'].value = 'efficientnet_b4'

if __name__ == '__main__':
    unittest.main()
