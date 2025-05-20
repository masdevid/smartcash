"""
File: tests/test_hyperparameters_ui.py
Deskripsi: Pengujian untuk komponen UI hyperparameter
"""

import unittest
from unittest.mock import MagicMock, patch
import ipywidgets as widgets

from smartcash.ui.training_config.hyperparameters.components.main_components import (
    create_hyperparameters_ui_components
)
from smartcash.ui.training_config.hyperparameters.components.info_panel_components import (
    create_hyperparameters_info_panel
)
from smartcash.ui.training_config.hyperparameters.handlers.config_handlers import (
    update_ui_from_config,
    update_config_from_ui
)

class TestHyperparametersUI(unittest.TestCase):
    """Pengujian untuk komponen UI hyperparameter"""
    
    def setUp(self):
        """Setup untuk pengujian"""
        # Mock config
        self.mock_config = {
            'hyperparameters': {
                'enabled': True,
                'batch_size': 16,
                'image_size': 640,
                'epochs': 100,
                'augment': True,
                'optimizer': {
                    'type': 'SGD',
                    'learning_rate': 0.01,
                    'momentum': 0.937,
                    'weight_decay': 0.0005
                },
                'scheduler': {
                    'enabled': True,
                    'type': 'cosine',
                    'warmup_epochs': 3,
                    'warmup_momentum': 0.8,
                    'warmup_bias_lr': 0.1
                },
                'early_stopping': {
                    'enabled': True,
                    'patience': 10,
                    'min_delta': 0.001
                },
                'save_best': {
                    'enabled': True,
                    'metric': 'mAP_0.5'
                }
            }
        }
        
        # Mock UI components
        self.mock_ui_components = {
            'enabled_checkbox': MagicMock(value=True),
            'batch_size_slider': MagicMock(value=16),
            'image_size_slider': MagicMock(value=640),
            'epochs_slider': MagicMock(value=100),
            'augment_checkbox': MagicMock(value=True),
            'optimizer_dropdown': MagicMock(value='SGD'),
            'learning_rate_slider': MagicMock(value=0.01),
            'momentum_slider': MagicMock(value=0.937),
            'weight_decay_slider': MagicMock(value=0.0005),
            'scheduler_checkbox': MagicMock(value=True),
            'scheduler_dropdown': MagicMock(value='cosine'),
            'warmup_epochs_slider': MagicMock(value=3),
            'warmup_momentum_slider': MagicMock(value=0.8),
            'warmup_bias_lr_slider': MagicMock(value=0.1),
            'early_stopping_checkbox': MagicMock(value=True),
            'patience_slider': MagicMock(value=10),
            'min_delta_slider': MagicMock(value=0.001),
            'save_best_checkbox': MagicMock(value=True),
            'checkpoint_metric_dropdown': MagicMock(value='mAP_0.5'),
            'status': MagicMock(),
            'update_hyperparameters_info': MagicMock()
        }
    
    def test_create_hyperparameters_ui_components(self):
        """Pengujian pembuatan komponen UI hyperparameter"""
        # Panggil fungsi
        ui_components = create_hyperparameters_ui_components()
        
        # Verifikasi hasil
        self.assertIsInstance(ui_components, dict)
        self.assertIn('enabled_checkbox', ui_components)
        self.assertIn('batch_size_slider', ui_components)
        self.assertIn('image_size_slider', ui_components)
        self.assertIn('epochs_slider', ui_components)
        self.assertIn('augment_checkbox', ui_components)
        self.assertIn('optimizer_dropdown', ui_components)
        self.assertIn('learning_rate_slider', ui_components)
        self.assertIn('momentum_slider', ui_components)
        self.assertIn('weight_decay_slider', ui_components)
        self.assertIn('scheduler_checkbox', ui_components)
        self.assertIn('scheduler_dropdown', ui_components)
        self.assertIn('warmup_epochs_slider', ui_components)
        self.assertIn('warmup_momentum_slider', ui_components)
        self.assertIn('warmup_bias_lr_slider', ui_components)
        self.assertIn('early_stopping_checkbox', ui_components)
        self.assertIn('patience_slider', ui_components)
        self.assertIn('min_delta_slider', ui_components)
        self.assertIn('save_best_checkbox', ui_components)
        self.assertIn('checkpoint_metric_dropdown', ui_components)
        self.assertIn('status', ui_components)
        self.assertIn('update_hyperparameters_info', ui_components)
    
    def test_create_hyperparameters_info_panel(self):
        """Pengujian pembuatan panel info hyperparameter"""
        # Panggil fungsi
        info_panel, update_func = create_hyperparameters_info_panel()
        
        # Verifikasi hasil
        self.assertIsInstance(info_panel, widgets.Output)
        self.assertIsNotNone(update_func)
        self.assertTrue(callable(update_func))
    
    @patch('smartcash.ui.training_config.hyperparameters.handlers.config_handlers.get_config_manager')
    def test_update_ui_from_config(self, mock_get_config_manager):
        """Pengujian update UI dari konfigurasi"""
        # Setup mock
        mock_config_manager = MagicMock()
        mock_config_manager.get_module_config.return_value = self.mock_config
        mock_get_config_manager.return_value = mock_config_manager
        
        # Panggil fungsi
        update_ui_from_config(self.mock_ui_components, self.mock_config)
        
        # Verifikasi hasil
        self.assertEqual(self.mock_ui_components['enabled_checkbox'].value, True)
        self.assertEqual(self.mock_ui_components['batch_size_slider'].value, 16)
        self.assertEqual(self.mock_ui_components['image_size_slider'].value, 640)
        self.assertEqual(self.mock_ui_components['epochs_slider'].value, 100)
        self.assertEqual(self.mock_ui_components['augment_checkbox'].value, True)
        self.assertEqual(self.mock_ui_components['optimizer_dropdown'].value, 'SGD')
        self.assertEqual(self.mock_ui_components['learning_rate_slider'].value, 0.01)
        self.assertEqual(self.mock_ui_components['momentum_slider'].value, 0.937)
        self.assertEqual(self.mock_ui_components['weight_decay_slider'].value, 0.0005)
        self.assertEqual(self.mock_ui_components['scheduler_checkbox'].value, True)
        self.assertEqual(self.mock_ui_components['scheduler_dropdown'].value, 'cosine')
        self.assertEqual(self.mock_ui_components['warmup_epochs_slider'].value, 3)
        self.assertEqual(self.mock_ui_components['warmup_momentum_slider'].value, 0.8)
        self.assertEqual(self.mock_ui_components['warmup_bias_lr_slider'].value, 0.1)
        self.assertEqual(self.mock_ui_components['early_stopping_checkbox'].value, True)
        self.assertEqual(self.mock_ui_components['patience_slider'].value, 10)
        self.assertEqual(self.mock_ui_components['min_delta_slider'].value, 0.001)
        self.assertEqual(self.mock_ui_components['save_best_checkbox'].value, True)
        self.assertEqual(self.mock_ui_components['checkpoint_metric_dropdown'].value, 'mAP_0.5')
    
    @patch('smartcash.ui.training_config.hyperparameters.handlers.config_handlers.get_config_manager')
    def test_update_config_from_ui(self, mock_get_config_manager):
        """Pengujian update konfigurasi dari UI"""
        # Setup mock
        mock_config_manager = MagicMock()
        mock_config_manager.get_module_config.return_value = self.mock_config
        mock_get_config_manager.return_value = mock_config_manager
        
        # Panggil fungsi
        config = update_config_from_ui(self.mock_ui_components)
        
        # Verifikasi hasil
        self.assertEqual(config['hyperparameters']['enabled'], True)
        self.assertEqual(config['hyperparameters']['batch_size'], 16)
        self.assertEqual(config['hyperparameters']['image_size'], 640)
        self.assertEqual(config['hyperparameters']['epochs'], 100)
        self.assertEqual(config['hyperparameters']['augment'], True)
        self.assertEqual(config['hyperparameters']['optimizer']['type'], 'SGD')
        self.assertEqual(config['hyperparameters']['optimizer']['learning_rate'], 0.01)
        self.assertEqual(config['hyperparameters']['optimizer']['momentum'], 0.937)
        self.assertEqual(config['hyperparameters']['optimizer']['weight_decay'], 0.0005)
        self.assertEqual(config['hyperparameters']['scheduler']['enabled'], True)
        self.assertEqual(config['hyperparameters']['scheduler']['type'], 'cosine')
        self.assertEqual(config['hyperparameters']['scheduler']['warmup_epochs'], 3)
        self.assertEqual(config['hyperparameters']['scheduler']['warmup_momentum'], 0.8)
        self.assertEqual(config['hyperparameters']['scheduler']['warmup_bias_lr'], 0.1)
        self.assertEqual(config['hyperparameters']['early_stopping']['enabled'], True)
        self.assertEqual(config['hyperparameters']['early_stopping']['patience'], 10)
        self.assertEqual(config['hyperparameters']['early_stopping']['min_delta'], 0.001)
        self.assertEqual(config['hyperparameters']['save_best']['enabled'], True)
        self.assertEqual(config['hyperparameters']['save_best']['metric'], 'mAP_0.5')

if __name__ == '__main__':
    unittest.main()
