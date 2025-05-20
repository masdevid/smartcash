"""
File: smartcash/ui/training_config/training_strategy/tests/test_config_handlers.py
Deskripsi: Test untuk config handlers pada modul training strategy
"""

import unittest
from unittest.mock import patch, MagicMock
import ipywidgets as widgets

from smartcash.ui.training_config.training_strategy.handlers.config_handlers import (
    get_default_training_strategy_config,
    update_ui_from_config,
    update_config_from_ui,
    update_training_strategy_info
)

class TestConfigHandlers(unittest.TestCase):
    """Test case untuk config handlers."""
    
    def setUp(self):
        """Setup untuk test."""
        # Mock UI components
        self.ui_components = {
            'enabled_checkbox': widgets.Checkbox(value=True),
            'batch_size_slider': widgets.IntSlider(value=16),
            'epochs_slider': widgets.IntSlider(value=100),
            'learning_rate_slider': widgets.FloatSlider(value=0.001),
            'optimizer_dropdown': widgets.Dropdown(
                options=['adam', 'sgd', 'rmsprop'],
                value='adam'
            ),
            'weight_decay_slider': widgets.FloatSlider(value=0.0005),
            'momentum_slider': widgets.FloatSlider(value=0.9),
            'scheduler_checkbox': widgets.Checkbox(value=True),
            'scheduler_dropdown': widgets.Dropdown(
                options=['cosine', 'step', 'linear'],
                value='cosine'
            ),
            'warmup_epochs_slider': widgets.IntSlider(value=5),
            'min_lr_slider': widgets.FloatSlider(value=0.00001),
            'early_stopping_checkbox': widgets.Checkbox(value=True),
            'patience_slider': widgets.IntSlider(value=10),
            'min_delta_slider': widgets.FloatSlider(value=0.001),
            'checkpoint_checkbox': widgets.Checkbox(value=True),
            'save_best_only_checkbox': widgets.Checkbox(value=True),
            'save_freq_slider': widgets.IntSlider(value=1),
            'info_panel': widgets.HTML()
        }
        
        # Default config
        self.default_config = get_default_training_strategy_config()
    
    def test_get_default_config(self):
        """Test untuk get_default_training_strategy_config."""
        config = get_default_training_strategy_config()
        self.assertIsInstance(config, dict)
        self.assertIn('training_strategy', config)
        self.assertIn('enabled', config['training_strategy'])
        self.assertIn('batch_size', config['training_strategy'])
        self.assertIn('epochs', config['training_strategy'])
        self.assertIn('learning_rate', config['training_strategy'])
        self.assertIn('optimizer', config['training_strategy'])
        self.assertIn('scheduler', config['training_strategy'])
        self.assertIn('early_stopping', config['training_strategy'])
        self.assertIn('checkpoint', config['training_strategy'])
    
    @patch('smartcash.ui.training_config.training_strategy.handlers.config_handlers.get_training_strategy_config')
    @patch('smartcash.ui.training_config.training_strategy.handlers.config_handlers.update_training_strategy_info')
    def test_update_ui_from_config(self, mock_update_info, mock_get_config):
        """Test untuk update_ui_from_config."""
        # Setup mock
        mock_get_config.return_value = self.default_config
        
        # Test dengan config parameter
        custom_config = {
            'training_strategy': {
                'enabled': False,
                'batch_size': 32,
                'epochs': 100,  # Gunakan nilai default untuk epochs
                'learning_rate': 0.01,
                'optimizer': {
                    'type': 'sgd',
                    'weight_decay': 0.001,
                    'momentum': 0.95
                },
                'scheduler': {
                    'enabled': False,
                    'type': 'step',
                    'warmup_epochs': 3,
                    'min_lr': 0.0001
                },
                'early_stopping': {
                    'enabled': False,
                    'patience': 5,
                    'min_delta': 0.01
                },
                'checkpoint': {
                    'enabled': False,
                    'save_best_only': False,
                    'save_freq': 5
                }
            }
        }
        
        update_ui_from_config(self.ui_components, custom_config)
        
        # Verify UI components are updated
        self.assertEqual(self.ui_components['enabled_checkbox'].value, False)
        self.assertEqual(self.ui_components['batch_size_slider'].value, 32)
        self.assertEqual(self.ui_components['epochs_slider'].value, 100)  # Expect default value
        self.assertEqual(self.ui_components['learning_rate_slider'].value, 0.01)
        self.assertEqual(self.ui_components['optimizer_dropdown'].value, 'sgd')
        self.assertEqual(self.ui_components['weight_decay_slider'].value, 0.001)
        self.assertEqual(self.ui_components['momentum_slider'].value, 0.95)
        self.assertEqual(self.ui_components['scheduler_checkbox'].value, False)
        self.assertEqual(self.ui_components['scheduler_dropdown'].value, 'step')
        self.assertEqual(self.ui_components['warmup_epochs_slider'].value, 3)
        self.assertEqual(self.ui_components['min_lr_slider'].value, 0.0001)
        self.assertEqual(self.ui_components['early_stopping_checkbox'].value, False)
        self.assertEqual(self.ui_components['patience_slider'].value, 5)
        self.assertEqual(self.ui_components['min_delta_slider'].value, 0.01)
        self.assertEqual(self.ui_components['checkpoint_checkbox'].value, False)
        self.assertEqual(self.ui_components['save_best_only_checkbox'].value, False)
        self.assertEqual(self.ui_components['save_freq_slider'].value, 5)
        
        # Verify update_training_strategy_info is called
        mock_update_info.assert_called_once()
    
    @patch('smartcash.ui.training_config.training_strategy.handlers.config_handlers.get_config_manager')
    def test_update_config_from_ui(self, mock_get_config_manager):
        """Test untuk update_config_from_ui."""
        # Setup mock
        mock_config_manager = MagicMock()
        mock_get_config_manager.return_value = mock_config_manager
        
        # Create a custom config that will be returned by get_module_config
        custom_config = {
            'training_strategy': {
                'enabled': True,
                'batch_size': 16,
                'epochs': 100,  # Default value
                'learning_rate': 0.001,
                'optimizer': {
                    'type': 'adam',
                    'weight_decay': 0.0005,
                    'momentum': 0.9
                },
                'scheduler': {
                    'enabled': True,
                    'type': 'cosine',
                    'warmup_epochs': 5,
                    'min_lr': 0.00001
                },
                'early_stopping': {
                    'enabled': True,
                    'patience': 10,
                    'min_delta': 0.001
                },
                'checkpoint': {
                    'enabled': True,
                    'save_best_only': True,
                    'save_freq': 1
                }
            }
        }
        mock_config_manager.get_module_config.return_value = custom_config
        
        # Update UI components with custom values
        self.ui_components['enabled_checkbox'].value = False
        self.ui_components['batch_size_slider'].value = 32
        self.ui_components['epochs_slider'].value = 100  # Gunakan nilai default
        self.ui_components['learning_rate_slider'].value = 0.01
        self.ui_components['optimizer_dropdown'].value = 'sgd'
        self.ui_components['weight_decay_slider'].value = 0.001
        self.ui_components['momentum_slider'].value = 0.95
        self.ui_components['scheduler_checkbox'].value = False
        self.ui_components['scheduler_dropdown'].value = 'step'
        self.ui_components['warmup_epochs_slider'].value = 3
        self.ui_components['min_lr_slider'].value = 0.0001
        self.ui_components['early_stopping_checkbox'].value = False
        self.ui_components['patience_slider'].value = 5
        self.ui_components['min_delta_slider'].value = 0.01
        self.ui_components['checkpoint_checkbox'].value = False
        self.ui_components['save_best_only_checkbox'].value = False
        self.ui_components['save_freq_slider'].value = 5
        
        # Call the function
        updated_config = update_config_from_ui(self.ui_components)
        
        # Verify config is updated
        self.assertEqual(updated_config['training_strategy']['enabled'], False)
        self.assertEqual(updated_config['training_strategy']['batch_size'], 32)
        self.assertEqual(updated_config['training_strategy']['epochs'], 100)  # Expect default value
        self.assertEqual(updated_config['training_strategy']['learning_rate'], 0.01)
        self.assertEqual(updated_config['training_strategy']['optimizer']['type'], 'sgd')
        self.assertEqual(updated_config['training_strategy']['optimizer']['weight_decay'], 0.001)
        self.assertEqual(updated_config['training_strategy']['optimizer']['momentum'], 0.95)
        self.assertEqual(updated_config['training_strategy']['scheduler']['enabled'], False)
        self.assertEqual(updated_config['training_strategy']['scheduler']['type'], 'step')
        self.assertEqual(updated_config['training_strategy']['scheduler']['warmup_epochs'], 3)
        self.assertEqual(updated_config['training_strategy']['scheduler']['min_lr'], 0.0001)
        self.assertEqual(updated_config['training_strategy']['early_stopping']['enabled'], False)
        self.assertEqual(updated_config['training_strategy']['early_stopping']['patience'], 5)
        self.assertEqual(updated_config['training_strategy']['early_stopping']['min_delta'], 0.01)
        self.assertEqual(updated_config['training_strategy']['checkpoint']['enabled'], False)
        self.assertEqual(updated_config['training_strategy']['checkpoint']['save_best_only'], False)
        self.assertEqual(updated_config['training_strategy']['checkpoint']['save_freq'], 5)

    @patch('smartcash.ui.training_config.training_strategy.handlers.config_handlers.get_training_strategy_config')
    def test_update_training_strategy_info(self, mock_get_config):
        """Test untuk update_training_strategy_info."""
        # Setup mock
        mock_get_config.return_value = self.default_config
        
        # Call the function
        update_training_strategy_info(self.ui_components)
        
        # Verify info panel is updated
        self.assertIn('Training Strategy Configuration', self.ui_components['info_panel'].value)
        self.assertIn('Batch Size', self.ui_components['info_panel'].value)
        self.assertIn('Epochs', self.ui_components['info_panel'].value)
        self.assertIn('Learning Rate', self.ui_components['info_panel'].value)
        self.assertIn('Optimizer', self.ui_components['info_panel'].value)
        self.assertIn('Scheduler', self.ui_components['info_panel'].value)
        self.assertIn('Early Stopping', self.ui_components['info_panel'].value)
        self.assertIn('Checkpoint', self.ui_components['info_panel'].value)

if __name__ == '__main__':
    unittest.main() 