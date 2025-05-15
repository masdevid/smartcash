"""
File: smartcash/ui/training_config/training_strategy/tests/test_ui.py
Deskripsi: Test untuk komponen UI strategi pelatihan model
"""

import unittest
from unittest.mock import patch, MagicMock
import ipywidgets as widgets

from smartcash.ui.training_config.training_strategy.components.training_strategy_components import create_training_strategy_ui_components
from smartcash.ui.training_config.training_strategy.handlers.form_handlers import setup_training_strategy_form_handlers
from smartcash.ui.training_config.training_strategy.handlers.button_handlers import setup_training_strategy_button_handlers
from smartcash.ui.training_config.training_strategy.training_strategy_initializer import initialize_training_strategy_ui, get_training_strategy_ui

class TestTrainingStrategyUI(unittest.TestCase):
    """Test case untuk komponen UI strategi pelatihan."""
    
    def setUp(self):
        """Setup untuk test."""
        # Buat mock environment dan config manager
        self.mock_env = MagicMock()
        self.mock_env.is_drive_mounted = True
        self.mock_env.drive_path = '/content/drive'
        
        self.mock_config_manager = MagicMock()
        self.mock_config_manager.get_module_config.return_value = {
            'validation': {
                'frequency': 1,
                'iou_thres': 0.6,
                'conf_thres': 0.001
            },
            'multi_scale': True,
            'training_utils': {
                'experiment_name': 'test_experiment',
                'checkpoint_dir': '/test/checkpoints',
                'tensorboard': True,
                'log_metrics_every': 10,
                'visualize_batch_every': 100,
                'gradient_clipping': 1.0,
                'mixed_precision': True,
                'layer_mode': 'single'
            }
        }
        self.mock_config_manager.save_module_config.return_value = True
        self.mock_config_manager.get_ui_components.return_value = None
    
    @patch('smartcash.ui.training_config.training_strategy.components.training_strategy_components.create_config_buttons')
    def test_create_training_strategy_ui_components(self, mock_create_config_buttons):
        """Test membuat komponen UI."""
        # Setup mock
        mock_create_config_buttons.return_value = {
            'save_button': MagicMock(),
            'reset_button': MagicMock(),
            'container': MagicMock()
        }
        
        # Panggil fungsi yang ditest
        ui_components = create_training_strategy_ui_components()
        
        # Verifikasi hasil
        self.assertIsInstance(ui_components, dict)
        self.assertIn('experiment_name', ui_components)
        self.assertIn('checkpoint_dir', ui_components)
        self.assertIn('tensorboard', ui_components)
        self.assertIn('log_metrics_every', ui_components)
        self.assertIn('visualize_batch_every', ui_components)
        self.assertIn('gradient_clipping', ui_components)
        self.assertIn('mixed_precision', ui_components)
        self.assertIn('layer_mode', ui_components)
        self.assertIn('validation_frequency', ui_components)
        self.assertIn('iou_threshold', ui_components)
        self.assertIn('conf_threshold', ui_components)
        self.assertIn('multi_scale', ui_components)
        self.assertIn('main_container', ui_components)
        self.assertIn('tabs', ui_components)
        self.assertIn('status', ui_components)
        
        # Verifikasi tipe komponen
        self.assertIsInstance(ui_components['experiment_name'], widgets.Text)
        self.assertIsInstance(ui_components['checkpoint_dir'], widgets.Text)
        self.assertIsInstance(ui_components['tensorboard'], widgets.Checkbox)
        self.assertIsInstance(ui_components['log_metrics_every'], widgets.IntSlider)
        self.assertIsInstance(ui_components['visualize_batch_every'], widgets.IntSlider)
        self.assertIsInstance(ui_components['gradient_clipping'], widgets.FloatSlider)
        self.assertIsInstance(ui_components['mixed_precision'], widgets.Checkbox)
        self.assertIsInstance(ui_components['layer_mode'], widgets.RadioButtons)
        self.assertIsInstance(ui_components['validation_frequency'], widgets.IntSlider)
        self.assertIsInstance(ui_components['iou_threshold'], widgets.FloatSlider)
        self.assertIsInstance(ui_components['conf_threshold'], widgets.FloatSlider)
        self.assertIsInstance(ui_components['multi_scale'], widgets.Checkbox)
        self.assertIsInstance(ui_components['main_container'], widgets.VBox)
        self.assertIsInstance(ui_components['tabs'], widgets.Tab)
        self.assertIsInstance(ui_components['status'], widgets.Output)
    
    def test_setup_training_strategy_form_handlers(self):
        """Test setup form handlers."""
        # Buat mock UI components
        ui_components = {
            'experiment_name': MagicMock(),
            'checkpoint_dir': MagicMock(),
            'tensorboard': MagicMock(),
            'log_metrics_every': MagicMock(),
            'visualize_batch_every': MagicMock(),
            'gradient_clipping': MagicMock(),
            'mixed_precision': MagicMock(),
            'layer_mode': MagicMock(),
            'validation_frequency': MagicMock(),
            'iou_threshold': MagicMock(),
            'conf_threshold': MagicMock(),
            'multi_scale': MagicMock(),
            'training_strategy_info': MagicMock(),
            'status': MagicMock()
        }
        
        # Panggil fungsi yang ditest
        result = setup_training_strategy_form_handlers(ui_components)
        
        # Verifikasi hasil
        self.assertEqual(result, ui_components)
        self.assertIn('cleanup', result)
        self.assertIn('update_training_strategy_info', result)
        
        # Verifikasi observe dipanggil untuk setiap komponen
        ui_components['experiment_name'].observe.assert_called_once()
        ui_components['checkpoint_dir'].observe.assert_called_once()
        ui_components['tensorboard'].observe.assert_called_once()
        ui_components['log_metrics_every'].observe.assert_called_once()
        ui_components['visualize_batch_every'].observe.assert_called_once()
        ui_components['gradient_clipping'].observe.assert_called_once()
        ui_components['mixed_precision'].observe.assert_called_once()
        ui_components['layer_mode'].observe.assert_called_once()
        ui_components['validation_frequency'].observe.assert_called_once()
        ui_components['iou_threshold'].observe.assert_called_once()
        ui_components['conf_threshold'].observe.assert_called_once()
        ui_components['multi_scale'].observe.assert_called_once()
    
    @patch('smartcash.ui.training_config.training_strategy.handlers.button_handlers.get_config_manager')
    @patch('smartcash.ui.training_config.training_strategy.handlers.button_handlers.get_environment_manager')
    def test_setup_training_strategy_button_handlers(self, mock_get_environment_manager, mock_get_config_manager):
        """Test setup button handlers."""
        # Setup mock
        mock_get_environment_manager.return_value = self.mock_env
        mock_get_config_manager.return_value = self.mock_config_manager
        
        # Buat mock UI components
        ui_components = {
            'save_button': MagicMock(),
            'reset_button': MagicMock(),
            'status': MagicMock(),
            'training_strategy_info': MagicMock()
        }
        
        # Panggil fungsi yang ditest
        result = setup_training_strategy_button_handlers(ui_components)
        
        # Verifikasi hasil
        self.assertEqual(result, ui_components)
        self.assertIn('on_save_click', result)
        self.assertIn('on_reset_click', result)
        
        # Verifikasi on_click dipanggil untuk setiap tombol
        ui_components['save_button'].on_click.assert_called_once()
        ui_components['reset_button'].on_click.assert_called_once()
    
    @patch('smartcash.ui.training_config.training_strategy.training_strategy_initializer.get_config_manager')
    @patch('smartcash.ui.training_config.training_strategy.training_strategy_initializer.get_environment_manager')
    @patch('smartcash.ui.training_config.training_strategy.training_strategy_initializer.create_training_strategy_ui_components')
    @patch('smartcash.ui.training_config.training_strategy.training_strategy_initializer.setup_training_strategy_button_handlers')
    @patch('smartcash.ui.training_config.training_strategy.training_strategy_initializer.setup_training_strategy_form_handlers')
    @patch('smartcash.ui.training_config.training_strategy.training_strategy_initializer.update_ui_from_config')
    @patch('smartcash.ui.training_config.training_strategy.training_strategy_initializer.display')
    def test_initialize_training_strategy_ui(self, mock_display, mock_update_ui, mock_setup_form, mock_setup_button, 
                                           mock_create_ui, mock_get_env, mock_get_config):
        """Test inisialisasi UI."""
        # Setup mock
        mock_get_env.return_value = self.mock_env
        mock_get_config.return_value = self.mock_config_manager
        mock_create_ui.return_value = {'main_container': MagicMock()}
        mock_setup_button.return_value = {'main_container': MagicMock()}
        mock_setup_form.return_value = {'main_container': MagicMock()}
        
        # Panggil fungsi yang ditest
        result = initialize_training_strategy_ui()
        
        # Verifikasi hasil
        self.assertIsInstance(result, dict)
        self.assertIn('main_container', result)
        
        # Verifikasi fungsi dipanggil
        mock_get_env.assert_called_once()
        mock_get_config.assert_called_once()
        mock_create_ui.assert_called_once()
        mock_setup_button.assert_called_once()
        mock_setup_form.assert_called_once()
        mock_update_ui.assert_called_once()
        mock_display.assert_called_once()
    
    @patch('smartcash.ui.training_config.training_strategy.training_strategy_initializer.get_config_manager')
    @patch('smartcash.ui.training_config.training_strategy.training_strategy_initializer.initialize_training_strategy_ui')
    def test_get_training_strategy_ui_new(self, mock_initialize, mock_get_config):
        """Test mendapatkan UI baru."""
        # Setup mock
        mock_get_config.return_value = self.mock_config_manager
        mock_initialize.return_value = {'main_container': MagicMock()}
        
        # Panggil fungsi yang ditest
        result = get_training_strategy_ui()
        
        # Verifikasi hasil
        self.assertIsInstance(result, dict)
        self.assertIn('main_container', result)
        
        # Verifikasi fungsi dipanggil
        mock_get_config.assert_called_once()
        mock_initialize.assert_called_once()
    
    @patch('smartcash.ui.training_config.training_strategy.training_strategy_initializer.get_config_manager')
    @patch('smartcash.ui.training_config.training_strategy.training_strategy_initializer.update_ui_from_config')
    @patch('smartcash.ui.training_config.training_strategy.training_strategy_initializer.initialize_training_strategy_ui')
    def test_get_training_strategy_ui_existing(self, mock_initialize, mock_update_ui, mock_get_config):
        """Test mendapatkan UI yang sudah ada."""
        # Setup mock
        mock_get_config.return_value = self.mock_config_manager
        self.mock_config_manager.get_ui_components.return_value = {'main_container': MagicMock(), 'config': {}}
        
        # Panggil fungsi yang ditest
        result = get_training_strategy_ui()
        
        # Verifikasi hasil
        self.assertIsInstance(result, dict)
        self.assertIn('main_container', result)
        
        # Verifikasi fungsi dipanggil
        mock_get_config.assert_called_once()
        mock_initialize.assert_not_called()
        mock_update_ui.assert_called_once()

if __name__ == '__main__':
    unittest.main()
