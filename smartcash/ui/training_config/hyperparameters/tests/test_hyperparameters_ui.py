"""
File: tests/test_hyperparameters_ui.py
Deskripsi: Pengujian untuk komponen UI hyperparameter
"""

import unittest
from unittest.mock import MagicMock, patch
import ipywidgets as widgets

from smartcash.ui.training_config.hyperparameters.components import (
    create_hyperparameters_ui_components,
    create_hyperparameters_info_panel
)
from smartcash.ui.training_config.hyperparameters.handlers.config_handlers import (
    update_ui_from_config,
    update_config_from_ui
)
from smartcash.ui.training_config.hyperparameters.handlers.button_handlers import (
    setup_hyperparameters_button_handlers
)
from smartcash.ui.training_config.hyperparameters.handlers.form_handlers import (
    setup_hyperparameters_form_handlers
)
from smartcash.ui.training_config.hyperparameters.hyperparameters_initializer import (
    initialize_hyperparameters_ui,
    get_hyperparameters_ui
)

class TestHyperparametersUI(unittest.TestCase):
    """Pengujian untuk komponen UI hyperparameter"""
    
    def setUp(self):
        """Setup untuk pengujian"""
        # Mock config
        self.mock_config = {
            'batch_size': 16,
            'image_size': 640,
            'epochs': 100,
            'augment': True,
            'optimizer': 'SGD',
            'learning_rate': 0.01,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'scheduler': 'cosine',
            'warmup_epochs': 3,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
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
        
        # Mock environment
        self.mock_env = MagicMock()
        self.mock_env.is_drive_mounted = True
        self.mock_env.drive_path = '/content/drive'
    
    @patch('smartcash.ui.training_config.hyperparameters.components.main_components.create_tab_widget')
    @patch('smartcash.ui.training_config.hyperparameters.components.main_components.create_header')
    @patch('smartcash.ui.training_config.hyperparameters.components.main_components.create_hyperparameters_info_panel')
    @patch('smartcash.ui.training_config.hyperparameters.components.main_components.create_hyperparameters_button_components')
    @patch('smartcash.ui.training_config.hyperparameters.components.main_components.create_hyperparameters_advanced_components')
    @patch('smartcash.ui.training_config.hyperparameters.components.main_components.create_hyperparameters_optimization_components')
    @patch('smartcash.ui.training_config.hyperparameters.components.main_components.create_hyperparameters_basic_components')
    @patch('smartcash.ui.training_config.hyperparameters.components.main_components.widgets')
    def test_create_hyperparameters_ui_components(self, mock_widgets, mock_basic, mock_optimization, mock_advanced, mock_button, mock_info_panel, mock_header, mock_tab_widget):
        """Pengujian pembuatan komponen UI hyperparameter"""
        # Setup mock
        mock_widgets.VBox.return_value = MagicMock()
        mock_widgets.HBox.return_value = MagicMock()
        mock_widgets.HTML.return_value = MagicMock()
        mock_widgets.Box.return_value = MagicMock()
        mock_widgets.Tab.return_value = MagicMock()
        mock_widgets.Accordion.return_value = MagicMock()
        mock_widgets.Layout.return_value = MagicMock()
        
        # Mock komponen
        mock_basic.return_value = {'basic_box': MagicMock()}
        mock_optimization.return_value = {'optimization_box': MagicMock()}
        mock_advanced.return_value = {'advanced_box': MagicMock()}
        mock_button.return_value = {'save_button': MagicMock(), 'reset_button': MagicMock(), 'status': MagicMock()}
        mock_info_panel.return_value = (MagicMock(), MagicMock())
        mock_header.return_value = MagicMock()
        mock_tab_widget.return_value = MagicMock()
        
        # Panggil fungsi
        ui_components = create_hyperparameters_ui_components()
        
        # Verifikasi hasil
        self.assertIsInstance(ui_components, dict)
        self.assertIn('basic_box', ui_components)
        self.assertIn('optimization_box', ui_components)
        self.assertIn('advanced_box', ui_components)
        self.assertIn('save_button', ui_components)
        self.assertIn('reset_button', ui_components)
        self.assertIn('status', ui_components)
        self.assertIn('info_panel', ui_components)
        self.assertIn('main_container', ui_components)
        self.assertIn('tabs', ui_components)
        self.assertIn('header', ui_components)
    
    @patch('smartcash.ui.training_config.hyperparameters.components.info_panel_components.widgets')
    def test_create_hyperparameters_info_panel(self, mock_widgets):
        """Pengujian pembuatan panel informasi hyperparameter"""
        # Setup mock
        mock_widgets.Output.return_value = MagicMock()
        mock_widgets.Layout.return_value = MagicMock()
        
        # Panggil fungsi
        info_panel, update_func = create_hyperparameters_info_panel()
        
        # Verifikasi hasil
        self.assertIsNotNone(info_panel)
        self.assertTrue(callable(update_func))
    
    @patch('smartcash.ui.training_config.hyperparameters.handlers.config_handlers.get_logger')
    def test_update_ui_from_config(self, mock_get_logger):
        """Pengujian update UI dari konfigurasi"""
        # Setup mock
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        # Buat UI components mock
        ui_components = {
            'batch_size_slider': MagicMock(),
            'image_size_slider': MagicMock(),
            'epochs_slider': MagicMock(),
            'augment_checkbox': MagicMock(),
            'optimizer_dropdown': MagicMock(),
            'learning_rate_slider': MagicMock(),
            'momentum_slider': MagicMock(),
            'weight_decay_slider': MagicMock(),
            'scheduler_dropdown': MagicMock(),
            'warmup_epochs_slider': MagicMock(),
            'warmup_momentum_slider': MagicMock(),
            'warmup_bias_lr_slider': MagicMock(),
            'early_stopping_enabled_checkbox': MagicMock(),
            'early_stopping_patience_slider': MagicMock(),
            'early_stopping_min_delta_slider': MagicMock(),
            'save_best_checkbox': MagicMock(),
            'checkpoint_metric_dropdown': MagicMock(),
            'update_hyperparameters_info': MagicMock()
        }
        
        # Panggil fungsi
        update_ui_from_config(ui_components, self.mock_config)
        
        # Verifikasi hasil
        ui_components['batch_size_slider'].value = self.mock_config['batch_size']
        ui_components['image_size_slider'].value = self.mock_config['image_size']
        ui_components['epochs_slider'].value = self.mock_config['epochs']
        ui_components['augment_checkbox'].value = self.mock_config['augment']
        ui_components['optimizer_dropdown'].value = self.mock_config['optimizer']
        ui_components['learning_rate_slider'].value = self.mock_config['learning_rate']
    
    @patch('smartcash.ui.training_config.hyperparameters.handlers.config_handlers.get_logger')
    def test_update_config_from_ui(self, mock_get_logger):
        """Pengujian update konfigurasi dari UI"""
        # Setup mock
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        # Buat UI components mock
        ui_components = {
            'batch_size_slider': MagicMock(value=32),
            'image_size_slider': MagicMock(value=800),
            'epochs_slider': MagicMock(value=200),
            'augment_checkbox': MagicMock(value=False),
            'optimizer_dropdown': MagicMock(value='Adam'),
            'learning_rate_slider': MagicMock(value=0.001),
            'momentum_slider': MagicMock(value=0.9),
            'weight_decay_slider': MagicMock(value=0.0001),
            'scheduler_dropdown': MagicMock(value='linear'),
            'warmup_epochs_slider': MagicMock(value=5),
            'warmup_momentum_slider': MagicMock(value=0.85),
            'warmup_bias_lr_slider': MagicMock(value=0.05),
            'early_stopping_enabled_checkbox': MagicMock(value=False),
            'early_stopping_patience_slider': MagicMock(value=15),
            'early_stopping_min_delta_slider': MagicMock(value=0.0005),
            'save_best_checkbox': MagicMock(value=True),
            'checkpoint_metric_dropdown': MagicMock(value='mAP_0.5:0.95')
        }
        
        # Panggil fungsi
        config = update_config_from_ui(ui_components, {})
        
        # Verifikasi hasil
        self.assertEqual(config['batch_size'], 32)
        self.assertEqual(config['image_size'], 800)
        self.assertEqual(config['epochs'], 200)
        self.assertEqual(config['augment'], False)
        self.assertEqual(config['optimizer'], 'Adam')
        self.assertEqual(config['learning_rate'], 0.001)
        self.assertEqual(config['momentum'], 0.9)
        self.assertEqual(config['weight_decay'], 0.0001)
        self.assertEqual(config['scheduler'], 'linear')
        self.assertEqual(config['warmup_epochs'], 5)
        self.assertEqual(config['warmup_momentum'], 0.85)
        self.assertEqual(config['warmup_bias_lr'], 0.05)
        self.assertEqual(config['early_stopping']['enabled'], False)
        self.assertEqual(config['early_stopping']['patience'], 15)
        self.assertEqual(config['early_stopping']['min_delta'], 0.0005)
        self.assertEqual(config['save_best']['enabled'], True)
        self.assertEqual(config['save_best']['metric'], 'mAP_0.5:0.95')
    
    @patch('smartcash.ui.training_config.hyperparameters.handlers.button_handlers.get_config_manager')
    @patch('smartcash.ui.training_config.hyperparameters.handlers.button_handlers.get_logger')
    def test_setup_hyperparameters_button_handlers(self, mock_get_logger, mock_get_config_manager):
        """Pengujian setup handler untuk tombol"""
        # Setup mock
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        mock_config_manager = MagicMock()
        mock_get_config_manager.return_value = mock_config_manager
        
        # Buat UI components mock
        ui_components = {
            'save_button': MagicMock(),
            'reset_button': MagicMock(),
            'sync_from_drive_button': MagicMock(),
            'sync_to_drive_button': MagicMock(),
            'status': MagicMock(),
            'update_hyperparameters_info': MagicMock()
        }
        
        # Panggil fungsi
        result = setup_hyperparameters_button_handlers(ui_components, self.mock_env, self.mock_config)
        
        # Verifikasi hasil
        self.assertIsInstance(result, dict)
        self.assertIn('on_save_click', result)
        self.assertIn('on_reset_click', result)
        
        # Verifikasi handler terpasang
        ui_components['save_button'].on_click.assert_called_once()
        ui_components['reset_button'].on_click.assert_called_once()
    
    @patch('smartcash.ui.training_config.hyperparameters.handlers.form_handlers.get_logger')
    def test_setup_hyperparameters_form_handlers(self, mock_get_logger):
        """Pengujian setup handler untuk form"""
        # Setup mock
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        # Buat UI components mock
        ui_components = {
            'optimizer_dropdown': MagicMock(),
            'scheduler_dropdown': MagicMock(),
            'early_stopping_enabled_checkbox': MagicMock(),
            'save_best_checkbox': MagicMock(),
            'update_hyperparameters_info': MagicMock()
        }
        
        # Panggil fungsi
        result = setup_hyperparameters_form_handlers(ui_components, self.mock_env, self.mock_config)
        
        # Verifikasi hasil
        self.assertIsInstance(result, dict)
        self.assertIn('on_optimizer_type_change', result)
        self.assertIn('on_scheduler_type_change', result)
        self.assertIn('on_early_stopping_enabled_change', result)
        self.assertIn('on_save_best_change', result)
        
        # Verifikasi handler terpasang
        ui_components['optimizer_dropdown'].observe.assert_called_once()
        ui_components['scheduler_dropdown'].observe.assert_called_once()
        ui_components['early_stopping_enabled_checkbox'].observe.assert_called_once()
        ui_components['save_best_checkbox'].observe.assert_called_once()
    
    @patch('smartcash.ui.training_config.hyperparameters.hyperparameters_initializer.get_config_manager')
    @patch('smartcash.ui.training_config.hyperparameters.hyperparameters_initializer.get_environment_manager')
    @patch('smartcash.ui.training_config.hyperparameters.hyperparameters_initializer.create_hyperparameters_ui_components')
    @patch('smartcash.ui.training_config.hyperparameters.hyperparameters_initializer.setup_hyperparameters_button_handlers')
    @patch('smartcash.ui.training_config.hyperparameters.hyperparameters_initializer.setup_hyperparameters_form_handlers')
    @patch('smartcash.ui.training_config.hyperparameters.hyperparameters_initializer.update_ui_from_config')
    @patch('smartcash.ui.training_config.hyperparameters.hyperparameters_initializer.get_logger')
    def test_initialize_hyperparameters_ui(
        self, mock_get_logger, mock_update_ui, mock_setup_form, mock_setup_button, 
        mock_create_ui, mock_get_env, mock_get_config_manager
    ):
        """Pengujian inisialisasi UI hyperparameter"""
        # Setup mock
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        mock_env = MagicMock()
        mock_get_env.return_value = mock_env
        
        mock_config_manager = MagicMock()
        mock_config_manager.get_module_config.return_value = self.mock_config
        mock_get_config_manager.return_value = mock_config_manager
        
        mock_ui_components = {
            'update_hyperparameters_info': MagicMock(),
            'main_layout': MagicMock()
        }
        mock_create_ui.return_value = mock_ui_components
        mock_setup_button.return_value = mock_ui_components
        mock_setup_form.return_value = mock_ui_components
        
        # Panggil fungsi
        result = initialize_hyperparameters_ui()
        
        # Verifikasi hasil
        self.assertEqual(result, mock_ui_components)
        mock_create_ui.assert_called_once()
        mock_setup_button.assert_called_once()
        mock_setup_form.assert_called_once()
        mock_update_ui.assert_called_once()
        mock_ui_components['update_hyperparameters_info'].assert_called_once()
        mock_config_manager.register_ui_components.assert_called_once()
    
    @patch('smartcash.ui.training_config.hyperparameters.hyperparameters_initializer.get_config_manager')
    @patch('smartcash.ui.training_config.hyperparameters.hyperparameters_initializer.initialize_hyperparameters_ui')
    @patch('smartcash.ui.training_config.hyperparameters.hyperparameters_initializer.update_ui_from_config')
    def test_get_hyperparameters_ui_existing(self, mock_update_ui, mock_initialize, mock_get_config_manager):
        """Pengujian mendapatkan UI hyperparameter yang sudah ada"""
        # Setup mock
        mock_config_manager = MagicMock()
        mock_ui_components = {
            'config': {},
            'update_hyperparameters_info': MagicMock()
        }
        mock_config_manager.get_ui_components.return_value = mock_ui_components
        mock_config_manager.get_module_config.return_value = self.mock_config
        mock_get_config_manager.return_value = mock_config_manager
        
        # Panggil fungsi
        result = get_hyperparameters_ui()
        
        # Verifikasi hasil
        self.assertEqual(result, mock_ui_components)
        mock_initialize.assert_not_called()
        mock_update_ui.assert_called_once()
        mock_ui_components['update_hyperparameters_info'].assert_called_once()
    
    @patch('smartcash.ui.training_config.hyperparameters.hyperparameters_initializer.get_config_manager')
    @patch('smartcash.ui.training_config.hyperparameters.hyperparameters_initializer.initialize_hyperparameters_ui')
    def test_get_hyperparameters_ui_new(self, mock_initialize, mock_get_config_manager):
        """Pengujian mendapatkan UI hyperparameter baru"""
        # Setup mock
        mock_config_manager = MagicMock()
        mock_config_manager.get_ui_components.return_value = None
        mock_get_config_manager.return_value = mock_config_manager
        
        mock_ui_components = {
            'main_layout': MagicMock()
        }
        mock_initialize.return_value = mock_ui_components
        
        # Panggil fungsi
        result = get_hyperparameters_ui()
        
        # Verifikasi hasil
        self.assertEqual(result, mock_ui_components)
        mock_initialize.assert_called_once()

if __name__ == '__main__':
    unittest.main()
