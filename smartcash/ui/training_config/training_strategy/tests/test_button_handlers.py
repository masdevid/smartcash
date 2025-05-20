"""
File: smartcash/ui/training_config/training_strategy/tests/test_button_handlers.py
Deskripsi: Test untuk button handlers pada modul training strategy
"""

import unittest
from unittest.mock import patch, MagicMock
import ipywidgets as widgets

from smartcash.ui.training_config.training_strategy.handlers.button_handlers import (
    on_save_click,
    on_reset_click,
    on_sync_to_drive_click,
    on_sync_from_drive_click,
    setup_training_strategy_button_handlers
)

class TestButtonHandlers(unittest.TestCase):
    """Test case untuk button handlers."""
    
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
            'info_panel': widgets.HTML(),
            'status_panel': widgets.Output(),
            'save_button': widgets.Button(),
            'reset_button': widgets.Button(),
            'sync_to_drive_button': widgets.Button(),
            'sync_from_drive_button': widgets.Button()
        }
    
    @patch('smartcash.ui.training_config.training_strategy.handlers.button_handlers.update_config_from_ui')
    @patch('smartcash.ui.training_config.training_strategy.handlers.button_handlers.update_ui_from_config')
    @patch('smartcash.ui.training_config.training_strategy.handlers.button_handlers.update_training_strategy_info')
    @patch('smartcash.ui.training_config.training_strategy.handlers.button_handlers.update_status_panel')
    @patch('smartcash.ui.training_config.training_strategy.handlers.button_handlers.get_config_manager')
    def test_on_save_click(self, mock_get_config_manager, mock_update_status, mock_update_info, mock_update_ui, mock_update_config):
        """Test untuk on_save_click."""
        # Setup mocks
        mock_config = {'training_strategy': {'enabled': True}}
        mock_update_config.return_value = mock_config
        mock_config_manager = MagicMock()
        mock_get_config_manager.return_value = mock_config_manager
        
        # Call the function
        button = widgets.Button()
        on_save_click(button, self.ui_components)
        
        # Verify calls
        mock_update_status.assert_any_call(self.ui_components, "Menyimpan konfigurasi strategi pelatihan...", "info")
        mock_update_config.assert_called_once_with(self.ui_components)
        mock_config_manager.save_module_config.assert_called_once_with('training_strategy', mock_config)
        mock_update_ui.assert_called_once_with(self.ui_components, mock_config)
        mock_update_info.assert_called_once_with(self.ui_components)
        mock_update_status.assert_any_call(self.ui_components, "Konfigurasi strategi pelatihan berhasil disimpan", "success")
    
    @patch('smartcash.ui.training_config.training_strategy.handlers.button_handlers.get_default_config')
    @patch('smartcash.ui.training_config.training_strategy.handlers.button_handlers.update_ui_from_config')
    @patch('smartcash.ui.training_config.training_strategy.handlers.button_handlers.update_training_strategy_info')
    @patch('smartcash.ui.training_config.training_strategy.handlers.button_handlers.update_status_panel')
    @patch('smartcash.ui.training_config.training_strategy.handlers.button_handlers.get_config_manager')
    def test_on_reset_click(self, mock_get_config_manager, mock_update_status, mock_update_info, mock_update_ui, mock_get_default):
        """Test untuk on_reset_click."""
        # Setup mocks
        mock_default_config = {'training_strategy': {'enabled': True}}
        mock_get_default.return_value = mock_default_config
        mock_config_manager = MagicMock()
        mock_get_config_manager.return_value = mock_config_manager
        
        # Call the function
        button = widgets.Button()
        on_reset_click(button, self.ui_components)
        
        # Verify calls
        mock_update_status.assert_any_call(self.ui_components, "Mereset konfigurasi strategi pelatihan...", "info")
        mock_get_default.assert_called_once()
        mock_update_ui.assert_called_once_with(self.ui_components, mock_default_config)
        mock_config_manager.save_module_config.assert_called_once_with('training_strategy', mock_default_config)
        mock_update_info.assert_called_once_with(self.ui_components)
        mock_update_status.assert_any_call(self.ui_components, "Konfigurasi strategi pelatihan berhasil direset ke default", "success")
    
    @patch('smartcash.ui.training_config.training_strategy.handlers.button_handlers.sync_to_drive')
    @patch('smartcash.ui.training_config.training_strategy.handlers.button_handlers.update_config_from_ui')
    @patch('smartcash.ui.training_config.training_strategy.handlers.button_handlers.update_status_panel')
    @patch('smartcash.ui.training_config.training_strategy.handlers.button_handlers.get_config_manager')
    def test_on_sync_to_drive_click_success(self, mock_get_config_manager, mock_update_status, mock_update_config, mock_sync_to_drive):
        """Test untuk on_sync_to_drive_click dengan sukses."""
        # Setup mocks
        mock_config = {'training_strategy': {'enabled': True}}
        mock_update_config.return_value = mock_config
        mock_config_manager = MagicMock()
        mock_get_config_manager.return_value = mock_config_manager
        mock_sync_to_drive.return_value = (True, "Berhasil sinkronisasi ke Google Drive")
        
        # Call the function
        button = widgets.Button()
        on_sync_to_drive_click(button, self.ui_components)
        
        # Verify calls
        mock_update_status.assert_any_call(self.ui_components, "Menyinkronkan konfigurasi strategi pelatihan ke Google Drive...", "info")
        mock_update_config.assert_called_once_with(self.ui_components)
        mock_config_manager.save_module_config.assert_called_once_with('training_strategy', mock_config)
        mock_sync_to_drive.assert_called_once_with(button, self.ui_components)
        mock_update_status.assert_any_call(self.ui_components, "Berhasil sinkronisasi ke Google Drive", "success")
    
    @patch('smartcash.ui.training_config.training_strategy.handlers.button_handlers.sync_from_drive')
    @patch('smartcash.ui.training_config.training_strategy.handlers.button_handlers.update_ui_from_config')
    @patch('smartcash.ui.training_config.training_strategy.handlers.button_handlers.update_training_strategy_info')
    @patch('smartcash.ui.training_config.training_strategy.handlers.button_handlers.update_status_panel')
    def test_on_sync_from_drive_click_success(self, mock_update_status, mock_update_info, mock_update_ui, mock_sync_from_drive):
        """Test untuk on_sync_from_drive_click dengan sukses."""
        # Setup mocks
        mock_drive_config = {'training_strategy': {'enabled': True}}
        mock_sync_from_drive.return_value = (True, "Berhasil sinkronisasi dari Google Drive", mock_drive_config)
        
        # Call the function
        button = widgets.Button()
        on_sync_from_drive_click(button, self.ui_components)
        
        # Verify calls
        mock_update_status.assert_any_call(self.ui_components, "Menyinkronkan konfigurasi strategi pelatihan dari Google Drive...", "info")
        mock_sync_from_drive.assert_called_once_with(button, self.ui_components)
        mock_update_ui.assert_called_once_with(self.ui_components, mock_drive_config)
        mock_update_info.assert_called_once_with(self.ui_components)
        mock_update_status.assert_any_call(self.ui_components, "Berhasil sinkronisasi dari Google Drive", "success")
    
    def test_setup_training_strategy_button_handlers(self):
        """Test untuk setup_training_strategy_button_handlers."""
        # Call the function
        ui_components = setup_training_strategy_button_handlers(self.ui_components)
        
        # Verify handler functions are added
        self.assertIn('on_save_click', ui_components)
        self.assertIn('on_reset_click', ui_components)
        self.assertIn('on_sync_to_drive_click', ui_components)
        self.assertIn('on_sync_from_drive_click', ui_components)
        
        # Verify handlers are callable
        self.assertTrue(callable(ui_components['on_save_click']))
        self.assertTrue(callable(ui_components['on_reset_click']))
        self.assertTrue(callable(ui_components['on_sync_to_drive_click']))
        self.assertTrue(callable(ui_components['on_sync_from_drive_click']))

if __name__ == '__main__':
    unittest.main() 