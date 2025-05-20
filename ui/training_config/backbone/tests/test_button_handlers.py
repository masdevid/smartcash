"""
File: smartcash/ui/training_config/backbone/tests/test_button_handlers.py
Deskripsi: Test untuk button handlers backbone
"""

import unittest
from unittest.mock import MagicMock, patch
import ipywidgets as widgets

from smartcash.ui.training_config.backbone.handlers.button_handlers import (
    on_save_click,
    on_reset_click
)

class TestBackboneButtonHandlers(unittest.TestCase):
    """
    Test untuk button handlers backbone
    """
    
    def setUp(self):
        """
        Setup test
        """
        # Create mock UI components
        self.mock_ui_components = {
            'model_type_dropdown': MagicMock(value='efficient_basic'),
            'backbone_dropdown': MagicMock(value='efficientnet_b4'),
            'use_attention_checkbox': MagicMock(value=True),
            'use_residual_checkbox': MagicMock(value=True),
            'use_ciou_checkbox': MagicMock(value=False),
            'info_panel': MagicMock(),
            'status_panel': MagicMock()
        }
        
        # Create mock button
        self.mock_button = MagicMock()
        
        # Create mock config
        self.mock_config = {
            'model': {
                'backbone': 'efficientnet_b4',
                'model_type': 'efficient_basic',
                'use_attention': True,
                'use_residual': True,
                'use_ciou': False
            }
        }
    
    @patch('smartcash.ui.training_config.backbone.handlers.button_handlers.update_config_from_ui')
    @patch('smartcash.ui.training_config.backbone.handlers.button_handlers.save_config')
    @patch('smartcash.ui.training_config.backbone.handlers.button_handlers.update_backbone_info')
    @patch('smartcash.ui.training_config.backbone.handlers.button_handlers.update_status_panel')
    def test_on_save_click_success(self, mock_update_status, mock_update_info, mock_save_config, mock_update_config):
        """
        Test on_save_click dengan skenario sukses
        """
        # Setup mocks
        mock_update_config.return_value = self.mock_config
        mock_save_config.return_value = self.mock_config
        
        # Call function
        on_save_click(self.mock_button, self.mock_ui_components)
        
        # Verify calls
        mock_update_status.assert_any_call(self.mock_ui_components, "Menyimpan konfigurasi backbone...", 'info')
        mock_update_config.assert_called_once_with(self.mock_ui_components)
        mock_save_config.assert_called_once_with(self.mock_config, self.mock_ui_components)
        mock_update_info.assert_called_once_with(self.mock_ui_components)
        mock_update_status.assert_any_call(self.mock_ui_components, "Konfigurasi backbone berhasil disimpan", 'success')
    
    @patch('smartcash.ui.training_config.backbone.handlers.button_handlers.update_config_from_ui')
    @patch('smartcash.ui.training_config.backbone.handlers.button_handlers.save_config')
    @patch('smartcash.ui.training_config.backbone.handlers.button_handlers.update_status_panel')
    def test_on_save_click_failure(self, mock_update_status, mock_save_config, mock_update_config):
        """
        Test on_save_click dengan skenario gagal
        """
        # Setup mocks untuk simulasi error
        mock_update_config.return_value = self.mock_config
        mock_save_config.side_effect = Exception("Test error")
        
        # Call function
        on_save_click(self.mock_button, self.mock_ui_components)
        
        # Verify calls
        mock_update_status.assert_any_call(self.mock_ui_components, "Menyimpan konfigurasi backbone...", 'info')
        mock_update_config.assert_called_once_with(self.mock_ui_components)
        mock_save_config.assert_called_once_with(self.mock_config, self.mock_ui_components)
        mock_update_status.assert_any_call(self.mock_ui_components, "Gagal menyimpan konfigurasi: Test error", 'error')
    
    @patch('smartcash.ui.training_config.backbone.handlers.button_handlers.get_default_backbone_config')
    @patch('smartcash.ui.training_config.backbone.handlers.button_handlers.save_config')
    @patch('smartcash.ui.training_config.backbone.handlers.button_handlers.update_ui_from_config')
    @patch('smartcash.ui.training_config.backbone.handlers.button_handlers.update_backbone_info')
    @patch('smartcash.ui.training_config.backbone.handlers.button_handlers.update_status_panel')
    def test_on_reset_click_success(self, mock_update_status, mock_update_info, mock_update_ui, mock_save_config, mock_get_default):
        """
        Test on_reset_click dengan skenario sukses
        """
        # Setup mocks
        mock_get_default.return_value = self.mock_config
        mock_save_config.return_value = self.mock_config
        
        # Call function
        on_reset_click(self.mock_button, self.mock_ui_components)
        
        # Verify calls
        mock_update_status.assert_any_call(self.mock_ui_components, "Mereset konfigurasi backbone...", 'info')
        mock_get_default.assert_called_once()
        mock_save_config.assert_called_once_with(self.mock_config, self.mock_ui_components)
        mock_update_ui.assert_called_once_with(self.mock_ui_components, self.mock_config)
        mock_update_info.assert_called_once_with(self.mock_ui_components)
        mock_update_status.assert_any_call(self.mock_ui_components, "Konfigurasi backbone berhasil direset ke default", 'success')
    
    @patch('smartcash.ui.training_config.backbone.handlers.button_handlers.get_default_backbone_config')
    @patch('smartcash.ui.training_config.backbone.handlers.button_handlers.save_config')
    @patch('smartcash.ui.training_config.backbone.handlers.button_handlers.update_status_panel')
    def test_on_reset_click_failure(self, mock_update_status, mock_save_config, mock_get_default):
        """
        Test on_reset_click dengan skenario gagal
        """
        # Setup mocks untuk simulasi error
        mock_get_default.return_value = self.mock_config
        mock_save_config.side_effect = Exception("Test error")
        
        # Call function
        on_reset_click(self.mock_button, self.mock_ui_components)
        
        # Verify calls
        mock_update_status.assert_any_call(self.mock_ui_components, "Mereset konfigurasi backbone...", 'info')
        mock_get_default.assert_called_once()
        mock_save_config.assert_called_once_with(self.mock_config, self.mock_ui_components)
        mock_update_status.assert_any_call(self.mock_ui_components, "Error saat reset konfigurasi backbone: Test error", 'error')

if __name__ == '__main__':
    unittest.main() 