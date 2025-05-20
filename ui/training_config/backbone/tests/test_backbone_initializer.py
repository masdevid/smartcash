"""
File: smartcash/ui/training_config/backbone/tests/test_backbone_initializer.py
Deskripsi: Test untuk backbone initializer
"""

import unittest
from unittest.mock import MagicMock, patch

class TestBackboneInitializer(unittest.TestCase):
    """
    Test untuk backbone initializer
    """
    
    @patch('smartcash.ui.training_config.backbone.backbone_initializer.display')
    @patch('smartcash.ui.training_config.backbone.backbone_initializer.clear_output')
    @patch('smartcash.ui.training_config.backbone.backbone_initializer.get_config_manager')
    @patch('smartcash.ui.training_config.backbone.handlers.config_handlers.update_ui_from_config')
    @patch('smartcash.ui.training_config.backbone.handlers.config_handlers.update_backbone_info')
    @patch('smartcash.ui.training_config.backbone.handlers.status_handlers.update_status_panel')
    @patch('smartcash.ui.training_config.backbone.handlers.status_handlers.add_status_panel')
    @patch('smartcash.ui.training_config.backbone.components.backbone_components.create_backbone_ui')
    def test_initialize_backbone_ui_success(self, mock_create_ui, mock_add_panel, mock_update_status, mock_update_info, mock_update_ui, mock_get_config, mock_clear_output, mock_display):
        """
        Test initialize_backbone_ui dengan skenario sukses
        """
        # Import di dalam test untuk menghindari circular import
        from smartcash.ui.training_config.backbone.backbone_initializer import initialize_backbone_ui
        
        # Setup mocks
        mock_ui_components = {
            'main_container': MagicMock(),
            'backbone_dropdown': MagicMock(),
            'model_type_dropdown': MagicMock(),
            'use_attention_checkbox': MagicMock(),
            'use_residual_checkbox': MagicMock(),
            'use_ciou_checkbox': MagicMock(),
            'save_button': MagicMock(),
            'reset_button': MagicMock(),
            'sync_info': MagicMock(),
            'status_panel': MagicMock()
        }
        mock_create_ui.return_value = mock_ui_components
        mock_add_panel.return_value = mock_ui_components
        
        mock_config_manager = MagicMock()
        mock_config_manager.get_module_config.return_value = {'backbone': 'efficientnet_b4'}
        mock_get_config.return_value = mock_config_manager
        
        # Call function
        result = initialize_backbone_ui()
        
        # Verify calls
        mock_create_ui.assert_called_once()
        mock_add_panel.assert_called_once_with(mock_ui_components)
        mock_clear_output.assert_called_once_with(wait=True)
        mock_display.assert_called_once_with(mock_ui_components['main_container'])
        mock_get_config.assert_called_once()
        mock_config_manager.get_module_config.assert_called_once_with('model')
        mock_update_ui.assert_called_once_with(mock_ui_components, {'backbone': 'efficientnet_b4'})
        mock_update_info.assert_called_once_with(mock_ui_components)
        mock_config_manager.register_ui_components.assert_called_once_with('backbone', mock_ui_components)
        mock_update_status.assert_called_once_with(mock_ui_components, "Konfigurasi backbone siap digunakan", 'info')
        
        # Verify result
        self.assertEqual(result, mock_ui_components)
    
    @patch('smartcash.ui.training_config.backbone.backbone_initializer.display')
    @patch('smartcash.ui.training_config.backbone.backbone_initializer.logger')
    @patch('smartcash.ui.training_config.backbone.components.backbone_components.create_backbone_ui')
    def test_initialize_backbone_ui_failure(self, mock_create_ui, mock_logger, mock_display):
        """
        Test initialize_backbone_ui dengan skenario gagal
        """
        # Import di dalam test untuk menghindari circular import
        from smartcash.ui.training_config.backbone.backbone_initializer import initialize_backbone_ui
        
        # Setup mocks untuk simulasi error
        mock_create_ui.side_effect = Exception("Test error")
        
        # Call function
        result = initialize_backbone_ui()
        
        # Verify calls
        mock_create_ui.assert_called_once()
        mock_logger.error.assert_called_once()
        mock_display.assert_called_once()
        
        # Verify result
        self.assertIn('main_container', result)

if __name__ == '__main__':
    unittest.main() 