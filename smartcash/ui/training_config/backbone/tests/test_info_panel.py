"""
File: smartcash/ui/training_config/backbone/tests/test_info_panel.py
Deskripsi: Test untuk info panel backbone
"""

import unittest
from unittest.mock import MagicMock, patch
import ipywidgets as widgets

from smartcash.ui.training_config.backbone.handlers.config_handlers import (
    get_default_backbone_config,
    update_backbone_info
)

class TestBackboneInfoPanel(unittest.TestCase):
    """
    Test untuk info panel backbone
    """
    
    def setUp(self):
        """
        Setup test
        """
        # Create mock UI components
        self.mock_ui_components = {
            'info_panel': MagicMock(),
        }
    
    @patch('smartcash.ui.training_config.backbone.handlers.config_handlers.get_backbone_config')
    def test_update_backbone_info_complete_config(self, mock_get_config):
        """
        Test update_backbone_info dengan konfigurasi lengkap
        """
        # Setup mock dengan konfigurasi lengkap
        mock_get_config.return_value = get_default_backbone_config()
        
        # Call function
        update_backbone_info(self.mock_ui_components)
        
        # Verify info_panel.value was set
        self.assertTrue(self.mock_ui_components['info_panel'].value is not None)
        
        # Verify all expected fields are in the info panel
        info_text = self.mock_ui_components['info_panel'].value
        self.assertIn('Type: efficientnet_b4', info_text)
        self.assertIn('Pretrained: True', info_text)
        self.assertIn('Freeze Backbone: False', info_text)
        self.assertIn('Freeze BatchNorm: False', info_text)
        self.assertIn('Dropout: 0.2', info_text)
        self.assertIn('Activation: relu', info_text)
        self.assertIn('Normalization: batch_norm', info_text)
        self.assertIn('BN Momentum: 0.1', info_text)
    
    @patch('smartcash.ui.training_config.backbone.handlers.config_handlers.get_backbone_config')
    def test_update_backbone_info_incomplete_config(self, mock_get_config):
        """
        Test update_backbone_info dengan konfigurasi tidak lengkap
        """
        # Setup mock dengan konfigurasi tidak lengkap
        incomplete_config = {
            'model': {
                'backbone': 'efficientnet_b4',
                'model_type': 'efficient_basic'
                # Missing other fields
            }
        }
        mock_get_config.return_value = incomplete_config
        
        # Call function
        update_backbone_info(self.mock_ui_components)
        
        # Verify info_panel.value was set
        self.assertTrue(self.mock_ui_components['info_panel'].value is not None)
        
        # Verify all expected fields are in the info panel with default values
        info_text = self.mock_ui_components['info_panel'].value
        self.assertIn('Type: efficientnet_b4', info_text)
        self.assertIn('Pretrained: True', info_text)  # Default value
        self.assertIn('Freeze Backbone: False', info_text)  # Default value
        self.assertIn('Freeze BatchNorm: False', info_text)  # Default value
        self.assertIn('Dropout: 0.2', info_text)  # Default value
        self.assertIn('Activation: relu', info_text)  # Default value
        self.assertIn('Normalization: batch_norm', info_text)  # Default value
        self.assertIn('BN Momentum: 0.1', info_text)  # Default value

if __name__ == '__main__':
    unittest.main() 