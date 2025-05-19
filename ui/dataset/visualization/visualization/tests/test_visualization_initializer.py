"""
File: smartcash/ui/dataset/augmentation/visualization/tests/test_visualization_initializer.py
Deskripsi: Test untuk initializer visualisasi augmentasi
"""

import unittest
from unittest.mock import patch, MagicMock, call
import ipywidgets as widgets
from IPython.display import display

from smartcash.ui.dataset.augmentation.visualization.visualization_initializer import initialize_augmentation_visualization


class TestVisualizationInitializer(unittest.TestCase):
    """Test untuk initializer visualisasi augmentasi"""
    
    @patch('smartcash.ui.dataset.augmentation.visualization.visualization_initializer.create_header')
    @patch('smartcash.ui.dataset.augmentation.visualization.visualization_initializer.display')
    @patch('smartcash.ui.dataset.augmentation.visualization.visualization_initializer.get_config_manager')
    @patch('smartcash.ui.dataset.augmentation.visualization.visualization_manager.AugmentationVisualizationManager.get_instance')
    def test_initialize_augmentation_visualization(self, mock_manager_get_instance, mock_get_config_manager, mock_display, mock_create_header):
        """Test fungsi initialize_augmentation_visualization"""
        # Mock header
        mock_header = MagicMock()
        mock_create_header.return_value = mock_header
        
        # Mock ConfigManager
        mock_config_manager = MagicMock()
        mock_config_manager.get_module_config.return_value = {
            'visualization': {
                'sample_count': 3,
                'show_bboxes': True,
                'show_original': True,
                'save_visualizations': False,
                'vis_dir': 'test_visualizations'
            }
        }
        mock_get_config_manager.return_value = mock_config_manager
        
        # Mock VisualizationManager
        mock_manager = MagicMock()
        mock_manager_get_instance.return_value = mock_manager
        
        # Test initialize_augmentation_visualization tanpa config
        initialize_augmentation_visualization()
        
        # Cek apakah create_header dipanggil dengan benar
        mock_create_header.assert_called_once()
        
        # Cek apakah display dipanggil dengan benar untuk header
        mock_display.assert_any_call(mock_header)
        
        # Cek apakah ConfigManager.get_instance dipanggil dengan benar
        mock_get_config_manager.assert_called_once()
        
        # Cek apakah get_module_config dipanggil dengan benar
        mock_config_manager.get_module_config.assert_called_once_with('augmentation')
        
        # Cek apakah VisualizationManager.get_instance dipanggil dengan benar
        mock_manager_get_instance.assert_called_once()
        
        # Cek apakah display_visualization_ui dipanggil dengan benar
        mock_manager.display_visualization_ui.assert_called_once()
        
        # Reset mock
        mock_create_header.reset_mock()
        mock_display.reset_mock()
        mock_get_config_manager.reset_mock()
        mock_config_manager.get_module_config.reset_mock()
        mock_manager_get_instance.reset_mock()
        mock_manager.display_visualization_ui.reset_mock()
        
        # Test initialize_augmentation_visualization dengan config
        config = {'test': 'config'}
        initialize_augmentation_visualization(config)
        
        # Cek apakah create_header dipanggil dengan benar
        mock_create_header.assert_called_once()
        
        # Cek apakah display dipanggil dengan benar untuk header
        mock_display.assert_any_call(mock_header)
        
        # Cek apakah ConfigManager.get_instance tidak dipanggil
        mock_get_config_manager.assert_not_called()
        
        # Cek apakah get_module_config tidak dipanggil
        mock_config_manager.get_module_config.assert_not_called()
        
        # Cek apakah VisualizationManager.get_instance dipanggil dengan benar
        mock_manager_get_instance.assert_called_once_with(config, unittest.mock.ANY)
        
        # Cek apakah display_visualization_ui dipanggil dengan benar
        mock_manager.display_visualization_ui.assert_called_once()
        
        # Test dengan error
        mock_manager_get_instance.side_effect = Exception("Test error")
        
        # Cek apakah exception dilempar
        with self.assertRaises(Exception):
            initialize_augmentation_visualization()


if __name__ == '__main__':
    unittest.main()
