"""
File: smartcash/ui/dataset/visualization/tests/test_visualization_tabs.py
Deskripsi: Test untuk tab visualisasi dataset
"""

import unittest
from unittest.mock import patch, MagicMock
import ipywidgets as widgets

from smartcash.ui.dataset.visualization.components.visualization_tabs import (
    create_distribution_tab, create_split_distribution_tab, 
    create_layer_distribution_tab, create_heatmap_tab,
    create_visualization_tabs
)
from smartcash.ui.dataset.visualization.handlers.visualization_tab_handler import (
    on_distribution_click, on_split_distribution_click,
    on_layer_distribution_click, on_heatmap_click,
    setup_visualization_tab_handlers
)


class TestVisualizationTabs(unittest.TestCase):
    """Test untuk tab visualisasi dataset."""
    
    def test_create_distribution_tab(self):
        """Test untuk fungsi create_distribution_tab."""
        # Panggil fungsi
        components = create_distribution_tab()
        
        # Verifikasi hasil
        self.assertIn('container', components)
        self.assertIn('output', components)
        self.assertIn('button', components)
        self.assertIsInstance(components['container'], widgets.VBox)
        self.assertIsInstance(components['output'], widgets.Output)
        self.assertIsInstance(components['button'], widgets.Button)
    
    def test_create_split_distribution_tab(self):
        """Test untuk fungsi create_split_distribution_tab."""
        # Panggil fungsi
        components = create_split_distribution_tab()
        
        # Verifikasi hasil
        self.assertIn('container', components)
        self.assertIn('output', components)
        self.assertIn('button', components)
        self.assertIsInstance(components['container'], widgets.VBox)
        self.assertIsInstance(components['output'], widgets.Output)
        self.assertIsInstance(components['button'], widgets.Button)
    
    def test_create_layer_distribution_tab(self):
        """Test untuk fungsi create_layer_distribution_tab."""
        # Panggil fungsi
        components = create_layer_distribution_tab()
        
        # Verifikasi hasil
        self.assertIn('container', components)
        self.assertIn('output', components)
        self.assertIn('button', components)
        self.assertIsInstance(components['container'], widgets.VBox)
        self.assertIsInstance(components['output'], widgets.Output)
        self.assertIsInstance(components['button'], widgets.Button)
    
    def test_create_heatmap_tab(self):
        """Test untuk fungsi create_heatmap_tab."""
        # Panggil fungsi
        components = create_heatmap_tab()
        
        # Verifikasi hasil
        self.assertIn('container', components)
        self.assertIn('output', components)
        self.assertIn('button', components)
        self.assertIsInstance(components['container'], widgets.VBox)
        self.assertIsInstance(components['output'], widgets.Output)
        self.assertIsInstance(components['button'], widgets.Button)
    
    def test_create_visualization_tabs(self):
        """Test untuk fungsi create_visualization_tabs."""
        # Panggil fungsi
        components = create_visualization_tabs()
        
        # Verifikasi hasil
        self.assertIn('tab', components)
        self.assertIn('distribution_tab', components)
        self.assertIn('split_tab', components)
        self.assertIn('layer_tab', components)
        self.assertIn('heatmap_tab', components)
        self.assertIsInstance(components['tab'], widgets.Tab)
    
    @patch('smartcash.ui.dataset.visualization.handlers.visualization_tab_handler.get_config_manager')
    @patch('smartcash.ui.dataset.visualization.handlers.visualization_tab_handler.clear_output')
    @patch('smartcash.ui.dataset.visualization.handlers.visualization_tab_handler.display')
    @patch('smartcash.ui.dataset.visualization.handlers.visualization_tab_handler.create_status_indicator')
    def test_on_distribution_click(self, mock_status, mock_display, mock_clear, mock_config):
        """Test untuk fungsi on_distribution_click."""
        # Setup mock
        mock_config.return_value.get.return_value = '/dummy/path'
        mock_status.return_value = "Status Indicator"
        
        # Setup UI components
        ui_components = {
            'visualization_components': {
                'distribution_tab': {
                    'output': widgets.Output()
                }
            }
        }
        
        # Panggil fungsi dengan patch untuk os.path.exists dan os.listdir
        with patch('os.path.exists', return_value=False):
            on_distribution_click(None, ui_components)
        
        # Verifikasi bahwa fungsi dipanggil
        mock_clear.assert_called()
        mock_display.assert_called()
    
    @patch('smartcash.ui.dataset.visualization.handlers.visualization_tab_handler.get_config_manager')
    @patch('smartcash.ui.dataset.visualization.handlers.visualization_tab_handler.clear_output')
    @patch('smartcash.ui.dataset.visualization.handlers.visualization_tab_handler.display')
    @patch('smartcash.ui.dataset.visualization.handlers.visualization_tab_handler.create_status_indicator')
    def test_on_split_distribution_click(self, mock_status, mock_display, mock_clear, mock_config):
        """Test untuk fungsi on_split_distribution_click."""
        # Setup mock
        mock_config.return_value.get.return_value = '/dummy/path'
        mock_status.return_value = "Status Indicator"
        
        # Setup UI components
        ui_components = {
            'visualization_components': {
                'split_tab': {
                    'output': widgets.Output()
                }
            }
        }
        
        # Panggil fungsi dengan patch untuk os.path.exists dan os.listdir
        with patch('os.path.exists', return_value=False):
            on_split_distribution_click(None, ui_components)
        
        # Verifikasi bahwa fungsi dipanggil
        mock_clear.assert_called()
        mock_display.assert_called()
    
    @patch('smartcash.ui.dataset.visualization.handlers.visualization_tab_handler.on_distribution_click')
    @patch('smartcash.ui.dataset.visualization.handlers.visualization_tab_handler.on_split_distribution_click')
    @patch('smartcash.ui.dataset.visualization.handlers.visualization_tab_handler.on_layer_distribution_click')
    @patch('smartcash.ui.dataset.visualization.handlers.visualization_tab_handler.on_heatmap_click')
    def test_setup_visualization_tab_handlers(self, mock_heatmap, mock_layer, mock_split, mock_distribution):
        """Test untuk fungsi setup_visualization_tab_handlers."""
        # Setup UI components
        ui_components = {
            'visualization_components': {
                'distribution_tab': {
                    'button': widgets.Button(),
                    'output': widgets.Output()
                },
                'split_tab': {
                    'button': widgets.Button(),
                    'output': widgets.Output()
                },
                'layer_tab': {
                    'button': widgets.Button(),
                    'output': widgets.Output()
                },
                'heatmap_tab': {
                    'button': widgets.Button(),
                    'output': widgets.Output()
                }
            }
        }
        
        # Panggil fungsi
        result = setup_visualization_tab_handlers(ui_components)
        
        # Verifikasi hasil
        self.assertEqual(result, ui_components)
        
        # Kita hanya perlu memverifikasi bahwa fungsi mengembalikan ui_components yang sama
        # Karena kita tidak dapat mengakses handler internal dari button secara langsung dalam test


if __name__ == '__main__':
    unittest.main()
