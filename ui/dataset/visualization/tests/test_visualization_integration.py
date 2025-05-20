"""
File: smartcash/ui/dataset/visualization/tests/test_visualization_integration.py
Deskripsi: Test integrasi untuk visualisasi dataset dengan fokus pada tampilan
"""

import unittest
import ipywidgets as widgets
from IPython.display import display
from unittest.mock import MagicMock, patch, Mock

class TestVisualizationIntegration(unittest.TestCase):
    """Test integrasi untuk visualisasi dataset"""
    
    @patch('smartcash.ui.dataset.visualization.setup.display')
    @patch('smartcash.ui.dataset.visualization.setup.create_visualization_layout')
    @patch('smartcash.ui.dataset.visualization.setup.setup_visualization_handlers')
    @patch('smartcash.ui.dataset.visualization.setup.update_dashboard_cards')
    def test_setup_visualization(self, mock_update_cards, mock_setup_handlers, mock_create_layout, mock_display):
        """Test setup visualisasi dataset"""
        try:
            from smartcash.ui.dataset.visualization.setup import setup_dataset_visualization
            
            # Mock komponen UI
            mock_ui_components = {
                'main_container': MagicMock(spec=widgets.VBox),
                'status': MagicMock(spec=widgets.Output),
                'refresh_button': MagicMock(spec=widgets.Button),
                'visualization_components': {
                    'distribution_tab': {
                        'button': MagicMock(spec=widgets.Button),
                        'output': MagicMock(spec=widgets.Output)
                    },
                    'split_tab': {
                        'button': MagicMock(spec=widgets.Button),
                        'output': MagicMock(spec=widgets.Output)
                    },
                    'layer_tab': {
                        'button': MagicMock(spec=widgets.Button),
                        'output': MagicMock(spec=widgets.Output)
                    },
                    'bbox_tab': {
                        'button': MagicMock(spec=widgets.Button),
                        'output': MagicMock(spec=widgets.Output)
                    },
                    'heatmap_tab': {
                        'button': MagicMock(spec=widgets.Button),
                        'output': MagicMock(spec=widgets.Output)
                    }
                }
            }
            
            # Setup mock returns
            mock_create_layout.return_value = mock_ui_components
            mock_setup_handlers.return_value = mock_ui_components
            
            # Panggil fungsi setup
            result = setup_dataset_visualization()
            
            # Verifikasi fungsi dipanggil
            mock_create_layout.assert_called_once()
            mock_setup_handlers.assert_called_once()
            mock_update_cards.assert_called_once()
            mock_display.assert_called_once()
            
            # Verifikasi hasil
            self.assertEqual(result, mock_ui_components)
        except ImportError:
            self.skipTest("Modul setup tidak tersedia")

class TestHandlerIntegration(unittest.TestCase):
    """Test integrasi untuk handler visualisasi"""
    
    def setUp(self):
        """Setup untuk setiap test case"""
        # Mock komponen UI
        self.ui_components = {
            'status': MagicMock(spec=widgets.Output),
            'refresh_button': MagicMock(spec=widgets.Button),
            'visualization_components': {
                'distribution_tab': {
                    'button': MagicMock(spec=widgets.Button),
                    'output': MagicMock(spec=widgets.Output)
                },
                'split_tab': {
                    'button': MagicMock(spec=widgets.Button),
                    'output': MagicMock(spec=widgets.Output)
                },
                'layer_tab': {
                    'button': MagicMock(spec=widgets.Button),
                    'output': MagicMock(spec=widgets.Output)
                },
                'bbox_tab': {
                    'button': MagicMock(spec=widgets.Button),
                    'output': MagicMock(spec=widgets.Output)
                },
                'heatmap_tab': {
                    'button': MagicMock(spec=widgets.Button),
                    'output': MagicMock(spec=widgets.Output)
                }
            },
            'split_cards_container': MagicMock(spec=widgets.Output),
            'preprocessing_cards': MagicMock(spec=widgets.Output),
            'augmentation_cards': MagicMock(spec=widgets.Output)
        }
    
    @patch('smartcash.ui.dataset.visualization.handlers.bbox_handlers.setup_bbox_handlers')
    @patch('smartcash.ui.dataset.visualization.handlers.layer_handlers.setup_layer_handlers')
    @patch('smartcash.ui.dataset.visualization.handlers.distribution_handlers.setup_distribution_handlers')
    @patch('smartcash.ui.dataset.visualization.handlers.split_handlers.setup_split_handlers')
    @patch('smartcash.ui.dataset.visualization.handlers.dashboard_handlers.setup_dashboard_handlers')
    def test_setup_handlers(self, mock_dashboard, mock_split, mock_distribution, mock_layer, mock_bbox):
        """Test setup semua handler"""
        try:
            from smartcash.ui.dataset.visualization.handlers.setup_handlers import setup_visualization_handlers
            
            # Setup mock returns
            mock_dashboard.return_value = self.ui_components
            mock_split.return_value = self.ui_components
            mock_distribution.return_value = self.ui_components
            mock_layer.return_value = self.ui_components
            mock_bbox.return_value = self.ui_components
            
            # Panggil fungsi setup
            result = setup_visualization_handlers(self.ui_components)
            
            # Verifikasi fungsi dipanggil
            mock_dashboard.assert_called_once()
            mock_split.assert_called_once()
            mock_distribution.assert_called_once()
            mock_layer.assert_called_once()
            mock_bbox.assert_called_once()
            
            # Verifikasi hasil
            self.assertEqual(result, self.ui_components)
        except ImportError:
            self.skipTest("Modul setup_handlers tidak tersedia")
    
    @patch('smartcash.ui.dataset.visualization.handlers.bbox_handlers.on_bbox_button_click')
    def test_bbox_button_click(self, mock_click_handler):
        """Test klik tombol bbox"""
        try:
            from smartcash.ui.dataset.visualization.handlers.bbox_handlers import setup_bbox_handlers
            
            # Setup handler
            setup_bbox_handlers(self.ui_components)
            
            # Simulasi klik tombol
            button = self.ui_components['visualization_components']['bbox_tab']['button']
            button.on_click.assert_called_once()
            
            # Dapatkan fungsi callback
            callback = button.on_click.call_args[0][0]
            
            # Panggil callback
            callback(button)
            
            # Verifikasi handler dipanggil
            mock_click_handler.assert_called_once()
        except ImportError:
            self.skipTest("Modul bbox_handlers tidak tersedia")
    
    @patch('smartcash.ui.dataset.visualization.handlers.layer_handlers.on_layer_button_click')
    def test_layer_button_click(self, mock_click_handler):
        """Test klik tombol layer"""
        try:
            from smartcash.ui.dataset.visualization.handlers.layer_handlers import setup_layer_handlers
            
            # Setup handler
            setup_layer_handlers(self.ui_components)
            
            # Simulasi klik tombol
            button = self.ui_components['visualization_components']['layer_tab']['button']
            button.on_click.assert_called_once()
            
            # Dapatkan fungsi callback
            callback = button.on_click.call_args[0][0]
            
            # Panggil callback
            callback(button)
            
            # Verifikasi handler dipanggil
            mock_click_handler.assert_called_once()
        except ImportError:
            self.skipTest("Modul layer_handlers tidak tersedia")

class TestRefreshIntegration(unittest.TestCase):
    """Test integrasi untuk refresh visualisasi"""
    
    def setUp(self):
        """Setup untuk setiap test case"""
        # Mock komponen UI
        self.ui_components = {
            'status': MagicMock(spec=widgets.Output),
            'refresh_button': MagicMock(spec=widgets.Button),
            'split_cards_container': MagicMock(spec=widgets.Output),
            'preprocessing_cards': MagicMock(spec=widgets.Output),
            'augmentation_cards': MagicMock(spec=widgets.Output)
        }
    
    @patch('smartcash.ui.dataset.visualization.handlers.dashboard_handlers.get_config_manager')
    @patch('smartcash.ui.dataset.visualization.handlers.dashboard_handlers.display')
    def test_update_dashboard_cards(self, mock_display, mock_get_config):
        """Test update dashboard cards"""
        try:
            from smartcash.ui.dataset.visualization.handlers.dashboard_handlers import update_dashboard_cards
            
            # Setup mock config
            mock_config = MagicMock()
            mock_config.get_module_config.return_value = {'dataset_path': '/dummy/path'}
            mock_get_config.return_value = mock_config
            
            # Panggil fungsi update
            update_dashboard_cards(self.ui_components)
            
            # Verifikasi display dipanggil untuk menampilkan cards
            self.assertTrue(mock_display.called)
        except ImportError:
            self.skipTest("Modul dashboard_handlers tidak tersedia")
    
    @patch('smartcash.ui.dataset.visualization.handlers.dashboard_handlers.update_dashboard_cards')
    def test_refresh_button_click(self, mock_update_cards):
        """Test klik tombol refresh"""
        try:
            from smartcash.ui.dataset.visualization.handlers.dashboard_handlers import setup_dashboard_handlers
            
            # Setup handler
            setup_dashboard_handlers(self.ui_components)
            
            # Simulasi klik tombol
            button = self.ui_components['refresh_button']
            button.on_click.assert_called_once()
            
            # Dapatkan fungsi callback
            callback = button.on_click.call_args[0][0]
            
            # Panggil callback
            callback(button)
            
            # Verifikasi update cards dipanggil
            mock_update_cards.assert_called_once()
        except ImportError:
            self.skipTest("Modul dashboard_handlers tidak tersedia")

if __name__ == '__main__':
    unittest.main() 