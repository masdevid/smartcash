"""
File: smartcash/ui/dataset/augmentation/visualization/tests/test_visualization_components.py
Deskripsi: Test untuk komponen UI visualisasi augmentasi
"""

import unittest
from unittest.mock import patch, MagicMock, call
import ipywidgets as widgets
from IPython.display import display

from smartcash.ui.dataset.augmentation.visualization.components.visualization_components import AugmentationVisualizationComponents


class TestAugmentationVisualizationComponents(unittest.TestCase):
    """Test untuk komponen UI visualisasi augmentasi"""
    
    def setUp(self):
        """Setup untuk test"""
        # Mock config
        self.config = {
            'visualization': {
                'sample_count': 3,
                'show_bboxes': True,
                'show_original': True,
                'save_visualizations': False,
                'vis_dir': 'test_visualizations'
            }
        }
        
        # Inisialisasi kelas yang diuji
        self.components = AugmentationVisualizationComponents(self.config)
    
    def test_init(self):
        """Test inisialisasi kelas"""
        # Cek apakah komponen UI diinisialisasi dengan benar
        self.assertIsInstance(self.components.visualization_tabs, widgets.Tab)
        self.assertIsInstance(self.components.sample_output, widgets.Output)
        self.assertIsInstance(self.components.compare_output, widgets.Output)
        self.assertIsInstance(self.components.status_output, widgets.Output)
        self.assertIsInstance(self.components.visualize_samples_button, widgets.Button)
        self.assertIsInstance(self.components.visualize_variations_button, widgets.Button)
        self.assertIsInstance(self.components.visualize_compare_button, widgets.Button)
        self.assertIsInstance(self.components.visualize_impact_button, widgets.Button)
        self.assertIsInstance(self.components.aug_type_dropdown, widgets.Dropdown)
        self.assertIsInstance(self.components.split_dropdown, widgets.Dropdown)
        self.assertIsInstance(self.components.sample_count_slider, widgets.IntSlider)
        self.assertIsInstance(self.components.show_bbox_checkbox, widgets.Checkbox)
        self.assertIsInstance(self.components.data_dir_text, widgets.Text)
        self.assertIsInstance(self.components.preprocessed_dir_text, widgets.Text)
        
        # Cek apakah nilai default diset dengan benar
        self.assertEqual(self.components.sample_count_slider.value, self.config['visualization']['sample_count'])
        self.assertEqual(self.components.show_bbox_checkbox.value, self.config['visualization']['show_bboxes'])
    
    def test_create_sample_tab(self):
        """Test metode create_sample_tab"""
        # Panggil metode yang diuji
        tab = self.components.create_sample_tab()
        
        # Cek apakah tab dibuat dengan benar
        self.assertIsInstance(tab, widgets.VBox)
        self.assertEqual(len(tab.children), 2)
        self.assertIsInstance(tab.children[0], widgets.VBox)  # controls
        self.assertIs(tab.children[1], self.components.sample_output)
    
    def test_create_compare_tab(self):
        """Test metode create_compare_tab"""
        # Panggil metode yang diuji
        tab = self.components.create_compare_tab()
        
        # Cek apakah tab dibuat dengan benar
        self.assertIsInstance(tab, widgets.VBox)
        self.assertEqual(len(tab.children), 2)
        self.assertIsInstance(tab.children[0], widgets.VBox)  # controls
        self.assertIs(tab.children[1], self.components.compare_output)
    
    def test_create_visualization_ui(self):
        """Test metode create_visualization_ui"""
        # Panggil metode yang diuji
        ui = self.components.create_visualization_ui()
        
        # Cek apakah UI dibuat dengan benar
        self.assertIsInstance(ui, widgets.VBox)
        self.assertEqual(len(ui.children), 2)
        self.assertIs(ui.children[0], self.components.visualization_tabs)
        self.assertIs(ui.children[1], self.components.status_output)
        
        # Cek apakah tab diset dengan benar
        self.assertEqual(len(self.components.visualization_tabs.children), 2)
    
    @patch('builtins.print')
    def test_show_status(self, mock_print):
        """Test metode show_status"""
        # Test show_status dengan exception untuk memicu fallback
        self.components.status_output = MagicMock()
        self.components.status_output.__enter__ = MagicMock(side_effect=Exception("Test exception"))
        
        # Panggil metode yang diuji
        self.components.show_status("Test message", "info")
        
        # Cek apakah print dipanggil dengan benar
        mock_print.assert_called_once_with("Status: info - Test message")
    
    @patch('builtins.print')
    def test_show_figure(self, mock_print):
        """Test metode show_figure"""
        # Mock figure
        mock_fig = MagicMock()
        
        # Mock output widget dengan exception untuk memicu fallback
        mock_output = MagicMock()
        mock_output.__enter__ = MagicMock(side_effect=Exception("Test exception"))
        
        # Test show_figure
        self.components.show_figure(mock_fig, mock_output)
        
        # Cek apakah print dipanggil dengan benar
        mock_print.assert_called_once_with("Menampilkan figure di output widget")
    
    def test_register_handlers(self):
        """Test metode register_handlers"""
        # Simpan referensi asli ke on_click untuk setiap tombol
        original_on_click_methods = {
            'samples': self.components.visualize_samples_button.on_click,
            'variations': self.components.visualize_variations_button.on_click,
            'compare': self.components.visualize_compare_button.on_click,
            'impact': self.components.visualize_impact_button.on_click
        }
        
        # Ganti metode on_click dengan mock
        self.components.visualize_samples_button.on_click = MagicMock()
        self.components.visualize_variations_button.on_click = MagicMock()
        self.components.visualize_compare_button.on_click = MagicMock()
        self.components.visualize_impact_button.on_click = MagicMock()
        
        # Mock handlers
        mock_on_visualize_samples = MagicMock()
        mock_on_visualize_variations = MagicMock()
        mock_on_visualize_compare = MagicMock()
        mock_on_visualize_impact = MagicMock()
        
        # Test register_handlers
        self.components.register_handlers(
            on_visualize_samples=mock_on_visualize_samples,
            on_visualize_variations=mock_on_visualize_variations,
            on_visualize_compare=mock_on_visualize_compare,
            on_visualize_impact=mock_on_visualize_impact
        )
        
        # Verifikasi bahwa on_click dipanggil dengan handler yang benar
        self.components.visualize_samples_button.on_click.assert_called_once_with(mock_on_visualize_samples)
        self.components.visualize_variations_button.on_click.assert_called_once_with(mock_on_visualize_variations)
        self.components.visualize_compare_button.on_click.assert_called_once_with(mock_on_visualize_compare)
        self.components.visualize_impact_button.on_click.assert_called_once_with(mock_on_visualize_impact)
        
        # Kembalikan metode on_click asli
        self.components.visualize_samples_button.on_click = original_on_click_methods['samples']
        self.components.visualize_variations_button.on_click = original_on_click_methods['variations']
        self.components.visualize_compare_button.on_click = original_on_click_methods['compare']
        self.components.visualize_impact_button.on_click = original_on_click_methods['impact']


if __name__ == '__main__':
    unittest.main()
