"""
File: smartcash/ui/dataset/augmentation/visualization/tests/test_visualization_manager.py
Deskripsi: Test untuk manager visualisasi augmentasi
"""

import unittest
import threading
from unittest.mock import patch, MagicMock, call
import ipywidgets as widgets
from IPython.display import display

from smartcash.ui.dataset.augmentation.visualization.visualization_manager import AugmentationVisualizationManager


class TestAugmentationVisualizationManager(unittest.TestCase):
    """Test untuk manager visualisasi augmentasi"""
    
    def setUp(self):
        """Setup untuk test"""
        # Reset singleton instance
        AugmentationVisualizationManager._instance = None
        
        # Inisialisasi kelas yang diuji
        self.manager = AugmentationVisualizationManager.get_instance()
        
        # Buat mock untuk handler dan komponen
        self.mock_sample_handler = MagicMock()
        self.mock_compare_handler = MagicMock()
        self.mock_components = MagicMock()
        
        # Ganti handler dan komponen dengan mock
        self.manager.sample_handler = self.mock_sample_handler
        self.manager.compare_handler = self.mock_compare_handler
        self.manager.ui_components = self.mock_components
        
        # Mock komponen UI
        self.mock_components.aug_type_dropdown = MagicMock()
        self.mock_components.aug_type_dropdown.value = 'combined'
        self.mock_components.split_dropdown = MagicMock()
        self.mock_components.split_dropdown.value = 'train'
        self.mock_components.sample_count_slider = MagicMock()
        self.mock_components.sample_count_slider.value = 3
        self.mock_components.data_dir_text = MagicMock()
        self.mock_components.data_dir_text.value = 'data'
        self.mock_components.preprocessed_dir_text = MagicMock()
        self.mock_components.preprocessed_dir_text.value = 'data/preprocessed'
        self.mock_components.sample_output = MagicMock()
        self.mock_components.compare_output = MagicMock()
    
    def test_singleton(self):
        """Test pattern singleton"""
        # Dapatkan instance lagi
        manager2 = AugmentationVisualizationManager.get_instance()
        
        # Cek apakah instance sama
        self.assertIs(self.manager, manager2)
    
    def test_init(self):
        """Test inisialisasi kelas"""
        # Cek apakah komponen diinisialisasi dengan benar
        self.assertIsNotNone(self.manager.ui_components)
        self.assertIsNotNone(self.manager.sample_handler)
        self.assertIsNotNone(self.manager.compare_handler)
        
        # Pastikan register_handlers dipanggil dengan benar
        # Karena register_handlers dipanggil di __init__, kita perlu memeriksa apakah mock_components dibuat
        # dan bukan memeriksa apakah register_handlers dipanggil
    
    @patch('threading.Thread')
    def test_on_visualize_samples(self, mock_thread):
        """Test metode _on_visualize_samples"""
        # Mock button
        mock_button = MagicMock()
        
        # Mock UI components
        self.manager.ui_components.aug_type_dropdown.value = 'combined'
        self.manager.ui_components.split_dropdown.value = 'train'
        self.manager.ui_components.sample_count_slider.value = 3
        self.manager.ui_components.data_dir_text.value = 'data'
        
        # Mock hasil visualisasi
        self.mock_sample_handler.visualize_augmentation_samples.return_value = {
            'status': 'success',
            'results': [{
                'figure': MagicMock()
            }]
        }
        
        # Panggil metode yang diuji
        self.manager._on_visualize_samples(mock_button)
        
        # Cek apakah thread dibuat dengan benar
        mock_thread.assert_called_once()
        mock_thread.return_value.start.assert_called_once()
        
        # Dapatkan fungsi yang dijalankan di thread dan jalankan
        thread_func = mock_thread.call_args[1]['target']
        thread_func()
        
        # Cek apakah visualize_augmentation_samples dipanggil dengan benar
        # Periksa parameter yang diberikan
        call_args = self.mock_sample_handler.visualize_augmentation_samples.call_args
        self.assertEqual(call_args[1]['data_dir'], 'data')
        self.assertEqual(call_args[1]['aug_types'], ['combined'])
        self.assertEqual(call_args[1]['split'], 'train')
        self.assertEqual(call_args[1]['num_samples'], 3)
        
        # Cek apakah show_figure dipanggil dengan benar
        self.mock_components.show_figure.assert_called_once()
        # Cek apakah show_figure dipanggil dengan benar
        self.mock_components.show_figure.assert_called_once()
        
        # Cek apakah show_status dipanggil dengan benar
        self.assertEqual(self.mock_components.show_status.call_count, 2)
        
        # Test dengan error
        self.mock_sample_handler.visualize_augmentation_samples.side_effect = Exception("Test error")
        
        # Reset mock
        self.mock_components.show_status.reset_mock()
        self.mock_components.show_figure.reset_mock()
        
        # Panggil fungsi yang dijalankan di thread
        thread_func()
        
        # Cek apakah show_status dipanggil dengan benar
        self.mock_components.show_status.assert_called_once()
        self.assertIn("error", self.mock_components.show_status.call_args[0][1].lower())
    
    @patch('threading.Thread')
    def test_on_visualize_variations(self, mock_thread):
        """Test metode _on_visualize_variations"""
        # Mock button
        mock_button = MagicMock()
        
        # Mock UI components
        self.manager.ui_components.aug_type_dropdown.value = 'combined'
        self.manager.ui_components.split_dropdown.value = 'train'
        self.manager.ui_components.data_dir_text.value = 'data'
        
        # Mock hasil visualisasi
        self.mock_sample_handler.visualize_augmentation_variations.return_value = {
            'status': 'success',
            'figure': MagicMock(),
            'filename': 'test.jpg'
        }
        
        # Panggil metode yang diuji
        self.manager._on_visualize_variations(mock_button)
        
        # Cek apakah thread dibuat dengan benar
        mock_thread.assert_called_once()
        mock_thread.return_value.start.assert_called_once()
        
        # Dapatkan fungsi yang dijalankan di thread
        thread_func = mock_thread.call_args[1]['target']
        
        # Jalankan fungsi thread
        thread_func()
        
        # Cek apakah visualize_augmentation_variations dipanggil dengan benar
        # Periksa parameter yang diberikan
        call_args = self.mock_sample_handler.visualize_augmentation_variations.call_args
        self.assertEqual(call_args[1]['data_dir'], 'data')
        self.assertEqual(call_args[1]['aug_type'], 'combined')
        self.assertEqual(call_args[1]['split'], 'train')
        
        # Cek apakah show_figure dipanggil dengan benar
        self.mock_components.show_figure.assert_called_once()
        
        # Cek apakah show_status dipanggil dengan benar
        self.assertEqual(self.mock_components.show_status.call_count, 2)
        
        # Test dengan error
        self.mock_sample_handler.visualize_augmentation_variations.side_effect = Exception("Test error")
        
        # Reset mock
        self.mock_components.show_status.reset_mock()
        self.mock_components.show_figure.reset_mock()
        
        # Panggil fungsi yang dijalankan di thread
        thread_func()
        
        # Cek apakah show_status dipanggil dengan benar
        self.mock_components.show_status.assert_called_once()
        self.assertIn("error", self.mock_components.show_status.call_args[0][1].lower())
    
    @patch('threading.Thread')
    def test_on_visualize_compare(self, mock_thread):
        """Test metode _on_visualize_compare"""
        # Mock button
        mock_button = MagicMock()
        
        # Mock UI components
        self.manager.ui_components.aug_type_dropdown.value = 'combined'
        self.manager.ui_components.split_dropdown.value = 'train'
        self.manager.ui_components.sample_count_slider.value = 3
        self.manager.ui_components.data_dir_text.value = 'data'
        self.manager.ui_components.preprocessed_dir_text.value = 'data/preprocessed'
        
        # Mock hasil visualisasi
        self.mock_compare_handler.visualize_preprocess_vs_augmentation.return_value = {
            'status': 'success',
            'figure': MagicMock()
        }
        
        # Panggil metode yang diuji
        self.manager._on_visualize_compare(mock_button)
        
        # Cek apakah thread dibuat dengan benar
        mock_thread.assert_called_once()
        mock_thread.return_value.start.assert_called_once()
        
        # Dapatkan fungsi yang dijalankan di thread dan jalankan
        thread_func = mock_thread.call_args[1]['target']
        thread_func()
        
        # Cek apakah visualize_preprocess_vs_augmentation dipanggil dengan benar
        # Periksa parameter yang diberikan
        call_args = self.mock_compare_handler.visualize_preprocess_vs_augmentation.call_args
        self.assertEqual(call_args[1]['data_dir'], 'data')
        self.assertEqual(call_args[1]['preprocessed_dir'], 'data/preprocessed')
        self.assertEqual(call_args[1]['aug_type'], 'combined')
        self.assertEqual(call_args[1]['split'], 'train')
        
        # Cek apakah show_figure dipanggil dengan benar
        self.mock_components.show_figure.assert_called_once()
        
        # Cek apakah show_status dipanggil dengan benar
        self.assertEqual(self.mock_components.show_status.call_count, 2)
        
        # Test dengan error
        self.mock_compare_handler.visualize_preprocess_vs_augmentation.side_effect = Exception("Test error")
        
        # Reset mock
        self.mock_components.show_status.reset_mock()
        self.mock_components.show_figure.reset_mock()
        
        # Panggil fungsi yang dijalankan di thread
        thread_func()
        
        # Cek apakah show_status dipanggil dengan benar
        self.mock_components.show_status.assert_called_once()
        self.assertIn("error", self.mock_components.show_status.call_args[0][1].lower())
    
    @patch('threading.Thread')
    def test_on_visualize_impact(self, mock_thread):
        """Test metode _on_visualize_impact"""
        # Mock button
        mock_button = MagicMock()
        
        # Mock UI components
        self.manager.ui_components.split_dropdown.value = 'train'
        self.manager.ui_components.data_dir_text.value = 'data'
        self.manager.ui_components.preprocessed_dir_text.value = 'data/preprocessed'
        
        # Mock hasil visualisasi
        self.mock_compare_handler.visualize_augmentation_impact.return_value = {
            'status': 'success',
            'figure': MagicMock(),
            'filename': 'test.jpg'
        }
        
        # Panggil metode yang diuji
        self.manager._on_visualize_impact(mock_button)
        
        # Cek apakah thread dibuat dengan benar
        mock_thread.assert_called_once()
        mock_thread.return_value.start.assert_called_once()
        
        # Dapatkan fungsi yang dijalankan di thread dan jalankan
        thread_func = mock_thread.call_args[1]['target']
        thread_func()
        
        # Cek apakah visualize_augmentation_impact dipanggil dengan benar
        # Periksa parameter yang diberikan
        call_args = self.mock_compare_handler.visualize_augmentation_impact.call_args
        self.assertTrue('preprocessed_dir' in call_args[1])
        self.assertEqual(call_args[1]['preprocessed_dir'], 'data/preprocessed')
        self.assertEqual(call_args[1]['aug_types'], ['combined', 'position', 'lighting'])
        self.assertEqual(call_args[1]['split'], 'train')
        
        # Cek apakah show_figure dipanggil dengan benar
        self.mock_components.show_figure.assert_called_once()
        
        # Cek apakah show_status dipanggil dengan benar
        self.assertEqual(self.mock_components.show_status.call_count, 2)
    
    def test_get_visualization_ui(self):
        """Test metode get_visualization_ui"""
        # Buat mock baru untuk create_visualization_ui
        mock_ui = MagicMock()
        self.manager.ui_components.create_visualization_ui = MagicMock(return_value=mock_ui)
        
        # Test get_visualization_ui
        ui = self.manager.get_visualization_ui()
        
        # Cek apakah create_visualization_ui dipanggil dengan benar
        self.manager.ui_components.create_visualization_ui.assert_called_once()
        
        # Cek apakah UI dikembalikan dengan benar
        self.assertEqual(ui, mock_ui)
    
    def test_display_visualization_ui(self):
        """Test metode display_visualization_ui"""
        # Buat mock untuk get_visualization_ui
        mock_ui = MagicMock()
        original_get_ui = self.manager.get_visualization_ui
        self.manager.get_visualization_ui = MagicMock(return_value=mock_ui)
        
        # Buat mock untuk display dengan patch yang benar
        with patch('smartcash.ui.dataset.augmentation.visualization.visualization_manager.display') as mock_display:
            # Test display_visualization_ui
            self.manager.display_visualization_ui()
            
            # Cek apakah get_visualization_ui dipanggil dengan benar
            self.manager.get_visualization_ui.assert_called_once()
            
            # Cek apakah display dipanggil dengan benar
            mock_display.assert_called_once_with(mock_ui)
        
        # Kembalikan metode asli
        self.manager.get_visualization_ui = original_get_ui
    



if __name__ == '__main__':
    unittest.main()
