"""
File: smartcash/ui/dataset/augmentation/visualization/tests/test_sample_visualization_handler.py
Deskripsi: Test untuk handler visualisasi sampel augmentasi
"""

import os
import unittest
import numpy as np
from unittest.mock import patch, MagicMock, call
import matplotlib.pyplot as plt

from smartcash.ui.dataset.augmentation.visualization.handlers.sample_visualization_handler import SampleVisualizationHandler


class TestSampleVisualizationHandler(unittest.TestCase):
    """Test untuk handler visualisasi sampel augmentasi"""
    
    def setUp(self):
        """Setup untuk test"""
        # Mock config
        self.config = {
            'visualization': {
                'sample_count': 3,
                'show_bboxes': True,
                'show_original': True,
                'save_visualizations': True,  # Ubah ke True agar save_figure dipanggil
                'vis_dir': 'test_visualizations'
            }
        }
        
        # Inisialisasi kelas yang diuji
        self.handler = SampleVisualizationHandler(self.config)
        
        # Buat dummy image untuk test
        self.dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
        self.dummy_image[30:70, 30:70, 0] = 255  # Buat kotak merah
        
        # Buat dummy labels untuk test
        self.dummy_labels = [
            [0, 0.5, 0.5, 0.4, 0.4]  # class_id, x_center, y_center, width, height
        ]
        
        # Buat dummy samples untuk test
        self.dummy_samples = [
            {
                'image': self.dummy_image,
                'labels': self.dummy_labels,
                'img_path': 'data/train/images/img1.jpg',
                'label_path': 'data/train/labels/img1.txt',
                'filename': 'img1.jpg'
            },
            {
                'image': self.dummy_image,
                'labels': self.dummy_labels,
                'img_path': 'data/train/images/img2.jpg',
                'label_path': 'data/train/labels/img2.txt',
                'filename': 'img2.jpg'
            }
        ]
    
    @patch('smartcash.ui.dataset.augmentation.visualization.handlers.sample_visualization_handler.SampleVisualizationHandler.load_sample_data')
    @patch('smartcash.ui.dataset.augmentation.visualization.handlers.sample_visualization_handler.SampleVisualizationHandler._visualize_single_augmentation_type')
    def test_visualize_augmentation_samples(self, mock_visualize_single, mock_load_sample_data):
        """Test metode visualize_augmentation_samples"""
        # Mock load_sample_data
        mock_load_sample_data.return_value = self.dummy_samples
        
        # Mock _visualize_single_augmentation_type
        mock_visualize_single.return_value = {
            'aug_type': 'combined',
            'figure': MagicMock(),
            'n_samples': 2
        }
        
        # Test visualize_augmentation_samples
        result = self.handler.visualize_augmentation_samples(
            data_dir='data',
            aug_types=['combined', 'position'],
            split='train',
            num_samples=2
        )
        
        # Cek apakah load_sample_data dipanggil dengan benar
        mock_load_sample_data.assert_called_once_with('data', 'train', 2)
        
        # Cek apakah _visualize_single_augmentation_type dipanggil dengan benar
        self.assertEqual(mock_visualize_single.call_count, 2)
        mock_visualize_single.assert_has_calls([
            call(self.dummy_samples, 'combined'),
            call(self.dummy_samples, 'position')
        ])
        
        # Cek hasil
        self.assertEqual(result['status'], 'success')
        self.assertEqual(len(result['results']), 2)
        
        # Test dengan samples kosong
        mock_load_sample_data.return_value = []
        result = self.handler.visualize_augmentation_samples(
            data_dir='data',
            aug_types=['combined'],
            split='train',
            num_samples=2
        )
        
        # Cek hasil
        self.assertEqual(result['status'], 'error')
    
    @patch('smartcash.ui.dataset.augmentation.visualization.handlers.sample_visualization_handler.SampleVisualizationHandler._create_visualization_figure')
    @patch('smartcash.ui.dataset.augmentation.visualization.handlers.sample_visualization_handler.SampleVisualizationHandler._visualize_image')
    @patch('smartcash.ui.dataset.augmentation.visualization.handlers.sample_visualization_handler.SampleVisualizationHandler.apply_augmentation')
    @patch('smartcash.ui.dataset.augmentation.visualization.handlers.sample_visualization_handler.SampleVisualizationHandler.save_figure')
    @patch('matplotlib.pyplot.tight_layout')
    def test_visualize_single_augmentation_type(self, mock_tight_layout, mock_save_figure, mock_apply_augmentation, mock_visualize_image, mock_create_figure):
        """Test metode _visualize_single_augmentation_type"""
        # Mock _create_visualization_figure
        mock_fig = MagicMock()
        mock_axes = np.array([[MagicMock(), MagicMock()], [MagicMock(), MagicMock()]])
        mock_create_figure.return_value = (mock_fig, mock_axes)
        
        # Mock apply_augmentation
        mock_apply_augmentation.return_value = (self.dummy_image, self.dummy_labels)
        
        # Test _visualize_single_augmentation_type
        result = self.handler._visualize_single_augmentation_type(
            samples=self.dummy_samples,
            aug_type='combined'
        )
        
        # Cek apakah _create_visualization_figure dipanggil dengan benar
        mock_create_figure.assert_called_once_with(2)  # 2 sampel
        
        # Cek apakah apply_augmentation dipanggil dengan benar
        self.assertEqual(mock_apply_augmentation.call_count, 2)
        
        # Cek apakah _visualize_image dipanggil dengan benar (4 kali: 2 sampel x 2 gambar)
        self.assertEqual(mock_visualize_image.call_count, 4)
        
        # Cek apakah tight_layout dipanggil
        mock_tight_layout.assert_called_once()
        
        # Cek apakah save_figure dipanggil dengan benar
        mock_save_figure.assert_called_once_with(mock_fig, 'augmentation_samples_combined.png')
        
        # Cek hasil
        self.assertEqual(result['aug_type'], 'combined')
        self.assertEqual(result['n_samples'], 2)
        self.assertEqual(result['figure'], mock_fig)
    
    @patch('matplotlib.pyplot.subplots')
    @patch('smartcash.ui.dataset.augmentation.visualization.handlers.sample_visualization_handler.SampleVisualizationHandler._visualize_image')
    @patch('smartcash.ui.dataset.augmentation.visualization.handlers.sample_visualization_handler.SampleVisualizationHandler.load_sample_data')
    @patch('smartcash.ui.dataset.augmentation.visualization.handlers.sample_visualization_handler.SampleVisualizationHandler.apply_augmentation')
    @patch('smartcash.ui.dataset.augmentation.visualization.handlers.sample_visualization_handler.SampleVisualizationHandler.save_figure')
    @patch('matplotlib.pyplot.tight_layout')
    def test_visualize_augmentation_variations(self, mock_tight_layout, mock_save_figure, mock_apply_augmentation, mock_load_sample_data, mock_visualize_image, mock_subplots):
        """Test metode visualize_augmentation_variations"""
        # Mock plt.subplots
        mock_fig = MagicMock()
        mock_axes = np.array([[MagicMock(), MagicMock(), MagicMock()], [MagicMock(), MagicMock(), MagicMock()]])
        mock_subplots.return_value = (mock_fig, mock_axes)
        
        # Mock load_sample_data
        mock_load_sample_data.return_value = [self.dummy_samples[0]]
        
        # Mock apply_augmentation
        mock_apply_augmentation.return_value = (self.dummy_image, self.dummy_labels)
        
        # Test visualize_augmentation_variations
        result = self.handler.visualize_augmentation_variations(
            data_dir='data',
            aug_type='combined',
            split='train',
            n_variations=5
        )
        
        # Cek apakah load_sample_data dipanggil dengan benar
        mock_load_sample_data.assert_called_once_with('data', 'train', 1)
        
        # Cek apakah plt.subplots dipanggil dengan benar
        mock_subplots.assert_called_once()
        
        # Cek apakah apply_augmentation dipanggil dengan benar
        self.assertEqual(mock_apply_augmentation.call_count, 5)
        
        # Cek apakah _visualize_image dipanggil dengan benar (6 kali: 1 original + 5 variations)
        self.assertEqual(mock_visualize_image.call_count, 6)
        
        # Cek apakah tight_layout dipanggil
        mock_tight_layout.assert_called_once()
        
        # Cek apakah save_figure dipanggil dengan benar
        mock_save_figure.assert_called_once_with(mock_fig, 'augmentation_variations_combined.png')
        
        # Cek hasil
        self.assertEqual(result['status'], 'success')
        self.assertEqual(result['aug_type'], 'combined')
        self.assertEqual(result['figure'], mock_fig)
        self.assertEqual(result['filename'], 'img1.jpg')
        
        # Test dengan samples kosong
        mock_load_sample_data.return_value = []
        result = self.handler.visualize_augmentation_variations(
            data_dir='data',
            aug_type='combined',
            split='train'
        )
        
        # Cek hasil
        self.assertEqual(result['status'], 'error')


if __name__ == '__main__':
    unittest.main()
