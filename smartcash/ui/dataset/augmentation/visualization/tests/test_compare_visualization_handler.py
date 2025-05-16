"""
File: smartcash/ui/dataset/augmentation/visualization/tests/test_compare_visualization_handler.py
Deskripsi: Test untuk handler visualisasi perbandingan preprocess vs augmentasi
"""

import os
import unittest
import numpy as np
from unittest.mock import patch, MagicMock, call
import matplotlib.pyplot as plt

from smartcash.ui.dataset.augmentation.visualization.handlers.compare_visualization_handler import CompareVisualizationHandler


class TestCompareVisualizationHandler(unittest.TestCase):
    """Test untuk handler visualisasi perbandingan preprocess vs augmentasi"""
    
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
        self.handler = CompareVisualizationHandler(self.config)
        
        # Buat dummy image untuk test
        self.dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
        self.dummy_image[30:70, 30:70, 0] = 255  # Buat kotak merah
        
        # Buat dummy labels untuk test
        self.dummy_labels = [
            [0, 0.5, 0.5, 0.4, 0.4]  # class_id, x_center, y_center, width, height
        ]
        
        # Buat dummy samples untuk test
        self.dummy_original_samples = [
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
        
        self.dummy_preprocessed_samples = [
            {
                'image': self.dummy_image,
                'labels': self.dummy_labels,
                'img_path': 'data/preprocessed/train/images/img1.jpg',
                'label_path': 'data/preprocessed/train/labels/img1.txt',
                'filename': 'img1.jpg'
            },
            {
                'image': self.dummy_image,
                'labels': self.dummy_labels,
                'img_path': 'data/preprocessed/train/images/img2.jpg',
                'label_path': 'data/preprocessed/train/labels/img2.txt',
                'filename': 'img2.jpg'
            }
        ]
    
    def test_visualize_preprocess_vs_augmentation(self):
        """Test metode visualize_preprocess_vs_augmentation"""
        # Patch load_sample_data dan _visualize_comparison
        with patch.object(self.handler, 'load_sample_data') as mock_load_sample_data, \
             patch.object(self.handler, '_visualize_comparison') as mock_visualize_comparison:
            
            # Test dengan data valid
            mock_load_sample_data.side_effect = [self.dummy_original_samples, self.dummy_preprocessed_samples]
            mock_visualize_comparison.return_value = {'status': 'success'}
            
            result = self.handler.visualize_preprocess_vs_augmentation(
                original_dir='data/original',
                preprocessed_dir='data/preprocessed',
                aug_type='combined',
                split='train',
                n_samples=2
            )
            
            # Cek apakah load_sample_data dipanggil dengan benar
            mock_load_sample_data.assert_has_calls([
                call('data/original', 'train', 2),
                call('data/preprocessed', 'train', 2)
            ])
            
            # Cek apakah _visualize_comparison dipanggil dengan benar
            mock_visualize_comparison.assert_called_once_with(
                self.dummy_original_samples,
                self.dummy_preprocessed_samples,
                'combined',
                f"Perbandingan Original vs Preprocessed vs Augmented (combined)"
            )
            
            # Cek hasil
            self.assertEqual(result, {'status': 'success'})
            
            # Test dengan original samples kosong
            mock_load_sample_data.reset_mock()
            mock_load_sample_data.side_effect = [[], self.dummy_preprocessed_samples]
            
            result = self.handler.visualize_preprocess_vs_augmentation(
                original_dir='data/original',
                preprocessed_dir='data/preprocessed',
                aug_type='combined',
                split='train'
            )
            
            # Cek hasil
            self.assertEqual(result['status'], 'error')
            
            # Test dengan preprocessed samples kosong
            mock_load_sample_data.reset_mock()
            mock_load_sample_data.side_effect = [self.dummy_original_samples, []]
            
            result = self.handler.visualize_preprocess_vs_augmentation(
                original_dir='data/original',
                preprocessed_dir='data/preprocessed',
                aug_type='combined',
                split='train'
            )
            
            # Cek hasil
            self.assertEqual(result['status'], 'error')
    
    @patch('smartcash.ui.dataset.augmentation.visualization.handlers.compare_visualization_handler.CompareVisualizationHandler._create_visualization_figure')
    @patch('smartcash.ui.dataset.augmentation.visualization.handlers.compare_visualization_handler.CompareVisualizationHandler._visualize_image')
    @patch('smartcash.ui.dataset.augmentation.visualization.handlers.compare_visualization_handler.CompareVisualizationHandler.apply_augmentation')
    @patch('smartcash.ui.dataset.augmentation.visualization.handlers.compare_visualization_handler.CompareVisualizationHandler.save_figure')
    @patch('matplotlib.pyplot.tight_layout')
    def test_visualize_comparison(self, mock_tight_layout, mock_save_figure, mock_apply_augmentation, mock_visualize_image, mock_create_figure):
        """Test metode _visualize_comparison"""
        # Mock _create_visualization_figure
        mock_fig = MagicMock()
        mock_axes = np.array([[MagicMock(), MagicMock(), MagicMock()], [MagicMock(), MagicMock(), MagicMock()]])
        mock_create_figure.return_value = (mock_fig, mock_axes)
        
        # Mock apply_augmentation
        mock_apply_augmentation.return_value = (self.dummy_image, self.dummy_labels)
        
        # Test _visualize_comparison
        result = self.handler._visualize_comparison(
            original_samples=self.dummy_original_samples,
            preprocessed_samples=self.dummy_preprocessed_samples,
            aug_type='combined'
        )
        
        # Cek apakah _create_visualization_figure dipanggil dengan benar
        mock_create_figure.assert_called_once_with(2, 3, None)  # 2 sampel, 3 kolom, tanpa judul
        
        # Cek apakah apply_augmentation dipanggil dengan benar
        self.assertEqual(mock_apply_augmentation.call_count, 2)
        
        # Cek apakah _visualize_image dipanggil dengan benar (6 kali: 2 sampel x 3 gambar)
        self.assertEqual(mock_visualize_image.call_count, 6)
        
        # Cek apakah tight_layout dipanggil
        mock_tight_layout.assert_called_once()
        
        # Cek apakah save_figure dipanggil dengan benar
        mock_save_figure.assert_called_once_with(mock_fig, 'comparison_combined.png')
        
        # Cek hasil
        self.assertEqual(result['aug_type'], 'combined')
        self.assertEqual(result['n_samples'], 2)
        self.assertEqual(result['figure'], mock_fig)
        self.assertEqual(result['n_samples'], 2)
    
    @patch('matplotlib.pyplot.subplots')
    @patch('smartcash.ui.dataset.augmentation.visualization.handlers.compare_visualization_handler.CompareVisualizationHandler._visualize_image')
    @patch('smartcash.ui.dataset.augmentation.visualization.handlers.compare_visualization_handler.CompareVisualizationHandler.load_sample_data')
    @patch('smartcash.ui.dataset.augmentation.visualization.handlers.compare_visualization_handler.CompareVisualizationHandler.apply_augmentation')
    @patch('smartcash.ui.dataset.augmentation.visualization.handlers.compare_visualization_handler.CompareVisualizationHandler.save_figure')
    @patch('matplotlib.pyplot.tight_layout')
    def test_visualize_augmentation_impact(self, mock_tight_layout, mock_save_figure, mock_apply_augmentation, mock_load_sample_data, mock_visualize_image, mock_subplots):
        """Test metode visualize_augmentation_impact"""
        # Mock plt.subplots
        mock_fig = MagicMock()
        mock_axes = np.array([MagicMock(), MagicMock(), MagicMock(), MagicMock()])
        mock_subplots.return_value = (mock_fig, mock_axes)
        
        # Mock load_sample_data
        mock_load_sample_data.return_value = [self.dummy_preprocessed_samples[0]]
        
        # Mock apply_augmentation
        mock_apply_augmentation.return_value = (self.dummy_image, self.dummy_labels)
        
        # Test visualize_augmentation_impact
        result = self.handler.visualize_augmentation_impact(
            preprocessed_dir='data/preprocessed',
            aug_types=['combined', 'position', 'lighting'],
            split='train'
        )
        
        # Cek apakah load_sample_data dipanggil dengan benar
        mock_load_sample_data.assert_called_once_with('data/preprocessed', 'train', 1)
        
        # Cek apakah plt.subplots dipanggil dengan benar
        mock_subplots.assert_called_once()
        
        # Cek apakah apply_augmentation dipanggil dengan benar
        self.assertEqual(mock_apply_augmentation.call_count, 3)  # 3 augmentation types
        
        # Cek apakah _visualize_image dipanggil dengan benar (4 kali: 1 original + 3 augmentation types)
        self.assertEqual(mock_visualize_image.call_count, 4)
        
        # Cek apakah tight_layout dipanggil
        mock_tight_layout.assert_called_once()
        
        # Cek apakah save_figure dipanggil dengan benar
        # Nama file sekarang menggunakan filename dari sampel
        mock_save_figure.assert_called_once_with(mock_fig, 'augmentation_impact_img1.jpg.png')
        
        # Cek hasil
        self.assertEqual(result['status'], 'success')
        self.assertEqual(result['figure'], mock_fig)
        self.assertEqual(result['filename'], 'img1.jpg')
        self.assertEqual(result['aug_types'], ['combined', 'position', 'lighting'])
        
        # Test dengan samples kosong
        mock_load_sample_data.return_value = []
        result = self.handler.visualize_augmentation_impact(
            preprocessed_dir='data/preprocessed',
            aug_types=['combined'],
            split='train'
        )
        
        # Cek hasil
        self.assertEqual(result['status'], 'error')


if __name__ == '__main__':
    unittest.main()
