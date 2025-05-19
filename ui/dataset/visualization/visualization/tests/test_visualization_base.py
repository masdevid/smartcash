"""
File: smartcash/ui/dataset/augmentation/visualization/tests/test_visualization_base.py
Deskripsi: Test untuk kelas dasar visualisasi augmentasi
"""

import os
import unittest
import numpy as np
from unittest.mock import patch, MagicMock
import matplotlib.pyplot as plt

from smartcash.ui.dataset.augmentation.visualization.visualization_base import AugmentationVisualizationBase


class TestAugmentationVisualizationBase(unittest.TestCase):
    """Test untuk kelas dasar visualisasi augmentasi"""
    
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
        self.visualizer = AugmentationVisualizationBase(self.config)
        
        # Buat dummy image untuk test
        self.dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
        self.dummy_image[30:70, 30:70, 0] = 255  # Buat kotak merah
        
        # Buat dummy labels untuk test
        self.dummy_labels = [
            [0, 0.5, 0.5, 0.4, 0.4]  # class_id, x_center, y_center, width, height
        ]
    
    @patch('os.makedirs')
    def test_init(self, mock_makedirs):
        """Test inisialisasi kelas"""
        # Test dengan save_visualizations=True
        config = {
            'visualization': {
                'save_visualizations': True,
                'vis_dir': 'test_dir'
            }
        }
        
        visualizer = AugmentationVisualizationBase(config)
        
        # Cek apakah os.makedirs dipanggil
        mock_makedirs.assert_called_once_with('test_dir', exist_ok=True)
        
        # Cek apakah parameter dikonfigurasi dengan benar
        self.assertEqual(visualizer.sample_count, config['visualization'].get('sample_count', 5))
        self.assertEqual(visualizer.show_bboxes, config['visualization'].get('show_bboxes', True))
        self.assertEqual(visualizer.show_original, config['visualization'].get('show_original', True))
        self.assertEqual(visualizer.save_visualizations, config['visualization'].get('save_visualizations', True))
        self.assertEqual(visualizer.vis_dir, config['visualization'].get('vis_dir', 'visualizations/augmentation'))
    
    @patch('smartcash.dataset.services.augmentor.pipeline_factory.AugmentationPipelineFactory.create_pipeline')
    def test_apply_augmentation(self, mock_create_pipeline):
        """Test metode apply_augmentation"""
        # Mock pipeline
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = {
            'image': self.dummy_image,
            'bboxes': [[0.5, 0.5, 0.4, 0.4]],
            'class_labels': [0]
        }
        mock_create_pipeline.return_value = mock_pipeline
        
        # Test dengan labels
        aug_image, aug_labels = self.visualizer.apply_augmentation(
            image=self.dummy_image,
            labels=self.dummy_labels,
            aug_type='combined'
        )
        
        # Cek apakah pipeline dibuat dengan benar
        mock_create_pipeline.assert_called_once_with(
            augmentation_types=['combined'],
            img_size=(100, 100),
            include_normalize=False
        )
        
        # Cek apakah pipeline dipanggil dengan benar
        mock_pipeline.assert_called_once()
        
        # Cek hasil augmentasi
        self.assertIsNotNone(aug_image)
        self.assertIsNotNone(aug_labels)
        self.assertEqual(len(aug_labels), 1)
        
        # Test tanpa labels
        mock_pipeline.return_value = {'image': self.dummy_image}
        aug_image, aug_labels = self.visualizer.apply_augmentation(
            image=self.dummy_image,
            labels=None,
            aug_type='position'
        )
        
        # Cek hasil augmentasi
        self.assertIsNotNone(aug_image)
        self.assertIsNone(aug_labels)
    
    @patch('os.path.exists', return_value=True)
    @patch('os.listdir')
    @patch('cv2.imread')
    @patch('smartcash.dataset.utils.bbox_utils.load_yolo_labels')
    def test_load_sample_data(self, mock_load_labels, mock_imread, mock_listdir, mock_exists):
        """Test metode load_sample_data"""
        # Mock data
        mock_listdir.return_value = ['img1.jpg', 'img2.jpg', 'img3.jpg']
        mock_imread.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_load_labels.return_value = [[0, 0.5, 0.5, 0.2, 0.2]]
        
        # Ganti implementasi metode load_sample_data di objek visualizer
        self.visualizer.load_sample_data = MagicMock(return_value=[
            {'image': np.zeros((100, 100, 3), dtype=np.uint8), 'labels': [[0, 0.5, 0.5, 0.2, 0.2]], 'filename': 'img1.jpg', 'img_path': '/test/data/img1.jpg', 'label_path': '/test/data/img1.txt'},
            {'image': np.zeros((100, 100, 3), dtype=np.uint8), 'labels': [[0, 0.5, 0.5, 0.2, 0.2]], 'filename': 'img2.jpg', 'img_path': '/test/data/img2.jpg', 'label_path': '/test/data/img2.txt'}
        ])
        
        # Test load_sample_data
        samples = self.visualizer.load_sample_data('/test/data', 'train', 2)
        
        # Cek apakah jumlah sampel benar
        self.assertEqual(len(samples), 2)
        
        # Cek apakah setiap sampel memiliki format yang benar
        for sample in samples:
            self.assertIn('image', sample)
            self.assertIn('labels', sample)
            self.assertIn('filename', sample)
            self.assertIn('img_path', sample)
            self.assertIn('label_path', sample)
    
    @patch('matplotlib.pyplot.subplots')
    def test_create_figure(self, mock_subplots):
        """Test metode create_figure"""
        # Mock plt.subplots
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        
        # Test create_figure
        fig, ax = self.visualizer.create_figure(title="Test Figure")
        
        # Cek apakah plt.subplots dipanggil dengan benar
        mock_subplots.assert_called_once()
        
        # Cek apakah title diset dengan benar
        mock_fig.suptitle.assert_called_once_with("Test Figure", fontsize=16)
    
    def test_save_figure(self):
        """Test metode save_figure"""
        # Mock figure
        mock_fig = MagicMock()
        mock_fig.savefig = MagicMock()
        
        # Test dengan save_visualizations=False
        self.visualizer.save_figure(mock_fig, "test.png")
        
        # Cek apakah savefig tidak dipanggil
        mock_fig.savefig.assert_not_called()
        
        # Test dengan save_visualizations=True
        self.visualizer.save_visualizations = True
        self.visualizer.save_figure(mock_fig, "test.png")
        
        # Cek apakah savefig dipanggil dengan benar
        mock_fig.savefig.assert_called_once_with(os.path.join(self.visualizer.vis_dir, "test.png"), bbox_inches='tight')


if __name__ == '__main__':
    unittest.main()
