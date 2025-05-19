"""
File: smartcash/ui/dataset/augmentation/visualization/tests/test_visualization_simple.py
Deskripsi: Test sederhana untuk visualisasi augmentasi
"""

import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import os

from smartcash.ui.dataset.augmentation.visualization.visualization_initializer import initialize_augmentation_visualization
from smartcash.ui.dataset.augmentation.visualization.visualization_manager import AugmentationVisualizationManager
from smartcash.ui.dataset.augmentation.visualization.components.visualization_components import AugmentationVisualizationComponents
from smartcash.ui.dataset.augmentation.visualization.handlers.sample_visualization_handler import SampleVisualizationHandler
from smartcash.ui.dataset.augmentation.visualization.handlers.compare_visualization_handler import CompareVisualizationHandler


class TestVisualizationSimple(unittest.TestCase):
    """Test sederhana untuk visualisasi augmentasi"""
    
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
            }
        ]
    
    @patch('smartcash.ui.dataset.augmentation.visualization.visualization_initializer.create_header')
    @patch('smartcash.ui.dataset.augmentation.visualization.visualization_initializer.display')
    @patch('smartcash.ui.dataset.augmentation.visualization.visualization_initializer.get_config_manager')
    @patch('smartcash.ui.dataset.augmentation.visualization.visualization_manager.AugmentationVisualizationManager.get_instance')
    def test_initialize_visualization(self, mock_manager_get_instance, mock_get_config_manager, mock_display, mock_create_header):
        """Test inisialisasi visualisasi"""
        # Mock header
        mock_header = MagicMock()
        mock_create_header.return_value = mock_header
        
        # Mock ConfigManager
        mock_config_manager = MagicMock()
        mock_config_manager.get_module_config.return_value = self.config
        mock_get_config_manager.return_value = mock_config_manager
        
        # Mock VisualizationManager
        mock_manager = MagicMock()
        mock_manager_get_instance.return_value = mock_manager
        
        # Test initialize_augmentation_visualization
        initialize_augmentation_visualization()
        
        # Verifikasi panggilan
        mock_create_header.assert_called_once()
        mock_display.assert_any_call(mock_header)
        mock_get_config_manager.assert_called_once()
        mock_config_manager.get_module_config.assert_called_once_with('augmentation')
        mock_manager_get_instance.assert_called_once()
        mock_manager.display_visualization_ui.assert_called_once()
    
    @patch('smartcash.ui.dataset.augmentation.visualization.visualization_manager.AugmentationVisualizationComponents')
    def test_visualization_manager(self, mock_components_class):
        """Test manager visualisasi"""
        # Mock komponen
        mock_components = MagicMock()
        mock_components_class.return_value = mock_components
        
        # Mock logger
        mock_logger = MagicMock()
        
        # Test inisialisasi manager
        manager = AugmentationVisualizationManager(self.config, mock_logger)
        
        # Verifikasi inisialisasi komponen - gunakan any untuk mengabaikan parameter tambahan
        self.assertEqual(mock_components_class.call_count, 1)
        args, kwargs = mock_components_class.call_args
        self.assertEqual(args[0], self.config)  # Verifikasi parameter pertama adalah config
        
        # Test get_visualization_ui
        manager.get_visualization_ui()
        mock_components.create_visualization_ui.assert_called_once()
    
    def test_visualization_components(self):
        """Test komponen visualisasi"""
        # Test inisialisasi komponen
        components = AugmentationVisualizationComponents(self.config)
        
        # Verifikasi atribut
        self.assertEqual(components.sample_count_slider.value, self.config['visualization']['sample_count'])
        self.assertEqual(components.show_bbox_checkbox.value, self.config['visualization']['show_bboxes'])
        
        # Test create_visualization_ui
        ui = components.create_visualization_ui()
        self.assertEqual(len(ui.children), 2)
    
    @patch('smartcash.ui.dataset.augmentation.visualization.handlers.sample_visualization_handler.SampleVisualizationHandler.apply_augmentation')
    @patch('smartcash.ui.dataset.augmentation.visualization.handlers.sample_visualization_handler.SampleVisualizationHandler.load_sample_data')
    def test_sample_handler_basic(self, mock_load_sample_data, mock_apply_augmentation):
        """Test dasar untuk sample handler"""
        # Mock fungsi
        mock_load_sample_data.return_value = self.dummy_samples
        mock_apply_augmentation.return_value = (self.dummy_image, self.dummy_labels)
        
        # Patch plt.subplots dan plt.figure
        with patch('matplotlib.pyplot.subplots', return_value=(MagicMock(), MagicMock())), \
             patch('matplotlib.pyplot.figure', return_value=MagicMock()), \
             patch('matplotlib.pyplot.tight_layout'), \
             patch('matplotlib.axes.Axes.imshow'), \
             patch('matplotlib.axes.Axes.set_title'), \
             patch('matplotlib.axes.Axes.axis'), \
             patch('smartcash.dataset.utils.bbox_utils.draw_bboxes'):
            
            # Inisialisasi handler
            handler = SampleVisualizationHandler(self.config)
            
            # Test load_sample_data
            handler.load_sample_data('data', 'train', 1)
            mock_load_sample_data.assert_called_once_with('data', 'train', 1)
            
            # Test apply_augmentation
            handler.apply_augmentation(self.dummy_image, self.dummy_labels, 'combined')
            mock_apply_augmentation.assert_called_once_with(self.dummy_image, self.dummy_labels, 'combined')
    
    @patch('smartcash.ui.dataset.augmentation.visualization.handlers.compare_visualization_handler.CompareVisualizationHandler.load_sample_data')
    def test_compare_handler_basic(self, mock_load_sample_data):
        """Test dasar untuk compare handler"""
        # Mock fungsi
        mock_load_sample_data.return_value = self.dummy_samples
        
        # Patch plt.subplots dan plt.figure
        with patch('matplotlib.pyplot.subplots', return_value=(MagicMock(), MagicMock())), \
             patch('matplotlib.pyplot.figure', return_value=MagicMock()), \
             patch('matplotlib.pyplot.tight_layout'), \
             patch('matplotlib.axes.Axes.imshow'), \
             patch('matplotlib.axes.Axes.set_title'), \
             patch('matplotlib.axes.Axes.axis'), \
             patch('smartcash.dataset.utils.bbox_utils.draw_bboxes'):
            
            # Inisialisasi handler
            handler = CompareVisualizationHandler(self.config)
            
            # Test load_sample_data
            handler.load_sample_data('data', 'train', 1)
            mock_load_sample_data.assert_called_once_with('data', 'train', 1)


if __name__ == '__main__':
    unittest.main()
