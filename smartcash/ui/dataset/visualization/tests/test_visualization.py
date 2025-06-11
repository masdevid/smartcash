"""
File: smartcash/ui/dataset/visualization/tests/test_visualization.py
Deskripsi: Unit test untuk modul visualisasi dataset
"""

import pytest
import os
import numpy as np
from unittest.mock import MagicMock, patch
import ipywidgets as widgets

from smartcash.ui.dataset.visualization import (
    VisualizationController,
    show_visualization
)
from smartcash.ui.dataset.visualization.components import (
    DatasetStatsComponent,
    AugmentationVisualizer
)

# Mock data untuk testing
MOCK_DATASET_STATS = {
    'summary': {
        'train': {
            'total_images': 100,
            'total_annotations': 500,
            'class_distribution': {
                'class1': 200,
                'class2': 200,
                'class3': 100
            }
        },
        'val': {
            'total_images': 20,
            'total_annotations': 100,
            'class_distribution': {
                'class1': 40,
                'class2': 40,
                'class3': 20
            }
        },
        'test': {
            'total_images': 30,
            'total_annotations': 150,
            'class_distribution': {
                'class1': 60,
                'class2': 60,
                'class3': 30
            }
        }
    },
    'image_sizes': {
        'train': [(640, 480), (800, 600), (1024, 768)],
        'val': [(640, 480), (800, 600)],
        'test': [(800, 600), (1024, 768)]
    },
    'class_distribution': {
        'train': {'class1': 200, 'class2': 200, 'class3': 100},
        'val': {'class1': 40, 'class2': 40, 'class3': 20},
        'test': {'class1': 60, 'class2': 60, 'class3': 30}
    }
}

# Fixture untuk controller
def mock_controller():
    """Fixture untuk membuat instance controller dengan mock"""
    with patch('smartcash.dataset.preprocessor.list_available_datasets') as mock_list, \
         patch('smartcash.dataset.preprocessor.get_preprocessing_stats') as mock_stats:
        
        # Setup mock
        mock_list.return_value = ['test_dataset']
        mock_stats.return_value = MOCK_DATASET_STATS
        
        # Buat controller
        controller = VisualizationController()
        
        return controller, mock_list, mock_stats

# Test VisualizationController
class TestVisualizationController:
    """Test untuk VisualizationController"""
    
    def test_initialization(self, mock_controller):
        """Test inisialisasi controller"""
        controller, mock_list, mock_stats = mock_controller
        
        # Verifikasi inisialisasi
        assert controller is not None
        assert isinstance(controller.stats_component, DatasetStatsComponent)
        assert controller.aug_visualizer is None  # Belum diinisialisasi sampai dataset dimuat
    
    def test_load_dataset_success(self, mock_controller):
        """Test memuat dataset berhasil"""
        controller, mock_list, mock_stats = mock_controller
        
        # Panggil method load_dataset
        result = controller.load_dataset('test_dataset')
        
        # Verifikasi hasil
        assert result is True
        assert controller.current_dataset == 'test_dataset'
        assert controller.dataset_stats == MOCK_DATASET_STATS
        assert isinstance(controller.aug_visualizer, AugmentationVisualizer)
    
    def test_load_dataset_failure(self, mock_controller):
        """Test gagal memuat dataset"""
        controller, mock_list, mock_stats = mock_controller
        
        # Setup mock untuk mengembalikan dataset tidak ditemukan
        mock_list.return_value = ['other_dataset']
        
        # Panggil method load_dataset dengan dataset yang tidak ada
        result = controller.load_dataset('nonexistent_dataset')
        
        # Verifikasi hasil
        assert result is False
        assert controller.current_dataset is None
    
    @patch('IPython.display.display')
    def test_display(self, mock_display, mock_controller):
        """Test menampilkan UI"""
        controller, _, _ = mock_controller
        
        # Panggil method display
        controller.display()
        
        # Verifikasi display dipanggil
        assert mock_display.called
        assert hasattr(controller, 'main_container')

# Test DatasetStatsComponent
class TestDatasetStatsComponent:
    """Test untuk DatasetStatsComponent"""
    
    def test_initialization(self):
        """Test inisialisasi komponen statistik"""
        component = DatasetStatsComponent()
        
        # Verifikasi inisialisasi
        assert component is not None
        assert component.stats_data == {}
    
    def test_update_stats(self):
        """Test memperbarui data statistik"""
        component = DatasetStatsComponent()
        
        # Panggil method update_stats
        component.update_stats(MOCK_DATASET_STATS)
        
        # Verifikasi data terupdate
        assert component.stats_data == MOCK_DATASET_STATS
    
    @patch('plotly.graph_objects.FigureWidget')
    def test_create_class_distribution_plot(self, mock_figure):
        """Test membuat plot distribusi kelas"""
        component = DatasetStatsComponent()
        component.update_stats(MOCK_DATASET_STATS)
        
        # Panggil method _create_class_distribution_plot
        result = component._create_class_distribution_plot()
        
        # Verifikasi hasil
        assert result is not None
    
    @patch('plotly.graph_objects.FigureWidget')
    def test_create_image_size_plot(self, mock_figure):
        """Test membuat plot ukuran gambar"""
        component = DatasetStatsComponent()
        component.update_stats(MOCK_DATASET_STATS)
        
        # Panggil method _create_image_size_plot
        result = component._create_image_size_plot()
        
        # Verifikasi hasil
        assert result is not None

# Test AugmentationVisualizer
class TestAugmentationVisualizer:
    """Test untuk AugmentationVisualizer"""
    
    @patch('os.path.exists')
    @patch('os.listdir')
    @patch('cv2.imread')
    def test_load_random_image(self, mock_imread, mock_listdir, mock_exists):
        """Test memuat gambar acak"""
        # Setup mock
        mock_exists.return_value = True
        mock_listdir.return_value = ['image1.jpg', 'image2.jpg']
        mock_imread.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Buat instance
        visualizer = AugmentationVisualizer(dataset_path='/test/dataset')
        
        # Panggil method load_random_image
        result = visualizer.load_random_image(split='train')
        
        # Verifikasi hasil
        assert result is True
        assert visualizer.current_image is not None
        assert visualizer.current_image_path.endswith('.jpg')
    
    @patch('os.path.exists')
    @patch('os.listdir')
    def test_load_random_image_no_images(self, mock_listdir, mock_exists):
        """Test memuat gambar saat tidak ada gambar"""
        # Setup mock
        mock_exists.return_value = True
        mock_listdir.return_value = []
        
        # Buat instance
        visualizer = AugmentationVisualizer(dataset_path='/test/dataset')
        
        # Panggil method load_random_image
        result = visualizer.load_random_image(split='train')
        
        # Verifikasi hasil
        assert result is False
        assert visualizer.current_image is None

# Test show_visualization function
@patch('smartcash.ui.dataset.visualization.VisualizationController')
def test_show_visualization(mock_controller_class):
    """Test fungsi show_visualization"""
    # Setup mock
    mock_controller = MagicMock()
    mock_controller_class.return_value = mock_controller
    
    # Panggil fungsi
    result = show_visualization()
    
    # Verifikasi
    mock_controller_class.assert_called_once()
    mock_controller.display.assert_called_once()
    assert result == mock_controller
