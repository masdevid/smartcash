"""
File: smartcash/ui/dataset/visualization/tests/test_visualization_display.py
Deskripsi: Test untuk tampilan visualisasi dataset tanpa validasi data
"""

import unittest
import ipywidgets as widgets
from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from unittest.mock import MagicMock, patch

# Fungsi dummy untuk data
def get_dummy_bbox_data():
    """Fungsi dummy untuk data bbox"""
    return {
        'positions': {
            'x_center': np.random.rand(100),
            'y_center': np.random.rand(100)
        },
        'sizes': {
            'width': np.random.rand(100) * 0.5,
            'height': np.random.rand(100) * 0.5
        },
        'aspect_ratios': np.random.rand(100) * 2,
        'classes': np.random.choice(['Rp1000', 'Rp2000', 'Rp5000', 'Rp10000', 'Rp20000', 'Rp50000', 'Rp100000'], 100)
    }

def get_dummy_layer_data():
    """Fungsi dummy untuk data layer"""
    layers = ['layer1', 'layer2', 'layer3']
    classes = ['Rp1000', 'Rp2000', 'Rp5000', 'Rp10000', 'Rp20000', 'Rp50000', 'Rp100000']
    
    # Buat data dummy untuk class_counts
    class_counts = {}
    for layer in layers:
        class_counts[layer] = {}
        for cls in classes:
            class_counts[layer][cls] = random.randint(10, 100)
    
    # Buat data dummy untuk feature_maps
    feature_maps = {}
    for layer in layers:
        feature_maps[layer] = np.random.rand(5, 5, 3)
    
    # Buat data dummy untuk layer_stats
    layer_stats = {}
    for layer in layers:
        layer_stats[layer] = {
            'mean_activation': random.uniform(0.1, 0.9),
            'std_activation': random.uniform(0.01, 0.2),
            'min_activation': random.uniform(0, 0.1),
            'max_activation': random.uniform(0.8, 1.0)
        }
    
    return {
        'class_counts': class_counts,
        'feature_maps': feature_maps,
        'layer_stats': layer_stats,
        'layers': layers,
        'classes': classes
    }

def get_dummy_class_distribution():
    """Fungsi dummy untuk distribusi kelas"""
    classes = ['Rp1000', 'Rp2000', 'Rp5000', 'Rp10000', 'Rp20000', 'Rp50000', 'Rp100000']
    return {cls: random.randint(50, 200) for cls in classes}

# Fungsi dummy untuk plotting
def plot_bbox_position_distribution(bbox_data, output, is_dummy=False):
    """Fungsi dummy untuk plotting distribusi posisi bbox"""
    with output:
        plt.figure(figsize=(10, 6))
        plt.scatter(bbox_data['positions']['x_center'], bbox_data['positions']['y_center'], alpha=0.5)
        plt.title('Distribusi Posisi Bounding Box')
        plt.xlabel('X Center')
        plt.ylabel('Y Center')
        plt.grid(True)
        plt.show()
        
        if is_dummy:
            show_dummy_data_warning(output)

def plot_bbox_size_distribution(bbox_data, output):
    """Fungsi dummy untuk plotting distribusi ukuran bbox"""
    with output:
        plt.figure(figsize=(10, 6))
        plt.scatter(bbox_data['sizes']['width'], bbox_data['sizes']['height'], alpha=0.5)
        plt.title('Distribusi Ukuran Bounding Box')
        plt.xlabel('Width')
        plt.ylabel('Height')
        plt.grid(True)
        plt.show()
        display("Bbox size distribution displayed")

def plot_layer_class_distribution(layer_data, output, is_dummy=False):
    """Fungsi dummy untuk plotting distribusi kelas per layer"""
    with output:
        plt.figure(figsize=(12, 6))
        for i, layer in enumerate(layer_data['layers']):
            counts = [layer_data['class_counts'][layer][cls] for cls in layer_data['classes']]
            plt.bar([x + i*0.25 for x in range(len(layer_data['classes']))], counts, width=0.25, label=layer)
        
        plt.title('Distribusi Kelas per Layer')
        plt.xlabel('Kelas')
        plt.ylabel('Jumlah')
        plt.xticks(range(len(layer_data['classes'])), layer_data['classes'], rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        if is_dummy:
            show_dummy_data_warning(output)

def plot_feature_maps(layer_data, output):
    """Fungsi dummy untuk plotting feature maps"""
    with output:
        for layer, feature_map in layer_data['feature_maps'].items():
            plt.figure(figsize=(10, 5))
            plt.imshow(feature_map)
            plt.title(f'Feature Map untuk {layer}')
            plt.colorbar()
            plt.show()
        display("Feature maps displayed")

def plot_layer_statistics(layer_data, output):
    """Fungsi dummy untuk plotting statistik layer"""
    with output:
        stats = []
        for layer in layer_data['layers']:
            stats.append({
                'Layer': layer,
                'Mean Activation': layer_data['layer_stats'][layer]['mean_activation'],
                'Std Activation': layer_data['layer_stats'][layer]['std_activation'],
                'Min Activation': layer_data['layer_stats'][layer]['min_activation'],
                'Max Activation': layer_data['layer_stats'][layer]['max_activation']
            })
        
        df = pd.DataFrame(stats)
        display(df)
        display("Layer statistics displayed")

def plot_class_distribution(class_counts, output, is_dummy=False):
    """Fungsi dummy untuk plotting distribusi kelas"""
    with output:
        plt.figure(figsize=(10, 6))
        classes = list(class_counts.keys())
        counts = list(class_counts.values())
        plt.bar(classes, counts)
        plt.title('Distribusi Kelas')
        plt.xlabel('Kelas')
        plt.ylabel('Jumlah')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        display("Class distribution displayed")

def show_dummy_data_warning(output):
    """Fungsi dummy untuk menampilkan peringatan data dummy"""
    with output:
        display(widgets.HTML(
            "<div style='background-color: #fff3cd; color: #856404; padding: 10px; border-left: 5px solid #ffeeba; margin: 10px 0;'>"
            "<strong>⚠️ Peringatan:</strong> Menggunakan data dummy karena dataset tidak tersedia."
            "</div>"
        ))

def show_loading_status(output, message):
    """Fungsi dummy untuk menampilkan status loading"""
    with output:
        display(widgets.HTML(
            f"<div style='background-color: #d1ecf1; color: #0c5460; padding: 10px; border-left: 5px solid #bee5eb; margin: 10px 0;'>"
            f"<strong>ℹ️ Info:</strong> {message}"
            "</div>"
        ))

def show_success_status(output, message):
    """Fungsi dummy untuk menampilkan status sukses"""
    with output:
        display(widgets.HTML(
            f"<div style='background-color: #d4edda; color: #155724; padding: 10px; border-left: 5px solid #c3e6cb; margin: 10px 0;'>"
            f"<strong>✅ Sukses:</strong> {message}"
            "</div>"
        ))

class TestVisualizationDisplay(unittest.TestCase):
    """Test untuk tampilan visualisasi dataset"""
    
    def setUp(self):
        """Setup untuk setiap test case"""
        # Mock output widget
        self.output = MagicMock(spec=widgets.Output)
        
        # Setup patches
        self.display_patcher = patch('IPython.display.display')
        self.plt_figure_patcher = patch('matplotlib.pyplot.figure')
        self.plt_gcf_patcher = patch('matplotlib.pyplot.gcf')
        self.plt_show_patcher = patch('matplotlib.pyplot.show')
        
        # Start patches
        self.mock_display = self.display_patcher.start()
        self.mock_figure = self.plt_figure_patcher.start()
        self.mock_gcf = self.plt_gcf_patcher.start()
        self.mock_show = self.plt_show_patcher.start()
        
        # Setup mock returns
        self.mock_figure.return_value = MagicMock()
        self.mock_gcf.return_value = MagicMock()
        
    def tearDown(self):
        """Cleanup setelah setiap test case"""
        # Stop patches
        self.display_patcher.stop()
        self.plt_figure_patcher.stop()
        self.plt_gcf_patcher.stop()
        self.plt_show_patcher.stop()
    
    def test_bbox_position_distribution_display(self):
        """Test tampilan distribusi posisi bbox"""
        # Patch show_dummy_data_warning
        with patch('smartcash.ui.dataset.visualization.tests.test_visualization_display.show_dummy_data_warning') as mock_warning:
            # Dapatkan data dummy
            bbox_data = get_dummy_bbox_data()
            
            # Panggil fungsi plot
            plot_bbox_position_distribution(bbox_data, self.output, is_dummy=True)
            
            # Verifikasi warning ditampilkan
            mock_warning.assert_called_once()
            
            # Verifikasi figure dibuat
            self.mock_figure.assert_called_once()
            
            # Verifikasi plot ditampilkan
            self.mock_show.assert_called_once()
    
    def test_bbox_size_distribution_display(self):
        """Test tampilan distribusi ukuran bbox"""
        # Dapatkan data dummy
        bbox_data = get_dummy_bbox_data()
        
        # Panggil fungsi plot
        plot_bbox_size_distribution(bbox_data, self.output)
        
        # Verifikasi figure dibuat
        self.mock_figure.assert_called_once()
        
        # Verifikasi plot ditampilkan
        self.mock_show.assert_called_once()
        
        # Verifikasi display dipanggil
        self.mock_display.assert_called_once()
    
    def test_layer_class_distribution_display(self):
        """Test tampilan distribusi kelas per layer"""
        # Patch show_dummy_data_warning
        with patch('smartcash.ui.dataset.visualization.tests.test_visualization_display.show_dummy_data_warning') as mock_warning:
            # Dapatkan data dummy
            layer_data = get_dummy_layer_data()
            
            # Panggil fungsi plot
            plot_layer_class_distribution(layer_data, self.output, is_dummy=True)
            
            # Verifikasi warning ditampilkan
            mock_warning.assert_called_once()
            
            # Verifikasi figure dibuat
            self.mock_figure.assert_called_once()
            
            # Verifikasi plot ditampilkan
            self.mock_show.assert_called_once()
    
    def test_feature_maps_display(self):
        """Test tampilan feature maps"""
        # Dapatkan data dummy
        layer_data = get_dummy_layer_data()
        
        # Panggil fungsi plot
        plot_feature_maps(layer_data, self.output)
        
        # Verifikasi figure dibuat (satu untuk setiap layer)
        self.assertEqual(self.mock_figure.call_count, len(layer_data['layers']))
        
        # Verifikasi plot ditampilkan (satu untuk setiap layer)
        self.assertEqual(self.mock_show.call_count, len(layer_data['layers']))
        
        # Verifikasi display dipanggil
        self.mock_display.assert_called_once()
    
    def test_layer_statistics_display(self):
        """Test tampilan statistik layer"""
        # Dapatkan data dummy
        layer_data = get_dummy_layer_data()
        
        # Panggil fungsi plot
        plot_layer_statistics(layer_data, self.output)
        
        # Verifikasi display dipanggil untuk dataframe dan pesan
        self.assertEqual(self.mock_display.call_count, 2)
    
    def test_class_distribution_display(self):
        """Test tampilan distribusi kelas"""
        # Dapatkan data dummy
        class_counts = get_dummy_class_distribution()
        
        # Panggil fungsi plot
        plot_class_distribution(class_counts, self.output, is_dummy=True)
        
        # Verifikasi figure dibuat
        self.mock_figure.assert_called_once()
        
        # Verifikasi plot ditampilkan
        self.mock_show.assert_called_once()
        
        # Verifikasi display dipanggil
        self.mock_display.assert_called_once()
    
    def test_status_display(self):
        """Test tampilan status"""
        # Panggil fungsi status
        show_loading_status(self.output, "Memuat data test...")
        show_dummy_data_warning(self.output)
        show_success_status(self.output, "Test berhasil")
        
        # Verifikasi display dipanggil untuk menampilkan status
        self.assertEqual(self.mock_display.call_count, 3)

class TestDummyDataGeneration(unittest.TestCase):
    """Test untuk pembuatan data dummy"""
    
    def test_dummy_bbox_data(self):
        """Test pembuatan data dummy untuk bbox"""
        bbox_data = get_dummy_bbox_data()
        
        # Verifikasi struktur data
        self.assertIn('positions', bbox_data)
        self.assertIn('sizes', bbox_data)
        self.assertIn('aspect_ratios', bbox_data)
        self.assertIn('classes', bbox_data)
        
        # Verifikasi data positions
        self.assertIn('x_center', bbox_data['positions'])
        self.assertIn('y_center', bbox_data['positions'])
    
    def test_dummy_layer_data(self):
        """Test pembuatan data dummy untuk layer"""
        layer_data = get_dummy_layer_data()
        
        # Verifikasi struktur data
        self.assertIn('class_counts', layer_data)
        self.assertIn('feature_maps', layer_data)
        self.assertIn('layer_stats', layer_data)
        self.assertIn('layers', layer_data)
        self.assertIn('classes', layer_data)
        
        # Verifikasi data layer stats
        for layer in layer_data['layers']:
            self.assertIn(layer, layer_data['layer_stats'])
            self.assertIn('mean_activation', layer_data['layer_stats'][layer])
            self.assertIn('std_activation', layer_data['layer_stats'][layer])
    
    def test_dummy_class_distribution(self):
        """Test pembuatan data dummy untuk distribusi kelas"""
        class_counts = get_dummy_class_distribution()
        
        # Verifikasi data kelas
        expected_classes = ['Rp1000', 'Rp2000', 'Rp5000', 'Rp10000', 'Rp20000', 'Rp50000', 'Rp100000']
        for cls in expected_classes:
            self.assertIn(cls, class_counts)

if __name__ == '__main__':
    unittest.main() 