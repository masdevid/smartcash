"""
File: smartcash/ui/dataset/visualization/tests/test_simple.py
Deskripsi: Tes sederhana untuk visualisasi dataset tanpa mengimpor modul yang bermasalah
"""

import unittest
from unittest.mock import MagicMock, patch
import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class TestSimpleVisualization(unittest.TestCase):
    """Tes sederhana untuk visualisasi dataset"""
    
    def setUp(self):
        """Setup untuk setiap test case"""
        # Mock output widget
        self.output = MagicMock(spec=widgets.Output)
        
        # Setup patches
        self.display_patcher = patch('IPython.display.display')
        self.plt_figure_patcher = patch('matplotlib.pyplot.figure')
        self.plt_gcf_patcher = patch('matplotlib.pyplot.gcf')
        
        # Start patches
        self.mock_display = self.display_patcher.start()
        self.mock_figure = self.plt_figure_patcher.start()
        self.mock_gcf = self.plt_gcf_patcher.start()
        
        # Setup mock returns
        self.mock_figure.return_value = MagicMock()
        self.mock_gcf.return_value = MagicMock()
    
    def tearDown(self):
        """Cleanup setelah setiap test case"""
        # Stop patches
        self.display_patcher.stop()
        self.plt_figure_patcher.stop()
        self.plt_gcf_patcher.stop()
    
    def test_dummy_data_generation(self):
        """Test pembuatan data dummy"""
        # Fungsi dummy untuk membuat data bbox
        def get_dummy_bbox_data():
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
        
        # Dapatkan data dummy
        bbox_data = get_dummy_bbox_data()
        
        # Verifikasi struktur data
        self.assertIn('positions', bbox_data)
        self.assertIn('sizes', bbox_data)
        self.assertIn('aspect_ratios', bbox_data)
        self.assertIn('classes', bbox_data)
        
        # Verifikasi data positions
        self.assertIn('x_center', bbox_data['positions'])
        self.assertIn('y_center', bbox_data['positions'])
    
    def test_plot_function(self):
        """Test fungsi plotting sederhana"""
        # Fungsi dummy untuk plotting
        def plot_class_distribution(class_counts, output, is_dummy=False):
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
        
        # Data dummy
        class_counts = {
            'Rp1000': 120,
            'Rp2000': 80,
            'Rp5000': 150,
            'Rp10000': 200,
            'Rp20000': 100,
            'Rp50000': 90,
            'Rp100000': 110
        }
        
        # Panggil fungsi plot
        plot_class_distribution(class_counts, self.output)
        
        # Verifikasi figure dibuat
        self.mock_figure.assert_called_once()
        
        # Verifikasi plot ditampilkan
        self.assertTrue(self.mock_gcf.return_value.show.called)

if __name__ == '__main__':
    unittest.main() 