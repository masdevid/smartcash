"""
File: smartcash/ui/dataset/visualization/tests/test_simple_visualization.py
Deskripsi: Tes sederhana untuk visualisasi dataset
"""

import unittest
import numpy as np
import random

class TestSimpleDataGeneration(unittest.TestCase):
    """Test untuk pembuatan data dummy"""
    
    def test_dummy_bbox_data(self):
        """Test pembuatan data dummy untuk bbox"""
        # Fungsi dummy untuk data bbox
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
        # Fungsi dummy untuk data layer
        def get_dummy_layer_data():
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
        # Fungsi dummy untuk distribusi kelas
        def get_dummy_class_distribution():
            classes = ['Rp1000', 'Rp2000', 'Rp5000', 'Rp10000', 'Rp20000', 'Rp50000', 'Rp100000']
            return {cls: random.randint(50, 200) for cls in classes}
        
        class_counts = get_dummy_class_distribution()
        
        # Verifikasi data kelas
        expected_classes = ['Rp1000', 'Rp2000', 'Rp5000', 'Rp10000', 'Rp20000', 'Rp50000', 'Rp100000']
        for cls in expected_classes:
            self.assertIn(cls, class_counts)

if __name__ == '__main__':
    unittest.main() 