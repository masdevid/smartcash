"""
File: smartcash/ui/dataset/visualization/tests/test_dashboard_visualization.py
Deskripsi: Test untuk dashboard visualisasi dataset
"""

import unittest
import os
import tempfile
import shutil
from unittest.mock import patch, MagicMock

from smartcash.ui.dataset.visualization.components.dashboard_component import create_dashboard_component
from smartcash.ui.dataset.visualization.handlers.dashboard_handler import (
    get_dataset_stats, get_preprocessing_stats, get_augmentation_stats, get_processing_status
)


class TestDashboardVisualization(unittest.TestCase):
    """Test untuk dashboard visualisasi dataset."""
    
    def setUp(self):
        """Setup untuk test."""
        # Buat direktori temporary untuk test
        self.test_dir = tempfile.mkdtemp()
        
        # Buat struktur direktori dataset
        os.makedirs(os.path.join(self.test_dir, 'images', 'train'), exist_ok=True)
        os.makedirs(os.path.join(self.test_dir, 'images', 'val'), exist_ok=True)
        os.makedirs(os.path.join(self.test_dir, 'images', 'test'), exist_ok=True)
        os.makedirs(os.path.join(self.test_dir, 'labels', 'train'), exist_ok=True)
        os.makedirs(os.path.join(self.test_dir, 'labels', 'val'), exist_ok=True)
        os.makedirs(os.path.join(self.test_dir, 'labels', 'test'), exist_ok=True)
        os.makedirs(os.path.join(self.test_dir, 'metadata', 'preprocessing'), exist_ok=True)
        os.makedirs(os.path.join(self.test_dir, 'metadata', 'augmentation'), exist_ok=True)
        
        # Buat beberapa file dummy
        for i in range(5):
            with open(os.path.join(self.test_dir, 'images', 'train', f'img_{i}.jpg'), 'w') as f:
                f.write('dummy')
            with open(os.path.join(self.test_dir, 'labels', 'train', f'img_{i}.txt'), 'w') as f:
                f.write('0 0.5 0.5 0.1 0.1\n1 0.7 0.7 0.2 0.2')
        
        for i in range(3):
            with open(os.path.join(self.test_dir, 'images', 'val', f'img_{i}.jpg'), 'w') as f:
                f.write('dummy')
            with open(os.path.join(self.test_dir, 'labels', 'val', f'img_{i}.txt'), 'w') as f:
                f.write('0 0.5 0.5 0.1 0.1')
        
        for i in range(2):
            with open(os.path.join(self.test_dir, 'images', 'test', f'img_{i}.jpg'), 'w') as f:
                f.write('dummy')
            with open(os.path.join(self.test_dir, 'labels', 'test', f'img_{i}.txt'), 'w') as f:
                f.write('0 0.5 0.5 0.1 0.1')
        
        # Buat file metadata
        with open(os.path.join(self.test_dir, 'metadata', 'preprocessing', 'processed_images.txt'), 'w') as f:
            f.write('img_0.jpg\nimg_1.jpg\nimg_2.jpg\nimg_3.jpg\nimg_4.jpg\nimg_5.jpg\nimg_6.jpg\nimg_7.jpg\nimg_8.jpg\nimg_9.jpg')
        
        with open(os.path.join(self.test_dir, 'metadata', 'preprocessing', 'train_processed.txt'), 'w') as f:
            f.write('processed')
        
        with open(os.path.join(self.test_dir, 'metadata', 'augmentation', 'augmented_images.txt'), 'w') as f:
            f.write('img_0.jpg\nimg_1.jpg\nimg_2.jpg\nimg_3.jpg\nimg_4.jpg')
        
        with open(os.path.join(self.test_dir, 'metadata', 'augmentation', 'train_augmented.txt'), 'w') as f:
            f.write('augmented')
    
    def tearDown(self):
        """Cleanup setelah test."""
        # Hapus direktori temporary
        shutil.rmtree(self.test_dir)
    
    def test_get_dataset_stats(self):
        """Test untuk fungsi get_dataset_stats."""
        with patch('smartcash.ui.dataset.visualization.handlers.dashboard_handler.get_config_manager') as mock_config:
            # Mock config manager
            mock_config.return_value.get.return_value = self.test_dir
            
            # Panggil fungsi
            stats = get_dataset_stats()
            
            # Verifikasi hasil
            self.assertEqual(stats['split_stats']['train']['images'], 5)
            self.assertEqual(stats['split_stats']['val']['images'], 3)
            self.assertEqual(stats['split_stats']['test']['images'], 2)
            self.assertEqual(stats['split_stats']['train']['labels'], 5)
            self.assertEqual(stats['split_stats']['train']['objects'], 10)  # 2 objek per file
    
    def test_get_preprocessing_stats(self):
        """Test untuk fungsi get_preprocessing_stats."""
        # Panggil fungsi
        stats = get_preprocessing_stats(self.test_dir)
        
        # Verifikasi hasil
        self.assertEqual(stats['processed_images'], 10)
    
    def test_get_augmentation_stats(self):
        """Test untuk fungsi get_augmentation_stats."""
        # Panggil fungsi
        stats = get_augmentation_stats(self.test_dir)
        
        # Verifikasi hasil
        self.assertEqual(stats['augmented_images'], 5)
    
    def test_get_processing_status(self):
        """Test untuk fungsi get_processing_status."""
        # Panggil fungsi
        status = get_processing_status(self.test_dir)
        
        # Verifikasi hasil
        self.assertTrue(status['preprocessing_status']['train'])
        self.assertFalse(status['preprocessing_status']['val'])
        self.assertTrue(status['augmentation_status']['train'])
        self.assertFalse(status['augmentation_status']['val'])
    
    def test_create_dashboard_component(self):
        """Test untuk fungsi create_dashboard_component."""
        # Panggil fungsi
        ui_components = create_dashboard_component()
        
        # Verifikasi hasil
        self.assertIn('main_container', ui_components)
        self.assertIn('status', ui_components)
        self.assertIn('refresh_button', ui_components)
        self.assertIn('split_cards_container', ui_components)
        self.assertIn('preprocessing_cards', ui_components)
        self.assertIn('augmentation_cards', ui_components)
        self.assertIn('visualization_components', ui_components)


if __name__ == '__main__':
    unittest.main()
