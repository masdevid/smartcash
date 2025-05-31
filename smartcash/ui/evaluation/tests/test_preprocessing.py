"""
File: /Users/masdevid/Projects/smartcash/smartcash/ui/evaluation/tests/test_preprocessing.py
Test suite untuk memverifikasi fungsi preprocessing gambar untuk modul evaluasi.
"""
import unittest
import numpy as np
import torch
import cv2
from typing import Tuple, Dict, Any

from smartcash.dataset.utils.augmentation_utils import preprocess_image_for_yolo
# Tidak perlu mengimpor ModelPreprocessor

class TestPreprocessingFunctions(unittest.TestCase):
    """Test suite untuk fungsi preprocessing gambar."""
    
    def setUp(self) -> None: self.test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    def test_preprocess_image_for_yolo_returns_tensor(self) -> None: self.assertIsInstance(preprocess_image_for_yolo(self.test_image)[0], torch.Tensor)
    
    def test_preprocess_image_for_yolo_returns_metadata(self) -> None: self.assertIsInstance(preprocess_image_for_yolo(self.test_image)[1], dict)
    
    def test_preprocess_image_for_yolo_tensor_shape(self) -> None: self.assertEqual(preprocess_image_for_yolo(self.test_image, img_size=416)[0].shape, (1, 3, 416, 416))
    
    def test_preprocess_image_for_yolo_metadata_keys(self) -> None: 
        expected_keys = {'original_shape', 'processed_shape', 'scale_factor', 'padding'}
        self.assertTrue(expected_keys.issubset(preprocess_image_for_yolo(self.test_image)[1].keys()))
    
    def test_preprocess_image_for_yolo_normalized_values(self) -> None:
        # Memverifikasi bahwa nilai hasil normalisasi berada dalam range yang diharapkan
        tensor, _ = preprocess_image_for_yolo(self.test_image, img_size=416, normalize=True)
        # Nilai setelah normalisasi dengan mean dan std ImageNet biasanya dalam range [-2.5, 2.5]
        self.assertTrue(torch.min(tensor) >= -3.0 and torch.max(tensor) <= 3.0)

if __name__ == '__main__':
    unittest.main()
