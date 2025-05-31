"""
/Users/masdevid/Projects/smartcash/tests/integration/model/test_optimized_models.py

Pengujian untuk memverifikasi bahwa semua model optimasi dapat dibangun dengan benar
menggunakan pretrained model dari Google Drive.
"""

import unittest
import torch
from pathlib import Path
import os

from smartcash.model.manager import ModelManager
from smartcash.model.config.model_constants import OPTIMIZED_MODELS
from smartcash.common.logger import get_logger

logger = get_logger(__name__)

class TestOptimizedModels(unittest.TestCase):
    """
    Test case untuk memverifikasi bahwa semua model optimasi dapat dibangun dengan benar
    menggunakan pretrained model dari Google Drive.
    """
    
    def setUp(self):
        """Setup untuk pengujian."""
        # Pastikan kita menggunakan CPU untuk testing
        self.device = 'cpu'
        # Direktori untuk model di drive
        self.drive_models_dir = Path('/drive/MyDrive/SmartCash/models')
        
        # Periksa apakah direktori model di drive ada
        self.drive_available = self.drive_models_dir.exists()
        if not self.drive_available:
            logger.warning(f"⚠️ Direktori model di drive tidak ditemukan: {self.drive_models_dir}")
            logger.warning("⚠️ Test akan dilewati jika Google Drive tidak tersedia")
    
    def test_build_all_optimized_models(self):
        """
        Menguji build model untuk semua model optimasi di OPTIMIZED_MODELS
        tanpa menggunakan testing mode.
        """
        if not self.drive_available:
            self.skipTest("Google Drive tidak tersedia, melewati test")
        
        # Iterasi melalui semua model optimasi
        for model_type, config in OPTIMIZED_MODELS.items():
            with self.subTest(model_type=model_type):
                logger.info(f"🔍 Menguji model optimasi: {model_type}")
                logger.info(f"📝 Deskripsi: {config['description']}")
                
                try:
                    # Buat model manager dengan model_type yang sesuai
                    # testing_mode=False untuk menggunakan pretrained model asli
                    model_manager = ModelManager(
                        model_type=model_type,
                        layer_mode='single',
                        detection_layers=['banknote'],
                        testing_mode=False
                    )
                    
                    # Build model
                    model = model_manager.build_model()
                    
                    # Verifikasi model telah dibangun dengan benar
                    self.assertIsNotNone(model, f"Model {model_type} tidak berhasil dibangun")
                    
                    # Verifikasi model memiliki struktur yang benar
                    self.assertIsNotNone(model.backbone, f"Backbone untuk {model_type} tidak ada")
                    self.assertIsNotNone(model.neck, f"Neck untuk {model_type} tidak ada")
                    self.assertIsNotNone(model.head, f"Head untuk {model_type} tidak ada")
                    
                    # Verifikasi model dapat melakukan forward pass
                    dummy_input = torch.randn(1, 3, 640, 640)
                    with torch.no_grad():
                        output = model(dummy_input)
                    
                    # Verifikasi output memiliki struktur yang benar
                    self.assertIsNotNone(output, f"Output untuk {model_type} adalah None")
                    
                    logger.success(f"✅ Model {model_type} berhasil dibangun dan diuji")
                    
                except Exception as e:
                    self.fail(f"Gagal membangun model {model_type}: {str(e)}")
    
    def test_build_multilayer_models(self):
        """
        Menguji build model untuk semua model optimasi di OPTIMIZED_MODELS
        dengan mode multilayer.
        """
        if not self.drive_available:
            self.skipTest("Google Drive tidak tersedia, melewati test")
        
        # Iterasi melalui semua model optimasi
        for model_type, config in OPTIMIZED_MODELS.items():
            with self.subTest(model_type=model_type):
                logger.info(f"🔍 Menguji model optimasi multilayer: {model_type}")
                
                try:
                    # Buat model manager dengan model_type yang sesuai dan mode multilayer
                    model_manager = ModelManager(
                        model_type=model_type,
                        layer_mode='multilayer',
                        detection_layers=['banknote', 'nominal', 'security'],
                        testing_mode=False
                    )
                    
                    # Build model
                    model = model_manager.build_model()
                    
                    # Verifikasi model telah dibangun dengan benar
                    self.assertIsNotNone(model, f"Model multilayer {model_type} tidak berhasil dibangun")
                    
                    # Verifikasi model dalam mode multilayer
                    self.assertEqual(model.layer_mode, 'multilayer', f"Model {model_type} tidak dalam mode multilayer")
                    
                    # Verifikasi model memiliki semua detection layers
                    self.assertListEqual(
                        sorted(model.detection_layers), 
                        sorted(['banknote', 'nominal', 'security']), 
                        f"Detection layers tidak sesuai untuk {model_type}"
                    )
                    
                    # Verifikasi model dapat melakukan forward pass
                    dummy_input = torch.randn(1, 3, 640, 640)
                    with torch.no_grad():
                        output = model(dummy_input)
                    
                    # Verifikasi output memiliki struktur yang benar
                    self.assertIsNotNone(output, f"Output untuk multilayer {model_type} adalah None")
                    
                    logger.success(f"✅ Model multilayer {model_type} berhasil dibangun dan diuji")
                    
                except Exception as e:
                    self.fail(f"Gagal membangun model multilayer {model_type}: {str(e)}")

if __name__ == '__main__':
    unittest.main()
