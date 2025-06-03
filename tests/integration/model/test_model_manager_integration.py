"""
/Users/masdevid/Projects/smartcash/tests/integration/model/test_model_manager_integration.py

Test integrasi untuk ModelManager, termasuk penggunaan pretrained model dari Google Drive.
"""

import unittest
import os
import tempfile
from unittest.mock import patch, MagicMock
import torch
import torch.nn as nn
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from smartcash.model.manager import ModelManager
from smartcash.common.logger import SmartCashLogger, LogLevel
from smartcash.model.utils.pretrained_model_utils import load_pretrained_model

# Konfigurasi logger untuk pengujian
logger = SmartCashLogger('test_model_manager_integration')
logger.set_level(LogLevel.INFO)

class DummyBackbone(nn.Module):
    """Backbone dummy untuk pengujian"""
    def __init__(self, out_channels=[128, 256, 512]):
        super().__init__()
        self.out_channels = out_channels
        
    def forward(self, x):
        # Mengembalikan 3 feature maps dengan ukuran berbeda
        batch_size = x.shape[0]
        f1 = torch.zeros(batch_size, self.out_channels[0], 80, 80)
        f2 = torch.zeros(batch_size, self.out_channels[1], 40, 40)
        f3 = torch.zeros(batch_size, self.out_channels[2], 20, 20)
        return [f1, f2, f3]
        
    def get_output_channels(self):
        # Metode ini dipanggil oleh ModelManager._build_neck()
        return self.out_channels
        
    def get_info(self):
        # Metode ini mungkin dipanggil untuk debugging
        return {"type": "dummy", "channels": self.out_channels}

class TestModelManagerIntegration(unittest.TestCase):
    """Test integrasi untuk ModelManager"""
    
    @classmethod
    def setUpClass(cls):
        """Setup yang dijalankan sekali di awal pengujian untuk semua test case"""
        logger.info("🧪 Memulai integration test untuk ModelManager")
        
        # Buat konfigurasi dasar yang akan digunakan di semua test
        cls.base_config = {
            'model_type': 'efficient_optimized',
            'layer_mode': 'single',
            'detection_layers': ['banknote'],
            'backbone': 'efficientnet_b4',
            'img_size': 640,
            'pretrained': True,
            'use_attention': True,
            'use_residual': False,
            'use_ciou': False,
            'num_classes': 7,
            'device': 'cpu',
            'confidence_threshold': 0.25,
            'nms_threshold': 0.45
        }
    
    def setUp(self):
        """Setup untuk setiap test case"""
        # Buat direktori sementara untuk checkpoint
        self.temp_dir = tempfile.mkdtemp()
        logger.info(f"📁 Direktori sementara dibuat: {self.temp_dir}")
        
        # Update konfigurasi dengan direktori sementara
        self.config = self.base_config.copy()
        self.config['checkpoint_dir'] = self.temp_dir
    
    def tearDown(self):
        """Cleanup setelah setiap test case"""
        # Hapus direktori sementara
        import shutil
        shutil.rmtree(self.temp_dir)
        logger.info("🧹 Direktori sementara dihapus")
    
    def test_build_model_with_testing_mode(self):
        """Test build model dengan testing mode"""
        logger.info("🔍 Testing build model dengan testing mode")
        
        # Buat model manager dengan testing_mode=True
        model_manager = ModelManager(self.config, testing_mode=True)
        
        # Patch _build_backbone untuk mengembalikan dummy backbone
        with patch.object(ModelManager, '_build_backbone', autospec=True) as mock_build_backbone:
            # Setup mock untuk _build_backbone
            def side_effect(self_instance):
                dummy_backbone = DummyBackbone()
                self_instance.backbone = dummy_backbone
                return dummy_backbone
            
            mock_build_backbone.side_effect = side_effect
            
            # Build model
            model = model_manager.build_model()
        
        # Verifikasi model telah dibangun dengan benar
        self.assertIsNotNone(model, "Model tidak berhasil dibangun")
        
        # Verifikasi konfigurasi model sesuai dengan yang diharapkan
        self.assertEqual(model_manager.model_type, self.config['model_type'], 
                        f"Model type tidak sesuai: {model_manager.model_type} != {self.config['model_type']}")
        
        # Verifikasi backbone sesuai dengan konfigurasi
        self.assertEqual(model_manager.config['backbone'], self.config['backbone'], 
                        f"Backbone tidak sesuai: {model_manager.config['backbone']} != {self.config['backbone']}")
        
        logger.info("✅ Test build model dengan testing mode berhasil")
    
    def test_build_model_multilayer(self):
        """Test build model dengan mode multilayer"""
        logger.info("🔍 Testing build model dengan mode multilayer")
        
        # Update konfigurasi untuk multilayer
        config = self.config.copy()
        config['layer_mode'] = 'multilayer'
        config['detection_layers'] = ['banknote', 'nominal', 'security']
        
        # Buat model manager dengan mode multilayer
        model_manager = ModelManager(config, testing_mode=True)
        
        # Patch _build_backbone untuk mengembalikan dummy backbone
        with patch.object(ModelManager, '_build_backbone', autospec=True) as mock_build_backbone:
            # Setup mock untuk _build_backbone
            def side_effect(self_instance):
                dummy_backbone = DummyBackbone()
                self_instance.backbone = dummy_backbone
                return dummy_backbone
            
            mock_build_backbone.side_effect = side_effect
            
            # Build model
            model = model_manager.build_model()
        
        # Verifikasi model telah dibangun dengan benar
        self.assertIsNotNone(model, "Model multilayer tidak berhasil dibangun")
        
        # Verifikasi konfigurasi model sesuai dengan yang diharapkan
        self.assertEqual(model_manager.config['layer_mode'], 'multilayer', 
                        f"Layer mode tidak sesuai: {model_manager.config['layer_mode']} != multilayer")
        
        # Verifikasi detection layers sesuai dengan konfigurasi
        self.assertEqual(model_manager.config['detection_layers'], ['banknote', 'nominal', 'security'], 
                        f"Detection layers tidak sesuai")
        
        logger.info("✅ Test build model dengan mode multilayer berhasil")
    
    def test_parallel_model_building(self):
        """Test pembangunan model secara paralel"""
        logger.info("🔍 Testing pembangunan model secara paralel")
        
        # Buat dua konfigurasi berbeda
        config1 = self.config.copy()
        config1['backbone'] = 'cspdarknet_s'
        
        config2 = self.config.copy()
        config2['backbone'] = 'efficientnet_b4'
        
        # Patch _build_backbone di luar thread
        original_build_backbone = ModelManager._build_backbone
        
        def mock_build_backbone(self_instance):
            dummy_backbone = DummyBackbone()
            self_instance.backbone = dummy_backbone
            return dummy_backbone
        
        # Ganti metode asli dengan mock
        ModelManager._build_backbone = mock_build_backbone
        
        try:
            # Fungsi untuk membangun model tanpa patching di dalam thread
            def build_model(config, idx, total):
                model_manager = ModelManager(config, testing_mode=True)
                model = model_manager.build_model()
                logger.info(f"🔄 Membangun model secara paralel: {idx}/{total} selesai ({idx/total*100:.1f}%)")
                return model
            
            # Bangun model secara paralel
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [
                    executor.submit(build_model, config1, 1, 2),
                    executor.submit(build_model, config2, 2, 2)
                ]
                
                # Ambil hasil
                models = [future.result() for future in futures]
        finally:
            # Kembalikan metode asli
            ModelManager._build_backbone = original_build_backbone
        
        # Verifikasi kedua model telah dibangun dengan benar
        self.assertEqual(len(models), 2, "Tidak semua model berhasil dibangun")
        self.assertIsNotNone(models[0], "Model pertama tidak berhasil dibangun")
        self.assertIsNotNone(models[1], "Model kedua tidak berhasil dibangun")
        
        logger.info("✅ Test pembangunan model paralel berhasil")
    
    def test_pretrained_model_from_drive(self):
        """Test penggunaan pretrained model dari drive jika tersedia."""
        logger.info("🔍 Testing penggunaan pretrained model dari drive")
        
        # Setup mock directory dan file untuk mensimulasikan drive
        with tempfile.TemporaryDirectory() as temp_dir:
            # Buat struktur direktori drive
            drive_models_dir = os.path.join(temp_dir, "drive", "MyDrive", "SmartCash", "models")
            os.makedirs(drive_models_dir, exist_ok=True)
            
            # Buat mock file untuk EfficientNet
            efficientnet_path = os.path.join(drive_models_dir, "efficientnet_b4_huggingface.bin")
            with open(efficientnet_path, 'wb') as f:
                f.write(b'mock_efficientnet_model_data')
                
            # Buat mock file untuk YOLOv5
            yolov5_path = os.path.join(drive_models_dir, "yolov5s.pt")
            with open(yolov5_path, 'wb') as f:
                f.write(b'mock_yolov5_model_data')
            
            # Test dengan EfficientNet
            config = self.base_config.copy()
            config['backbone'] = 'efficientnet_b4'
            
            # Patch fungsi di lokasi di mana ia diimpor (manager.py)
            with patch('smartcash.model.manager.check_pretrained_model_in_drive', return_value=efficientnet_path):
                # Patch fungsi load_pretrained_model di lokasi di mana ia diimpor
                with patch('smartcash.model.manager.load_pretrained_model') as mock_load:
                    # Patch _build_backbone untuk mengembalikan dummy backbone
                    with patch.object(ModelManager, '_build_backbone', autospec=True) as mock_build_backbone:
                        # Setup mock untuk _build_backbone
                        def side_effect_eff(self_instance):
                            dummy_backbone = DummyBackbone()
                            self_instance.backbone = dummy_backbone
                            return dummy_backbone
                        
                        mock_build_backbone.side_effect = side_effect_eff
                        
                        # Buat model manager
                        manager_eff = ModelManager(config, testing_mode=False)
                        manager_eff.build_model()
                        
                        # Verifikasi bahwa load_pretrained_model dipanggil dengan parameter yang benar
                        mock_load.assert_called_once()
                        args, _ = mock_load.call_args
                        self.assertEqual(args[1], efficientnet_path)  # Verifikasi path model
                        self.assertEqual(args[2], config['device'])   # Verifikasi device
            
            # Test dengan CSPDarknet
            config = self.base_config.copy()
            config['backbone'] = 'cspdarknet_s'
            
            # Patch fungsi di lokasi di mana ia diimpor (manager.py)
            with patch('smartcash.model.manager.check_pretrained_model_in_drive', return_value=yolov5_path):
                # Patch fungsi load_pretrained_model di lokasi di mana ia diimpor
                with patch('smartcash.model.manager.load_pretrained_model') as mock_load:
                    # Patch _build_backbone untuk mengembalikan dummy backbone
                    with patch.object(ModelManager, '_build_backbone', autospec=True) as mock_build_backbone:
                        # Setup mock untuk _build_backbone
                        def side_effect_csp(self_instance):
                            dummy_backbone = DummyBackbone()
                            self_instance.backbone = dummy_backbone
                            return dummy_backbone
                        
                        mock_build_backbone.side_effect = side_effect_csp
                        
                        # Buat model manager
                        manager_csp = ModelManager(config, testing_mode=False)
                        manager_csp.build_model()
                        
                        # Verifikasi bahwa load_pretrained_model dipanggil dengan parameter yang benar
                        mock_load.assert_called_once()
                        args, _ = mock_load.call_args
                        self.assertEqual(args[1], yolov5_path)  # Verifikasi path model
                        self.assertEqual(args[2], config['device'])   # Verifikasi device
        
        logger.info("✅ Test penggunaan pretrained model dari drive berhasil")

if __name__ == '__main__':
    unittest.main()
