"""
/Users/masdevid/Projects/smartcash/tests/integration/model/test_build_optimized_models_fixed.py

Pengujian untuk memverifikasi bahwa semua model optimasi dapat dibangun dengan benar
menggunakan pretrained model dari Google Drive dengan mocking.
"""

import unittest
from unittest.mock import patch, MagicMock
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm

from smartcash.model.manager import ModelManager
from smartcash.model.config.model_constants import OPTIMIZED_MODELS
from smartcash.common.logger import SmartCashLogger, LogLevel

# Konfigurasi logger untuk pengujian
logger = SmartCashLogger('test_optimized_models')
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

class TestBuildOptimizedModels(unittest.TestCase):
    """
    Test case untuk memverifikasi bahwa semua model optimasi dapat dibangun dengan benar
    menggunakan pretrained model dari Google Drive.
    """
    
    # Variabel kelas untuk menyimpan model yang telah diunduh
    pretrained_models = {}
    
    @classmethod
    def setUpClass(cls):
        """Setup yang dijalankan sekali di awal pengujian untuk semua test case"""
        logger.info("🔄 Menyiapkan model pretrained untuk semua pengujian...")
        
        # Simulasikan direktori Google Drive untuk model pretrained
        cls.drive_models_dir = Path("/drive/MyDrive/SmartCash/models")
        
        # Siapkan dummy state dict untuk semua backbone yang mungkin digunakan
        for model_type, config in OPTIMIZED_MODELS.items():
            backbone = config['backbone']
            if backbone not in cls.pretrained_models:
                logger.info(f"🔄 Menyiapkan pretrained model untuk backbone: {backbone}")
                cls.pretrained_models[backbone] = {}
                
        logger.success("✅ Semua model pretrained telah disiapkan!")
    
    def setUp(self):
        """Setup untuk setiap test case"""
        # Gunakan drive_models_dir dari kelas
        self.drive_models_dir = self.__class__.drive_models_dir
        
    @patch('smartcash.model.utils.pretrained_model_utils.check_pretrained_model_in_drive')
    @patch('smartcash.model.manager.ModelManager._build_backbone')
    @patch('torch.load')
    def test_build_all_optimized_models(self, mock_torch_load, mock_build_backbone, mock_check_drive):
        """
        Menguji build model untuk semua model optimasi di OPTIMIZED_MODELS
        tanpa menggunakan testing mode, dengan mocking untuk pretrained model.
        """
        # Setup mock untuk check_pretrained_model_in_drive
        mock_check_drive.side_effect = lambda backbone: str(self.drive_models_dir / f"{backbone}.pt")
        
        # Setup mock untuk torch.load
        mock_torch_load.side_effect = lambda path, **kwargs: self.__class__.pretrained_models.get(Path(path).stem, {})
        
        # Setup mock untuk _build_backbone dengan lambda function
        # Lambda ini akan mengatur self.backbone pada instance ModelManager dan mengembalikan DummyBackbone
        mock_build_backbone.side_effect = lambda self_instance: self_instance.__setattr__('backbone', DummyBackbone()) or DummyBackbone()
        
        logger.info(f"🧪 Menguji build model untuk semua model optimasi")
        
        # Iterasi melalui semua model optimasi dengan progress bar
        for model_type, config in tqdm(OPTIMIZED_MODELS.items(), desc="Menguji model optimasi"):
            with self.subTest(model_type=model_type):
                logger.info(f"🔍 Menguji model optimasi: {model_type}")
                if 'description' in config:
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
                    
                    # Verifikasi konfigurasi model sesuai dengan yang diharapkan
                    self.assertEqual(model_manager.model_type, model_type, 
                                    f"Model type tidak sesuai: {model_manager.model_type} != {model_type}")
                    
                    # Verifikasi backbone sesuai dengan konfigurasi
                    expected_backbone = config['backbone']
                    actual_backbone = model_manager.config['backbone']
                    self.assertEqual(actual_backbone, expected_backbone, 
                                    f"Backbone tidak sesuai: {actual_backbone} != {expected_backbone}")
                    
                    # Verifikasi parameter optimasi sesuai dengan konfigurasi
                    self.assertEqual(model_manager.config.get('use_attention', False), config.get('use_attention', False),
                                    "Parameter use_attention tidak sesuai")
                    self.assertEqual(model_manager.config.get('use_residual', False), config.get('use_residual', False),
                                    "Parameter use_residual tidak sesuai")
                    self.assertEqual(model_manager.config.get('use_ciou', False), config.get('use_ciou', False),
                                    "Parameter use_ciou tidak sesuai")
                    
                    logger.success(f"✅ Model {model_type} berhasil dibangun dan diuji")
                    
                except Exception as e:
                    self.fail(f"Gagal membangun model {model_type}: {str(e)}")
    
    @patch('smartcash.model.utils.pretrained_model_utils.check_pretrained_model_in_drive')
    @patch('smartcash.model.manager.ModelManager._build_backbone')
    @patch('torch.load')
    def test_build_multilayer_models(self, mock_torch_load, mock_build_backbone, mock_check_drive):
        """
        Menguji build model untuk semua model optimasi di OPTIMIZED_MODELS
        dengan mode multilayer, dengan mocking untuk pretrained model.
        """
        # Setup mock untuk check_pretrained_model_in_drive
        mock_check_drive.side_effect = lambda backbone: str(self.drive_models_dir / f"{backbone}.pt")
        
        # Setup mock untuk torch.load - gunakan model yang sudah disiapkan di setUpClass
        mock_torch_load.side_effect = lambda path, **kwargs: self.__class__.pretrained_models.get(Path(path).stem, {})
        
        # Setup mock untuk _build_backbone dengan lambda function
        # Lambda ini akan mengatur self.backbone pada instance ModelManager dan mengembalikan DummyBackbone
        mock_build_backbone.side_effect = lambda self_instance: self_instance.__setattr__('backbone', DummyBackbone()) or DummyBackbone()
        
        logger.info(f"🧪 Menguji build model multilayer untuk semua model optimasi")
        
        # Iterasi melalui semua model optimasi dengan progress bar
        for model_type, config in tqdm(OPTIMIZED_MODELS.items(), desc="Menguji model multilayer"):
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
                    
                    # Verifikasi konfigurasi model sesuai dengan yang diharapkan
                    self.assertEqual(model_manager.model_type, model_type, 
                                    f"Model type tidak sesuai: {model_manager.model_type} != {model_type}")
                    
                    # Verifikasi backbone sesuai dengan konfigurasi
                    expected_backbone = config['backbone']
                    actual_backbone = model_manager.config['backbone']
                    self.assertEqual(actual_backbone, expected_backbone, 
                                    f"Backbone tidak sesuai: {actual_backbone} != {expected_backbone}")
                    
                    # Verifikasi parameter optimasi sesuai dengan konfigurasi
                    self.assertEqual(model_manager.config.get('use_attention', False), config.get('use_attention', False),
                                    "Parameter use_attention tidak sesuai")
                    self.assertEqual(model_manager.config.get('use_residual', False), config.get('use_residual', False),
                                    "Parameter use_residual tidak sesuai")
                    self.assertEqual(model_manager.config.get('use_ciou', False), config.get('use_ciou', False),
                                    "Parameter use_ciou tidak sesuai")
                    
                    logger.success(f"✅ Model multilayer {model_type} berhasil dibangun dan diuji")
                    
                except Exception as e:
                    self.fail(f"Gagal membangun model multilayer {model_type}: {str(e)}")

if __name__ == '__main__':
    unittest.main()
