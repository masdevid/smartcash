"""
File: tests/integration/model/test_detection_layers.py
Deskripsi: Integration test untuk fitur detection layers dan layer mode pada ModelManager
"""

import unittest
import torch
import tempfile
import shutil
from unittest.mock import patch

from smartcash.model.manager import ModelManager
from smartcash.model.models.yolov5_model import YOLOv5Model
from smartcash.model.architectures.heads import DetectionHead
from smartcash.model.config.model_constants import DETECTION_LAYERS, LAYER_CONFIG
from smartcash.model.utils.layer_validator import validate_layer_params, get_num_classes_for_layers
from smartcash.common.logger import SmartCashLogger

class TestDetectionLayersIntegration(unittest.TestCase):
    """Integration test untuk fitur detection layers dan layer mode pada ModelManager."""
    
    def setUp(self):
        """Setup untuk test dengan inisialisasi variabel umum."""
        self.logger = SmartCashLogger("test_detection_layers")
        self.logger.info("🧪 Memulai integration test untuk fitur detection layers")
        
        # Buat direktori sementara untuk menyimpan checkpoint
        self.temp_dir = tempfile.mkdtemp()
        self.logger.info(f"📁 Direktori sementara dibuat: {self.temp_dir}")
        
        # Konfigurasi dasar untuk semua test
        self.base_config = {
            'model_type': 'yolov5',
            'backbone': 'efficientnet_b4',  # Gunakan EfficientNet sebagai default
            'img_size': 640,
            'confidence_threshold': 0.25,
            'nms_threshold': 0.45,
            'checkpoint_dir': self.temp_dir,
            'device': 'cpu'  # Gunakan CPU untuk test
        }
        
        # Ukuran batch dan input dummy untuk testing
        self.batch_size = 2
        self.input_shape = (self.batch_size, 3, 640, 640)
        
    def tearDown(self):
        """Cleanup setelah test selesai."""
        # Hapus direktori sementara
        shutil.rmtree(self.temp_dir)
        self.logger.info("🧹 Direktori sementara dihapus")
        
        # Hapus cache CUDA jika tersedia
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    def _create_dummy_input(self):
        """Buat tensor input dummy untuk testing."""
        return torch.rand(self.input_shape)
    
    def test_single_layer_mode_default(self):
        """Test ModelManager dengan mode single layer default."""
        self.logger.info("🔍 Testing ModelManager dengan mode single layer default")
        
        # Buat ModelManager tanpa parameter layer_mode dan detection_layers
        model_manager = ModelManager(self.base_config.copy(), testing_mode=True)
        model_manager.build_model()
        
        # Verifikasi bahwa model dibangun dengan benar
        self.assertIsInstance(model_manager.model, YOLOv5Model)
        
        # Verifikasi parameter layer_mode dan detection_layers
        self.assertEqual(model_manager.get_layer_mode(), 'single')
        self.assertEqual(model_manager.get_detection_layers(), ['banknote'])
        self.assertFalse(model_manager.is_multilayer())
        
        # Verifikasi bahwa DetectionHead dikonfigurasi dengan benar
        self.assertIsInstance(model_manager.model.head, DetectionHead)
        head_config = model_manager.model.head.get_config()
        self.assertEqual(head_config['layer_mode'], 'single')
        self.assertEqual(head_config['layers'], ['banknote'])
        
        # Verifikasi num_classes
        self.assertEqual(model_manager.model.num_classes, LAYER_CONFIG['banknote']['num_classes'])
        
        # Test forward pass
        input_tensor = self._create_dummy_input()
        output = model_manager.model(input_tensor)
        
        # Verifikasi output hanya memiliki satu layer
        self.assertIn('banknote', output)
        self.assertEqual(len(output), 1)
        
        self.logger.info("✅ Test mode single layer default berhasil")
    
    def test_single_layer_mode_explicit(self):
        """Test ModelManager dengan mode single layer yang diberikan secara eksplisit."""
        self.logger.info("🔍 Testing ModelManager dengan mode single layer eksplisit")
        
        # Buat ModelManager dengan parameter layer_mode dan detection_layers
        config = self.base_config.copy()
        config['layer_mode'] = 'single'
        config['detection_layers'] = ['nominal']
        
        model_manager = ModelManager(config, testing_mode=True)
        model_manager.build_model()
        
        # Verifikasi parameter layer_mode dan detection_layers
        self.assertEqual(model_manager.get_layer_mode(), 'single')
        self.assertEqual(model_manager.get_detection_layers(), ['nominal'])
        self.assertFalse(model_manager.is_multilayer())
        
        # Verifikasi bahwa DetectionHead dikonfigurasi dengan benar
        head_config = model_manager.model.head.get_config()
        self.assertEqual(head_config['layer_mode'], 'single')
        self.assertEqual(head_config['layers'], ['nominal'])
        
        # Verifikasi num_classes
        self.assertEqual(model_manager.model.num_classes, LAYER_CONFIG['nominal']['num_classes'])
        
        # Test forward pass
        input_tensor = self._create_dummy_input()
        output = model_manager.model(input_tensor)
        
        # Verifikasi output hanya memiliki satu layer
        self.assertIn('nominal', output)
        self.assertEqual(len(output), 1)
        
        self.logger.info("✅ Test mode single layer eksplisit berhasil")
    
    def test_multilayer_mode(self):
        """Test ModelManager dengan mode multilayer."""
        self.logger.info("🔍 Testing ModelManager dengan mode multilayer")
        
        # Buat ModelManager dengan parameter layer_mode multilayer
        config = self.base_config.copy()
        config['layer_mode'] = 'multilayer'
        config['detection_layers'] = ['banknote', 'nominal', 'security']
        
        self.logger.info(f"📋 Config untuk test_multilayer_mode: {config}")
        
        model_manager = ModelManager(config, testing_mode=True)
        model_manager.build_model()
        
        # Debug logging
        self.logger.info(f"🔍 Setelah build_model, layer_mode: {model_manager.get_layer_mode()}, detection_layers: {model_manager.get_detection_layers()}")
        if hasattr(model_manager, 'head') and model_manager.head is not None:
            self.logger.info(f"🔍 Head layer_mode: {model_manager.head.layer_mode}")
        if hasattr(model_manager, 'model') and model_manager.model is not None:
            self.logger.info(f"🔍 Model layer_mode: {model_manager.model.layer_mode}")
        
        # Verifikasi parameter layer_mode dan detection_layers
        self.assertEqual(model_manager.get_layer_mode(), 'multilayer')
        self.assertEqual(model_manager.get_detection_layers(), ['banknote', 'nominal', 'security'])
        self.assertTrue(model_manager.is_multilayer())
        
        # Verifikasi bahwa DetectionHead dikonfigurasi dengan benar
        head_config = model_manager.model.head.get_config()
        self.assertEqual(head_config['layer_mode'], 'multilayer')
        self.assertEqual(head_config['layers'], ['banknote', 'nominal', 'security'])
        
        # Verifikasi total_classes
        expected_total_classes = sum(LAYER_CONFIG[layer]['num_classes'] for layer in ['banknote', 'nominal', 'security'])
        self.assertEqual(head_config['total_classes'], expected_total_classes)
        
        # Test forward pass
        input_tensor = self._create_dummy_input()
        output = model_manager.model(input_tensor)
        
        # Verifikasi output memiliki semua layer
        for layer in ['banknote', 'nominal', 'security']:
            self.assertIn(layer, output)
        self.assertEqual(len(output), 3)
        
        self.logger.info("✅ Test mode multilayer berhasil")
    
    def test_auto_fix_layer_mode(self):
        """Test fitur auto-fix layer_mode pada ModelManager."""
        self.logger.info("🔍 Testing fitur auto-fix layer_mode")
        
        # Test case 1: layer_mode = 'multilayer' dengan hanya satu detection_layer
        # Seharusnya diubah menjadi 'single'
        config = self.base_config.copy()
        config['layer_mode'] = 'multilayer'
        config['detection_layers'] = ['banknote']
        
        model_manager = ModelManager(config, testing_mode=True)
        model_manager.build_model()
        
        # Verifikasi bahwa layer_mode diubah menjadi 'single'
        self.assertEqual(model_manager.get_layer_mode(), 'single')
        self.assertEqual(model_manager.get_detection_layers(), ['banknote'])
        self.assertFalse(model_manager.is_multilayer())
        
        # Test case 2: layer_mode = 'single' dengan multiple detection_layers
        # Seharusnya tetap 'single' tapi hanya menggunakan layer pertama
        config = self.base_config.copy()
        config['layer_mode'] = 'single'
        config['detection_layers'] = ['banknote', 'nominal', 'security']
        
        model_manager = ModelManager(config, testing_mode=True)
        model_manager.build_model()
        
        # Verifikasi bahwa layer_mode tetap 'single'
        self.assertEqual(model_manager.get_layer_mode(), 'single')
        # Verifikasi bahwa semua detection_layers disimpan tapi hanya layer pertama yang digunakan
        self.assertEqual(model_manager.get_detection_layers(), ['banknote', 'nominal', 'security'])
        self.assertFalse(model_manager.is_multilayer())
        
        # Test forward pass untuk memastikan hanya layer pertama yang digunakan
        input_tensor = self._create_dummy_input()
        output = model_manager.model(input_tensor)
        
        # Verifikasi output hanya memiliki layer pertama
        self.assertIn('banknote', output)
        self.assertEqual(len(output), 1)
        
        self.logger.info("✅ Test fitur auto-fix layer_mode berhasil")
    
    def test_invalid_detection_layers(self):
        """Test penanganan detection_layers yang tidak valid."""
        self.logger.info("🔍 Testing penanganan detection_layers yang tidak valid")
        
        # Test dengan detection_layers yang tidak valid
        config = self.base_config.copy()
        config['detection_layers'] = ['invalid_layer']
        
        model_manager = ModelManager(config, testing_mode=True)
        model_manager.build_model()
        
        # Verifikasi bahwa detection_layers diubah menjadi default
        self.assertEqual(model_manager.get_detection_layers(), ['banknote'])
        
        self.logger.info("✅ Test penanganan detection_layers yang tidak valid berhasil")
    
    def test_constructor_with_params(self):
        """Test konstruktor ModelManager dengan parameter layer_mode dan detection_layers."""
        self.logger.info("🔍 Testing konstruktor ModelManager dengan parameter layer_mode dan detection_layers")
        
        # Test konstruktor dengan parameter layer_mode dan detection_layers
        config = {
            'model_type': 'efficient_optimized',
            'layer_mode': 'multilayer',
            'detection_layers': ['banknote', 'nominal'],
            'backbone': 'efficientnet_b4',
            'img_size': 640,
            'device': 'cpu'
        }
        
        self.logger.info(f"📋 Config untuk test_constructor_with_params: {config}")
        
        model_manager = ModelManager(config, testing_mode=True)
        model_manager.build_model()
        
        # Debug logging
        self.logger.info(f"🔍 Setelah build_model, layer_mode: {model_manager.get_layer_mode()}, detection_layers: {model_manager.get_detection_layers()}")
        if hasattr(model_manager, 'head') and model_manager.head is not None:
            self.logger.info(f"🔍 Head layer_mode: {model_manager.head.layer_mode}")
        if hasattr(model_manager, 'model') and model_manager.model is not None:
            self.logger.info(f"🔍 Model layer_mode: {model_manager.model.layer_mode}")
        
        # Verifikasi parameter layer_mode dan detection_layers
        self.assertEqual(model_manager.get_layer_mode(), 'multilayer')
        self.assertEqual(model_manager.get_detection_layers(), ['banknote', 'nominal'])
        self.assertTrue(model_manager.is_multilayer())
        
        # Verifikasi bahwa DetectionHead dikonfigurasi dengan benar
        head_config = model_manager.model.head.get_config()
        self.assertEqual(head_config['layer_mode'], 'multilayer')
        self.assertEqual(head_config['layers'], ['banknote', 'nominal'])
        
        # Verifikasi total_classes
        expected_total_classes = sum(LAYER_CONFIG[layer]['num_classes'] for layer in ['banknote', 'nominal'])
        self.assertEqual(head_config['total_classes'], expected_total_classes)
        
        self.logger.info("✅ Test factory method berhasil")


if __name__ == '__main__':
    unittest.main()
