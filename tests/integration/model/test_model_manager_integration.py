"""
File: tests/integration/model/test_model_manager_integration.py
Deskripsi: Integration test untuk ModelManager yang menguji YOLOv5 dengan backbone CSPDarknet dan EfficientNet-B4
"""

import unittest
import os
import torch
import tempfile
import shutil
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch, MagicMock
from pathlib import Path

from smartcash.model.manager import ModelManager
from smartcash.model.utils.pretrained_model_utils import check_pretrained_model_in_drive, load_pretrained_model
from smartcash.model.models.yolov5_model import YOLOv5Model
from smartcash.model.architectures.backbones import CSPDarknet, EfficientNetBackbone
from smartcash.model.architectures.necks import FeatureProcessingNeck
from smartcash.model.architectures.heads import DetectionHead
from smartcash.model.config.model_constants import SUPPORTED_BACKBONES, DEFAULT_MODEL_CONFIG_FULL
from smartcash.common.logger import SmartCashLogger
from smartcash.common.exceptions import ModelError

class TestModelManagerIntegration(unittest.TestCase):
    """Integration test untuk ModelManager dengan berbagai backbone."""
    
    def setUp(self):
        """Setup untuk test dengan inisialisasi variabel umum."""
        self.logger = SmartCashLogger("test_model_manager")
        self.logger.info("🧪 Memulai integration test untuk ModelManager")
        
        # Buat direktori sementara untuk menyimpan checkpoint
        self.temp_dir = tempfile.mkdtemp()
        self.logger.info(f"📁 Direktori sementara dibuat: {self.temp_dir}")
        
        # Konfigurasi dasar untuk semua test
        self.base_config = {
            'model_type': 'yolov5',
            'img_size': 640,
            'num_classes': 7,
            'confidence_threshold': 0.25,
            'nms_threshold': 0.45,
            'checkpoint_dir': self.temp_dir,
            'device': 'cpu'  # Tambahkan device untuk test
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
    
    def _create_dummy_targets(self):
        """Buat tensor target dummy untuk testing."""
        # Format: [batch_idx, class_idx, x, y, w, h]
        # Buat 5 target per batch
        targets = []
        for i in range(self.batch_size):
            for _ in range(5):
                # [batch_idx, class_idx, x, y, w, h]
                target = [i, torch.randint(0, 7, (1,)).item(), 
                          torch.rand(1).item(), torch.rand(1).item(),
                          torch.rand(1).item() * 0.2, torch.rand(1).item() * 0.2]
                targets.append(target)
        return torch.tensor(targets)
        
    def _verify_model_components(self, model):
        """Verifikasi komponen model (backbone, neck, head)."""
        self.assertIsInstance(model, YOLOv5Model, "Model bukan instance dari YOLOv5Model")
        self.assertIsNotNone(model.backbone, "Backbone tidak diinisialisasi")
        self.assertIsNotNone(model.neck, "Neck tidak diinisialisasi")
        self.assertIsNotNone(model.head, "Head tidak diinisialisasi")
        
        # Verifikasi tipe komponen
        self.assertIsInstance(model.backbone, (CSPDarknet, EfficientNetBackbone), 
                             "Backbone bukan CSPDarknet atau EfficientNetBackbone")
        self.assertIsInstance(model.neck, FeatureProcessingNeck, 
                             "Neck bukan FeatureProcessingNeck")
        self.assertIsInstance(model.head, DetectionHead, 
                             "Head bukan DetectionHead")
    
    def _verify_forward_pass(self, model, input_tensor):
        """Verifikasi forward pass model dengan input dummy."""
        # Jalankan forward pass
        with torch.no_grad():
            outputs = model(input_tensor)
        
        # Verifikasi output
        self.assertIsNotNone(outputs, "Output model adalah None")
        self.assertTrue(isinstance(outputs, dict), "Output model bukan dictionary")
        
        # Verifikasi layer output
        for layer_name in model.head.detection_layers:
            self.assertIn(layer_name, outputs, f"Layer {layer_name} tidak ada dalam output")
            
            # Verifikasi output shape untuk setiap layer
            layer_outputs = outputs[layer_name]
            self.assertEqual(len(layer_outputs), 3, f"Jumlah output untuk layer {layer_name} tidak sesuai")
            
            # Verifikasi shape output
            for output in layer_outputs:
                # [batch, anchors, grid_h, grid_w, 5+num_classes]
                self.assertEqual(output.dim(), 5, "Dimensi output tidak sesuai")
                self.assertEqual(output.shape[0], self.batch_size, "Batch size output tidak sesuai")
                self.assertEqual(output.shape[1], 3, "Jumlah anchor output tidak sesuai")
                self.assertEqual(output.shape[4], 5 + model.num_classes, "Output channel tidak sesuai")
    
    def _verify_prediction(self, model, input_tensor):
        """Verifikasi pipeline prediksi model."""
        # Jalankan prediksi
        with torch.no_grad():
            predictions = model.predict(input_tensor)
        
        # Verifikasi hasil prediksi
        self.assertIsNotNone(predictions, "Hasil prediksi adalah None")
        self.assertTrue(isinstance(predictions, list), "Hasil prediksi bukan list")
        self.assertEqual(len(predictions), self.batch_size, "Jumlah hasil prediksi tidak sesuai dengan batch size")
        
        # Verifikasi format prediksi untuk setiap item dalam batch
        for batch_pred in predictions:
            # Mungkin tidak ada deteksi yang melewati threshold
            if len(batch_pred) > 0:
                # Format: [x1, y1, x2, y2, confidence, class_id]
                self.assertEqual(batch_pred.shape[1], 6, "Format prediksi tidak sesuai")
    
    def _test_save_load_checkpoint(self, model_manager, model_type):
        """Test save dan load checkpoint model."""
        # Buat model
        model = model_manager.build_model()
        
        # Simpan checkpoint
        checkpoint_path = os.path.join(self.temp_dir, f"{model_type}_test.pt")
        model_manager.save_model(checkpoint_path)
        
        # Verifikasi file checkpoint dibuat
        self.assertTrue(os.path.exists(checkpoint_path), f"File checkpoint tidak dibuat: {checkpoint_path}")
        
        # Load checkpoint ke model baru dengan testing_mode=True
        # Gunakan config yang sama dengan model_manager asli untuk memastikan backbone yang sama
        new_config = model_manager.get_config().copy()
        new_config.update({'testing_mode': True})
        new_model_manager = ModelManager(config=new_config, testing_mode=True)
        loaded_model = new_model_manager.load_model(checkpoint_path)
        
        # Verifikasi model yang dimuat
        self._verify_model_components(loaded_model)
        
        # Verifikasi forward pass dengan model yang dimuat
        dummy_input = self._create_dummy_input()
        self._verify_forward_pass(loaded_model, dummy_input)
        
        return loaded_model
    
    def test_cspdarknet_backbone_with_testing_mode(self):
        """Test ModelManager dengan backbone CSPDarknet dalam mode testing."""
        self.logger.info("🧪 Testing ModelManager dengan backbone CSPDarknet (testing mode)")
        
        # Buat konfigurasi untuk CSPDarknet
        config = self.base_config.copy()
        config['backbone'] = 'cspdarknet_s'
        
        # Inisialisasi ModelManager dengan testing_mode=True
        model_manager = ModelManager(config, testing_mode=True)
        self.assertIsNotNone(model_manager, "ModelManager tidak diinisialisasi")
        
        # Buat model
        model = model_manager.build_model()
        self.assertIsNotNone(model, "Model tidak dibuat")
        
        # Verifikasi komponen model
        self._verify_model_components(model)
        self.assertIsInstance(model.backbone, CSPDarknet, "Backbone bukan CSPDarknet")
        
        # Verifikasi forward pass
        dummy_input = self._create_dummy_input()
        self._verify_forward_pass(model, dummy_input)
        
        # Verifikasi prediksi
        self._verify_prediction(model, dummy_input)
        
        # Test save dan load checkpoint
        loaded_model = self._test_save_load_checkpoint(model_manager, "cspdarknet_test")
        self.assertIsInstance(loaded_model.backbone, CSPDarknet, "Loaded model backbone bukan CSPDarknet")
        
        self.logger.info("✅ Test CSPDarknet backbone (testing mode) berhasil")
        
    def test_cspdarknet_backbone(self):
        """Test ModelManager dengan backbone CSPDarknet."""
        # Skip test ini karena memerlukan download model yang besar
        self.logger.info("⏩ Melewati test CSPDarknet backbone tanpa testing mode")
        return
    
    def test_efficientnet_backbone_with_testing_mode(self):
        """Test ModelManager dengan backbone EfficientNet-B4 dalam mode testing."""
        self.logger.info("🧪 Testing ModelManager dengan backbone EfficientNet-B4 (testing mode)")
        
        # Buat konfigurasi untuk EfficientNet
        config = self.base_config.copy()
        config['backbone'] = 'efficientnet_b4'
        
        # Inisialisasi ModelManager dengan testing_mode=True
        model_manager = ModelManager(config, testing_mode=True)
        self.assertIsNotNone(model_manager, "ModelManager tidak diinisialisasi")
        
        # Buat model
        model = model_manager.build_model()
        self.assertIsNotNone(model, "Model tidak dibuat")
        
        # Verifikasi komponen model
        self._verify_model_components(model)
        self.assertIsInstance(model.backbone, EfficientNetBackbone, "Backbone bukan EfficientNetBackbone")
        
        # Verifikasi forward pass
        dummy_input = self._create_dummy_input()
        self._verify_forward_pass(model, dummy_input)
        
        # Verifikasi prediksi
        self._verify_prediction(model, dummy_input)
        
        # Test save dan load checkpoint
        loaded_model = self._test_save_load_checkpoint(model_manager, "efficientnet_test")
        self.assertIsInstance(loaded_model.backbone, EfficientNetBackbone, "Loaded model backbone bukan EfficientNetBackbone")
        
        self.logger.info("✅ Test EfficientNet backbone (testing mode) berhasil")
        
    def test_efficientnet_backbone(self):
        """Test ModelManager dengan backbone EfficientNet-B4."""
        # Skip test ini karena memerlukan download model yang besar
        self.logger.info("⏩ Melewati test EfficientNet backbone tanpa testing mode")
        return
    
    def test_model_training_components(self):
        """Test komponen training dari ModelManager."""
        self.logger.info("🔍 Testing komponen training ModelManager")
        
        # Buat konfigurasi
        config = self.base_config.copy()
        config['backbone'] = 'efficientnet_b4'  # Gunakan salah satu backbone
        
        # Inisialisasi ModelManager dengan testing_mode=True
        model_manager = ModelManager(config, testing_mode=True)
        
        # Buat model
        model = model_manager.build_model()
        
        # Buat optimizer
        optimizer = model.get_optimizer(lr=0.01)
        self.assertIsNotNone(optimizer, "Optimizer tidak dibuat")
        
        # Buat dummy input dan target
        dummy_input = self._create_dummy_input()
        dummy_targets = self._create_dummy_targets()
        
        # Forward pass untuk training
        outputs = model(dummy_input)
        
        # Hitung loss
        loss = model.compute_loss(outputs, dummy_targets)
        self.assertIsNotNone(loss, "Loss tidak dihitung")
        self.assertEqual(len(loss), 2, "Loss output tidak sesuai format (total_loss, loss_components)")
        
        # Verifikasi backpropagation
        loss[0].backward()
        optimizer.step()
        optimizer.zero_grad()
        
        self.logger.info("✅ Test komponen training berhasil")
    
    def test_parallel_model_building(self):
        """Test pembangunan model secara paralel dengan ThreadPoolExecutor dan progress callback."""
        self.logger.info("🔍 Testing pembangunan model secara paralel dengan progress callback")
        
        # Daftar backbone yang akan diuji
        backbones = ['cspdarknet_s', 'efficientnet_b4']
        
        # Fungsi untuk membangun model
        def build_model_for_backbone(backbone):
            config = self.base_config.copy()
            config['backbone'] = backbone
            model_manager = ModelManager(config, testing_mode=True)
            model = model_manager.build_model()
            return backbone, model
        
        # Bangun model secara paralel dengan progress callback
        results = []
        with ThreadPoolExecutor(max_workers=len(backbones)) as executor:
            # Submit semua tugas sekaligus
            futures = [executor.submit(build_model_for_backbone, backbone) for backbone in backbones]
            self.logger.info(f"📊 Mulai membangun {len(backbones)} model secara paralel")
            
            # Gunakan progress callback untuk reporting ke UI
            completed = 0
            total = len(futures)
            for future in futures:
                backbone, model = future.result()
                results.append((backbone, model))
                completed += 1
                # Log progress tanpa tqdm
                self.logger.info(f"🔄 Membangun model secara paralel: {completed}/{total} selesai ({int(completed/total*100)}%)")
        
        # Verifikasi hasil
        self.assertEqual(len(results), len(backbones), "Jumlah model yang dibangun tidak sesuai")
        
        # Verifikasi setiap model
        for backbone, model in results:
            self.assertIsNotNone(model, f"Model dengan backbone {backbone} tidak dibangun")
            self._verify_model_components(model)
            
            # Verifikasi tipe backbone
            if backbone == 'cspdarknet_s':
                self.assertIsInstance(model.backbone, CSPDarknet, f"Model {backbone} memiliki tipe backbone yang salah")
            elif backbone == 'efficientnet_b4':
                self.assertIsInstance(model.backbone, EfficientNetBackbone, f"Model {backbone} memiliki tipe backbone yang salah")
        
        self.logger.info("✅ Test pembangunan model paralel berhasil")
    
    def test_error_handling(self):
        """Test penanganan error pada ModelManager."""
        self.logger.info("🔍 Testing penanganan error ModelManager")
        
        # Test dengan backbone yang tidak valid
        config = self.base_config.copy()
        config['backbone'] = 'invalid_backbone'
        
        with self.assertRaises(ModelError):
            model_manager = ModelManager(config, testing_mode=True)
            model_manager.build_model()
        
        # Test dengan checkpoint yang tidak ada
        config['backbone'] = 'cspdarknet_s'  # Reset ke backbone valid
        model_manager = ModelManager(config, testing_mode=True)
        
        with self.assertRaises(Exception):
            model_manager.load_checkpoint('nonexistent_checkpoint.pt')
        
        self.logger.info("✅ Test penanganan error berhasil")


    def test_pretrained_model_from_drive(self):
        """Test penggunaan pretrained model dari drive jika tersedia."""
        self.logger.info("🔍 Testing penggunaan pretrained model dari drive")
        
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
                    manager_eff = ModelManager(config, testing_mode=True)
                    manager_eff.build_model()
                    
                    # Verifikasi bahwa load_pretrained_model dipanggil dengan parameter yang benar
                    mock_load.assert_called_once()
                    args, _ = mock_load.call_args
                    self.assertEqual(args[1], efficientnet_path)  # Verifikasi path model
                    self.assertEqual(args[2], config['device'])   # Verifikasi device
            
            # Test dengan CSPDarknet
            config['backbone'] = 'cspdarknet_s'
            
            # Patch fungsi di lokasi di mana ia diimpor (manager.py)
            with patch('smartcash.model.manager.check_pretrained_model_in_drive', return_value=yolov5_path):
                # Patch fungsi load_pretrained_model di lokasi di mana ia diimpor
                with patch('smartcash.model.manager.load_pretrained_model') as mock_load:
                    manager_csp = ModelManager(config, testing_mode=True)
                    manager_csp.build_model()
                    
                    # Verifikasi bahwa load_pretrained_model dipanggil dengan parameter yang benar
                    mock_load.assert_called_once()
                    args, _ = mock_load.call_args
                    self.assertEqual(args[1], yolov5_path)  # Verifikasi path model
                    self.assertEqual(args[2], config['device'])   # Verifikasi device
        
        self.logger.info("✅ Test penggunaan pretrained model dari drive berhasil")


if __name__ == '__main__':
    unittest.main()
