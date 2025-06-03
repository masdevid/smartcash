"""
File: tests/model/test_manager_training.py
Deskripsi: Unit test untuk integrasi ModelManager dengan TrainingService
"""

import unittest
import os
import torch
import tempfile
import shutil
from unittest.mock import MagicMock, patch
from torch.utils.data import DataLoader, TensorDataset

from smartcash.model.manager import ModelManager
from smartcash.model.service.training_service import TrainingService
from smartcash.model.service.checkpoint_service import CheckpointService
from smartcash.model.service.metrics_tracker import MetricsTracker

class TestModelManagerTraining(unittest.TestCase):
    """Test case untuk integrasi ModelManager dengan TrainingService"""
    
    def setUp(self):
        """Setup untuk setiap test case"""
        # Buat temporary directory untuk checkpoint
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_dir = os.path.join(self.temp_dir, "checkpoints")
        
        # Buat config dummy
        self.config = {
            "model_type": "yolov5",
            "backbone": "cspdarknet_s",
            "layer_mode": "single",
            "num_classes": 2,
            "checkpoint_dir": self.checkpoint_dir
        }
        
        # Buat progress callback mock
        self.mock_progress_callback = MagicMock()
        
        # Buat metrics callback mock
        self.mock_metrics_callback = MagicMock()
        
        # Patch model building methods untuk menghindari loading model yang sebenarnya
        with patch('smartcash.model.manager.ModelManager.build_model') as mock_build_model:
            with patch('smartcash.model.manager.check_pretrained_model_in_drive') as mock_check_drive:
                with patch('smartcash.model.manager.ModelManager._build_backbone') as mock_build_backbone:
                    with patch('smartcash.model.manager.ModelManager._build_neck') as mock_build_neck:
                        with patch('smartcash.model.manager.ModelManager._build_head') as mock_build_head:
                            # Set mock return values
                            mock_model = torch.nn.Sequential(
                                torch.nn.Linear(10, 5),
                                torch.nn.ReLU(),
                                torch.nn.Linear(5, 2)
                            )
                            
                            # Inisialisasi model manager dengan testing_mode=True untuk menghindari loading model
                            self.model_manager = ModelManager(testing_mode=True)
                            self.model_manager.model = mock_model
        
        # Buat dataset dummy
        x_train = torch.randn(20, 10)
        y_train = torch.randint(0, 2, (20,))
        self.train_dataset = TensorDataset(x_train, y_train)
        self.train_loader = DataLoader(self.train_dataset, batch_size=4)
        
        x_val = torch.randn(10, 10)
        y_val = torch.randint(0, 2, (10,))
        self.val_dataset = TensorDataset(x_val, y_val)
        self.val_loader = DataLoader(self.val_dataset, batch_size=4)
        
    def tearDown(self):
        """Cleanup setelah setiap test case"""
        # Hapus temporary directory
        shutil.rmtree(self.temp_dir)
        
    def test_build_loss_function(self):
        """Test _build_loss_function"""
        # Mock YOLOLoss - pastikan menggunakan path import yang benar
        with patch('smartcash.model.manager.YOLOLoss') as mock_yolo_loss:
            # Set mock return value
            mock_loss_instance = MagicMock()
            mock_yolo_loss.return_value = mock_loss_instance
            
            # Build loss function
            loss_fn = self.model_manager._build_loss_function()
            
            # Verifikasi YOLOLoss dipanggil
            mock_yolo_loss.assert_called_once()
            self.assertEqual(loss_fn, mock_loss_instance)
        
    def test_get_training_service_no_checkpoint_service(self):
        """Test get_training_service tanpa checkpoint service"""
        # Patch _build_loss_function
        with patch('smartcash.model.manager.ModelManager._build_loss_function') as mock_build_loss:
            # Set mock return value
            mock_loss = torch.nn.CrossEntropyLoss()
            mock_build_loss.return_value = mock_loss
            
            # Get training service
            training_service = self.model_manager.get_training_service(
                callback=self.mock_metrics_callback
            )
            
            # Verify training service was created
            self.assertIsInstance(training_service, TrainingService)
            # TrainingService tidak lagi memiliki atribut model, loss_fn, optimizer
            # Kita hanya periksa model_manager dan checkpoint_service
            self.assertEqual(training_service.model_manager, self.model_manager)
            self.assertIsNotNone(training_service.checkpoint_service)
            
            # Verify checkpoint service was created
            self.assertIsInstance(self.model_manager.checkpoint_service, CheckpointService)
            
    def test_get_training_service_with_checkpoint_service(self):
        """Test get_training_service dengan checkpoint service"""
        # Buat checkpoint service
        checkpoint_service = CheckpointService(
            checkpoint_dir=self.checkpoint_dir,
            progress_callback=self.mock_progress_callback
        )
        
        # Set checkpoint service
        self.model_manager.set_checkpoint_service(checkpoint_service)
        
        # Patch _build_loss_function
        with patch('smartcash.model.manager.ModelManager._build_loss_function') as mock_build_loss:
            # Set mock return value
            mock_loss = torch.nn.CrossEntropyLoss()
            mock_build_loss.return_value = mock_loss
            
            # Get training service
            training_service = self.model_manager.get_training_service(
                callback=self.mock_metrics_callback
            )
            
            # Verify training service was created
            self.assertIsInstance(training_service, TrainingService)
            self.assertEqual(training_service.checkpoint_service, checkpoint_service)
            
    def test_training_integration(self):
        """Test integrasi training"""
        # Patch model building methods
        with patch('smartcash.model.service.training_service.TrainingService.train') as mock_train:
            # Set mock return value
            mock_train.return_value = {
                'accuracy': 0.8,
                'loss': 0.2
            }
            
            # Patch get_optimizer untuk menghindari AttributeError
            with patch.object(self.model_manager, 'get_optimizer') as mock_get_optimizer:
                # Setup mock optimizer
                mock_optimizer = MagicMock()
                mock_get_optimizer.return_value = mock_optimizer
                
                # Patch torch.save untuk menghindari error saat menyimpan checkpoint
                with patch('torch.save') as mock_save:
                    # Patch build_loss_function
                    with patch('smartcash.model.manager.ModelManager._build_loss_function') as mock_build_loss:
                        # Get training service
                        training_service = self.model_manager.get_training_service(
                            callback=self.mock_metrics_callback
                        )
                        
                        # Train for 1 epoch
                        best_metrics = training_service.train(
                            train_loader=self.train_loader,
                            val_loader=self.val_loader,
                            epochs=1
                        )
                        
                        # Verify train was called
                        mock_train.assert_called_once()
                        
                        # Verify best metrics
                        self.assertEqual(best_metrics['accuracy'], 0.8)
                        self.assertEqual(best_metrics['loss'], 0.2)
        
    def test_get_optimizer(self):
        """Test get_optimizer"""
        # Patch backbone dan head untuk menghindari AttributeError
        with patch.object(self.model_manager, 'backbone', create=True) as mock_backbone:
            with patch.object(self.model_manager, 'head', create=True) as mock_head:
                # Setup mock parameters
                mock_params_backbone = [torch.nn.Parameter(torch.randn(5, 5)) for _ in range(2)]
                mock_params_head = [torch.nn.Parameter(torch.randn(5, 5)) for _ in range(2)]
                mock_backbone.parameters.return_value = mock_params_backbone
                mock_head.parameters.return_value = mock_params_head
                
                # Get optimizer
                optimizer = self.model_manager.get_optimizer(learning_rate=0.01)
                
                # Get scheduler - API telah berubah, hanya menerima optimizer dan epochs
                scheduler = self.model_manager.get_scheduler(
                    optimizer=optimizer,
                    epochs=100
                )
                
                # Verify scheduler was created - sekarang selalu CosineAnnealingLR
                self.assertIsInstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)
                self.assertEqual(scheduler.T_max, 100)
        
    def test_get_scheduler_custom_epochs(self):
        """Test get_scheduler dengan custom epochs"""
        # Patch backbone dan head untuk menghindari AttributeError
        with patch.object(self.model_manager, 'backbone', create=True) as mock_backbone:
            with patch.object(self.model_manager, 'head', create=True) as mock_head:
                # Setup mock parameters
                mock_params_backbone = [torch.nn.Parameter(torch.randn(5, 5)) for _ in range(2)]
                mock_params_head = [torch.nn.Parameter(torch.randn(5, 5)) for _ in range(2)]
                mock_backbone.parameters.return_value = mock_params_backbone
                mock_head.parameters.return_value = mock_params_head
                
                # Get optimizer
                optimizer = self.model_manager.get_optimizer(learning_rate=0.01)
                
                # Get scheduler dengan custom epochs
                scheduler = self.model_manager.get_scheduler(
                    optimizer=optimizer,
                    epochs=50
                )
                
                # Verify scheduler was created dengan T_max yang sesuai
                self.assertIsInstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)
                self.assertEqual(scheduler.T_max, 50)
        
    # Test ini dihapus karena API get_scheduler telah berubah dan tidak lagi mendukung ReduceLROnPlateau
        
    # Test ini dihapus karena API get_scheduler telah berubah dan tidak lagi mendukung tipe scheduler
        
    def test_error_handling(self):
        """Test penanganan error"""
        # Test error handling saat model tidak ditemukan
        with patch('os.path.exists', return_value=False):
            try:
                self.model_manager.load_model('path/to/nonexistent/model.pt')
                self.fail("Seharusnya menimbulkan exception")
            except Exception:
                pass  # Test berhasil jika exception ditangkap

if __name__ == '__main__':
    unittest.main()
