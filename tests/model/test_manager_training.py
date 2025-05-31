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
        
        # Patch model building methods
        with patch('smartcash.model.manager.ModelManager._build_model') as mock_build_model:
            with patch('smartcash.model.manager.ModelManager._check_pretrained_model_in_drive') as mock_check_drive:
                # Set mock return values
                mock_build_model.return_value = torch.nn.Sequential(
                    torch.nn.Linear(10, 5),
                    torch.nn.ReLU(),
                    torch.nn.Linear(5, 2)
                )
                mock_check_drive.return_value = None
                
                # Buat ModelManager
                self.model_manager = ModelManager(self.config)
        
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
        
    @patch('torch.nn.CrossEntropyLoss')
    def test_build_loss_function(self, mock_loss):
        """Test _build_loss_function"""
        # Set mock return value
        mock_loss_instance = MagicMock()
        mock_loss.return_value = mock_loss_instance
        
        # Build loss function
        loss_fn = self.model_manager._build_loss_function()
        
        # Verify loss function was created
        self.assertEqual(loss_fn, mock_loss_instance)
        
    def test_get_training_service_no_checkpoint_service(self):
        """Test get_training_service tanpa checkpoint service"""
        # Patch _build_loss_function
        with patch('smartcash.model.manager.ModelManager._build_loss_function') as mock_build_loss:
            # Set mock return value
            mock_loss = MagicMock()
            mock_build_loss.return_value = mock_loss
            
            # Get training service
            training_service = self.model_manager.get_training_service(
                progress_callback=self.mock_progress_callback,
                metrics_callback=self.mock_metrics_callback
            )
            
            # Verify training service was created
            self.assertIsInstance(training_service, TrainingService)
            self.assertEqual(training_service.model, self.model_manager.model)
            self.assertEqual(training_service.loss_fn, mock_loss)
            self.assertIsNotNone(training_service.optimizer)
            self.assertIsNotNone(training_service.checkpoint_service)
            
            # Verify checkpoint service was created
            self.assertIsInstance(self.model_manager.checkpoint_service, CheckpointService)
            self.assertEqual(self.model_manager.checkpoint_service.checkpoint_dir, self.checkpoint_dir)
            
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
            mock_loss = MagicMock()
            mock_build_loss.return_value = mock_loss
            
            # Get training service
            training_service = self.model_manager.get_training_service(
                progress_callback=self.mock_progress_callback,
                metrics_callback=self.mock_metrics_callback
            )
            
            # Verify training service was created
            self.assertIsInstance(training_service, TrainingService)
            self.assertEqual(training_service.checkpoint_service, checkpoint_service)
            
    def test_get_training_service_with_optimizer(self):
        """Test get_training_service dengan optimizer"""
        # Buat optimizer
        optimizer = torch.optim.SGD(self.model_manager.model.parameters(), lr=0.01)
        
        # Patch _build_loss_function
        with patch('smartcash.model.manager.ModelManager._build_loss_function') as mock_build_loss:
            # Set mock return value
            mock_loss = MagicMock()
            mock_build_loss.return_value = mock_loss
            
            # Get training service
            training_service = self.model_manager.get_training_service(
                optimizer=optimizer,
                progress_callback=self.mock_progress_callback,
                metrics_callback=self.mock_metrics_callback
            )
            
            # Verify training service was created
            self.assertIsInstance(training_service, TrainingService)
            self.assertEqual(training_service.optimizer, optimizer)
            
    def test_get_training_service_with_scheduler(self):
        """Test get_training_service dengan scheduler"""
        # Buat optimizer
        optimizer = torch.optim.SGD(self.model_manager.model.parameters(), lr=0.01)
        
        # Buat scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
        
        # Patch _build_loss_function
        with patch('smartcash.model.manager.ModelManager._build_loss_function') as mock_build_loss:
            # Set mock return value
            mock_loss = MagicMock()
            mock_build_loss.return_value = mock_loss
            
            # Get training service
            training_service = self.model_manager.get_training_service(
                optimizer=optimizer,
                scheduler=scheduler,
                progress_callback=self.mock_progress_callback,
                metrics_callback=self.mock_metrics_callback
            )
            
            # Verify training service was created
            self.assertIsInstance(training_service, TrainingService)
            self.assertEqual(training_service.scheduler, scheduler)
            
    @patch('torch.save')
    @patch('smartcash.model.manager.ModelManager._build_loss_function')
    def test_training_integration(self, mock_build_loss, mock_save):
        """Test integrasi training"""
        # Set mock return value
        mock_loss = torch.nn.CrossEntropyLoss()
        mock_build_loss.return_value = mock_loss
        
        # Get training service
        training_service = self.model_manager.get_training_service(
            progress_callback=self.mock_progress_callback,
            metrics_callback=self.mock_metrics_callback
        )
        
        # Train for 1 epoch
        best_metrics = training_service.train(
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            epochs=1
        )
        
        # Verify training completed
        self.assertIsInstance(best_metrics, dict)
        self.assertTrue(self.mock_progress_callback.complete.called)
        
        # Verify checkpoint was saved
        self.assertTrue(mock_save.called)
        
    def test_get_optimizer(self):
        """Test get_optimizer"""
        # Get optimizer
        optimizer = self.model_manager.get_optimizer(lr=0.01)
        
        # Verify optimizer was created
        self.assertIsInstance(optimizer, torch.optim.Adam)
        self.assertEqual(optimizer.param_groups[0]['lr'], 0.01)
        
    def test_get_scheduler(self):
        """Test get_scheduler"""
        # Get optimizer
        optimizer = self.model_manager.get_optimizer(lr=0.01)
        
        # Get scheduler
        scheduler = self.model_manager.get_scheduler(
            optimizer=optimizer,
            scheduler_type="step",
            step_size=5,
            gamma=0.5
        )
        
        # Verify scheduler was created
        self.assertIsInstance(scheduler, torch.optim.lr_scheduler.StepLR)
        self.assertEqual(scheduler.step_size, 5)
        self.assertEqual(scheduler.gamma, 0.5)
        
    def test_get_scheduler_cosine(self):
        """Test get_scheduler dengan cosine annealing"""
        # Get optimizer
        optimizer = self.model_manager.get_optimizer(lr=0.01)
        
        # Get scheduler
        scheduler = self.model_manager.get_scheduler(
            optimizer=optimizer,
            scheduler_type="cosine",
            T_max=10
        )
        
        # Verify scheduler was created
        self.assertIsInstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)
        self.assertEqual(scheduler.T_max, 10)
        
    def test_get_scheduler_reduce_on_plateau(self):
        """Test get_scheduler dengan reduce on plateau"""
        # Get optimizer
        optimizer = self.model_manager.get_optimizer(lr=0.01)
        
        # Get scheduler
        scheduler = self.model_manager.get_scheduler(
            optimizer=optimizer,
            scheduler_type="plateau",
            patience=3,
            factor=0.1
        )
        
        # Verify scheduler was created
        self.assertIsInstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
        self.assertEqual(scheduler.patience, 3)
        self.assertEqual(scheduler.factor, 0.1)
        
    def test_get_scheduler_unknown_type(self):
        """Test get_scheduler dengan tipe yang tidak dikenal"""
        # Get optimizer
        optimizer = self.model_manager.get_optimizer(lr=0.01)
        
        # Get scheduler with unknown type
        scheduler = self.model_manager.get_scheduler(
            optimizer=optimizer,
            scheduler_type="unknown"
        )
        
        # Verify None was returned
        self.assertIsNone(scheduler)
        
    def test_error_handling(self):
        """Test penanganan error"""
        # Patch _build_loss_function to raise exception
        with patch('smartcash.model.manager.ModelManager._build_loss_function') as mock_build_loss:
            # Set mock to raise exception
            mock_build_loss.side_effect = Exception("Test error")
            
            # Try to get training service
            with self.assertRaises(Exception):
                self.model_manager.get_training_service()

if __name__ == '__main__':
    unittest.main()
