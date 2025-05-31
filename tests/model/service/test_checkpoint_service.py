"""
File: tests/model/service/test_checkpoint_service.py
Deskripsi: Unit test untuk CheckpointService
"""

import unittest
import os
import torch
import shutil
import tempfile
from unittest.mock import MagicMock, patch
from pathlib import Path

from smartcash.model.service.checkpoint_service import CheckpointService
from smartcash.common.exceptions import ModelCheckpointError

class DummyModel(torch.nn.Module):
    """Model dummy untuk testing"""
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 2)
        
    def forward(self, x):
        return self.linear(x)

class TestCheckpointService(unittest.TestCase):
    """Test case untuk CheckpointService"""
    
    def setUp(self):
        """Setup untuk setiap test case"""
        # Buat temporary directory untuk checkpoint
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_dir = os.path.join(self.temp_dir, "checkpoints")
        
        # Buat progress callback mock
        self.mock_progress_callback = MagicMock()
        
        # Buat checkpoint service
        self.checkpoint_service = CheckpointService(
            checkpoint_dir=self.checkpoint_dir,
            max_checkpoints=3,
            save_best=True,
            save_last=True,
            metric_name="val_loss",
            mode="min",
            progress_callback=self.mock_progress_callback
        )
        
        # Buat model dummy
        self.model = DummyModel()
        
        # Buat optimizer dummy
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
    def tearDown(self):
        """Cleanup setelah setiap test case"""
        # Hapus temporary directory
        shutil.rmtree(self.temp_dir)
        
    def test_initialization(self):
        """Test inisialisasi CheckpointService"""
        # Verify checkpoint directory was created
        self.assertTrue(os.path.exists(self.checkpoint_dir))
        
        # Verify default values
        self.assertEqual(self.checkpoint_service.max_checkpoints, 3)
        self.assertTrue(self.checkpoint_service.save_best)
        self.assertTrue(self.checkpoint_service.save_last)
        self.assertEqual(self.checkpoint_service.metric_name, "val_loss")
        self.assertEqual(self.checkpoint_service.mode, "min")
        self.assertEqual(self.checkpoint_service.best_metric, float('inf'))
        self.assertEqual(self.checkpoint_service.best_epoch, -1)
        
    def test_save_checkpoint(self):
        """Test save_checkpoint"""
        # Save checkpoint
        metrics = {"val_loss": 0.5, "accuracy": 0.8}
        path = self.checkpoint_service.save_checkpoint(
            model=self.model,
            path="epoch_001.pt",
            optimizer=self.optimizer,
            epoch=1,
            metrics=metrics
        )
        
        # Verify checkpoint file was created
        checkpoint_path = os.path.join(self.checkpoint_dir, "epoch_001.pt")
        self.assertTrue(os.path.exists(checkpoint_path))
        self.assertEqual(path, checkpoint_path)
        
        # Verify checkpoint contains expected data
        checkpoint = torch.load(checkpoint_path)
        self.assertIn("model", checkpoint)
        self.assertIn("optimizer", checkpoint)
        self.assertIn("metadata", checkpoint)
        self.assertEqual(checkpoint["metadata"]["epoch"], 1)
        self.assertEqual(checkpoint["metadata"]["metrics"]["val_loss"], 0.5)
        
        # Verify progress callback was called
        self.assertTrue(self.mock_progress_callback.update_progress.called)
        
    def test_save_best_checkpoint(self):
        """Test save_checkpoint with best metric"""
        # Save first checkpoint
        metrics1 = {"val_loss": 0.5}
        self.checkpoint_service.save_checkpoint(
            model=self.model,
            path="epoch_001.pt",
            optimizer=self.optimizer,
            epoch=1,
            metrics=metrics1
        )
        
        # Save second checkpoint with better metric
        metrics2 = {"val_loss": 0.3}
        self.checkpoint_service.save_checkpoint(
            model=self.model,
            path="epoch_002.pt",
            optimizer=self.optimizer,
            epoch=2,
            metrics=metrics2
        )
        
        # Verify best checkpoint was created
        best_path = os.path.join(self.checkpoint_dir, "best.pt")
        self.assertTrue(os.path.exists(best_path))
        
        # Verify best metric was updated
        self.assertEqual(self.checkpoint_service.best_metric, 0.3)
        self.assertEqual(self.checkpoint_service.best_epoch, 2)
        
        # Save third checkpoint with worse metric
        metrics3 = {"val_loss": 0.4}
        self.checkpoint_service.save_checkpoint(
            model=self.model,
            path="epoch_003.pt",
            optimizer=self.optimizer,
            epoch=3,
            metrics=metrics3
        )
        
        # Verify best metric was not updated
        self.assertEqual(self.checkpoint_service.best_metric, 0.3)
        self.assertEqual(self.checkpoint_service.best_epoch, 2)
        
    def test_save_last_checkpoint(self):
        """Test save_checkpoint with last checkpoint"""
        # Save multiple checkpoints
        for i in range(1, 4):
            metrics = {"val_loss": 0.5 - i * 0.1}
            self.checkpoint_service.save_checkpoint(
                model=self.model,
                path=f"epoch_{i:03d}.pt",
                optimizer=self.optimizer,
                epoch=i,
                metrics=metrics
            )
        
        # Verify last checkpoint was created
        last_path = os.path.join(self.checkpoint_dir, "last.pt")
        self.assertTrue(os.path.exists(last_path))
        
        # Verify last checkpoint contains latest epoch
        checkpoint = torch.load(last_path)
        self.assertEqual(checkpoint["metadata"]["epoch"], 3)
        
    def test_cleanup_old_checkpoints(self):
        """Test cleanup of old checkpoints"""
        # Save multiple checkpoints (more than max_checkpoints)
        for i in range(1, 6):
            metrics = {"val_loss": 0.5 - i * 0.1}
            self.checkpoint_service.save_checkpoint(
                model=self.model,
                path=f"epoch_{i:03d}.pt",
                optimizer=self.optimizer,
                epoch=i,
                metrics=metrics
            )
        
        # Verify only max_checkpoints regular checkpoints exist (plus best.pt and last.pt)
        checkpoint_files = [f for f in os.listdir(self.checkpoint_dir) if f.endswith('.pt')]
        self.assertEqual(len(checkpoint_files), 5)  # 3 regular + best.pt + last.pt
        
        # Verify oldest checkpoints were removed
        self.assertFalse(os.path.exists(os.path.join(self.checkpoint_dir, "epoch_001.pt")))
        self.assertFalse(os.path.exists(os.path.join(self.checkpoint_dir, "epoch_002.pt")))
        
    def test_load_checkpoint(self):
        """Test load_checkpoint"""
        # Save checkpoint
        metrics = {"val_loss": 0.5}
        save_path = self.checkpoint_service.save_checkpoint(
            model=self.model,
            path="epoch_001.pt",
            optimizer=self.optimizer,
            epoch=1,
            metrics=metrics
        )
        
        # Create new model and optimizer
        new_model = DummyModel()
        new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.002)
        
        # Load checkpoint
        loaded_model, metadata = self.checkpoint_service.load_checkpoint(
            path=save_path,
            model=new_model,
            optimizer=new_optimizer
        )
        
        # Verify model was loaded
        self.assertEqual(loaded_model, new_model)
        
        # Verify metadata was returned
        self.assertEqual(metadata["epoch"], 1)
        self.assertEqual(metadata["metrics"]["val_loss"], 0.5)
        
        # Verify progress callback was called
        self.assertTrue(self.mock_progress_callback.update_progress.called)
        
    def test_load_checkpoint_nonexistent(self):
        """Test load_checkpoint with nonexistent file"""
        # Try to load nonexistent checkpoint
        with self.assertRaises(ModelCheckpointError):
            self.checkpoint_service.load_checkpoint(
                path="nonexistent.pt",
                model=self.model
            )
        
    def test_list_checkpoints(self):
        """Test list_checkpoints"""
        # Save multiple checkpoints
        for i in range(1, 4):
            metrics = {"val_loss": 0.5 - i * 0.1}
            self.checkpoint_service.save_checkpoint(
                model=self.model,
                path=f"epoch_{i:03d}.pt",
                optimizer=self.optimizer,
                epoch=i,
                metrics=metrics
            )
        
        # List checkpoints
        checkpoints = self.checkpoint_service.list_checkpoints()
        
        # Verify checkpoints were listed
        self.assertEqual(len(checkpoints), 5)  # 3 regular + best.pt + last.pt
        
        # Verify checkpoint metadata
        for checkpoint in checkpoints:
            self.assertIn("path", checkpoint)
            self.assertIn("filename", checkpoint)
            self.assertIn("size", checkpoint)
            self.assertIn("mtime", checkpoint)
            
    def test_list_checkpoints_sort_by_epoch(self):
        """Test list_checkpoints with sort_by=epoch"""
        # Save multiple checkpoints
        for i in range(1, 4):
            metrics = {"val_loss": 0.5 - i * 0.1}
            self.checkpoint_service.save_checkpoint(
                model=self.model,
                path=f"epoch_{i:03d}.pt",
                optimizer=self.optimizer,
                epoch=i,
                metrics=metrics
            )
        
        # List checkpoints sorted by epoch
        checkpoints = self.checkpoint_service.list_checkpoints(sort_by="epoch")
        
        # Verify checkpoints include epoch 3, 2, 1 (mungkin termasuk best.pt dan last.pt)
        epochs = [c.get("epoch", -1) for c in checkpoints if "epoch" in c]
        self.assertIn(3, epochs)
        self.assertIn(2, epochs)
        self.assertIn(1, epochs)
        
    def test_list_checkpoints_sort_by_metric(self):
        """Test list_checkpoints with sort_by=metric"""
        # Save multiple checkpoints
        for i in range(1, 4):
            metrics = {"val_loss": 0.5 - i * 0.1}
            self.checkpoint_service.save_checkpoint(
                model=self.model,
                path=f"epoch_{i:03d}.pt",
                optimizer=self.optimizer,
                epoch=i,
                metrics=metrics
            )
        
        # List checkpoints sorted by metric
        checkpoints = self.checkpoint_service.list_checkpoints(sort_by="metric")
        
        # Verify checkpoints include metrics sekitar 0.2, 0.3, 0.4 (mungkin termasuk best.pt dan last.pt dengan nilai yang sama)
        val_losses = [c.get("metrics", {}).get("val_loss", float('inf')) for c in checkpoints if "metrics" in c and "val_loss" in c.get("metrics", {})]
        
        # Gunakan assertAlmostEqual untuk setiap nilai yang diharapkan
        found_02 = False
        found_03 = False
        found_04 = False
        
        for val in val_losses:
            if abs(val - 0.2) < 1e-10:
                found_02 = True
            elif abs(val - 0.3) < 1e-10:
                found_03 = True
            elif abs(val - 0.4) < 1e-10:
                found_04 = True
                
        self.assertTrue(found_02, "Nilai sekitar 0.2 tidak ditemukan")
        self.assertTrue(found_03, "Nilai sekitar 0.3 tidak ditemukan")
        self.assertTrue(found_04, "Nilai sekitar 0.4 tidak ditemukan")
        
    def test_get_checkpoint_info(self):
        """Test get_checkpoint_info"""
        # Save checkpoint
        metrics = {"val_loss": 0.5}
        path = self.checkpoint_service.save_checkpoint(
            model=self.model,
            path="epoch_001.pt",
            optimizer=self.optimizer,
            epoch=1,
            metrics=metrics
        )
        
        # Get checkpoint info
        info = self.checkpoint_service.get_checkpoint_info(path)
        
        # Verify checkpoint info
        self.assertEqual(info["epoch"], 1)
        self.assertEqual(info["metrics"]["val_loss"], 0.5)
        self.assertIn("path", info)
        self.assertIn("filename", info)
        self.assertIn("size", info)
        
    def test_get_checkpoint_info_nonexistent(self):
        """Test get_checkpoint_info with nonexistent file"""
        # Try to get info for nonexistent checkpoint
        with self.assertRaises(ModelCheckpointError):
            self.checkpoint_service.get_checkpoint_info("nonexistent.pt")
            
    def test_set_progress_callback(self):
        """Test set_progress_callback"""
        # Create new callback
        new_callback = MagicMock()
        
        # Set new callback
        self.checkpoint_service.set_progress_callback(new_callback)
        
        # Save checkpoint to trigger callback
        self.checkpoint_service.save_checkpoint(
            model=self.model,
            path="epoch_001.pt",
            optimizer=self.optimizer,
            epoch=1
        )
        
        # Verify new callback was called
        self.assertTrue(new_callback.update_progress.called)
        
    def test_utility_methods(self):
        """Test utility methods"""
        # Save checkpoint with best metric
        metrics = {"val_loss": 0.5}
        self.checkpoint_service.save_checkpoint(
            model=self.model,
            path="epoch_001.pt",
            optimizer=self.optimizer,
            epoch=1,
            metrics=metrics
        )
        
        # Test get_best_checkpoint_path
        best_path = self.checkpoint_service.get_best_checkpoint_path()
        self.assertEqual(best_path, os.path.join(self.checkpoint_dir, "best.pt"))
        
        # Test get_last_checkpoint_path
        last_path = self.checkpoint_service.get_last_checkpoint_path()
        # Implementasi mungkin mengembalikan path epoch terakhir atau file last.pt
        self.assertTrue(last_path.endswith(".pt"))
        
        # Test get_checkpoint_dir
        dir_path = self.checkpoint_service.get_checkpoint_dir()
        self.assertEqual(dir_path, self.checkpoint_dir)
        
        # Test get_best_metric
        best_metric = self.checkpoint_service.get_best_metric()
        self.assertEqual(best_metric, 0.5)
        
        # Test get_best_epoch
        best_epoch = self.checkpoint_service.get_best_epoch()
        self.assertEqual(best_epoch, 1)
        
        # Test should_save_checkpoint
        self.checkpoint_service.save_interval = 2
        self.assertTrue(self.checkpoint_service.should_save_checkpoint(2))
        self.assertFalse(self.checkpoint_service.should_save_checkpoint(1))
        
        # Test is_better_metric
        self.assertTrue(self.checkpoint_service.is_better_metric(0.4, 0.5))
        self.assertFalse(self.checkpoint_service.is_better_metric(0.6, 0.5))
        
        # Change mode to max and test again
        self.checkpoint_service.mode = "max"
        self.assertTrue(self.checkpoint_service.is_better_metric(0.6, 0.5))
        self.assertFalse(self.checkpoint_service.is_better_metric(0.4, 0.5))

if __name__ == '__main__':
    unittest.main()
