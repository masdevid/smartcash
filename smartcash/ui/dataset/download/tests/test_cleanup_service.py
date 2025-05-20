"""
File: smartcash/ui/dataset/download/tests/test_cleanup_service.py
Deskripsi: Test suite untuk cleanup_service.py
"""

import unittest
from unittest.mock import MagicMock, patch, call
import sys
import os
import tempfile
import shutil
from pathlib import Path

# Tambahkan path ke sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))

from smartcash.dataset.services.downloader.cleanup_service import CleanupService

class TestCleanupService(unittest.TestCase):
    """Test suite untuk CleanupService."""
    
    def setUp(self):
        """Setup test environment."""
        # Buat temporary directory untuk test
        self.test_dir = tempfile.mkdtemp()
        
        # Buat dataset dummy
        self.dataset_dir = os.path.join(self.test_dir, "test_dataset")
        os.makedirs(self.dataset_dir)
        
        # Buat beberapa file dummy di dataset
        for i in range(5):
            with open(os.path.join(self.dataset_dir, f"file_{i}.txt"), "w") as f:
                f.write(f"Test content {i}")
        
        # Buat subdirectory
        os.makedirs(os.path.join(self.dataset_dir, "images"))
        for i in range(3):
            with open(os.path.join(self.dataset_dir, "images", f"image_{i}.jpg"), "w") as f:
                f.write(f"Image content {i}")
        
        # Setup mock
        self.mock_observer_manager = MagicMock()
        self.mock_logger = MagicMock()
        
        # Inisialisasi service
        self.cleanup_service = CleanupService(
            logger=self.mock_logger,
            observer_manager=self.mock_observer_manager
        )
    
    def tearDown(self):
        """Cleanup after test."""
        # Hapus temporary directory
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_cleanup_dataset_success(self):
        """Test cleanup_dataset berhasil."""
        # Jalankan cleanup
        result = self.cleanup_service.cleanup_dataset(
            self.dataset_dir,
            backup_before_delete=False,
            show_progress=False
        )
        
        # Verifikasi hasil
        self.assertEqual(result["status"], "success")
        self.assertIn("Dataset berhasil dihapus", result["message"])
        
        # Verifikasi direktori sudah dihapus
        self.assertFalse(os.path.exists(self.dataset_dir))
        
        # Verifikasi notifikasi dikirim
        self.mock_logger.info.assert_any_call(f"🗑️ Menghapus dataset: {self.dataset_dir}")
        self.mock_logger.info.assert_any_call(unittest.mock.ANY)  # Untuk log success
    
    def test_cleanup_dataset_with_backup(self):
        """Test cleanup_dataset dengan backup."""
        # Patch backup_service.backup_dataset
        with patch('smartcash.dataset.services.downloader.backup_service.BackupService.backup_dataset') as mock_backup:
            # Setup mock untuk mengembalikan hasil sukses
            mock_backup.return_value = {
                "status": "success",
                "message": "Backup berhasil",
                "backup_path": f"{self.dataset_dir}_backup.zip"
            }
            
            # Jalankan cleanup
            result = self.cleanup_service.cleanup_dataset(
                self.dataset_dir,
                backup_before_delete=True,
                show_progress=False
            )
            
            # Verifikasi hasil
            self.assertEqual(result["status"], "success")
            self.assertIn("Dataset berhasil dihapus", result["message"])
            
            # Verifikasi direktori sudah dihapus
            self.assertFalse(os.path.exists(self.dataset_dir))
            
            # Verifikasi backup dipanggil
            mock_backup.assert_called_once()
            
            # Verifikasi notifikasi dikirim
            self.mock_logger.info.assert_any_call(f"💾 Membuat backup sebelum menghapus: {self.dataset_dir}")
    
    def test_cleanup_dataset_nonexistent(self):
        """Test cleanup_dataset dengan direktori yang tidak ada."""
        # Path yang tidak ada
        nonexistent_path = os.path.join(self.test_dir, "nonexistent")
        
        # Jalankan cleanup
        result = self.cleanup_service.cleanup_dataset(
            nonexistent_path,
            backup_before_delete=False,
            show_progress=False
        )
        
        # Verifikasi hasil
        self.assertEqual(result["status"], "error")
        self.assertIn("Dataset tidak ditemukan", result["message"])
        
        # Verifikasi warning log
        self.mock_logger.warning.assert_called_with(f"⚠️ Dataset tidak ditemukan: {nonexistent_path}")
    
    def test_cleanup_multiple_datasets(self):
        """Test cleanup_multiple_datasets."""
        # Buat dataset dummy kedua
        dataset_dir2 = os.path.join(self.test_dir, "test_dataset2")
        os.makedirs(dataset_dir2)
        with open(os.path.join(dataset_dir2, "file.txt"), "w") as f:
            f.write("Test content")
        
        # Jalankan cleanup multiple
        result = self.cleanup_service.cleanup_multiple_datasets(
            [self.dataset_dir, dataset_dir2],
            backup_before_delete=False,
            show_progress=False
        )
        
        # Verifikasi hasil
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["success_count"], 2)
        self.assertEqual(result["error_count"], 0)
        
        # Verifikasi direktori sudah dihapus
        self.assertFalse(os.path.exists(self.dataset_dir))
        self.assertFalse(os.path.exists(dataset_dir2))
    
    def test_cleanup_multiple_datasets_partial_failure(self):
        """Test cleanup_multiple_datasets dengan sebagian gagal."""
        # Buat dataset dummy kedua
        dataset_dir2 = os.path.join(self.test_dir, "test_dataset2")
        os.makedirs(dataset_dir2)
        
        # Path yang tidak ada
        nonexistent_path = os.path.join(self.test_dir, "nonexistent")
        
        # Jalankan cleanup multiple
        result = self.cleanup_service.cleanup_multiple_datasets(
            [self.dataset_dir, dataset_dir2, nonexistent_path],
            backup_before_delete=False,
            show_progress=False
        )
        
        # Verifikasi hasil
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["success_count"], 2)
        self.assertEqual(result["error_count"], 1)
        
        # Verifikasi direktori sudah dihapus
        self.assertFalse(os.path.exists(self.dataset_dir))
        self.assertFalse(os.path.exists(dataset_dir2))
        
        # Verifikasi warning log untuk error count
        self.mock_logger.warning.assert_called_with(f"⚠️ {result['error_count']}/{len([self.dataset_dir, dataset_dir2, nonexistent_path])} dataset gagal dihapus")
    
    def test_set_observer_manager(self):
        """Test set_observer_manager."""
        # Buat observer manager baru
        new_observer_manager = MagicMock()
        
        # Set observer manager
        self.cleanup_service.set_observer_manager(new_observer_manager)
        
        # Verifikasi observer manager diupdate
        self.assertEqual(self.cleanup_service.observer_manager, new_observer_manager)

if __name__ == '__main__':
    unittest.main() 