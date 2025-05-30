"""
File: smartcash/ui/model/tests/test_simple_download.py
Deskripsi: Test case untuk fungsi download dan sync model pretrained yang disederhanakan
"""

import os
import sys
import unittest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

from smartcash.ui.utils.constants import ICONS
from smartcash.ui.model.handlers.simple_download import (
    sync_drive_to_local,
    sync_local_to_drive,
    process_download_sync
)

class TestSimpleDownload(unittest.TestCase):
    """Test case untuk fungsi download dan sync model pretrained"""
    
    def setUp(self):
        """Setup untuk test case"""
        # Buat direktori sementara untuk test
        self.temp_dir = tempfile.mkdtemp()
        self.local_dir = os.path.join(self.temp_dir, 'local')
        self.drive_dir = os.path.join(self.temp_dir, 'drive')
        
        # Buat direktori lokal dan drive
        os.makedirs(self.local_dir, exist_ok=True)
        os.makedirs(self.drive_dir, exist_ok=True)
        
        # Buat file dummy untuk test
        with open(os.path.join(self.local_dir, 'yolov5s.pt'), 'w') as f:
            f.write('dummy yolov5s model')
        
        with open(os.path.join(self.local_dir, 'efficientnet_b4.pt'), 'w') as f:
            f.write('dummy efficientnet model')
        
        # Mock untuk log function
        self.log_messages = []
        self.log_func = lambda msg: self.log_messages.append(msg)
    
    def tearDown(self):
        """Cleanup setelah test case"""
        # Hapus direktori sementara
        shutil.rmtree(self.temp_dir)
    
    def test_sync_local_to_drive(self):
        """Test sinkronisasi dari lokal ke drive"""
        # Jalankan fungsi sync
        model_info = {
            'models': {
                'yolov5s': {
                    'path': os.path.join(self.local_dir, 'yolov5s.pt'),
                    'size_mb': 0.01
                },
                'efficientnet_b4': {
                    'path': os.path.join(self.local_dir, 'efficientnet_b4.pt'),
                    'size_mb': 0.01
                }
            }
        }
        
        sync_local_to_drive(self.local_dir, self.drive_dir, model_info, self.log_func)
        
        # Cek apakah file berhasil disinkronkan
        self.assertTrue(os.path.exists(os.path.join(self.drive_dir, 'yolov5s.pt')))
        self.assertTrue(os.path.exists(os.path.join(self.drive_dir, 'efficientnet_b4.pt')))
        
        # Cek log messages
        success_message = f"{ICONS.get('success', '✅')} Sinkronisasi ke Google Drive selesai!"
        self.assertIn(success_message, self.log_messages)
    
    def test_sync_drive_to_local(self):
        """Test sinkronisasi dari drive ke lokal"""
        # Buat file dummy di drive
        with open(os.path.join(self.drive_dir, 'yolov5s.pt'), 'w') as f:
            f.write('dummy yolov5s model from drive')
        
        with open(os.path.join(self.drive_dir, 'efficientnet-b4_notop.h5'), 'w') as f:
            f.write('dummy efficientnet model from drive')
        
        # Hapus file di lokal untuk memastikan sinkronisasi bekerja
        os.remove(os.path.join(self.local_dir, 'yolov5s.pt'))
        
        # Jalankan fungsi sync
        sync_drive_to_local(self.local_dir, self.drive_dir, self.log_func)
        
        # Cek apakah file berhasil disinkronkan
        self.assertTrue(os.path.exists(os.path.join(self.local_dir, 'yolov5s.pt')))
        self.assertTrue(os.path.exists(os.path.join(self.local_dir, 'efficientnet-b4_notop.h5')))
        
        # Cek log messages
        success_message = f"{ICONS.get('success', '✅')} Sinkronisasi dari Drive ke lokal selesai!"
        self.assertIn(success_message, self.log_messages)
    
    @patch('smartcash.ui.model.handlers.simple_download.is_drive_mounted')
    def test_process_download_sync_no_drive(self, mock_is_drive_mounted):
        """Test proses download dan sync tanpa Google Drive"""
        # Mock is_drive_mounted untuk mengembalikan False
        mock_is_drive_mounted.return_value = False
        
        # Mock PretrainedModelDownloader dengan patch dalam fungsi test
        with patch('smartcash.model.services.pretrained_downloader.PretrainedModelDownloader') as mock_downloader_class:
            # Mock downloader instance
            mock_downloader = MagicMock()
            mock_downloader_class.return_value = mock_downloader
            
            # Mock return values untuk method downloader
            mock_downloader.download_yolov5.return_value = {'path': os.path.join(self.local_dir, 'yolov5s.pt')}
            mock_downloader.download_efficientnet.return_value = {'path': os.path.join(self.local_dir, 'efficientnet_b4.pt')}
            mock_downloader.get_model_info.return_value = {
                'models': {
                    'yolov5s': {'size_mb': 14.5, 'path': os.path.join(self.local_dir, 'yolov5s.pt')},
                    'efficientnet_b4': {'size_mb': 75.2, 'path': os.path.join(self.local_dir, 'efficientnet_b4.pt')}
                }
            }
        
            # Buat UI components mock
            ui_components = {
                'status': MagicMock(),
                'log': MagicMock(),
                'models_dir': self.local_dir,
                'drive_models_dir': self.drive_dir
            }
        
            # Jalankan fungsi process_download_sync
            process_download_sync(ui_components)
            
            # Verifikasi bahwa method downloader dipanggil
            mock_downloader.download_yolov5.assert_called_once()
            mock_downloader.download_efficientnet.assert_called_once()
            mock_downloader.get_model_info.assert_called_once()

def run_test_case(test_case):
    """Menjalankan test case tertentu"""
    suite = unittest.TestLoader().loadTestsFromTestCase(test_case)
    runner = unittest.TextTestRunner()
    return runner.run(suite)

class TestPretrainedInitializer(unittest.TestCase):
    """Test case untuk fungsi di pretrained_initializer.py"""
    
    def test_is_drive_mounted(self):
        """Test fungsi is_drive_mounted"""
        from smartcash.ui.model.pretrained_initializer import is_drive_mounted
        
        # Patch os.path.exists untuk mengembalikan True
        with patch('os.path.exists', return_value=True):
            self.assertTrue(is_drive_mounted())
        
        # Patch os.path.exists untuk mengembalikan False
        with patch('os.path.exists', return_value=False):
            self.assertFalse(is_drive_mounted())
    
    def test_mount_drive_success(self):
        """Test fungsi mount_drive saat berhasil"""
        from smartcash.ui.model.pretrained_initializer import mount_drive
        
        # Mock the import of google.colab
        with patch('builtins.__import__') as mock_import:
            # Setup mock to return our mocked module
            mock_drive = MagicMock()
            mock_import.return_value = MagicMock(drive=mock_drive)
            
            # Call the function
            success, message = mount_drive()
            
            # Verify the results
            self.assertTrue(success)
            self.assertIn("berhasil", message)
            mock_drive.mount.assert_called_once_with('/content/drive')
    
    def test_mount_drive_failure(self):
        """Test fungsi mount_drive saat gagal"""
        from smartcash.ui.model.pretrained_initializer import mount_drive
        
        # Mock the import of google.colab
        with patch('builtins.__import__') as mock_import:
            # Setup mock to raise exception when importing
            mock_import.side_effect = Exception("Test error")
            
            # Call the function
            success, message = mount_drive()
            
            # Verify the results
            self.assertFalse(success)
            self.assertIn("Gagal", message)
            self.assertIn("Test error", message)

class TestDownloadWithDrive(unittest.TestCase):
    """Test case untuk skenario download dengan Google Drive"""
    
    def setUp(self):
        """Setup untuk test case"""
        # Buat direktori sementara untuk test
        self.temp_dir = tempfile.mkdtemp()
        self.local_dir = os.path.join(self.temp_dir, 'local')
        self.drive_dir = os.path.join(self.temp_dir, 'drive')
        
        # Buat direktori lokal dan drive
        os.makedirs(self.local_dir, exist_ok=True)
        os.makedirs(self.drive_dir, exist_ok=True)
        
        # Buat file dummy di drive untuk test
        with open(os.path.join(self.drive_dir, 'yolov5s.pt'), 'w') as f:
            f.write('dummy yolov5s model from drive')
        
        with open(os.path.join(self.drive_dir, 'efficientnet-b4_notop.h5'), 'w') as f:
            f.write('dummy efficientnet model from drive')
        
        # Mock untuk log function
        self.log_messages = []
        self.log_func = lambda msg: self.log_messages.append(msg)
    
    def tearDown(self):
        """Cleanup setelah test case"""
        # Hapus direktori sementara
        shutil.rmtree(self.temp_dir)
    
    @patch('smartcash.ui.model.handlers.simple_download.is_drive_mounted')
    def test_process_download_sync_with_drive(self, mock_is_drive_mounted):
        """Test proses download dan sync dengan Google Drive"""
        # Mock is_drive_mounted untuk mengembalikan True
        mock_is_drive_mounted.return_value = True
        
        # Buat UI components mock
        ui_components = {
            'status': MagicMock(),
            'log': MagicMock(),
            'models_dir': self.local_dir,
            'drive_models_dir': self.drive_dir
        }
        
        # Import fungsi yang akan diuji
        from smartcash.ui.model.handlers.simple_download import process_download_sync
        
        # Jalankan fungsi process_download_sync
        process_download_sync(ui_components)
        
        # Verifikasi bahwa file berhasil disinkronkan dari drive ke lokal
        self.assertTrue(os.path.exists(os.path.join(self.local_dir, 'yolov5s.pt')))
        self.assertTrue(os.path.exists(os.path.join(self.local_dir, 'efficientnet-b4_notop.h5')))

def run_all_tests():
    """Menjalankan semua test case"""
    print("🧪 Menjalankan test case untuk fungsi download dan sync model pretrained...")
    
    # Jalankan semua test case
    test_classes = [TestSimpleDownload, TestPretrainedInitializer, TestDownloadWithDrive]
    success = True
    
    for test_class in test_classes:
        print(f"\n🧪 Menjalankan {test_class.__name__}...")
        result = run_test_case(test_class)
        if not result.wasSuccessful():
            success = False
    
    if success:
        print(f"\n{ICONS.get('success', '✅')} Semua test berhasil!")
    else:
        print(f"\n{ICONS.get('error', '❌')} Ada test yang gagal!")
    
    return success

if __name__ == '__main__':
    run_all_tests()
