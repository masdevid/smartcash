"""
File: smartcash/ui/setup/env_config/tests/test_env_utils.py
Deskripsi: Unit test untuk env_utils.py
"""

import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path

from smartcash.ui.setup.env_config.utils.env_utils import get_env_status, format_env_info

class TestEnvUtils(unittest.TestCase):
    """
    Test untuk env_utils.py
    """
    
    def setUp(self):
        """
        Setup untuk test
        """
        # Sample env_status untuk testing format_env_info
        self.env_status = {
            'system_info': {
                'os': 'Linux',
                'python_version': '3.8.0',
                'is_colab': True,
                'is_kaggle': False
            },
            'drive_status': {
                'is_mounted': True,
                'drive_path': '/content/drive'
            },
            'directory_status': {
                'configs': True,
                'data': True,
                'data/raw': False,
                'data/processed': False,
                'models': True,
                'models/checkpoints': False,
                'models/weights': False,
                'output': True,
                'logs': True
            }
        }
    
    @patch('smartcash.ui.setup.env_config.utils.env_utils.Path')
    def test_get_env_status_with_attributes(self, mock_path_class):
        """
        Test get_env_status dengan environment manager yang memiliki semua atribut
        """
        # Setup mocks
        env_manager = MagicMock()
        
        # Mock system_info
        env_manager.get_system_info.return_value = {
            'os': 'Linux',
            'python_version': '3.8.0',
            'is_colab': True
        }
        
        # Mock drive status
        env_manager.is_drive_mounted = True
        env_manager.drive_path = '/content/drive'
        
        # Mock base_dir dan Path.exists
        env_manager.base_dir = '/content'
        mock_path_instance = MagicMock()
        mock_path_class.return_value = mock_path_instance
        mock_path_instance.__truediv__.return_value = mock_path_instance
        
        # Konfigurasi perilaku Path.exists berdasarkan nama direktori
        def mock_exists():
            # Dapatkan path dari __truediv__ terakhir
            path_str = mock_path_instance.__truediv__.call_args[0][0]
            return path_str in ['configs', 'data', 'models', 'output', 'logs']
            
        mock_path_instance.exists.side_effect = mock_exists
        
        # Call function to test
        result = get_env_status(env_manager)
        
        # Verify hasil berisi keys yang diharapkan
        self.assertIn('system_info', result)
        self.assertIn('drive_status', result)
        self.assertIn('directory_status', result)
        
        # Verifikasi system_info
        self.assertEqual(result['system_info'], env_manager.get_system_info.return_value)
        
        # Verifikasi drive_status
        self.assertEqual(result['drive_status']['is_mounted'], env_manager.is_drive_mounted)
        self.assertEqual(result['drive_status']['drive_path'], env_manager.drive_path)
    
    @patch('smartcash.ui.setup.env_config.utils.env_utils.hasattr')
    def test_get_env_status_minimal(self, mock_hasattr):
        """
        Test get_env_status dengan environment manager yang minimal
        """
        # Mock hasattr untuk mengembalikan False untuk semua atribut
        mock_hasattr.return_value = False
        
        # Buat environment manager minimal
        env_manager = MagicMock()
        
        # Call function to test
        result = get_env_status(env_manager)
        
        # Verify hasil berisi keys yang diharapkan
        self.assertIn('system_info', result)
        self.assertIn('drive_status', result)
        self.assertIn('directory_status', result)
        
        # Verifikasi system_info kosong
        self.assertEqual(result['system_info'], {})
        
        # Verifikasi drive_status memiliki nilai default
        self.assertFalse(result['drive_status']['is_mounted'])
        self.assertIsNone(result['drive_status']['drive_path'])
        
        # Verifikasi directory_status kosong
        self.assertEqual(result['directory_status'], {})
    
    def test_format_env_info(self):
        """
        Test format_env_info
        """
        # Call function to test
        result = format_env_info(self.env_status)
        
        # Verify result is a list of tuples
        self.assertIsInstance(result, list)
        for item in result:
            self.assertIsInstance(item, tuple)
            self.assertEqual(len(item), 2)
        
        # Verify expected labels are in result
        labels = [item[0] for item in result]
        self.assertIn("Sistem Operasi", labels)
        self.assertIn("Python Version", labels)
        self.assertIn("Colab", labels)
        self.assertIn("Google Drive", labels)
        self.assertIn("Drive Path", labels)
        self.assertIn("Direktori Tersedia", labels)
        self.assertIn("Direktori Tidak Tersedia", labels)
        
        # Verify expected values
        values_dict = dict(result)
        self.assertEqual(values_dict["Sistem Operasi"], "Linux")
        self.assertEqual(values_dict["Python Version"], "3.8.0")
        self.assertEqual(values_dict["Colab"], "Ya")
        self.assertEqual(values_dict["Google Drive"], "Terhubung")
        self.assertEqual(values_dict["Drive Path"], "/content/drive")
        
        # Check direktori tersedia contains expected directories
        dir_tersedia = values_dict["Direktori Tersedia"]
        self.assertIn("configs", dir_tersedia)
        self.assertIn("data", dir_tersedia)
        self.assertIn("models", dir_tersedia)
        self.assertIn("output", dir_tersedia)
        self.assertIn("logs", dir_tersedia)
        
        # Check direktori tidak tersedia contains expected directories
        dir_tidak_tersedia = values_dict["Direktori Tidak Tersedia"]
        self.assertIn("data/raw", dir_tidak_tersedia)
        self.assertIn("data/processed", dir_tidak_tersedia)
        self.assertIn("models/checkpoints", dir_tidak_tersedia)
        self.assertIn("models/weights", dir_tidak_tersedia)
    
    def test_format_env_info_empty(self):
        """
        Test format_env_info with empty env_status
        """
        # Call function with empty dict
        result = format_env_info({})
        
        # Verify result is an empty list
        self.assertEqual(result, [])
        
    def test_format_env_info_partial(self):
        """
        Test format_env_info with partial env_status
        """
        # Create partial env_status with only system_info
        partial_env_status = {
            'system_info': {
                'os': 'Windows',
                'python_version': '3.9.0'
            }
        }
        
        # Call function to test
        result = format_env_info(partial_env_status)
        
        # Verify result contains expected system info
        labels = [item[0] for item in result]
        self.assertIn("Sistem Operasi", labels)
        self.assertIn("Python Version", labels)
        
        # Verify values
        values_dict = dict(result)
        self.assertEqual(values_dict["Sistem Operasi"], "Windows")
        self.assertEqual(values_dict["Python Version"], "3.9.0")
        
        # Verify drive and directory info not included
        self.assertNotIn("Google Drive", labels)
        self.assertNotIn("Direktori Tersedia", labels)

if __name__ == '__main__':
    unittest.main() 