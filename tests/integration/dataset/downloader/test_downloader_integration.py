#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: /Users/masdevid/Projects/smartcash/tests/integration/dataset/downloader/test_downloader_integration.py
# Deskripsi: Test integrasi antara UI dataset downloader dan dataset downloader

import unittest
import os
import sys
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

# Pastikan path smartcash tersedia
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))

# Mock untuk environment sebelum import
with patch('smartcash.common.environment._is_colab_environment', return_value=False):
    from smartcash.dataset.downloader.download_service import DownloadService, create_download_service
    from smartcash.ui.dataset.downloader.handlers.download_handler import _execute_download_sync_safe
    from smartcash.common.logger import get_logger
    from smartcash.dataset.downloader import get_downloader_instance
    from smartcash.common.environment import get_environment_manager, EnvironmentManager


class TestDownloaderIntegration(unittest.TestCase):
    """Test integrasi antara UI dataset downloader dan dataset downloader"""
    
    @patch('smartcash.common.environment._is_colab_environment', return_value=False)
    @patch('smartcash.common.environment.EnvironmentManager')
    def setUp(self, mock_env_manager, mock_is_colab):
        """Setup untuk setiap test case"""
        # Mock environment manager
        self.mock_env_manager = mock_env_manager.return_value
        self.mock_env_manager.is_colab = False
        self.mock_env_manager.is_drive_mounted = False
        
        self.logger = get_logger('test_integration')
        
        # Buat direktori temporary untuk test
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = os.path.join(self.temp_dir, 'data')
        self.downloads_dir = os.path.join(self.temp_dir, 'downloads')
        
        # Buat direktori yang diperlukan
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.downloads_dir, exist_ok=True)
        
        # Mock config untuk test
        self.config = {
            'endpoint': 'roboflow',
            'workspace': 'test-workspace',
            'project': 'test-project',
            'version': '1',
            'api_key': 'test-api-key-12345',
            'output_format': 'yolov5pytorch',
            'validate_download': True,
            'organize_dataset': True,
            'backup_existing': False,
            'data': {
                'dir': self.data_dir
            },
            'downloads': {
                'dir': self.downloads_dir
            }
        }
        
        # Mock UI components
        self.ui_components = {
            'logger': self.logger,
            'progress_tracker': MagicMock(),
            'status_area': MagicMock(),
            'show_for_operation': MagicMock(),
            'complete_operation': MagicMock(),
            'error_operation': MagicMock()
        }
        
        # Setup progress tracker mock
        self.ui_components['progress_tracker'].show = MagicMock()
        self.ui_components['progress_tracker'].update = MagicMock()
        self.ui_components['progress_tracker'].complete = MagicMock()
        self.ui_components['progress_tracker'].error = MagicMock()
    
    def tearDown(self):
        """Cleanup setelah setiap test case"""
        # Hapus direktori temporary
        shutil.rmtree(self.temp_dir)
    
    @patch('smartcash.common.environment._is_colab_environment', return_value=False)
    @patch('smartcash.common.environment.get_environment_manager')
    @patch('smartcash.ui.dataset.downloader.handlers.download_handler.get_downloader_instance')
    def test_download_service_creation(self, mock_get_downloader, mock_env_manager, mock_is_colab):
        """Test pembuatan download service dari UI handler"""
        # Setup mock environment
        mock_env = MagicMock()
        mock_env.is_colab = False
        mock_env.is_drive_mounted = False
        mock_env_manager.return_value = mock_env
        
        # Setup mock downloader
        mock_service = MagicMock()
        mock_get_downloader.return_value = mock_service
        mock_service.download_dataset.return_value = {
            'status': 'success',
            'output_dir': self.data_dir,
            'stats': {'total_images': 100},
            'duration': 1.5
        }
        mock_service.set_progress_callback = MagicMock()
        
        # Jalankan fungsi yang ditest
        _execute_download_sync_safe(self.ui_components, self.config, self.logger)
        
        # Verifikasi bahwa get_downloader_instance dipanggil dengan parameter yang benar
        mock_get_downloader.assert_called_once()
        args, kwargs = mock_get_downloader.call_args
        
        # Verifikasi config yang diberikan ke get_downloader_instance
        # Periksa apakah logger diberikan sebagai positional atau keyword argument
        if len(args) > 1:
            self.assertEqual(args[1], self.logger)
        elif 'logger' in kwargs:
            self.assertEqual(kwargs['logger'], self.logger)
        
        # Verifikasi config yang diberikan ke get_downloader_instance
        config_arg = args[0]
        self.assertEqual(config_arg['endpoint'], 'roboflow')
        self.assertEqual(config_arg['workspace'], 'test-workspace')
        self.assertEqual(config_arg['project'], 'test-project')
        self.assertEqual(config_arg['version'], '1')
        self.assertEqual(config_arg['api_key'], 'test-api-key-12345')
        
        # Verifikasi bahwa download_dataset dipanggil
        mock_service.download_dataset.assert_called_once()
        
        # Verifikasi bahwa progress tracker diupdate dengan benar
        self.ui_components['progress_tracker'].show.assert_called_once()
        self.ui_components['progress_tracker'].complete.assert_called_once()
    
    @patch('smartcash.common.environment._is_colab_environment', return_value=False)
    @patch('smartcash.common.environment.get_environment_manager')
    @patch('smartcash.dataset.downloader.download_service.DownloadService._validate_parameters')
    def test_download_service_parameter_extraction(self, mock_validate, mock_env_manager, mock_is_colab):
        """Test ekstraksi parameter dari config di DownloadService"""
        # Setup mock environment
        mock_env = MagicMock()
        mock_env.is_colab = False
        mock_env.is_drive_mounted = False
        mock_env_manager.return_value = mock_env
        
        # Setup mock validate
        mock_validate.return_value = {'valid': True, 'errors': []}
        
        # Buat service dengan config
        with patch('smartcash.dataset.downloader.download_service.RoboflowClient'), \
             patch('smartcash.dataset.downloader.download_service.FileProcessor'), \
             patch('smartcash.dataset.downloader.download_service.DatasetValidator'), \
             patch('smartcash.dataset.downloader.download_service.DownloadProgressTracker'):
            service = DownloadService(self.config, self.logger)
            
            # Mock metode yang dipanggil oleh download_dataset
            service._setup_download_paths = MagicMock(return_value={
                'temp_dir': Path(self.downloads_dir) / 'temp',
                'final_dir': Path(self.data_dir),
                'dataset_name': 'test_dataset',
                'base_downloads': Path(self.downloads_dir)
            })
            service.roboflow_client.get_dataset_metadata = MagicMock(return_value={
                'status': 'success',
                'data': {},
                'download_url': 'https://example.com/dataset.zip'
            })
            service.roboflow_client.download_dataset = MagicMock(return_value={
                'status': 'success',
                'file_path': os.path.join(self.downloads_dir, 'dataset.zip')
            })
            service.file_processor.extract_zip = MagicMock(return_value={
                'status': 'success'
            })
            service._organize_dataset_flow = MagicMock(return_value={
                'total_images': 100
            })
            service._cleanup_temp_files = MagicMock()
            service._success_result = MagicMock(return_value={
                'status': 'success',
                'output_dir': self.data_dir,
                'stats': {'total_images': 100}
            })
            
            # Jalankan metode yang ditest
            result = service.download_dataset()
            
            # Verifikasi bahwa parameter diekstrak dengan benar dari config
            mock_validate.assert_called_once_with(
                'test-workspace', 'test-project', '1', 'test-api-key-12345', 'yolov5pytorch'
            )
    
    @patch('smartcash.common.environment._is_colab_environment', return_value=False)
    @patch('smartcash.common.environment.get_environment_manager')
    def test_create_download_service_factory(self, mock_env_manager, mock_is_colab):
        """Test factory function create_download_service"""
        # Setup mock environment
        mock_env = MagicMock()
        mock_env.is_colab = False
        mock_env.is_drive_mounted = False
        mock_env_manager.return_value = mock_env
        
        # Mock komponen yang digunakan dalam DownloadService
        with patch('smartcash.dataset.downloader.download_service.RoboflowClient'), \
             patch('smartcash.dataset.downloader.download_service.FileProcessor'), \
             patch('smartcash.dataset.downloader.download_service.DatasetValidator'), \
             patch('smartcash.dataset.downloader.download_service.DownloadProgressTracker'):
            # Jalankan fungsi yang ditest
            service = create_download_service(self.config, self.logger)
            
            # Verifikasi bahwa service dibuat dengan benar
            self.assertIsInstance(service, DownloadService)
            self.assertEqual(service.config, self.config)
            self.assertEqual(service.logger, self.logger)
    
    @patch('smartcash.common.environment._is_colab_environment', return_value=False)
    @patch('smartcash.common.environment.get_environment_manager')
    @patch('smartcash.dataset.downloader.download_service.create_download_service')
    def test_get_downloader_instance(self, mock_create_service, mock_env_manager, mock_is_colab):
        """Test fungsi get_downloader_instance"""
        # Setup mock environment
        mock_env = MagicMock()
        mock_env.is_colab = False
        mock_env.is_drive_mounted = False
        mock_env_manager.return_value = mock_env
        
        # Setup mock service
        mock_service = MagicMock()
        mock_create_service.return_value = mock_service
        
        # Jalankan fungsi yang ditest
        result = get_downloader_instance(self.config, self.logger)
        
        # Verifikasi bahwa create_download_service dipanggil dengan parameter yang benar
        mock_create_service.assert_called_once_with(self.config, self.logger)
        
        # Verifikasi bahwa hasil yang dikembalikan adalah service yang benar
        self.assertEqual(result, mock_service)


if __name__ == '__main__':
    unittest.main()
