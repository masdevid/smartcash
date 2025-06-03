"""
File: smartcash/ui/training/tests/test_training_service_integration.py
Deskripsi: Pengujian integrasi untuk TrainingServiceAdapter dan TrainingServiceManager
"""

import unittest
from unittest.mock import MagicMock, patch
import os
import sys
from typing import Dict, Any

# Import modul yang akan diuji
from smartcash.ui.training.adapters.training_service_adapter import TrainingServiceAdapter
from smartcash.ui.training.services.training_service_manager import TrainingServiceManager
from smartcash.common.logger import get_logger


class TestTrainingServiceAdapter(unittest.TestCase):
    """Pengujian untuk TrainingServiceAdapter"""
    
    def setUp(self):
        """Setup untuk pengujian"""
        # Mock backend training service
        self.mock_backend_service = MagicMock()
        self.mock_backend_service.train = MagicMock()
        self.mock_backend_service.stop_training = MagicMock()
        self.mock_backend_service.reset_training_state = MagicMock()
        self.mock_backend_service.checkpoint_service = MagicMock()
        self.mock_backend_service.metrics_tracker = MagicMock()
        self.mock_backend_service.set_callback = MagicMock()
        
        # Mock logger
        self.mock_logger = MagicMock()
        
        # Buat adapter
        self.adapter = TrainingServiceAdapter(self.mock_backend_service, self.mock_logger)
    
    def test_start_training(self):
        """Test start_training memanggil backend service dengan benar"""
        # Setup
        mock_train_loader = MagicMock()
        mock_val_loader = MagicMock()
        mock_config = {'epochs': 10, 'batch_size': 16}
        mock_callbacks = {
            'progress_callback': MagicMock(),
            'metrics_callback': MagicMock(),
            'checkpoint_callback': MagicMock()
        }
        
        # Setup mock untuk _get_data_loaders dan _get_training_config
        self.adapter._get_data_loaders = MagicMock(return_value=(mock_train_loader, mock_val_loader))
        self.adapter._get_training_config = MagicMock(return_value=mock_config)
        
        # Setup model_manager
        self.mock_backend_service.model_manager = MagicMock()
        
        # Panggil method yang diuji
        self.adapter.set_progress_callbacks(
            mock_callbacks['progress_callback'],
            mock_callbacks['metrics_callback'],
            mock_callbacks['checkpoint_callback']
        )
        self.adapter.start_training()
        
        # Verifikasi
        self.mock_backend_service.train.assert_called_once()
        # Verifikasi callbacks diatur
        self.assertEqual(self.adapter.progress_callback, mock_callbacks['progress_callback'])
        self.assertEqual(self.adapter.metrics_callback, mock_callbacks['metrics_callback'])
        self.assertEqual(self.adapter.checkpoint_callback, mock_callbacks['checkpoint_callback'])
    
    def test_stop_training(self):
        """Test stop_training memanggil backend service dengan benar"""
        # Setup - set training_running ke True
        self.adapter._training_running = True
        
        # Panggil method yang diuji
        self.adapter.stop_training()
        
        # Verifikasi
        self.mock_backend_service.stop_training.assert_called_once()
        self.assertTrue(self.adapter._stop_requested)
    
    def test_reset_training_state(self):
        """Test reset_training_state memanggil backend service dengan benar"""
        # Setup - tambahkan atribut yang diperlukan
        self.adapter._training_running = True
        self.adapter._stop_requested = True
        
        # Panggil method yang diuji
        self.adapter.reset_training_state()
        
        # Verifikasi state direset
        self.assertFalse(self.adapter._training_running)
        self.assertFalse(self.adapter._stop_requested)


class TestTrainingServiceManager(unittest.TestCase):
    """Pengujian untuk TrainingServiceManager"""
    
    def setUp(self):
        """Setup untuk pengujian"""
        # Mock komponen UI
        self.mock_ui_components = {
            'training_service': MagicMock(),
            'model_manager': MagicMock(),
            'status_panel': MagicMock(),
            'metrics_chart': MagicMock(),
            'epoch_progress': MagicMock(),
            'batch_progress': MagicMock(),
            'logger': MagicMock()
        }
        
        # Mock config
        self.mock_config = {
            'training': {
                'epochs': 10,
                'batch_size': 16,
                'learning_rate': 0.001
            }
        }
        
        # Buat manager
        self.manager = TrainingServiceManager(self.mock_ui_components, self.mock_config)
    
    def test_register_model_manager(self):
        """Test register_model_manager menyimpan model manager dengan benar"""
        # Setup
        mock_model_manager = MagicMock()
        mock_model_manager.model_type = 'efficient_optimized'
        
        # Mock untuk _create_training_service
        self.manager._create_training_service = MagicMock()
        
        # Panggil method yang diuji
        self.manager.register_model_manager(mock_model_manager)
        
        # Verifikasi
        self.assertEqual(self.manager.ui_components['model_manager'], mock_model_manager)
        # Verifikasi _create_training_service dipanggil
        self.manager._create_training_service.assert_called_once_with(mock_model_manager)
    
    def test_register_config(self):
        """Test register_config menyimpan config dengan benar"""
        # Setup
        mock_config = {'training': {'epochs': 20}}
        mock_training_service = MagicMock()
        self.manager.ui_components['training_service'] = mock_training_service
        
        # Panggil method yang diuji
        self.manager.register_config(mock_config)
        
        # Verifikasi
        self.assertEqual(self.manager.config, mock_config)
        # Verifikasi config diupdate ke training service
        self.assertEqual(mock_training_service.config, mock_config.get('training', {}))
    
    def test_validate_training_readiness_success(self):
        """Test validate_training_readiness berhasil jika semua komponen tersedia"""
        # Setup
        mock_model_manager = MagicMock()
        mock_model_manager.model = MagicMock()  # Model sudah dibuat
        mock_training_service = MagicMock()
        
        # Tambahkan ke ui_components
        self.manager.ui_components['model_manager'] = mock_model_manager
        self.manager.ui_components['training_service'] = mock_training_service
        
        # Mock _validate_dataset
        self.manager._validate_dataset = MagicMock(return_value=True)
        
        # Panggil method yang diuji
        result = self.manager.validate_training_readiness()
        
        # Verifikasi
        self.assertTrue(result)
    
    def test_validate_training_readiness_failure_no_model_manager(self):
        """Test validate_training_readiness gagal jika model manager tidak tersedia"""
        # Setup
        # Pastikan model_manager tidak ada di ui_components
        if 'model_manager' in self.manager.ui_components:
            del self.manager.ui_components['model_manager']
        
        # Mock _notify_error
        self.manager._notify_error = MagicMock()
        
        # Panggil method yang diuji
        result = self.manager.validate_training_readiness()
        
        # Verifikasi
        self.assertFalse(result)
        # Verifikasi _notify_error dipanggil
        self.manager._notify_error.assert_called_once()
    
    def test_validate_training_readiness_failure_no_training_service(self):
        """Test validate_training_readiness gagal jika training service tidak tersedia"""
        # Setup
        mock_model_manager = MagicMock()
        mock_model_manager.model = MagicMock()
        
        # Tambahkan model_manager ke ui_components
        self.manager.ui_components['model_manager'] = mock_model_manager
        
        # Pastikan training_service tidak ada di ui_components
        if 'training_service' in self.manager.ui_components:
            del self.manager.ui_components['training_service']
        
        # Mock _notify_error
        self.manager._notify_error = MagicMock()
        
        # Panggil method yang diuji
        result = self.manager.validate_training_readiness()
        
        # Verifikasi
        self.assertFalse(result)
        # Verifikasi _notify_error dipanggil
        self.manager._notify_error.assert_called_once()
    
    def test_start_training(self):
        """Test start_training memanggil training service dengan benar"""
        # Setup
        mock_training_service = MagicMock()
        mock_model_manager = MagicMock()
        
        # Tambahkan ke ui_components
        self.manager.ui_components['training_service'] = mock_training_service
        self.manager.ui_components['model_manager'] = mock_model_manager
        
        # Mock validate_training_readiness
        self.manager.validate_training_readiness = MagicMock(return_value=True)
        
        # Panggil method yang diuji
        self.manager.start_training()
        
        # Verifikasi
        mock_training_service.start_training.assert_called_once()
    
    def test_stop_training(self):
        """Test stop_training memanggil training service dengan benar"""
        # Setup
        mock_training_service = MagicMock()
        
        # Tambahkan ke ui_components
        self.manager.ui_components['training_service'] = mock_training_service
        
        # Panggil method yang diuji
        self.manager.stop_training()
        
        # Verifikasi
        mock_training_service.stop_training.assert_called_once()
    
    def test_reset_training_state(self):
        """Test reset_training_state memanggil training service dengan benar"""
        # Setup
        mock_training_service = MagicMock()
        
        # Tambahkan ke ui_components
        self.manager.ui_components['training_service'] = mock_training_service
        
        # Panggil method yang diuji
        self.manager.reset_training_state()
        
        # Verifikasi
        mock_training_service.reset_training_state.assert_called_once()


if __name__ == '__main__':
    unittest.main()
