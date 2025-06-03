"""
File: smartcash/ui/training/tests/test_training_initializer_integration.py
Deskripsi: Pengujian integrasi untuk TrainingInitializer dengan TrainingServiceManager
"""

import unittest
from unittest.mock import patch, MagicMock, call
import os
import sys
from pathlib import Path

from smartcash.ui.training.training_init import TrainingInitializer

# Tambahkan mock classes untuk pengujian
class MockTrainingServiceAdapter:
    """Mock class untuk TrainingServiceAdapter"""
    def __init__(self, *args, **kwargs):
        self.backend_service = None
        self.model_manager = None
        self.config = {}
        self.callbacks = {}
    
    def set_progress_callbacks(self, *args):
        pass
        
    def start_training(self, *args, **kwargs):
        pass
        
    def stop_training(self):
        pass
        
    def reset_training(self):
        pass

class MockTrainingServiceManager:
    """Mock class untuk TrainingServiceManager"""
    def __init__(self, *args, **kwargs):
        self.adapter = None
        self.notification_manager = None
        self.config = {}
        self.model_manager = None
    
    def register_adapter(self, adapter):
        self.adapter = adapter
        
    def register_ui_components(self, components):
        pass
        
    def register_config(self, config):
        self.config = config
        
    def register_model_manager(self, manager):
        self.model_manager = manager
        
    def start_training(self, *args, **kwargs):
        pass
        
    def stop_training(self):
        pass
        
    def reset_training(self):
        pass
        
    def validate_training_readiness(self):
        return True

from smartcash.ui.training.services.training_service_manager import TrainingServiceManager
from smartcash.ui.training.adapters.training_service_adapter import TrainingServiceAdapter


class TestTrainingInitializerIntegration(unittest.TestCase):
    """Pengujian integrasi untuk TrainingInitializer dengan TrainingServiceManager"""
    
    def setUp(self):
        """Setup untuk pengujian"""
        # Mock logger
        self.mock_logger = MagicMock()
        
        # Patch CommonInitializer dan konstanta untuk menghindari inisialisasi yang tidak perlu
        with patch('smartcash.ui.utils.common_initializer.CommonInitializer.__init__') as mock_init, \
             patch('smartcash.ui.training.training_init.MODULE_LOGGER_NAME', 'training'), \
             patch('smartcash.ui.training.training_init.TRAINING_LOGGER_NAMESPACE', 'smartcash.ui.training'):
            mock_init.return_value = None
            self.initializer = TrainingInitializer()
        
        # Set logger
        self.initializer.logger = self.mock_logger
        
        # Mock config
        self.mock_config = {
            'training': {
                'model_type': 'efficient_optimized',
                'epochs': 10,
                'batch_size': 16,
                'learning_rate': 0.001
            }
        }
        
        # Mock UI components
        self.mock_ui_components = {
            'status_panel': MagicMock(),
            'metrics_chart': MagicMock(),
            'epoch_progress': MagicMock(),
            'batch_progress': MagicMock(),
            'validation_progress': MagicMock(),
            'logger': self.mock_logger
        }
    
    def test_create_training_manager(self):
        """Test pembuatan TrainingServiceManager dengan konfigurasi yang benar"""
        # Setup mock
        mock_config = {'training': {'batch_size': 16}}
        mock_ui_components = {
            'status_panel': MagicMock(),
            'model_manager': MagicMock(),
            'training_service': MagicMock()
        }
        
        # Patch TrainingServiceManager dengan mock class
        with patch('smartcash.ui.training.services.TrainingServiceManager') as mock_manager_class:
            mock_manager = MagicMock()
            mock_manager_class.return_value = mock_manager
            
            # Call method
            result = self.initializer._create_training_manager(mock_ui_components, mock_config)
            
            # Verify
            mock_manager_class.assert_called_once_with(mock_ui_components, mock_config, self.initializer.logger)
            mock_manager.register_model_manager.assert_called_once_with(mock_ui_components['model_manager'])
            mock_manager.register_config.assert_called_once_with(mock_config)
        
        # Verifikasi hasil adalah manager yang dibuat
        self.assertEqual(result, mock_manager)
    
    def test_create_training_manager_no_model_manager(self):
        """Test _create_training_manager mengembalikan None jika model manager tidak tersedia"""
        # Setup - tidak ada model manager di UI components
        self.mock_ui_components['training_service'] = MagicMock()
        
        # Panggil method yang diuji
        result = self.initializer._create_training_manager(self.mock_ui_components, self.mock_config)
        
        # Verifikasi
        self.assertIsNone(result)
        # Verifikasi warning dicatat
        self.mock_logger.warning.assert_called_once()
    
    def test_create_training_manager_no_training_service(self):
        """Test _create_training_manager mengembalikan None jika training service tidak tersedia"""
        # Setup - tidak ada training service di UI components
        self.mock_ui_components['model_manager'] = MagicMock()
        
        # Panggil method yang diuji
        result = self.initializer._create_training_manager(self.mock_ui_components, self.mock_config)
        
        # Verifikasi
        self.assertIsNone(result)
        # Verifikasi warning dicatat
        self.mock_logger.warning.assert_called_once()
    
    @patch('smartcash.ui.training.adapters.TrainingServiceAdapter')
    def test_create_training_services(self, mock_adapter_class):
        """Test _create_training_services membuat TrainingServiceAdapter dengan benar"""
        # Setup
        mock_model_manager = MagicMock()
        mock_model_manager.model_type = 'efficient_optimized'
        mock_backend_service = MagicMock()
        mock_model_manager.get_training_service.return_value = mock_backend_service
        
        # Setup mock untuk checkpoint service dan metrics tracker
        mock_backend_service.checkpoint_service = MagicMock()
        mock_backend_service.metrics_tracker = MagicMock()
        
        # Setup mock untuk adapter yang dibuat
        mock_adapter = MagicMock()
        mock_adapter_class.return_value = mock_adapter
        
        # Panggil method yang diuji
        result = self.initializer._create_training_services(mock_model_manager, self.mock_config)
        
        # Verifikasi
        # Verifikasi get_training_service dipanggil pada model manager
        mock_model_manager.get_training_service.assert_called_once()
        
        # Verifikasi TrainingServiceAdapter dibuat dengan parameter yang benar
        mock_adapter_class.assert_called_once_with(mock_backend_service, self.initializer.logger)
        
        # Verifikasi hasil berisi adapter dan services yang benar
        self.assertEqual(result['training_service'], mock_adapter)
        self.assertEqual(result['checkpoint_service'], mock_backend_service.checkpoint_service)
        self.assertEqual(result['metrics_tracker'], mock_backend_service.metrics_tracker)
        self.assertEqual(result['backend_training_service'], mock_backend_service)
    
    @patch('smartcash.ui.training.services.TrainingServiceManager')
    @patch('smartcash.ui.training.adapters.TrainingServiceAdapter')
    def test_initialize_model_services(self, mock_adapter_class, mock_manager_class):
        """Test _initialize_model_services membuat model manager, services, dan training manager dengan benar"""
        # Setup
        # Mock untuk _create_model_manager
        mock_model_manager = MagicMock()
        self.initializer._create_model_manager = MagicMock(return_value=mock_model_manager)
        
        # Mock untuk _create_training_services
        mock_services = {
            'training_service': MagicMock(),
            'checkpoint_service': MagicMock(),
            'metrics_tracker': MagicMock(),
            'backend_training_service': MagicMock()
        }
        self.initializer._create_training_services = MagicMock(return_value=mock_services)
        
        # Mock untuk _create_training_manager
        mock_manager = MagicMock()
        self.initializer._create_training_manager = MagicMock(return_value=mock_manager)
        
        # Mock untuk _update_initialization_progress
        self.initializer._update_initialization_progress = MagicMock()
        
        # Panggil method yang diuji
        result = self.initializer._initialize_model_services(self.mock_ui_components, self.mock_config)
        
        # Verifikasi
        # Verifikasi _create_model_manager dipanggil dengan parameter yang benar
        self.initializer._create_model_manager.assert_called_once_with(
            self.mock_config.get('training', {}), 
            self.mock_config.get('training', {}).get('model_type', 'efficient_optimized')
        )
        
        # Verifikasi _create_training_services dipanggil dengan parameter yang benar
        self.initializer._create_training_services.assert_called_once_with(mock_model_manager, self.mock_config)
        
        # Verifikasi _create_training_manager dipanggil dengan parameter yang benar
        self.initializer._create_training_manager.assert_called_once_with(self.mock_ui_components, self.mock_config)
        
        # Verifikasi UI components diupdate dengan model manager, services, dan training manager
        self.assertEqual(result['model_manager'], mock_model_manager)
        self.assertEqual(result['training_manager'], mock_manager)
        for key, value in mock_services.items():
            self.assertEqual(result[key], value)


if __name__ == '__main__':
    unittest.main()
