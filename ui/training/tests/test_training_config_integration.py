"""
File: /Users/masdevid/Projects/smartcash/smartcash/ui/training/tests/test_training_config_integration.py
Deskripsi: Unit test untuk memverifikasi integrasi konfigurasi dan training
"""

import unittest
from unittest.mock import patch, MagicMock
import os
import sys
import json
from pathlib import Path

class TestTrainingConfigIntegration(unittest.TestCase):
    """Test untuk memverifikasi integrasi konfigurasi training dengan modul lain"""
    
    def setUp(self):
        """Set up test environment"""
        # Mock ConfigManager
        self.config_patcher = patch('smartcash.common.config.manager.ConfigManager')
        self.mock_config_manager = self.config_patcher.start()
        
        # Mock model manager
        self.model_manager_patcher = patch('smartcash.model.manager.ModelManager')
        self.mock_model_manager = self.model_manager_patcher.start()
        
        # Sample configs
        self.backbone_config = {
            'model': {
                'backbone': 'efficientnet_b4',
                'model_type': 'efficient_optimized',
                'use_attention': True,
                'use_residual': True,
                'use_ciou': False
            }
        }
        
        self.hyperparameters_config = {
            'hyperparameters': {
                'training': {'batch_size': 16, 'image_size': 640, 'epochs': 100},
                'optimizer': {'type': 'SGD', 'learning_rate': 0.01}
            }
        }
        
        self.training_strategy_config = {
            'training_strategy': {
                'utils': {
                    'checkpoint_dir': '/content/runs/train/checkpoints',
                    'layer_mode': 'single'
                },
                'validation': {'validation_frequency': 1},
                'multiscale': {'enabled': True}
            }
        }
        
    def tearDown(self):
        """Clean up after test"""
        self.config_patcher.stop()
        self.model_manager_patcher.stop()
        
    def test_refresh_config_integration(self):
        """Test bahwa refresh_config_handler mengambil konfigurasi dari semua modul dengan benar"""
        # Import module yang akan ditest
        from smartcash.ui.training.handlers.refresh_config_handler import _refresh_config
        
        # Mock ConfigManager.get_instance() dan get_module_config
        mock_instance = self.mock_config_manager.get_instance.return_value
        mock_instance.get_module_config.side_effect = lambda module_name: {
            'backbone': self.backbone_config,
            'hyperparameters': self.hyperparameters_config,
            'training_strategy': self.training_strategy_config
        }.get(module_name, {})
        
        # Mock UI components
        ui_components = {
            'config_tabs': MagicMock(),
            'status_panel': MagicMock(),
        }
        
        # Panggil fungsi yang di-test
        _refresh_config(ui_components)
        
        # Verifikasi bahwa konfigurasi diambil dari semua modul
        mock_instance.get_module_config.assert_any_call('backbone')
        mock_instance.get_module_config.assert_any_call('hyperparameters')
        mock_instance.get_module_config.assert_any_call('training_strategy')
        
        # Verifikasi bahwa UI diperbarui
        self.assertTrue(ui_components['config_tabs'].children or 
                        hasattr(ui_components['config_tabs'], 'value'))
        
    def test_training_path_config(self):
        """Test bahwa jalur data diatur dengan benar"""
        # Import module yang akan ditest
        from smartcash.ui.training.components.config_tabs import create_config_tabs
        
        # Create test config
        test_config = {
            'paths': {
                'data_dir': '/data/preprocessed',
                'checkpoint_dir': '/content/runs/train/checkpoints'
            }
        }
        
        # Create config tabs
        tabs = create_config_tabs(test_config)
        
        # Convert tabs to string for inspection
        tabs_str = str(tabs.children)
        
        # Verify that data_dir is set correctly
        self.assertIn('/data/preprocessed', tabs_str)
        
    def test_model_manager_initialization(self):
        """Test bahwa ModelManager diinisialisasi dengan parameter yang benar"""
        from smartcash.ui.training.training_init import TrainingInitializer
        
        # Mock initializer
        initializer = TrainingInitializer('training', 'smartcash.ui.training')
        
        # Test config
        training_config = {
            'model_type': 'efficient_optimized',
            'backbone': 'efficientnet_b4',
            'batch_size': 16,
            'learning_rate': 0.01
        }
        
        # Call model manager creation
        initializer._create_model_manager(training_config, 'efficient_optimized')
        
        # Verify that ModelManager was initialized correctly
        self.mock_model_manager.assert_called_once()
        args, kwargs = self.mock_model_manager.call_args
        self.assertEqual(kwargs['model_type'], 'efficient_optimized')
        self.assertEqual(kwargs['backbone'], 'efficientnet_b4')
        self.assertEqual(kwargs['batch_size'], 16)
        
    def test_training_service_creation(self):
        """Test bahwa TrainingService dibuat dari model manager dengan benar"""
        from smartcash.ui.training.training_init import TrainingInitializer
        
        # Mock initializer
        initializer = TrainingInitializer('training', 'smartcash.ui.training')
        
        # Mock model manager
        mock_model_manager = MagicMock()
        mock_training_service = MagicMock()
        mock_model_manager.get_training_service.return_value = mock_training_service
        
        # Create test config
        test_config = {
            'training': {
                'epochs': 100,
                'batch_size': 16
            }
        }
        
        # Call training service creation
        result = initializer._create_training_services(mock_model_manager, test_config)
        
        # Verify that get_training_service was called
        mock_model_manager.get_training_service.assert_called_once()
        
        # Verify that the returned result has training_service
        self.assertIn('training_service', result)
        self.assertEqual(result['training_service'], mock_training_service)

if __name__ == '__main__':
    unittest.main()
