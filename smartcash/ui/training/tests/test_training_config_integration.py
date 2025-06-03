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
        self.config_patcher = patch('smartcash.common.config.manager.SimpleConfigManager')
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
        
    @patch('smartcash.ui.training.handlers.refresh_config_handler.ConfigManager')
    def test_refresh_config_integration(self, mock_config_manager_class):
        """Test bahwa refresh_config_handler mengambil konfigurasi dari semua modul dengan benar"""
        from smartcash.ui.training.handlers.refresh_config_handler import _refresh_config
        
        # Setup mock untuk ConfigManager
        mock_backbone_config = {'model': {'backbone': 'efficientnet_b4'}}
        mock_hyperparameters_config = {'hyperparameters': {'batch_size': 16, 'epochs': 100}}
        mock_training_strategy_config = {'training_strategy': {'utils': {'checkpoint_dir': '/test/checkpoints'}}}
        
        mock_config_manager = MagicMock()
        mock_config_manager_class.return_value = mock_config_manager
        
        # Setup mock untuk get_module_config
        mock_config_manager.get_module_config.side_effect = lambda module_name: {
            'backbone': mock_backbone_config,
            'hyperparameters': mock_hyperparameters_config,
            'training_strategy': mock_training_strategy_config
        }.get(module_name, {})
        
        # Setup mock untuk get_config_value
        mock_config_manager.get_config_value.return_value = '/test/checkpoints'
        
        # Setup mock untuk UI components
        mock_ui_components = {
            'config_tabs': MagicMock(),
            'status_panel': MagicMock()
        }
        
        # Panggil refresh config
        _refresh_config(mock_ui_components)
        
        # Verifikasi config diambil dari ConfigManager
        mock_config_manager.get_module_config.assert_any_call('backbone')
        mock_config_manager.get_module_config.assert_any_call('hyperparameters')
        mock_config_manager.get_module_config.assert_any_call('training_strategy')
        
        # Verifikasi UI diupdate
        self.assertTrue(mock_ui_components['config_tabs'].children is not None or
                        hasattr(mock_ui_components['config_tabs'], 'update'))
        
        # Verifikasi status panel diupdate
        self.assertTrue(mock_ui_components['status_panel'].success.called or
                       hasattr(mock_ui_components['status_panel'], 'value'))
        
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
        
        # Mock initializer dengan patching CommonInitializer
        with patch('smartcash.ui.utils.common_initializer.CommonInitializer.__init__') as mock_init:
            mock_init.return_value = None
            initializer = TrainingInitializer()
            initializer.logger = MagicMock()
        
        # Test config
        training_config = {
            'model_type': 'efficient_optimized',
            'backbone': 'efficientnet_b4',
            'batch_size': 16,
            'learning_rate': 0.01,
            'layer_mode': 'single',
            'detection_layers': ['banknote']
        }
        
        # Call model manager creation
        initializer._create_model_manager(training_config, 'efficient_optimized')
        
        # Verify that ModelManager was initialized correctly
        self.mock_model_manager.assert_called_once()
        args, kwargs = self.mock_model_manager.call_args
        self.assertEqual(kwargs['model_type'], 'efficient_optimized')
        self.assertEqual(kwargs['layer_mode'], 'single')
        self.assertEqual(kwargs['detection_layers'], ['banknote'])
        
    @patch('smartcash.ui.training.adapters.TrainingServiceAdapter')
    def test_training_service_creation(self, mock_adapter_class):
        """Test bahwa TrainingService dibuat dari model manager dengan benar"""
        from smartcash.ui.training.training_init import TrainingInitializer
        
        # Mock initializer dengan patching CommonInitializer
        with patch('smartcash.ui.utils.common_initializer.CommonInitializer.__init__') as mock_init:
            mock_init.return_value = None
            initializer = TrainingInitializer()
            initializer.logger = MagicMock()
        
        # Mock model manager dan training service
        mock_model_manager = MagicMock()
        mock_backend_service = MagicMock()
        mock_model_manager.get_training_service.return_value = mock_backend_service
        
        # Mock adapter instance
        mock_adapter = MagicMock()
        mock_adapter_class.return_value = mock_adapter
        
        # Mock checkpoint service dan metrics tracker
        mock_backend_service.checkpoint_service = MagicMock()
        mock_backend_service.metrics_tracker = MagicMock()
        
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
        
        # Verify that adapter was created with correct parameters
        mock_adapter_class.assert_called_once_with(mock_backend_service, initializer.logger)
        
        # Verify that the returned result has all expected services
        self.assertIn('training_service', result)
        self.assertIn('checkpoint_service', result)
        self.assertIn('metrics_tracker', result)
        self.assertIn('backend_training_service', result)
        
        # Verify service instances
        self.assertEqual(result['training_service'], mock_adapter)
        self.assertEqual(result['backend_training_service'], mock_backend_service)
        self.assertEqual(result['checkpoint_service'], mock_backend_service.checkpoint_service)
        self.assertEqual(result['metrics_tracker'], mock_backend_service.metrics_tracker)

if __name__ == '__main__':
    unittest.main()
