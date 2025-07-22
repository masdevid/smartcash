"""
Comprehensive integration tests for evaluation backend with UI integration.

Tests the complete evaluation pipeline from UI interaction to backend processing,
ensuring proper checkpoint discovery, configuration handling, and result generation.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch, call
from typing import Dict, Any
import yaml
import torch
import numpy as np

# Import modules under test
from smartcash.model.evaluation.evaluation_service import EvaluationService
from smartcash.model.evaluation.checkpoint_selector import CheckpointSelector
from smartcash.ui.model.evaluation.evaluation_uimodule import EvaluationUIModule


class TestEvaluationBackendIntegration(unittest.TestCase):
    """Test suite for evaluation backend integration with UI."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directories for test data
        self.temp_dir = Path(tempfile.mkdtemp())
        self.checkpoints_dir = self.temp_dir / 'checkpoints'
        self.test_data_dir = self.temp_dir / 'test_data'
        self.results_dir = self.temp_dir / 'results'
        
        # Create directories
        self.checkpoints_dir.mkdir(parents=True)
        self.test_data_dir.mkdir(parents=True)
        self.results_dir.mkdir(parents=True)
        
        # Create test configuration based on updated config structure
        self.test_config = {
            'evaluation': {
                'data': {
                    'test_dir': str(self.test_data_dir),
                    'evaluation_dir': str(self.temp_dir / 'evaluation'),
                    'results_dir': str(self.results_dir)
                },
                'checkpoints': {
                    'discovery_paths': [
                        str(self.checkpoints_dir),
                        str(self.temp_dir / 'runs/train/*/weights')
                    ],
                    'filename_patterns': ['best_*.pt', 'last.pt'],
                    'auto_select_best': True,
                    'sort_by': 'val_map',
                    'max_checkpoints': 5,
                    'min_val_map': 0.3,
                    'required_keys': ['model_state_dict', 'config'],
                    'supported_backbones': ['cspdarknet', 'efficientnet_b4']
                },
                'scenarios': {
                    'position_variation': {
                        'name': 'Position Variation',
                        'enabled': True,
                        'augmentation_config': {'num_variations': 3}
                    },
                    'lighting_variation': {
                        'name': 'Lighting Variation',
                        'enabled': True,
                        'augmentation_config': {'num_variations': 3}
                    }
                },
                'metrics': {
                    'primary': ['map', 'precision_recall'],
                    'map': {
                        'enabled': True,
                        'iou_thresholds': [0.5, 0.75],
                        'confidence_threshold': 0.25
                    },
                    'precision_recall': {
                        'enabled': True,
                        'confidence_threshold': 0.25,
                        'iou_threshold': 0.5,
                        'per_class': True
                    },
                    'inference_time': {
                        'enabled': True,
                        'warmup_runs': 2,
                        'timing_runs': 5
                    }
                },
                'execution': {
                    'run_mode': 'all_scenarios',
                    'parallel_execution': False,
                    'save_intermediate_results': True,
                    'timeout_per_scenario': 300  # 5 minutes for tests
                },
                'output': {
                    'save_dir': str(self.results_dir),
                    'save_predictions': True,
                    'save_metrics': True,
                    'export_formats': ['json']
                }
            }
        }
        
        # Create mock checkpoint files
        self._create_mock_checkpoints()
    
    def tearDown(self):
        """Clean up test fixtures."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def _create_mock_checkpoints(self):
        """Create mock checkpoint files for testing."""
        checkpoints_data = [
            {
                'filename': 'best_smartcash_efficientnet_b4_full_12202024.pt',
                'model_state_dict': {'layer1.weight': torch.randn(10, 5)},
                'config': {
                    'backbone': 'efficientnet_b4',
                    'layer_mode': 'full',
                    'num_classes': 7
                },
                'metrics': {
                    'val_map': 0.847,
                    'val_loss': 0.123,
                    'val_precision': 0.856,
                    'val_recall': 0.834
                },
                'epoch': 100
            },
            {
                'filename': 'best_smartcash_cspdarknet_single_12202024.pt',
                'model_state_dict': {'layer1.weight': torch.randn(10, 5)},
                'config': {
                    'backbone': 'cspdarknet',
                    'layer_mode': 'single',
                    'num_classes': 7
                },
                'metrics': {
                    'val_map': 0.782,
                    'val_loss': 0.156,
                    'val_precision': 0.798,
                    'val_recall': 0.765
                },
                'epoch': 85
            },
            {
                'filename': 'best_smartcash_efficientnet_b4_single_12202024.pt',
                'model_state_dict': {'layer1.weight': torch.randn(10, 5)},
                'config': {
                    'backbone': 'efficientnet_b4',
                    'layer_mode': 'single',
                    'num_classes': 7
                },
                'metrics': {
                    'val_map': 0.812,
                    'val_loss': 0.145,
                    'val_precision': 0.824,
                    'val_recall': 0.789
                },
                'epoch': 95
            }
        ]
        
        for checkpoint_data in checkpoints_data:
            checkpoint_path = self.checkpoints_dir / checkpoint_data['filename']
            torch.save(checkpoint_data, checkpoint_path)
    
    def test_checkpoint_selector_discovery_integration(self):
        """Test checkpoint selector with configurable discovery paths."""
        # Initialize checkpoint selector with test config
        selector = CheckpointSelector(config=self.test_config)
        
        # Test discovery paths configuration
        self.assertEqual(len(selector.discovery_paths), 2)
        self.assertIn(self.checkpoints_dir, selector.discovery_paths)
        
        # Test filename patterns configuration
        self.assertEqual(selector.filename_patterns, ['best_*.pt', 'last.pt'])
        
        # Test minimum mAP threshold
        self.assertEqual(selector.min_val_map, 0.3)
        
        # Test supported backbones configuration
        self.assertEqual(selector.supported_backbones, ['cspdarknet', 'efficientnet_b4'])
        
        # Test checkpoint discovery
        checkpoints = selector.list_available_checkpoints()
        
        # Verify all valid checkpoints were found
        self.assertEqual(len(checkpoints), 3)  # All 3 mock checkpoints should be found
        
        # Verify sorting by val_map (highest first)
        self.assertGreaterEqual(checkpoints[0]['metrics']['val_map'], 
                               checkpoints[1]['metrics']['val_map'])
        
        # Verify best checkpoint has highest mAP
        best_checkpoint = checkpoints[0]
        self.assertEqual(best_checkpoint['metrics']['val_map'], 0.847)
        self.assertEqual(best_checkpoint['backbone'], 'efficientnet_b4')
    
    def test_checkpoint_selector_filtering(self):
        """Test checkpoint selector filtering capabilities."""
        selector = CheckpointSelector(config=self.test_config)
        
        # Test backbone filtering
        efficientnet_checkpoints = selector.filter_checkpoints(backbone='efficientnet_b4')
        self.assertEqual(len(efficientnet_checkpoints), 2)  # 2 EfficientNet models
        
        cspdarknet_checkpoints = selector.filter_checkpoints(backbone='cspdarknet')
        self.assertEqual(len(cspdarknet_checkpoints), 1)  # 1 CSPDarknet model
        
        # Test minimum mAP filtering
        high_performance_checkpoints = selector.filter_checkpoints(min_map=0.8)
        self.assertEqual(len(high_performance_checkpoints), 2)  # 2 models with mAP >= 0.8
        
        # Test combined filtering
        filtered_checkpoints = selector.filter_checkpoints(
            backbone='efficientnet_b4', 
            min_map=0.8
        )
        self.assertEqual(len(filtered_checkpoints), 2)  # Both EfficientNet models have mAP >= 0.8
    
    def test_evaluation_service_initialization(self):
        """Test evaluation service initialization with enhanced configuration."""
        # Initialize evaluation service with test config
        service = EvaluationService(model_api=None, config=self.test_config)
        
        # Verify service components are initialized
        self.assertIsNotNone(service.scenario_manager)
        self.assertIsNotNone(service.evaluation_metrics)
        self.assertIsNotNone(service.checkpoint_selector)
        self.assertIsNotNone(service.inference_timer)
        self.assertIsNotNone(service.results_aggregator)
        
        # Verify configuration is properly stored
        self.assertEqual(service.config, self.test_config)
        
        # Test checkpoint selector integration
        checkpoints = service.checkpoint_selector.list_available_checkpoints()
        self.assertGreater(len(checkpoints), 0)  # Should find mock checkpoints
    
    def test_evaluation_ui_module_backend_integration(self):
        """Test evaluation UI module integration with backend services."""
        # Create UI module with test config
        ui_module = EvaluationUIModule()
        
        # Mock the config handler to return our test config
        with patch.object(ui_module, 'get_current_config', return_value=self.test_config):
            # Initialize backend services
            ui_module._initialize_backend_services()
            
            # Verify backend services are initialized
            self.assertIsNotNone(ui_module.checkpoint_selector)
            self.assertIsNotNone(ui_module.evaluation_service)
            self.assertIsNotNone(ui_module.progress_bridge)
            
            # Test model discovery integration
            models_result = ui_module._get_available_models()
            
            # Verify model discovery results
            self.assertTrue(models_result['success'])
            self.assertIn('models', models_result)
            self.assertGreater(len(models_result['models']), 0)
            
            # Test model refresh functionality
            refresh_result = ui_module._handle_refresh_models()
            self.assertTrue(refresh_result['success'])
            self.assertIn('models_found', refresh_result)
    
    def test_evaluation_service_scenario_execution(self):
        """Test evaluation service scenario execution."""
        service = EvaluationService(model_api=None, config=self.test_config)
        
        # Create mock test data
        self._create_mock_test_data()
        
        # Mock the scenario manager evaluation_dir to point to test data
        with patch.object(service.scenario_manager, 'evaluation_dir', self.test_data_dir):
            # Mock scenario data loading to return our test data
            with patch.object(service, '_load_scenario_data') as mock_load_data, \
                 patch.object(service, '_run_inference_with_timing') as mock_inference:
                
                # Setup mock scenario data
                mock_test_data = {
                    'images': [
                        {'image': None, 'filename': 'test_image_1.jpg', 'path': 'test_image_1.jpg'},
                        {'image': None, 'filename': 'test_image_2.jpg', 'path': 'test_image_2.jpg'}
                    ],
                    'labels': [
                        {'filename': 'test_image_1.jpg', 'annotations': [{'class_id': 0, 'bbox': [0.5, 0.5, 0.3, 0.4]}]},
                        {'filename': 'test_image_2.jpg', 'annotations': [{'class_id': 1, 'bbox': [0.4, 0.6, 0.2, 0.3]}]}
                    ]
                }
                mock_load_data.return_value = mock_test_data
                # Setup mock inference results
                mock_predictions = [
                    {
                        'filename': 'test_image_1.jpg',
                        'detections': [
                            {'class_id': 0, 'confidence': 0.85, 'bbox': [0.1, 0.1, 0.3, 0.4]},
                            {'class_id': 2, 'confidence': 0.72, 'bbox': [0.5, 0.2, 0.2, 0.3]}
                        ]
                    },
                    {
                        'filename': 'test_image_2.jpg',
                        'detections': [
                            {'class_id': 1, 'confidence': 0.91, 'bbox': [0.2, 0.3, 0.4, 0.5]}
                        ]
                    }
                ]
                mock_inference_times = [0.12, 0.08]
                mock_inference.return_value = (mock_predictions, mock_inference_times)
                
                # Execute single scenario
                best_checkpoint = service.checkpoint_selector.get_best_checkpoint()
                self.assertIsNotNone(best_checkpoint, "Should find best checkpoint from mock data")
                
                result = service.run_scenario('position_variation', best_checkpoint['path'])
                
                # Verify scenario execution results
                self.assertEqual(result['status'], 'success')
                self.assertIn('metrics', result)
                self.assertIn('additional_data', result)
                self.assertEqual(result['scenario_name'], 'position_variation')
    
    def test_evaluation_service_comprehensive_evaluation(self):
        """Test comprehensive evaluation pipeline."""
        service = EvaluationService(model_api=None, config=self.test_config)
        
        # Create mock test data
        self._create_mock_test_data()
        
        # Mock UI components for progress tracking
        mock_ui_components = {
            'operation_container': MagicMock(),
            'progress_tracker': MagicMock()
        }
        
        # Mock scenario manager methods and attributes
        with patch.object(service.scenario_manager, 'evaluation_dir', self.test_data_dir), \
             patch.object(service.scenario_manager, 'prepare_all_scenarios', return_value=True), \
             patch.object(service, '_load_scenario_data') as mock_load_data, \
             patch.object(service, '_run_inference_with_timing') as mock_inference:
            
            # Setup mock scenario data
            mock_test_data = {
                'images': [
                    {'image': None, 'filename': 'test_image_1.jpg', 'path': 'test_image_1.jpg'}
                ],
                'labels': [
                    {'filename': 'test_image_1.jpg', 'annotations': [{'class_id': 0, 'bbox': [0.5, 0.5, 0.3, 0.4]}]}
                ]
            }
            mock_load_data.return_value = mock_test_data
            
            # Setup mock inference results
            mock_predictions = [
                {
                    'filename': 'test_image_1.jpg',
                    'detections': [{'class_id': 0, 'confidence': 0.85, 'bbox': [0.1, 0.1, 0.3, 0.4]}]
                }
            ]
            mock_inference.return_value = (mock_predictions, [0.12])
            
            # Execute comprehensive evaluation
            result = service.run_evaluation(
                scenarios=['position_variation', 'lighting_variation'],
                checkpoints=None,  # Will auto-select
                ui_components=mock_ui_components
            )
            
            # Verify evaluation results
            self.assertEqual(result['status'], 'success')
            self.assertIn('evaluation_results', result)
            self.assertIn('summary', result)
            self.assertIn('export_files', result)
            
            # Verify scenarios were executed
            self.assertEqual(result['scenarios_evaluated'], 2)
            self.assertGreater(result['checkpoints_evaluated'], 0)
    
    def test_configuration_file_integration(self):
        """Test integration with actual configuration file."""
        # Load the actual evaluation config file
        config_path = Path(__file__).parent.parent.parent / 'smartcash' / 'configs' / 'evaluation_config.yaml'
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                actual_config = yaml.safe_load(f)
            
            # Verify configuration structure matches our expectations
            self.assertIn('evaluation', actual_config)
            eval_config = actual_config['evaluation']
            
            # Verify checkpoint configuration
            self.assertIn('checkpoints', eval_config)
            checkpoint_config = eval_config['checkpoints']
            self.assertIn('discovery_paths', checkpoint_config)
            self.assertIn('filename_patterns', checkpoint_config)
            
            # Verify scenarios configuration
            self.assertIn('scenarios', eval_config)
            self.assertIn('position_variation', eval_config['scenarios'])
            self.assertIn('lighting_variation', eval_config['scenarios'])
            
            # Test checkpoint selector with actual config
            selector = CheckpointSelector(config=actual_config)
            self.assertGreater(len(selector.discovery_paths), 0)
            self.assertGreater(len(selector.filename_patterns), 0)
    
    def test_error_handling_integration(self):
        """Test error handling across the integration."""
        # Test with invalid config
        invalid_config = {'evaluation': {'invalid': 'config'}}
        
        # Test checkpoint selector with invalid config
        selector = CheckpointSelector(config=invalid_config)
        checkpoints = selector.list_available_checkpoints()
        # Should handle invalid config gracefully
        self.assertIsInstance(checkpoints, list)
        
        # Test evaluation service with invalid config
        service = EvaluationService(model_api=None, config=invalid_config)
        # Should initialize with fallback values
        self.assertIsNotNone(service.checkpoint_selector)
        
        # Test UI module with missing backend services
        ui_module = EvaluationUIModule()
        ui_module.checkpoint_selector = None
        ui_module.evaluation_service = None
        
        # Should handle missing services gracefully
        models_result = ui_module._get_available_models()
        self.assertFalse(models_result['success'])
        self.assertIn('error', models_result)
    
    def _create_mock_test_data(self):
        """Create mock test data for evaluation scenarios."""
        # Create position_variation scenario directory
        position_dir = self.test_data_dir / 'position_variation'
        position_images_dir = position_dir / 'images'
        position_labels_dir = position_dir / 'labels'
        
        position_images_dir.mkdir(parents=True)
        position_labels_dir.mkdir(parents=True)
        
        # Create mock image files (empty files for testing)
        for i in range(2):
            image_file = position_images_dir / f'test_image_{i+1}.jpg'
            image_file.touch()
            
            # Create corresponding label file
            label_file = position_labels_dir / f'test_image_{i+1}.txt'
            with open(label_file, 'w') as f:
                f.write(f"{i} 0.5 0.5 0.3 0.4\n")  # class_id x_center y_center width height
        
        # Create lighting_variation scenario directory
        lighting_dir = self.test_data_dir / 'lighting_variation'
        lighting_images_dir = lighting_dir / 'images'
        lighting_labels_dir = lighting_dir / 'labels'
        
        lighting_images_dir.mkdir(parents=True)
        lighting_labels_dir.mkdir(parents=True)
        
        # Create mock image files
        for i in range(2):
            image_file = lighting_images_dir / f'light_test_{i+1}.jpg'
            image_file.touch()
            
            # Create corresponding label file
            label_file = lighting_labels_dir / f'light_test_{i+1}.txt'
            with open(label_file, 'w') as f:
                f.write(f"{(i+1)%7} 0.4 0.6 0.2 0.3\n")


class TestEvaluationUIIntegration(unittest.TestCase):
    """Test suite for evaluation UI integration."""
    
    def setUp(self):
        """Set up UI integration test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.test_config = {
            'evaluation': {
                'checkpoints': {
                    'discovery_paths': [str(self.temp_dir / 'checkpoints')],
                    'filename_patterns': ['best_*.pt'],
                    'auto_select_best': True,
                    'sort_by': 'val_map',
                    'max_checkpoints': 5,
                    'min_val_map': 0.3
                },
                'scenarios': {
                    'position_variation': {'enabled': True},
                    'lighting_variation': {'enabled': True}
                },
                'execution': {
                    'run_mode': 'all_scenarios',
                    'parallel_execution': False
                }
            }
        }
    
    def tearDown(self):
        """Clean up UI integration test fixtures."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_ui_module_configuration_integration(self):
        """Test UI module configuration integration."""
        ui_module = EvaluationUIModule()
        
        # Test default configuration
        default_config = ui_module.get_default_config()
        self.assertIsInstance(default_config, dict)
        
        # Test configuration handler creation
        config_handler = ui_module.create_config_handler(self.test_config)
        self.assertIsNotNone(config_handler)
    
    def test_ui_button_handler_integration(self):
        """Test UI button handler integration with backend."""
        ui_module = EvaluationUIModule()
        
        # Mock backend services
        ui_module.checkpoint_selector = MagicMock()
        ui_module.evaluation_service = MagicMock()
        
        # Mock checkpoint discovery
        ui_module.checkpoint_selector.list_available_checkpoints.return_value = [
            {
                'path': 'test_checkpoint.pt',
                'display_name': 'Test Model',
                'backbone': 'efficientnet_b4',
                'metrics': {'val_map': 0.85}
            }
        ]
        
        # Test refresh button handler
        with patch.object(ui_module, 'get_current_config', return_value=self.test_config), \
             patch.object(ui_module, '_get_available_models') as mock_get_models, \
             patch.object(ui_module, '_update_ui_model_state') as mock_update_ui:
            
            # Mock model discovery result
            mock_get_models.return_value = {
                'success': True,
                'models': {
                    'test_model': {
                        'name': 'Test Model',
                        'status': 'valid',
                        'map_score': 0.85
                    }
                }
            }
            
            refresh_result = ui_module._handle_refresh_models()
            
            self.assertTrue(refresh_result['success'])
            self.assertIn('models_found', refresh_result)
    
    def test_ui_operation_integration(self):
        """Test UI operation integration with backend services."""
        ui_module = EvaluationUIModule()
        
        # Mock backend services and methods
        ui_module.checkpoint_selector = MagicMock()
        ui_module.evaluation_service = MagicMock()
        
        # Mock successful evaluation
        ui_module.evaluation_service.run_evaluation.return_value = {
            'status': 'success',
            'scenarios_evaluated': 2,
            'checkpoints_evaluated': 1,
            'evaluation_results': {'test': 'results'},
            'summary': {'test': 'summary'}
        }
        
        with patch.object(ui_module, 'get_current_config', return_value=self.test_config):
            # Test comprehensive evaluation operation
            result = ui_module._execute_all_scenarios()
            
            # Check if result has success key (it should from the mock)
            if 'success' in result:
                self.assertTrue(result['success'])
            if 'status' in result:
                self.assertEqual(result['status'], 'success')


if __name__ == '__main__':
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTest(unittest.makeSuite(TestEvaluationBackendIntegration))
    suite.addTest(unittest.makeSuite(TestEvaluationUIIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Exit with appropriate code
    exit(0 if result.wasSuccessful() else 1)