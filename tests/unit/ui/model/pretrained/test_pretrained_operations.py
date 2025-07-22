"""
Comprehensive operation functionality tests for PretrainedUIModule.

This test suite focuses on testing all operations (download, validate, refresh, cleanup)
to ensure they work correctly with proper error handling.
"""

import unittest
from unittest.mock import MagicMock, patch, call, ANY
from typing import Dict, Any
from datetime import datetime
import os
import tempfile

# Import test base class
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from test_pretrained_uimodule import MockPretrainedUIModule


class TestPretrainedOperations(unittest.TestCase):
    """Test suite for pretrained module operations."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a fresh mock module instance
        self.module = MockPretrainedUIModule()
        
        # Set up mock UI components
        self.mock_ui_components = {
            'main_container': MagicMock(),
            'header_container': MagicMock(),
            'form_container': MagicMock(),
            'action_container': MagicMock(),
            'operation_container': MagicMock(),
            'progress_tracker': MagicMock(),
            'log_accordion': MagicMock()
        }
        
        # Set up module attributes
        self.module._ui_components = self.mock_ui_components
        self.module._is_initialized = True
        
        # Mock config
        self.test_config = {
            'models_dir': '/tmp/test_models',
            'pretrained_models': ['yolov5s', 'efficientnet_b4'],
            'model_urls': {
                'yolov5s': 'https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5s.pt',
                'efficientnet_b4': 'https://example.com/efficientnet_b4.pth'
            },
            'auto_download': True,
            'validate_downloads': True
        }
        
        # Mock the get_current_config method
        self.module.get_current_config = MagicMock(return_value=self.test_config)
        
        # Mock the get_component method
        self.module.get_component = MagicMock(side_effect=lambda name: self.mock_ui_components.get(name))
        
        # Mock the logging
        self.module.log = MagicMock()
    
    def test_download_operation_success(self):
        """Test successful download operation."""
        # Mock successful download result
        mock_download_result = {
            'success': True,
            'message': 'Models downloaded successfully',
            'downloaded_files': ['yolov5s.pt', 'efficientnet_b4.pth'],
            'download_time': '2024-01-01T12:00:00',
            'total_size': '100MB'
        }
        
        with patch('smartcash.ui.model.pretrained.pretrained_uimodule.PretrainedOperationFactory.execute_operation',
                  return_value=mock_download_result) as mock_factory:
            
            # Execute download operation
            result = self.module._execute_download_operation(self.test_config)
            
            # Verify the result
            self.assertTrue(result['success'])
            self.assertEqual(result['message'], 'Models downloaded successfully')
            self.assertEqual(len(result['downloaded_files']), 2)
            self.assertIn('yolov5s.pt', result['downloaded_files'])
            self.assertIn('efficientnet_b4.pth', result['downloaded_files'])
            
            # Verify factory was called with correct parameters
            mock_factory.assert_called_once()
            call_args = mock_factory.call_args[1]
            self.assertEqual(call_args['operation_type'], 'download')
            self.assertEqual(call_args['config']['models_dir'], '/tmp/test_models')
            self.assertTrue(call_args['config']['auto_download'])
    
    def test_download_operation_failure(self):
        """Test download operation failure handling."""
        # Mock download failure
        with patch('smartcash.ui.model.pretrained.pretrained_uimodule.PretrainedOperationFactory.execute_operation',
                  side_effect=Exception('Network error')) as mock_factory:
            
            # Execute download operation
            result = self.module._execute_download_operation(self.test_config)
            
            # Verify error handling
            self.assertFalse(result['success'])
            self.assertIn('error', result)
            self.assertIn('Network error', result['error'])
            self.assertEqual(result['message'], 'Failed to execute download operation')
            
            # Verify error was logged
            self.module.log.assert_called_with(
                'Error in download operation: Network error', 'error'
            )
    
    def test_validate_operation_success(self):
        """Test successful validation operation."""
        # Mock successful validation result
        mock_validation_result = {
            'success': True,
            'message': 'All models validated successfully',
            'validation_results': {
                'yolov5s.pt': {
                    'valid': True,
                    'size': '14.1MB',
                    'checksum': 'abc123def456'
                },
                'efficientnet_b4.pth': {
                    'valid': True,
                    'size': '19.3MB',
                    'checksum': 'def456ghi789'
                }
            },
            'total_models': 2,
            'valid_models': 2,
            'invalid_models': 0
        }
        
        with patch('smartcash.ui.model.pretrained.pretrained_uimodule.PretrainedOperationFactory.execute_operation',
                  return_value=mock_validation_result) as mock_factory:
            
            # Execute validation operation
            result = self.module._execute_validate_operation()
            
            # Verify the result
            self.assertTrue(result['success'])
            self.assertEqual(result['message'], 'All models validated successfully')
            self.assertEqual(result['total_models'], 2)
            self.assertEqual(result['valid_models'], 2)
            self.assertEqual(result['invalid_models'], 0)
            
            # Verify all models are valid
            for model_name, model_result in result['validation_results'].items():
                self.assertTrue(model_result['valid'])
                self.assertIn('size', model_result)
                self.assertIn('checksum', model_result)
            
            # Verify factory was called correctly
            mock_factory.assert_called_once()
            call_args = mock_factory.call_args[1]
            self.assertEqual(call_args['operation_type'], 'validate')
    
    def test_validate_operation_partial_failure(self):
        """Test validation operation with some invalid models."""
        # Mock partial validation failure
        mock_validation_result = {
            'success': True,
            'message': 'Validation completed with issues',
            'validation_results': {
                'yolov5s.pt': {
                    'valid': True,
                    'size': '14.1MB',
                    'checksum': 'abc123def456'
                },
                'efficientnet_b4.pth': {
                    'valid': False,
                    'error': 'Corrupted file',
                    'size': '0MB'
                }
            },
            'total_models': 2,
            'valid_models': 1,
            'invalid_models': 1
        }
        
        with patch('smartcash.ui.model.pretrained.pretrained_uimodule.PretrainedOperationFactory.execute_operation',
                  return_value=mock_validation_result) as mock_factory:
            
            # Execute validation operation
            result = self.module._execute_validate_operation()
            
            # Verify the result
            self.assertTrue(result['success'])  # Operation succeeded but found issues
            self.assertEqual(result['total_models'], 2)
            self.assertEqual(result['valid_models'], 1)
            self.assertEqual(result['invalid_models'], 1)
            
            # Verify specific model results
            yolo_result = result['validation_results']['yolov5s.pt']
            self.assertTrue(yolo_result['valid'])
            
            efficientnet_result = result['validation_results']['efficientnet_b4.pth']
            self.assertFalse(efficientnet_result['valid'])
            self.assertIn('error', efficientnet_result)
    
    def test_refresh_operation_success(self):
        """Test successful refresh operation."""
        # Mock successful refresh result
        mock_refresh_result = {
            'success': True,
            'message': 'Model status refreshed successfully',
            'models_found': ['yolov5s.pt', 'efficientnet_b4.pth'],
            'validation_results': {
                'yolov5s.pt': True,
                'efficientnet_b4.pth': True
            },
            'refresh_time': '2024-01-01T12:00:00',
            'last_refresh': '2024-01-01T12:00:00',
            'directory_status': {
                'exists': True,
                'readable': True,
                'total_files': 2
            }
        }
        
        with patch('smartcash.ui.model.pretrained.pretrained_uimodule.PretrainedOperationFactory.execute_operation',
                  return_value=mock_refresh_result) as mock_factory:
            
            # Execute refresh operation
            result = self.module._execute_refresh_operation()
            
            # Verify the result
            self.assertTrue(result['success'])
            self.assertEqual(result['message'], 'Model status refreshed successfully')
            self.assertEqual(len(result['models_found']), 2)
            self.assertIn('yolov5s.pt', result['models_found'])
            self.assertIn('efficientnet_b4.pth', result['models_found'])
            
            # Verify model status was updated in module
            self.assertEqual(self.module._model_status['models_found'], result['models_found'])
            self.assertEqual(self.module._model_status['validation_results'], result['validation_results'])
            self.assertEqual(self.module._model_status['last_refresh'], result['last_refresh'])
    
    def test_refresh_operation_empty_directory(self):
        """Test refresh operation with empty models directory."""
        # Mock refresh result for empty directory
        mock_refresh_result = {
            'success': True,
            'message': 'No models found in directory',
            'models_found': [],
            'validation_results': {},
            'refresh_time': '2024-01-01T12:00:00',
            'last_refresh': '2024-01-01T12:00:00',
            'directory_status': {
                'exists': True,
                'readable': True,
                'total_files': 0
            }
        }
        
        with patch('smartcash.ui.model.pretrained.pretrained_uimodule.PretrainedOperationFactory.execute_operation',
                  return_value=mock_refresh_result) as mock_factory:
            
            # Execute refresh operation
            result = self.module._execute_refresh_operation()
            
            # Verify the result
            self.assertTrue(result['success'])
            self.assertEqual(len(result['models_found']), 0)
            self.assertEqual(len(result['validation_results']), 0)
            
            # Verify model status reflects empty state
            self.assertEqual(len(self.module._model_status['models_found']), 0)
    
    def test_cleanup_operation_success(self):
        """Test successful cleanup operation."""
        # Mock successful cleanup result
        mock_cleanup_result = {
            'success': True,
            'message': 'Cleanup completed successfully',
            'cleaned_files': ['corrupted_model.pt', 'incomplete_download.tmp'],
            'space_freed': '50MB',
            'remaining_models': ['yolov5s.pt', 'efficientnet_b4.pth'],
            'cleanup_time': '2024-01-01T12:00:00'
        }
        
        with patch('smartcash.ui.model.pretrained.pretrained_uimodule.PretrainedOperationFactory.execute_operation',
                  return_value=mock_cleanup_result) as mock_factory:
            
            # Execute cleanup operation
            result = self.module._execute_cleanup_operation()
            
            # Verify the result
            self.assertTrue(result['success'])
            self.assertEqual(result['message'], 'Cleanup completed successfully')
            self.assertEqual(len(result['cleaned_files']), 2)
            self.assertEqual(result['space_freed'], '50MB')
            self.assertEqual(len(result['remaining_models']), 2)
            
            # Verify factory was called correctly
            mock_factory.assert_called_once()
            call_args = mock_factory.call_args[1]
            self.assertEqual(call_args['operation_type'], 'cleanup')
    
    def test_cleanup_operation_nothing_to_clean(self):
        """Test cleanup operation when no cleanup is needed."""
        # Mock cleanup result with nothing to clean
        mock_cleanup_result = {
            'success': True,
            'message': 'No files needed cleanup',
            'cleaned_files': [],
            'space_freed': '0MB',
            'remaining_models': ['yolov5s.pt', 'efficientnet_b4.pth'],
            'cleanup_time': '2024-01-01T12:00:00'
        }
        
        with patch('smartcash.ui.model.pretrained.pretrained_uimodule.PretrainedOperationFactory.execute_operation',
                  return_value=mock_cleanup_result) as mock_factory:
            
            # Execute cleanup operation
            result = self.module._execute_cleanup_operation()
            
            # Verify the result
            self.assertTrue(result['success'])
            self.assertEqual(len(result['cleaned_files']), 0)
            self.assertEqual(result['space_freed'], '0MB')
    
    def test_operation_with_custom_config(self):
        """Test operations with custom configuration overrides."""
        # Custom config for test
        custom_config = {
            'models_dir': '/custom/path/models',
            'auto_download': False,
            'validate_downloads': False
        }
        
        mock_result = {
            'success': True,
            'message': 'Operation completed with custom config'
        }
        
        with patch('smartcash.ui.model.pretrained.pretrained_uimodule.PretrainedOperationFactory.execute_operation',
                  return_value=mock_result) as mock_factory:
            
            # Execute operation with custom config
            result = self.module._execute_pretrained_operation('download', custom_config)
            
            # Verify the result
            self.assertTrue(result['success'])
            
            # Verify custom config was merged
            call_args = mock_factory.call_args[1]
            final_config = call_args['config']
            self.assertEqual(final_config['models_dir'], '/custom/path/models')
            self.assertFalse(final_config['auto_download'])
            self.assertFalse(final_config['validate_downloads'])
    
    def test_operation_ui_component_integration(self):
        """Test that operations properly integrate with UI components."""
        mock_result = {
            'success': True,
            'message': 'Operation completed'
        }
        
        with patch('smartcash.ui.model.pretrained.pretrained_uimodule.PretrainedOperationFactory.execute_operation',
                  return_value=mock_result) as mock_factory:
            
            # Execute operation
            result = self.module._execute_pretrained_operation('validate')
            
            # Verify UI components were passed to operation
            call_args = mock_factory.call_args[1]
            ui_components = call_args['ui_components']
            
            # Verify required UI components are present
            required_components = ['main_container', 'header_container', 'form_container', 
                                 'action_container', 'operation_container']
            for component_name in required_components:
                self.assertIn(component_name, ui_components)
    
    def test_operation_error_logging(self):
        """Test that operation errors are properly logged."""
        # Test different types of errors
        error_scenarios = [
            ('FileNotFoundError', 'Model file not found'),
            ('PermissionError', 'Permission denied'),
            ('ConnectionError', 'Network connection failed'),
            ('ValueError', 'Invalid configuration'),
            ('RuntimeError', 'Operation runtime error')
        ]
        
        for error_type, error_message in error_scenarios:
            with self.subTest(error_type=error_type):
                # Reset log mock
                self.module.log.reset_mock()
                
                with patch('smartcash.ui.model.pretrained.pretrained_uimodule.PretrainedOperationFactory.execute_operation',
                          side_effect=Exception(error_message)) as mock_factory:
                    
                    # Execute operation
                    result = self.module._execute_pretrained_operation('download')
                    
                    # Verify error handling
                    self.assertFalse(result['success'])
                    self.assertIn('error', result)
                    self.assertIn(error_message, result['error'])
                    
                    # Verify error was logged
                    self.module.log.assert_called_once()
                    log_call = self.module.log.call_args
                    self.assertIn(error_message, log_call[0][0])
                    self.assertEqual(log_call[0][1], 'error')
    
    def test_model_status_persistence_across_operations(self):
        """Test that model status persists correctly across multiple operations."""
        # Initial state
        initial_status = self.module._model_status.copy()
        
        # Mock refresh operation that updates model status
        refresh_result = {
            'success': True,
            'models_found': ['yolov5s.pt'],
            'validation_results': {'yolov5s.pt': True},
            'last_refresh': '2024-01-01T12:00:00'
        }
        
        with patch('smartcash.ui.model.pretrained.pretrained_uimodule.PretrainedOperationFactory.execute_operation',
                  return_value=refresh_result) as mock_factory:
            
            # Execute refresh operation
            self.module._execute_refresh_operation()
            
            # Verify model status was updated
            self.assertNotEqual(self.module._model_status, initial_status)
            self.assertEqual(self.module._model_status['models_found'], ['yolov5s.pt'])
            self.assertEqual(self.module._model_status['validation_results'], {'yolov5s.pt': True})
            
        # Mock download operation that doesn't update model status
        download_result = {
            'success': True,
            'message': 'Download completed'
        }
        
        with patch('smartcash.ui.model.pretrained.pretrained_uimodule.PretrainedOperationFactory.execute_operation',
                  return_value=download_result) as mock_factory:
            
            # Store current status
            status_before_download = self.module._model_status.copy()
            
            # Execute download operation
            self.module._execute_download_operation({})
            
            # Verify model status was not changed by download operation
            self.assertEqual(self.module._model_status, status_before_download)


if __name__ == '__main__':
    unittest.main()