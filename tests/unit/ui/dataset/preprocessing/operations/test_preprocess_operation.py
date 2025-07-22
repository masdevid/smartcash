"""
File: tests/unit/ui/dataset/preprocessing/operations/test_preprocess_operation.py
Description: Comprehensive unit tests for the preprocessing operations.
"""
import unittest
from unittest.mock import MagicMock, patch, call, ANY
import os
import shutil
import cv2
import numpy as np
from pathlib import Path

# Import the operation to test
from smartcash.ui.dataset.preprocessing.operations.preprocess_operation import PreprocessOperation
from smartcash.ui.dataset.preprocessing.operations.preprocessing_operation_base import BasePreprocessingOperation

class TestPreprocessOperation(unittest.TestCase):
    """Test suite for PreprocessOperation."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a mock UI module with required attributes
        self.mock_ui = MagicMock()
        self.mock_ui.logger = MagicMock()
        self.mock_ui._operation_container = MagicMock()
        self.mock_ui._operation_container.log = MagicMock()
        self.mock_ui._ui_components = {}
        self.mock_ui._operation_manager = MagicMock()
        
        # Set up the update_progress method
        self.mock_ui.update_progress = MagicMock()
        
        # Test configuration
        self.test_config = {
            'input_dir': 'data/raw',
            'output_dir': 'data/processed',
            'batch_size': 32,
            'move_invalid': True,
            'invalid_dir': 'data/invalid',
            'backup_enabled': True,
            'splits': ['train', 'valid'],  
            'preprocessing': {
                'resize': {'width': 224, 'height': 224},
                'normalize': True,
                'invalid_dir': 'data/invalid',
                'backup_enabled': True
            }
        }
        
        # Create the operation instance
        self.operation = PreprocessOperation(self.mock_ui, self.test_config)
        
        # Mock file system operations
        self.patcher_os = patch('os.makedirs')
        self.mock_os_makedirs = self.patcher_os.start()
        
        self.patcher_path = patch('pathlib.Path')
        self.mock_path = self.patcher_path.start()
        
        # Mock image processing libraries
        self.patcher_cv2 = patch('cv2.imread')
        self.mock_cv2_imread = self.patcher_cv2.start()
        
        self.patcher_cv2_resize = patch('cv2.resize')
        self.mock_cv2_resize = self.patcher_cv2_resize.start()
        
        # Mock numpy
        self.patcher_np = patch('numpy.array')
        self.mock_np_array = self.patcher_np.start()
        
        # Setup return values for mocks
        self.mock_image = np.zeros((480, 640, 3), dtype=np.uint8)
        self.mock_cv2_imread.return_value = self.mock_image
        self.mock_cv2_resize.return_value = self.mock_image
        self.mock_np_array.return_value = self.mock_image
        
        # Mock file listing
        self.mock_dir_entries = [
            'image1.jpg', 'image2.jpg', 'image3.jpg',
            'label1.txt', 'label2.txt', 'label3.txt'
        ]
        
        def mock_glob(pattern):
            if pattern.endswith('.jpg'):
                return [f'data/raw/{f}' for f in self.mock_dir_entries if f.endswith('.jpg')]
            elif pattern.endswith('.txt'):
                return [f'data/raw/{f}' for f in self.mock_dir_entries if f.endswith('.txt')]
            return []
            
        self.mock_path.return_value.glob.side_effect = mock_glob
        self.mock_path.return_value.is_file.return_value = True
        self.mock_path.return_value.parent = Path('data/raw')
        self.mock_path.return_value.name = 'image1.jpg'
        
    def tearDown(self):
        """Clean up after each test method."""
        self.patcher_os.stop()
        self.patcher_path.stop()
        self.patcher_cv2.stop()
        self.patcher_cv2_resize.stop()
        self.patcher_np.stop()
    
    def test_initialization(self):
        """Test that the operation initializes correctly."""
        # Verify the operation is initialized with the correct config
        self.assertEqual(self.operation.config, self.test_config)
        
        # Verify the operation has the required attributes from BasePreprocessingOperation
        self.assertTrue(hasattr(self.operation, '_ui_module'))
        self.assertTrue(hasattr(self.operation, 'config'))
        self.assertTrue(hasattr(self.operation, 'callbacks'))
        self.assertTrue(hasattr(self.operation, '_operation_container'))
        self.assertTrue(hasattr(self.operation, 'logger'))
        
        # Verify the operation has the required methods
        self.assertTrue(callable(getattr(self.operation, 'execute', None)))
        self.assertTrue(callable(getattr(self.operation, '_progress_adapter', None)))
        self.assertTrue(callable(getattr(self.operation, '_execute_callback', None)))
        
        # Verify the operation is an instance of BasePreprocessingOperation
        self.assertIsInstance(self.operation, BasePreprocessingOperation)
    
    @patch('smartcash.ui.dataset.preprocessing.operations.preprocess_operation.preprocess_dataset')
    def test_execute_success(self, mock_preprocess):
        """Test successful execution of the preprocessing operation."""
        # Setup
        mock_result = {
            'success': True,
            'message': 'Success',
            'statistics': {
                'files_processed': 10,
                'files_skipped': 2,
                'files_failed': 0
            },
            'total_time_seconds': 5.5
        }
        mock_preprocess.return_value = mock_result
        
        # Execute with callback monitoring
        with patch.object(self.operation, '_execute_callback') as mock_callback:
            result = self.operation.execute()
            
            # Verify result
            self.assertTrue(result['success'])
            self.assertEqual(result['message'], 'Preprocessing berhasil diselesaikan')
            
            # Verify backend was called with correct arguments
            mock_preprocess.assert_called_once()
            # Get the call arguments (positional and keyword)
            args, kwargs = mock_preprocess.call_args
            # Check if config was passed as a keyword argument
            self.assertEqual(kwargs.get('config'), self.test_config)  # Config passed correctly
            self.assertIsNotNone(kwargs.get('progress_callback'))  # Progress callback provided
            
            # Verify callbacks were called with correct arguments
            mock_callback.assert_any_call('on_success', ANY)  # Success summary
            mock_callback.assert_called_with('on_complete')  # Completion called last
    
    @patch('smartcash.ui.dataset.preprocessing.operations.preprocess_operation.preprocess_dataset')
    def test_execute_with_errors(self, mock_preprocess):
        """Test execution with processing errors from the backend."""
        # Setup
        mock_result = {
            'success': False,
            'message': 'Backend error occurred',
            'statistics': {
                'files_processed': 5,
                'files_skipped': 2,
                'files_failed': 3
            },
            'total_time_seconds': 3.2
        }
        mock_preprocess.return_value = mock_result
        
        # Execute with callback monitoring
        with patch.object(self.operation, '_execute_callback') as mock_callback:
            result = self.operation.execute()
            
            # Verify result - The handler should return success=False when the backend fails
            self.assertFalse(result['success'])
            self.assertIn('Preprocessing gagal', result['message'])
            
            # Verify backend was called
            mock_preprocess.assert_called_once()
            
            # Verify callbacks were called with correct arguments
            mock_callback.assert_any_call('on_failure', ANY)  # Failure summary
            mock_callback.assert_called_with('on_complete')  # Completion called last
    
    @patch('smartcash.ui.dataset.preprocessing.operations.preprocess_operation.preprocess_dataset')
    def test_execute_with_exception(self, mock_preprocess):
        """Test execution when an exception occurs during processing."""
        # Setup
        mock_preprocess.side_effect = Exception('Connection error')
        
        # Execute with callback monitoring
        with patch.object(self.operation, '_execute_callback') as mock_callback:
            result = self.operation.execute()
            
            # Verify result - The handler should return success=False when an exception occurs
            self.assertFalse(result['success'])
            self.assertIn('Error', result['message'])
            
            # Verify backend was called
            mock_preprocess.assert_called_once()
            
            # Verify callbacks were called with correct arguments
            mock_callback.assert_any_call('on_failure', 'Gagal memanggil backend pra-pemrosesan: Connection error')
            mock_callback.assert_called_with('on_complete')  # Completion called last
    
    def test_format_preprocess_summary(self):
        """Test the format_preprocess_summary method."""
        # Test data
        result = {
            'success': True,
            'message': 'Processing completed',
            'statistics': {
                'files_processed': 10,
                'files_skipped': 2,
                'files_failed': 1
            },
            'total_time_seconds': 5.5
        }
        
        # Call the method
        summary = self.operation._format_preprocess_summary(result)
        
        # Verify the summary contains expected information
        self.assertIn('Ringkasan Operasi Pra-pemrosesan', summary)
        self.assertIn('✅ Berhasil', summary)
        self.assertIn('✔️ 10', summary)  # files processed
        self.assertIn('⏭️ 2', summary)   # files skipped
        self.assertIn('❌ 1', summary)    # files failed
        self.assertIn('5.50 detik', summary)  # total time
        self.assertIn('Processing completed', summary)  # backend message
    
    @patch.object(BasePreprocessingOperation, 'update_progress')
    def test_progress_adapter_single_progress(self, mock_update_progress):
        """Test the progress adapter with single progress."""
        # Setup
        self.operation._progress_adapter('overall', 25, 100, 'Processing...')
        
        # Verify
        mock_update_progress.assert_called_once_with(
            progress=25,
            message='Processing...'
        )
    
    @patch.object(BasePreprocessingOperation, 'update_progress')
    def test_progress_adapter_dual_progress(self, mock_update_progress):
        """Test the progress adapter with dual progress."""
        # Setup
        self.operation._progress_adapter(
            'overall',
            current=25,
            total=100,
            message='Overall progress',
            secondary_current=5,
            secondary_total=10,
            secondary_message='Current file'
        )
        
        # Verify
        mock_update_progress.assert_called_once_with(
            progress=25,
            message='Overall progress',
            secondary_progress=50,  # 5/10 = 50%
            secondary_message='Current file'
        )


    def test_execute_with_partial_results(self):
        """Test execution with partial success results."""
        # Setup mock result with some failures
        mock_result = {
            'success': True,
            'message': 'Completed with some warnings',
            'statistics': {
                'files_processed': 8,
                'files_skipped': 3,
                'files_failed': 2
            },
            'total_time_seconds': 4.2
        }
        
        with patch('smartcash.ui.dataset.preprocessing.operations.preprocess_operation.preprocess_dataset') as mock_preprocess:
            mock_preprocess.return_value = mock_result
            
            with patch.object(self.operation, '_execute_callback') as mock_callback:
                result = self.operation.execute()
                
                # Should still be successful overall
                self.assertTrue(result['success'])
                self.assertEqual(result['message'], 'Preprocessing berhasil diselesaikan')
                
                # Verify callbacks were called correctly
                mock_callback.assert_any_call('on_success', ANY)
                mock_callback.assert_called_with('on_complete')

    def test_execute_with_empty_config(self):
        """Test execution with minimal/empty configuration."""
        # Setup operation with minimal config
        minimal_config = {
            'input_dir': 'data/raw',
            'output_dir': 'data/processed'
        }
        minimal_operation = PreprocessOperation(self.mock_ui, minimal_config)
        
        mock_result = {
            'success': True,
            'message': 'Success with minimal config',
            'statistics': {
                'files_processed': 5,
                'files_skipped': 0,
                'files_failed': 0
            },
            'total_time_seconds': 2.1
        }
        
        with patch('smartcash.ui.dataset.preprocessing.operations.preprocess_operation.preprocess_dataset') as mock_preprocess:
            mock_preprocess.return_value = mock_result
            
            result = minimal_operation.execute()
            
            # Verify config was passed correctly
            call_kwargs = mock_preprocess.call_args[1]
            self.assertEqual(call_kwargs['config'], minimal_config)
            self.assertTrue(result['success'])

    def test_execute_with_network_timeout(self):
        """Test execution with network timeout exception."""
        # Create a custom timeout exception instead of importing requests
        class TimeoutException(Exception):
            pass
        
        with patch('smartcash.ui.dataset.preprocessing.operations.preprocess_operation.preprocess_dataset') as mock_preprocess:
            mock_preprocess.side_effect = TimeoutException('Request timeout')
            
            with patch.object(self.operation, '_execute_callback') as mock_callback:
                result = self.operation.execute()
                
                # Should handle timeout gracefully
                self.assertFalse(result['success'])
                self.assertIn('Request timeout', result['message'])
                
                # Should call failure and complete callbacks
                mock_callback.assert_any_call('on_failure', ANY)
                mock_callback.assert_called_with('on_complete')

    def test_progress_adapter_with_zero_total(self):
        """Test progress adapter edge case with zero total."""
        with patch.object(BasePreprocessingOperation, 'update_progress') as mock_update_progress:
            # Test with zero total (should handle division by zero)
            self.operation._progress_adapter('overall', 10, 0, 'Processing with zero total')
            
            # Should call update_progress with 0% progress
            mock_update_progress.assert_called_once_with(
                progress=0,
                message='Processing with zero total'
            )

    def test_progress_adapter_with_invalid_values(self):
        """Test progress adapter with invalid/negative values."""
        with patch.object(BasePreprocessingOperation, 'update_progress') as mock_update_progress:
            # Test with negative values
            self.operation._progress_adapter('overall', -5, 100, 'Negative progress')
            
            # Should handle gracefully and calculate properly
            mock_update_progress.assert_called_once_with(
                progress=-5,  # Should calculate correctly even with negative
                message='Negative progress'
            )

    def test_format_preprocess_summary_with_missing_data(self):
        """Test summary formatting with missing or incomplete data."""
        # Test with minimal data
        minimal_result = {
            'success': True,
            'message': 'Minimal data result'
        }
        
        summary = self.operation._format_preprocess_summary(minimal_result)
        
        # Should handle missing statistics gracefully
        self.assertIn('Ringkasan Operasi Pra-pemrosesan', summary)
        self.assertIn('✅ Berhasil', summary)
        self.assertIn('✔️ 0', summary)  # Default values for missing stats
        self.assertIn('⏭️ 0', summary)
        self.assertIn('❌ 0', summary)
        self.assertIn('0.00 detik', summary)
        self.assertIn('Minimal data result', summary)

    def test_format_preprocess_summary_failure_case(self):
        """Test summary formatting for failure cases."""
        failure_result = {
            'success': False,
            'message': 'Processing failed due to invalid data',
            'statistics': {
                'files_processed': 3,
                'files_skipped': 1,
                'files_failed': 5
            },
            'total_time_seconds': 1.8
        }
        
        summary = self.operation._format_preprocess_summary(failure_result)
        
        # Should show failure status
        self.assertIn('❌ Gagal', summary)
        self.assertIn('✔️ 3', summary)
        self.assertIn('⏭️ 1', summary)
        self.assertIn('❌ 5', summary)
        self.assertIn('1.80 detik', summary)
        self.assertIn('Processing failed due to invalid data', summary)

    def test_callback_execution_with_invalid_callback(self):
        """Test callback execution with invalid/None callbacks."""
        # Create operation without callbacks
        operation_no_callbacks = PreprocessOperation(self.mock_ui, self.test_config)
        
        # Should not raise exception when callbacks are None/missing
        operation_no_callbacks._execute_callback('on_success', 'test_data')
        operation_no_callbacks._execute_callback('non_existent', 'test_data')
        
        # Test with operation that has some callbacks but not others
        partial_callbacks = {'on_success': MagicMock()}
        operation_partial = PreprocessOperation(self.mock_ui, self.test_config, partial_callbacks)
        
        # Should execute existing callback
        operation_partial._execute_callback('on_success', 'test_data')
        partial_callbacks['on_success'].assert_called_once_with('test_data')
        
        # Should handle missing callback gracefully
        operation_partial._execute_callback('on_failure', 'test_data')

    def test_callback_execution_with_exception(self):
        """Test callback execution when callback itself raises exception."""
        def failing_callback(*args, **kwargs):
            raise ValueError("Callback failed")
        
        callbacks_with_error = {'on_success': failing_callback}
        operation_with_error = PreprocessOperation(self.mock_ui, self.test_config, callbacks_with_error)
        
        # Should handle callback exceptions gracefully
        with patch.object(operation_with_error, 'log_error') as mock_log_error:
            operation_with_error._execute_callback('on_success', 'test_data')
            
            # Should log the callback error
            mock_log_error.assert_called_once()
            error_msg = mock_log_error.call_args[0][0]
            self.assertIn("Error executing callback 'on_success'", error_msg)
            self.assertIn("Callback failed", error_msg)


if __name__ == '__main__':
    unittest.main()
