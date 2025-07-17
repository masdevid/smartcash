"""
Unit tests for AugmentOperation class.
"""

import unittest
from unittest.mock import MagicMock, patch

from smartcash.ui.dataset.augmentation.operations.augment_operation import AugmentOperation

class TestAugmentOperation(unittest.TestCase):
    """Test cases for AugmentOperation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_ui_module = MagicMock()
        self.mock_ui_components = {}
        self.mock_ui_module._ui_components = self.mock_ui_components
        
        self.config = {
            'input_dir': '/test/input',
            'output_dir': '/test/output',
            'augmentations': [{'type': 'flip', 'probability': 0.5}]
        }
        
        # Create the operation instance
        self.operation = AugmentOperation(self.mock_ui_module, self.config)
    
    @patch('smartcash.ui.dataset.augmentation.operations.augment_operation.AugmentationBaseOperation.get_backend_api')
    def test_execute_success(self, mock_get_backend_api):
        """Test successful execution of augmentation."""
        # Setup
        mock_service = MagicMock()
        mock_service.return_value = {
            'status': 'success',
            'output_dir': '/test/output',
            'processed_count': 10,
            'generated_count': 50
        }
        mock_get_backend_api.return_value = mock_service
        
        # Execute
        result = self.operation.execute()
        
        # Assert
        self.assertEqual(result['status'], 'success')
        self.assertEqual(result['message'], 'Augmentasi dataset berhasil')
        self.assertEqual(result['output_dir'], '/test/output')
        self.assertEqual(result['processed_count'], 10)
        self.assertEqual(result['generated_count'], 50)
        
        # Verify the service was called with correct arguments
        mock_service.assert_called_once_with(
            config=self.config,
            progress_callback=self.operation._progress_adapter
        )
    
    @patch('smartcash.ui.dataset.augmentation.operations.augment_operation.AugmentationBaseOperation.get_backend_api')
    def test_execute_backend_error(self, mock_get_backend_api):
        """Test handling of backend errors."""
        # Setup
        mock_service = MagicMock()
        mock_service.return_value = {
            'status': 'error',
            'message': 'Invalid configuration',
            'error': 'ConfigurationError'
        }
        mock_get_backend_api.return_value = mock_service
        
        # Execute
        result = self.operation.execute()
        
        # Assert
        self.assertEqual(result['status'], 'error')
        self.assertIn('Gagal menjalankan augmentasi', result['message'])
    
    @patch('smartcash.ui.dataset.augmentation.operations.augment_operation.AugmentationBaseOperation.get_backend_api')
    def test_execute_exception(self, mock_get_backend_api):
        """Test handling of unexpected exceptions."""
        # Setup
        mock_service = MagicMock()
        mock_service.side_effect = Exception("Unexpected error")
        mock_get_backend_api.return_value = mock_service
        
        # Execute
        result = self.operation.execute()
        
        # Assert
        self.assertEqual(result['status'], 'error')
        self.assertIn('Terjadi kesalahan saat augmentasi', result['message'])
    
    def test_progress_adapter(self):
        """Test the progress adapter method."""
        # Setup
        self.operation.update_progress = MagicMock()
        
        # Execute
        self.operation._progress_adapter(0.75, "Processing image 5/10")
        
        # Assert
        self.operation.update_progress.assert_called_once_with(75.0, "Processing image 5/10")

if __name__ == '__main__':
    unittest.main()
