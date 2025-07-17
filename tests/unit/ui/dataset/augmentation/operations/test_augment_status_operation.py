"""
Unit tests for AugmentStatusOperation class.
"""

import unittest
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta

from smartcash.ui.dataset.augmentation.operations.augment_status_operation import AugmentStatusOperation

class TestAugmentStatusOperation(unittest.TestCase):
    """Test cases for AugmentStatusOperation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_ui_module = MagicMock()
        self.mock_ui_components = {}
        self.mock_ui_module._ui_components = self.mock_ui_components
        
        self.config = {
            'job_id': 'test_job_123',
            'poll_interval': 1
        }
        
        # Create the operation instance
        self.operation = AugmentStatusOperation(self.mock_ui_module, self.config)
    
    def test_format_running_status(self):
        """Test formatting of running status."""
        # Setup
        status_data = {
            'progress': 42.5,
            'processed': 17,
            'total': 100,
            'current_operation': 'applying_flip',
            'start_time': (datetime.now() - timedelta(minutes=2, seconds=30)).isoformat()
        }
        
        # Execute
        result = self.operation._format_running_status(status_data)
        
        # Assert
        self.assertIn("Proses: 42.5%", result)
        self.assertIn("Diproses: 17/100 gambar", result)
        self.assertIn("Operasi: applying_flip", result)
        self.assertIn("Berjalan selama:", result)
    
    @patch('smartcash.ui.dataset.augmentation.operations.augment_status_operation.AugmentationBaseOperation.get_backend_api')
    def test_execute_success(self, mock_get_backend_api):
        """Test successful status check."""
        # Setup
        mock_service = MagicMock()
        mock_service.return_value = {
            'status': 'completed',
            'message': 'Augmentation completed',
            'processed_count': 100,
            'generated_count': 500
        }
        mock_get_backend_api.return_value = mock_service
        
        # Execute
        result = self.operation.execute()
        
        # Assert
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(result['message'], 'Augmentation completed')
        self.assertEqual(result['processed_count'], 100)
        self.assertEqual(result['generated_count'], 500)
    
    @patch('smartcash.ui.dataset.augmentation.operations.augment_status_operation.AugmentationBaseOperation.get_backend_api')
    def test_execute_running(self, mock_get_backend_api):
        """Test running status check."""
        # Setup
        mock_service = MagicMock()
        mock_service.return_value = {
            'status': 'running',
            'progress': 42.5,
            'processed': 17,
            'total': 40,
            'current_operation': 'applying_flip',
            'start_time': (datetime.now() - timedelta(minutes=2, seconds=30)).isoformat()
        }
        mock_get_backend_api.return_value = mock_service
        
        # Execute
        result = self.operation.execute()
        
        # Assert
        self.assertEqual(result['status'], 'running')
        self.assertIn('Proses: 42.5%', result['details'])
        self.assertIn('Diproses: 17/40', result['details'])
    
    @patch('smartcash.ui.dataset.augmentation.operations.augment_status_operation.AugmentationBaseOperation.get_backend_api')
    def test_execute_error(self, mock_get_backend_api):
        """Test error status check."""
        # Setup
        mock_service = MagicMock()
        mock_service.return_value = {
            'status': 'error',
            'message': 'Processing error',
            'error': 'RuntimeError',
            'details': 'Error details here'
        }
        mock_get_backend_api.return_value = mock_service
        
        # Execute
        result = self.operation.execute()
        
        # Assert
        self.assertEqual(result['status'], 'error')
        self.assertEqual(result['message'], 'Processing error')
        self.assertEqual(result['error'], 'RuntimeError')

if __name__ == '__main__':
    unittest.main()
