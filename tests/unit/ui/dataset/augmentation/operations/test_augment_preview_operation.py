"""
Unit tests for AugmentPreviewOperation class.
"""

import os
import unittest
from unittest.mock import MagicMock, patch, mock_open
from pathlib import Path

from smartcash.ui.dataset.augmentation.operations.augment_preview_operation import AugmentPreviewOperation

class TestAugmentPreviewOperation(unittest.TestCase):
    """Test cases for AugmentPreviewOperation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_ui_module = MagicMock()
        self.mock_ui_components = {}
        self.mock_ui_module._ui_components = self.mock_ui_components
        
        self.config = {
            'target_split': 'train',
            'preview_config': {}
        }
        
        # Create a temporary directory for test files
        self.test_dir = Path(__file__).parent / 'test_data'
        self.test_dir.mkdir(exist_ok=True)
        
        # Create a test image file
        self.test_image_path = self.test_dir / 'test_preview.jpg'
        with open(self.test_image_path, 'wb') as f:
            f.write(b'test image data')
        
        # Create the operation instance
        self.operation = AugmentPreviewOperation(self.mock_ui_module, self.config)
    
    def tearDown(self):
        """Clean up after tests."""
        if self.test_image_path.exists():
            self.test_image_path.unlink()
        if self.test_dir.exists():
            self.test_dir.rmdir()
    
    @patch('builtins.open', new_callable=mock_open, read_data=b'test image data')
    @patch('os.path.exists', return_value=True)
    @patch('os.path.getsize', return_value=100)
    def test_load_preview_to_widget_success(self, mock_getsize, mock_exists, mock_file):
        """Test loading preview to widget successfully."""
        # Setup
        self.operation._preview_path = str(self.test_image_path)
        self.mock_ui_components['preview_image'] = MagicMock()
        
        # Execute
        result = self.operation._load_preview_to_widget()
        
        # Assert
        self.assertTrue(result)
        self.mock_ui_components['preview_image'].value = b'test image data'
    
    @patch('os.path.exists', return_value=False)
    def test_load_preview_to_widget_invalid_path(self, mock_exists):
        """Test loading preview with invalid path."""
        # Setup
        self.operation._preview_path = '/invalid/path'
        
        # Execute
        result = self.operation._load_preview_to_widget()
        
        # Assert
        self.assertFalse(result)
    
    @patch('builtins.open', side_effect=IOError("File read error"))
    @patch('os.path.exists', return_value=True)
    @patch('os.path.getsize', return_value=100)
    def test_load_preview_to_widget_read_error(self, mock_getsize, mock_exists, mock_file):
        """Test handling file read error."""
        # Setup
        self.operation._preview_path = str(self.test_image_path)
        self.mock_ui_components['preview_image'] = MagicMock()
        
        # Execute
        result = self.operation._load_preview_to_widget()
        
        # Assert
        self.assertFalse(result)
    
    @patch('smartcash.ui.dataset.augmentation.operations.augment_preview_operation.AugmentPreviewOperation._load_preview_to_widget', return_value=True)
    @patch('smartcash.ui.dataset.augmentation.operations.augment_preview_operation.AugmentationBaseOperation.get_backend_api')
    def test_execute_success(self, mock_get_backend_api, mock_load_preview):
        """Test successful execution of preview operation."""
        # Setup
        mock_service = MagicMock()
        mock_service.return_value = {'preview_path': str(self.test_image_path), 'status': 'success'}
        mock_get_backend_api.return_value = mock_service
        
        # Execute
        result = self.operation.execute()
        
        # Assert
        self.assertEqual(result['status'], 'success')
        self.assertEqual(result['message'], 'Preview berhasil dibuat')
        mock_service.assert_called_once_with(target_split='train', config=self.config)
    
    @patch('smartcash.ui.dataset.augmentation.operations.augment_preview_operation.AugmentationBaseOperation.get_backend_api')
    def test_execute_no_preview_path(self, mock_get_backend_api):
        """Test execution when no preview path is returned."""
        # Setup
        mock_service = MagicMock()
        mock_service.return_value = {'status': 'error', 'message': 'No preview generated'}
        mock_get_backend_api.return_value = mock_service
        
        # Execute
        result = self.operation.execute()
        
        # Assert
        self.assertEqual(result['status'], 'error')
        self.assertIn('Gagal membuat preview', result['message'])

if __name__ == '__main__':
    unittest.main()
