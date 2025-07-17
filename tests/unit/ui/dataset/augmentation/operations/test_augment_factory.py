"""
Unit tests for augment_factory module.
"""

import unittest
from unittest.mock import MagicMock, patch

from smartcash.ui.dataset.augmentation.operations.augment_factory import (
    create_operation, create_augment_operation, create_augment_preview_operation,
    create_augment_status_operation, create_augment_cleanup_operation
)

class TestAugmentFactory(unittest.TestCase):
    """Test cases for the augmentation operation factory functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_ui_module = MagicMock()
        self.config = {'test': 'config'}
        self.callbacks = {'on_complete': MagicMock()}
    
    @patch('smartcash.ui.dataset.augmentation.operations.augment_factory.AugmentOperation')
    def test_create_augment_operation(self, mock_operation_class):
        """Test creating an AugmentOperation instance."""
        # Setup
        mock_operation = MagicMock()
        mock_operation_class.return_value = mock_operation
        
        # Execute
        result = create_augment_operation(
            self.mock_ui_module,
            self.config,
            self.callbacks
        )
        
        # Assert
        self.assertEqual(result, mock_operation)
        mock_operation_class.assert_called_once_with(
            self.mock_ui_module,
            self.config,
            self.callbacks
        )
    
    @patch('smartcash.ui.dataset.augmentation.operations.augment_factory.AugmentPreviewOperation')
    def test_create_augment_preview_operation(self, mock_operation_class):
        """Test creating an AugmentPreviewOperation instance."""
        # Setup
        mock_operation = MagicMock()
        mock_operation_class.return_value = mock_operation
        
        # Execute
        result = create_augment_preview_operation(
            self.mock_ui_module,
            self.config,
            self.callbacks
        )
        
        # Assert
        self.assertEqual(result, mock_operation)
        mock_operation_class.assert_called_once_with(
            self.mock_ui_module,
            self.config,
            self.callbacks
        )
    
    @patch('smartcash.ui.dataset.augmentation.operations.augment_factory.AugmentStatusOperation')
    def test_create_augment_status_operation(self, mock_operation_class):
        """Test creating an AugmentStatusOperation instance."""
        # Setup
        mock_operation = MagicMock()
        mock_operation_class.return_value = mock_operation
        
        # Execute
        result = create_augment_status_operation(
            self.mock_ui_module,
            self.config,
            self.callbacks
        )
        
        # Assert
        self.assertEqual(result, mock_operation)
        mock_operation_class.assert_called_once_with(
            self.mock_ui_module,
            self.config,
            self.callbacks
        )
    
    @patch('smartcash.ui.dataset.augmentation.operations.augment_factory.AugmentCleanupOperation')
    def test_create_augment_cleanup_operation(self, mock_operation_class):
        """Test creating an AugmentCleanupOperation instance."""
        # Setup
        mock_operation = MagicMock()
        mock_operation_class.return_value = mock_operation
        
        # Execute
        result = create_augment_cleanup_operation(
            self.mock_ui_module,
            self.config,
            self.callbacks
        )
        
        # Assert
        self.assertEqual(result, mock_operation)
        mock_operation_class.assert_called_once_with(
            self.mock_ui_module,
            self.config,
            self.callbacks
        )
    
    @patch('smartcash.ui.dataset.augmentation.operations.augment_factory.create_augment_operation')
    def test_create_operation_augment(self, mock_create):
        """Test create_operation with 'augment' operation type."""
        # Setup
        mock_operation = MagicMock()
        mock_create.return_value = mock_operation
        
        # Execute
        result = create_operation(
            'augment',
            self.mock_ui_module,
            self.config,
            self.callbacks
        )
        
        # Assert
        self.assertEqual(result, mock_operation)
        mock_create.assert_called_once_with(
            self.mock_ui_module,
            self.config,
            self.callbacks
        )
    
    def test_create_operation_invalid_type(self):
        """Test create_operation with an invalid operation type."""
        # Execute & Assert
        with self.assertRaises(ValueError) as context:
            create_operation(
                'invalid_operation',
                self.mock_ui_module,
                self.config,
                self.callbacks
            )
        
        self.assertIn('Unknown operation type', str(context.exception))

if __name__ == '__main__':
    unittest.main()
