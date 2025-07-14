"""
Test cases for backbone progress tracking functionality.
"""

import unittest
from unittest.mock import MagicMock, patch
from smartcash.ui.model.backbone.operations.backbone_operation_manager import BackboneOperationManager
from smartcash.ui.components.operation_container import create_operation_container, OperationContainer

class TestBackboneProgress(unittest.TestCase):
    """Test cases for backbone progress tracking."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_config = {
            'backbone': {
                'model_type': 'efficientnet_b4',
                'input_size': 640,
                'num_classes': 7,
                'pretrained': True
            }
        }
    
        # Create mock operation container
        self.mock_operation_container = OperationContainer(
            show_progress=True,
            show_logs=True,
            log_module_name='test_backbone',
            log_height="200px",
            log_entry_style='compact'
        )
        
        # Mock update_progress
        self.mock_operation_container.update_progress = MagicMock()
        
        # Create mock service
        self.mock_service = MagicMock()
        self.mock_service.validate_backbone_config = MagicMock(
            return_value={'valid': True, 'message': 'Validation successful'}
        )
        self.mock_service.build_backbone_architecture = MagicMock(
            return_value={'success': True, 'message': 'Build successful'}
        )
        
        # Create operation manager
        self.operation_manager = BackboneOperationManager(
            config=self.mock_config,
            operation_container=self.mock_operation_container
        )
        
        # Set mock service
        self.operation_manager._service = self.mock_service
        
        # Mock asyncio.run
        self.asyncio_run_mock = MagicMock()
        self.asyncio_run_mock.side_effect = lambda coro: coro
        
        # Patch asyncio.run
        self.asyncio_patch = patch('asyncio.run', self.asyncio_run_mock)
        self.asyncio_patch.start()
        
        # Clean up
        self.addCleanup(self.asyncio_patch.stop)
        
    def test_progress_tracking_validate(self):
        """Test progress tracking during validation."""
        # Check initial visibility
        self.assertTrue(self.mock_operation_container.is_progress_tracker_visible())
        
        # Execute validation
        result = self.operation_manager.execute_validate()
        
        # Check progress updates
        self.mock_operation_container.update_progress.assert_any_call(0, "Initializing validation...", "primary")
        self.mock_operation_container.update_progress.assert_any_call(0, "Validation failed", "primary")
        
        # Check result
        self.assertFalse(result['success'])
        self.assertIn('coroutine', result['message'])
        
        # Check final visibility
        self.assertTrue(self.mock_operation_container.is_progress_tracker_visible())
        
    def test_progress_tracking_build(self):
        """Test progress tracking during build operation."""
        # Check initial visibility
        self.assertTrue(self.mock_operation_container.is_progress_tracker_visible())
        
        # Execute build
        result = self.operation_manager.execute_build()
        
        # Check progress updates
        self.mock_operation_container.update_progress.assert_any_call(0, "Initializing model build...", "primary")
        self.mock_operation_container.update_progress.assert_any_call(0, "Model build failed", "primary")
        
        # Check result
        self.assertFalse(result['success'])
        self.assertIn('dict', result['message'])
        
        # Check final visibility
        self.assertTrue(self.mock_operation_container.is_progress_tracker_visible())
        
            
    def test_progress_level_consistency(self):
        """Test that progress updates use correct level."""
        # Execute validation
        self.operation_manager.execute_validate()
        
        # Get all progress updates
        progress_calls = self.mock_operation_container.update_progress.call_args_list
        
        # Check that all progress updates use level="primary"
        for call in progress_calls:
            args, kwargs = call
            if len(args) >= 3:
                self.assertEqual(args[2], 'primary')
            elif kwargs:
                self.assertEqual(kwargs.get('level'), 'primary')

if __name__ == '__main__':
    unittest.main()
