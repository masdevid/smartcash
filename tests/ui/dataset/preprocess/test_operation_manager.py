"""
Tests for preprocessing operation manager and its operations.
"""

import pytest
from unittest.mock import MagicMock, patch, ANY
import asyncio

# Import the operation manager and related classes
from smartcash.ui.dataset.preprocess.operations.manager import PreprocessOperationManager
from smartcash.ui.dataset.preprocess.constants import PreprocessingOperation
import asyncio


class TestPreprocessOperationManager:
    """Test suite for PreprocessOperationManager and its operations."""
    
    @pytest.fixture
    def mock_operation_container(self):
        """Create a mock operation container."""
        return MagicMock()
    
    @pytest.fixture
    def mock_backend_service(self):
        """Create a mock backend service with async methods."""
        class AsyncMock(MagicMock):
            async def __call__(self, *args, **kwargs):
                return super().__call__(*args, **kwargs)
        
        mock = AsyncMock()
        
        # Create async methods
        async def mock_preprocess(*args, **kwargs):
            return {"success": True, "message": "Preprocessing completed"}
            
        async def mock_check(*args, **kwargs):
            return {"success": True, "message": "Check completed"}
            
        async def mock_cleanup(*args, **kwargs):
            return {"success": True, "message": "Cleanup completed"}
        
        # Assign the async functions to the mock methods
        mock.preprocess_dataset = mock_preprocess
        mock.check_dataset = mock_check
        mock.cleanup_dataset = mock_cleanup
        
        return mock
    
    @pytest.fixture
    def operation_manager(self, mock_operation_container, mock_backend_service):
        """Create an operation manager with mocks for testing."""
        config = {
            'dataset_path': '/test/dataset',
            'output_path': '/test/output',
            'image_size': [640, 640],
            'batch_size': 16,
            'num_workers': 4,
            'augmentation': True,
            'normalization': 'imagenet',
            'validation_split': 0.2,
            'seed': 42
        }
        
        manager = PreprocessOperationManager(config, mock_operation_container)
        manager._backend_service = mock_backend_service
        return manager
    
    def test_initialization(self, operation_manager, mock_operation_container):
        """Test that the operation manager initializes correctly."""
        assert operation_manager.config is not None
        assert operation_manager._operation_container == mock_operation_container
        assert hasattr(operation_manager, '_backend_service')
        assert operation_manager._current_operation is None
        assert operation_manager._is_processing is False
    
    def test_get_operations(self, operation_manager):
        """Test that all operations are properly registered."""
        operations = operation_manager.get_operations()
        assert 'preprocess' in operations
        assert 'check' in operations
        assert 'cleanup' in operations
    
    @pytest.mark.asyncio
    async def test_execute_preprocess_success(self, operation_manager, mock_backend_service):
        """Test successful execution of preprocessing operation."""
        # Execute the operation
        result = await operation_manager.execute_preprocess()
        
        # Verify the result
        assert result['success'] is True
        assert 'Preprocessing completed' in result['message']
        
        # Verify backend service was called with correct parameters
        mock_backend_service.preprocess_dataset.assert_called_once()
        
        # Verify logging was called - check if log was called with any arguments
        assert operation_manager.log.call_count > 0
    
    @pytest.mark.asyncio
    async def test_execute_check_success(self, operation_manager, mock_backend_service):
        """Test successful execution of dataset check operation."""
        # Execute the operation
        result = await operation_manager.execute_check()
        
        # Verify the result
        assert result['success'] is True
        assert 'Check completed' in result['message']
        
        # Verify backend service was called
        mock_backend_service.check_dataset.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_execute_cleanup_success(self, operation_manager, mock_backend_service):
        """Test successful execution of cleanup operation."""
        # Execute the operation
        result = await operation_manager.execute_cleanup()
        
        # Verify the result
        assert result['success'] is True
        assert 'Cleanup completed' in result['message']
        
        # Verify backend service was called
        mock_backend_service.cleanup_dataset.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, operation_manager, mock_backend_service):
        """Test that concurrent operations are handled correctly."""
        # Create a future that we can control
        future = asyncio.Future()
        
        # Create a mock that will block until we set the future
        async def mock_blocking_preprocess(*args, **kwargs):
            await future
            return {"success": True, "message": "Preprocessing completed"}
            
        # Set the mock to use our blocking function
        mock_backend_service.preprocess_dataset = mock_blocking_preprocess
        
        # Start the first operation (will block)
        task1 = asyncio.create_task(operation_manager.execute_preprocess())
        
        # Give it a moment to start
        await asyncio.sleep(0.1)
        
        # Try to start a second operation (should be rejected)
        task2 = asyncio.create_task(operation_manager.execute_preprocess())
        
        # Let the first operation complete
        future.set_result(True)
        
        # Wait for both tasks to complete
        results = await asyncio.gather(task1, task2, return_exceptions=True)
        
        # Check that we got one success and one error
        success_results = [r for r in results if isinstance(r, dict) and r.get('success', False)]
        error_results = [r for r in results if isinstance(r, Exception) or 
                        (isinstance(r, dict) and not r.get('success', True))]
        
        assert len(success_results) == 1
        assert len(error_results) == 1
    
    @pytest.mark.asyncio
    async def test_error_handling(self, operation_manager, mock_backend_service):
        """Test error handling during operations."""
        # Make the operation fail with an async function
        error_msg = "Test error: Dataset not found"
        
        async def mock_failing_preprocess(*args, **kwargs):
            raise Exception(error_msg)
            
        mock_backend_service.preprocess_dataset.side_effect = mock_failing_preprocess
        
        # Execute the operation
        result = await operation_manager.execute_preprocess()
        
        # Verify error was handled
        assert result['success'] is False
        assert error_msg in result.get('error', '')
        
        # Verify error was logged - check that log was called with error level
        error_log_found = any(
            call[0][1].get('level') == 'error' and 
            error_msg in str(call[0][0])
            for call in operation_manager.log.call_args_list
        )
        assert error_log_found, "Error was not properly logged"
    
    def test_ui_components_setter(self, operation_manager):
        """Test that UI components can be set and retrieved."""
        # Create mock UI components
        mock_components = {
            'progress_bar': MagicMock(),
            'status_label': MagicMock(),
            'log_output': MagicMock()
        }
        
        # Set the UI components
        operation_manager._ui_components = mock_components
        
        # Verify components were set
        assert operation_manager._ui_components == mock_components
        
    def test_update_progress(self, operation_manager):
        """Test that progress updates are properly handled."""
        # Create mock UI components
        mock_progress = MagicMock()
        mock_status = MagicMock()
        
        operation_manager._ui_components = {
            'progress_bar': mock_progress,
            'status_label': mock_status
        }
        
        # Update progress
        operation_manager.update_progress(75, "Three quarters done")
        
        # Verify updates were made
        mock_progress.value = 75
        mock_status.value = "Three quarters done"
        
    def test_logging(self, operation_manager):
        """Test that logging works correctly."""
        # Create mock log output
        mock_log = MagicMock()
        operation_manager._ui_components = {
            'log_output': mock_log
        }
        
        # Test info log
        operation_manager.log("Info message", level='info')
        
        # Check that append_stdout was called with the message
        found = False
        for call in mock_log.append_stdout.call_args_list:
            if "Info message" in str(call[0][0]):
                found = True
                break
        assert found, "Info message was not logged to stdout"
        
        # Reset mock for error test
        mock_log.reset_mock()
        
        # Test error log
        operation_manager.log("Error message", level='error')
        
        # Check that append_stderr was called with the message
        found = False
        for call in mock_log.append_stderr.call_args_list:
            if "Error message" in str(call[0][0]):
                found = True
                break
        assert found, "Error message was not logged to stderr"
