"""
File: tests/unit/ui/model/backbone/test_operation_manager.py
Description: Unit tests for backbone operation manager
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from smartcash.ui.model.backbone.handlers.operation_manager import BackboneOperationManager
from smartcash.ui.model.backbone.constants import BackboneOperation


@pytest.fixture
def mock_operation_container():
    """Mock operation container for testing."""
    container = Mock()
    container.update_progress = Mock()
    container.log_message = Mock()
    return container


@pytest.fixture
def operation_manager(mock_operation_container):
    """Create operation manager instance for testing."""
    return BackboneOperationManager(mock_operation_container)


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        'backbone_type': 'efficientnet_b4',
        'pretrained': True,
        'feature_optimization': False,
        'advanced_settings': {
            'num_classes': 17,
            'input_size': 640
        }
    }


class TestBackboneOperationManager:
    """Test cases for BackboneOperationManager."""
    
    def test_init(self, mock_operation_container):
        """Test operation manager initialization."""
        manager = BackboneOperationManager(mock_operation_container)
        
        assert manager is not None
        assert manager.operation_container == mock_operation_container
        assert hasattr(manager, 'operations')
        assert len(manager.operations) == 4
        
        # Check all operations are registered
        for op_type in [BackboneOperation.VALIDATE.value, BackboneOperation.LOAD.value,
                       BackboneOperation.BUILD.value, BackboneOperation.SUMMARY.value]:
            assert op_type in manager.operations
    
    def test_init_without_container(self):
        """Test operation manager initialization without container."""
        manager = BackboneOperationManager()
        
        assert manager is not None
        assert manager.operation_container is None
        assert len(manager.operations) == 4
    
    @pytest.mark.asyncio
    async def test_execute_validate_operation(self, operation_manager, sample_config):
        """Test executing validate operation."""
        with patch.object(operation_manager.operations[BackboneOperation.VALIDATE.value], 'validate_config') as mock_validate:
            mock_validate.return_value = {
                'valid': True,
                'errors': [],
                'warnings': []
            }
            
            result = await operation_manager.execute_operation(
                BackboneOperation.VALIDATE.value, sample_config
            )
            
            assert result['valid'] is True
            assert mock_validate.called
    
    @pytest.mark.asyncio
    async def test_execute_load_operation(self, operation_manager, sample_config):
        """Test executing load operation."""
        with patch.object(operation_manager.operations[BackboneOperation.LOAD.value], 'load_model') as mock_load:
            mock_load.return_value = {
                'success': True,
                'backbone': Mock(),
                'info': {}
            }
            
            result = await operation_manager.execute_operation(
                BackboneOperation.LOAD.value, sample_config
            )
            
            assert result['success'] is True
            assert mock_load.called
    
    @pytest.mark.asyncio
    async def test_execute_build_operation(self, operation_manager, sample_config):
        """Test executing build operation."""
        with patch.object(operation_manager.operations[BackboneOperation.BUILD.value], 'build_architecture') as mock_build:
            mock_build.return_value = {
                'success': True,
                'model': Mock(),
                'layer_info': {},
                'stats': {}
            }
            
            result = await operation_manager.execute_operation(
                BackboneOperation.BUILD.value, sample_config
            )
            
            assert result['success'] is True
            assert mock_build.called
    
    @pytest.mark.asyncio
    async def test_execute_summary_operation(self, operation_manager, sample_config):
        """Test executing summary operation."""
        with patch.object(operation_manager.operations[BackboneOperation.SUMMARY.value], 'generate_summary') as mock_summary:
            mock_summary.return_value = {
                'success': True,
                'summary': {},
                'analysis': {}
            }
            
            result = await operation_manager.execute_operation(
                BackboneOperation.SUMMARY.value, sample_config
            )
            
            assert result['success'] is True
            assert mock_summary.called
    
    @pytest.mark.asyncio
    async def test_execute_unknown_operation(self, operation_manager, sample_config):
        """Test executing unknown operation type."""
        result = await operation_manager.execute_operation(
            'unknown_operation', sample_config
        )
        
        assert result['success'] is False
        assert 'Unknown operation type' in result['error']
    
    @pytest.mark.asyncio
    async def test_execute_operation_with_callbacks(self, operation_manager, sample_config):
        """Test executing operation with callbacks."""
        progress_callback = AsyncMock()
        log_callback = AsyncMock()
        
        with patch.object(operation_manager.operations[BackboneOperation.VALIDATE.value], 'validate_config') as mock_validate:
            mock_validate.return_value = {'valid': True, 'errors': [], 'warnings': []}
            
            result = await operation_manager.execute_operation(
                BackboneOperation.VALIDATE.value,
                sample_config,
                progress_callback,
                log_callback
            )
            
            assert result['valid'] is True
            # Check that callbacks were passed to the operation
            call_args = mock_validate.call_args
            assert call_args[0][0] == sample_config
            assert call_args[0][1] == progress_callback
            assert call_args[0][2] == log_callback
    
    @pytest.mark.asyncio
    async def test_execute_operation_exception(self, operation_manager, sample_config):
        """Test executing operation that raises exception."""
        with patch.object(operation_manager.operations[BackboneOperation.VALIDATE.value], 'validate_config') as mock_validate:
            mock_validate.side_effect = Exception("Operation failed")
            
            result = await operation_manager.execute_operation(
                BackboneOperation.VALIDATE.value, sample_config
            )
            
            assert result['success'] is False
            assert 'Operation failed' in result['error']
    
    def test_get_available_operations(self, operation_manager):
        """Test getting available operations."""
        operations = operation_manager.get_available_operations()
        
        assert len(operations) == 4
        assert BackboneOperation.VALIDATE.value in operations
        assert BackboneOperation.LOAD.value in operations
        assert BackboneOperation.BUILD.value in operations
        assert BackboneOperation.SUMMARY.value in operations
        
        # Check descriptions are provided
        for description in operations.values():
            assert len(description) > 0
    
    def test_is_operation_running(self, operation_manager):
        """Test checking if operation is running."""
        with patch.object(operation_manager.operations[BackboneOperation.VALIDATE.value], 'is_operation_running') as mock_running:
            mock_running.return_value = True
            
            is_running = operation_manager.is_operation_running(BackboneOperation.VALIDATE.value)
            
            assert is_running is True
            assert mock_running.called
    
    def test_is_operation_running_unknown(self, operation_manager):
        """Test checking if unknown operation is running."""
        is_running = operation_manager.is_operation_running('unknown_operation')
        
        assert is_running is False
    
    def test_cancel_operation(self, operation_manager):
        """Test cancelling operation."""
        with patch.object(operation_manager.operations[BackboneOperation.VALIDATE.value], 'cancel_operation') as mock_cancel:
            mock_cancel.return_value = True
            
            cancelled = operation_manager.cancel_operation(BackboneOperation.VALIDATE.value)
            
            assert cancelled is True
            assert mock_cancel.called
    
    def test_cancel_operation_unknown(self, operation_manager):
        """Test cancelling unknown operation."""
        cancelled = operation_manager.cancel_operation('unknown_operation')
        
        assert cancelled is False
    
    def test_shutdown(self, operation_manager):
        """Test shutting down operation manager."""
        # Mock all operation shutdown methods
        for operation in operation_manager.operations.values():
            operation.shutdown = Mock()
        
        operation_manager.shutdown()
        
        # Check all operations were shut down
        for operation in operation_manager.operations.values():
            operation.shutdown.assert_called_once()
    
    def test_shutdown_with_exception(self, operation_manager):
        """Test shutting down with one operation raising exception."""
        # Mock operations, one raises exception
        operation_manager.operations[BackboneOperation.VALIDATE.value].shutdown = Mock(side_effect=Exception("Shutdown error"))
        operation_manager.operations[BackboneOperation.LOAD.value].shutdown = Mock()
        operation_manager.operations[BackboneOperation.BUILD.value].shutdown = Mock()
        operation_manager.operations[BackboneOperation.SUMMARY.value].shutdown = Mock()
        
        # Should not raise exception
        operation_manager.shutdown()
        
        # All operations should still be called
        for operation in operation_manager.operations.values():
            operation.shutdown.assert_called_once()


class TestOperationManagerIntegration:
    """Integration test cases for operation manager."""
    
    @pytest.mark.asyncio
    async def test_sequential_operations(self, operation_manager, sample_config):
        """Test running multiple operations sequentially."""
        # Mock all operations to return success
        with patch.object(operation_manager.operations[BackboneOperation.VALIDATE.value], 'validate_config') as mock_validate, \
             patch.object(operation_manager.operations[BackboneOperation.LOAD.value], 'load_model') as mock_load, \
             patch.object(operation_manager.operations[BackboneOperation.BUILD.value], 'build_architecture') as mock_build:
            
            mock_validate.return_value = {'valid': True, 'errors': [], 'warnings': []}
            mock_load.return_value = {'success': True, 'backbone': Mock(), 'info': {}}
            mock_build.return_value = {'success': True, 'model': Mock(), 'layer_info': {}, 'stats': {}}
            
            # Run operations sequentially
            validate_result = await operation_manager.execute_operation(
                BackboneOperation.VALIDATE.value, sample_config
            )
            load_result = await operation_manager.execute_operation(
                BackboneOperation.LOAD.value, sample_config
            )
            build_result = await operation_manager.execute_operation(
                BackboneOperation.BUILD.value, sample_config
            )
            
            assert validate_result['valid'] is True
            assert load_result['success'] is True
            assert build_result['success'] is True
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, operation_manager, sample_config):
        """Test running multiple operations concurrently."""
        # Mock all operations to return success
        with patch.object(operation_manager.operations[BackboneOperation.VALIDATE.value], 'validate_config') as mock_validate, \
             patch.object(operation_manager.operations[BackboneOperation.SUMMARY.value], 'generate_summary') as mock_summary:
            
            async def slow_validate(*args, **kwargs):
                await asyncio.sleep(0.01)
                return {'valid': True, 'errors': [], 'warnings': []}
            
            async def slow_summary(*args, **kwargs):
                await asyncio.sleep(0.01)
                return {'success': True, 'summary': {}, 'analysis': {}}
            
            mock_validate.side_effect = slow_validate
            mock_summary.side_effect = slow_summary
            
            # Run operations concurrently
            tasks = [
                operation_manager.execute_operation(BackboneOperation.VALIDATE.value, sample_config),
                operation_manager.execute_operation(BackboneOperation.SUMMARY.value, sample_config)
            ]
            
            results = await asyncio.gather(*tasks)
            
            assert len(results) == 2
            assert results[0]['valid'] is True
            assert results[1]['success'] is True


if __name__ == '__main__':
    pytest.main([__file__])