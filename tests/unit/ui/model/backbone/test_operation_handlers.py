"""
File: tests/unit/ui/model/backbone/test_operation_handlers.py
Description: Unit tests for backbone operation handlers
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from smartcash.ui.model.backbone.operations import (
    ValidateOperation, LoadOperation, BuildOperation, SummaryOperation
)


@pytest.fixture
def mock_operation_container():
    """Mock operation container for testing."""
    container = Mock()
    container.update_progress = Mock()
    container.log_message = Mock()
    return container


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


class TestValidateOperation:
    """Test cases for ValidateOperation."""
    
    def test_init(self, mock_operation_container):
        """Test validate operation initialization."""
        operation = ValidateOperation(mock_operation_container)
        
        assert operation is not None
        assert hasattr(operation, 'backbone_service')
        assert operation._operation_container == mock_operation_container
    
    def test_get_operations(self, mock_operation_container):
        """Test getting available operations."""
        operation = ValidateOperation(mock_operation_container)
        ops = operation.get_operations()
        
        assert 'validate' in ops
        assert callable(ops['validate'])
    
    @pytest.mark.asyncio
    async def test_validate_config_success(self, mock_operation_container, sample_config):
        """Test successful configuration validation."""
        operation = ValidateOperation(mock_operation_container)
        
        with patch.object(operation.backbone_service, 'validate_backbone_config') as mock_validate:
            mock_validate.return_value = {
                'valid': True,
                'errors': [],
                'warnings': []
            }
            
            result = await operation.validate_config(sample_config)
            
            assert result['valid'] is True
            assert mock_validate.called
            assert mock_validate.call_args[1]['config'] == sample_config
    
    @pytest.mark.asyncio
    async def test_validate_config_failure(self, mock_operation_container, sample_config):
        """Test configuration validation failure."""
        operation = ValidateOperation(mock_operation_container)
        
        with patch.object(operation.backbone_service, 'validate_backbone_config') as mock_validate:
            mock_validate.return_value = {
                'valid': False,
                'errors': ['Invalid backbone type'],
                'warnings': []
            }
            
            result = await operation.validate_config(sample_config)
            
            assert result['valid'] is False
            assert len(result['errors']) > 0
    
    @pytest.mark.asyncio
    async def test_validate_config_exception(self, mock_operation_container, sample_config):
        """Test configuration validation with exception."""
        operation = ValidateOperation(mock_operation_container)
        
        with patch.object(operation.backbone_service, 'validate_backbone_config') as mock_validate:
            mock_validate.side_effect = Exception("Service error")
            
            result = await operation.validate_config(sample_config)
            
            assert result['valid'] is False
            assert 'error' in result
            assert 'Service error' in result['error']


class TestLoadOperation:
    """Test cases for LoadOperation."""
    
    def test_init(self, mock_operation_container):
        """Test load operation initialization."""
        operation = LoadOperation(mock_operation_container)
        
        assert operation is not None
        assert hasattr(operation, 'backbone_service')
    
    def test_get_operations(self, mock_operation_container):
        """Test getting available operations."""
        operation = LoadOperation(mock_operation_container)
        ops = operation.get_operations()
        
        assert 'load' in ops
        assert callable(ops['load'])
    
    @pytest.mark.asyncio
    async def test_load_model_success(self, mock_operation_container, sample_config):
        """Test successful model loading."""
        operation = LoadOperation(mock_operation_container)
        
        with patch.object(operation.backbone_service, 'load_backbone_model') as mock_load:
            mock_load.return_value = {
                'success': True,
                'backbone': Mock(),
                'info': {
                    'total_parameters': 19000000,
                    'model_size_mb': 76.0
                },
                'message': 'Model loaded successfully'
            }
            
            result = await operation.load_model(sample_config)
            
            assert result['success'] is True
            assert 'backbone' in result
            assert 'info' in result
    
    @pytest.mark.asyncio
    async def test_load_model_failure(self, mock_operation_container, sample_config):
        """Test model loading failure."""
        operation = LoadOperation(mock_operation_container)
        
        with patch.object(operation.backbone_service, 'load_backbone_model') as mock_load:
            mock_load.return_value = {
                'success': False,
                'error': 'Failed to load model',
                'message': 'Loading failed'
            }
            
            result = await operation.load_model(sample_config)
            
            assert result['success'] is False
            assert 'error' in result


class TestBuildOperation:
    """Test cases for BuildOperation."""
    
    def test_init(self, mock_operation_container):
        """Test build operation initialization."""
        operation = BuildOperation(mock_operation_container)
        
        assert operation is not None
        assert hasattr(operation, 'backbone_service')
    
    def test_get_operations(self, mock_operation_container):
        """Test getting available operations."""
        operation = BuildOperation(mock_operation_container)
        ops = operation.get_operations()
        
        assert 'build' in ops
        assert callable(ops['build'])
    
    @pytest.mark.asyncio
    async def test_build_architecture_success(self, mock_operation_container, sample_config):
        """Test successful architecture building."""
        operation = BuildOperation(mock_operation_container)
        
        with patch.object(operation.backbone_service, 'build_backbone_architecture') as mock_build:
            mock_build.return_value = {
                'success': True,
                'model': Mock(),
                'layer_info': {
                    'total_layers': 10,
                    'backbone_layers': 8,
                    'head_layers': 2
                },
                'stats': {
                    'total_parameters': 19000000,
                    'model_size_mb': 76.0
                },
                'message': 'Architecture built successfully'
            }
            
            result = await operation.build_architecture(sample_config)
            
            assert result['success'] is True
            assert 'model' in result
            assert 'layer_info' in result
            assert 'stats' in result
    
    @pytest.mark.asyncio
    async def test_build_architecture_failure(self, mock_operation_container, sample_config):
        """Test architecture building failure."""
        operation = BuildOperation(mock_operation_container)
        
        with patch.object(operation.backbone_service, 'build_backbone_architecture') as mock_build:
            mock_build.return_value = {
                'success': False,
                'error': 'Failed to build architecture',
                'message': 'Build failed'
            }
            
            result = await operation.build_architecture(sample_config)
            
            assert result['success'] is False
            assert 'error' in result


class TestSummaryOperation:
    """Test cases for SummaryOperation."""
    
    def test_init(self, mock_operation_container):
        """Test summary operation initialization."""
        operation = SummaryOperation(mock_operation_container)
        
        assert operation is not None
        assert hasattr(operation, 'backbone_service')
    
    def test_get_operations(self, mock_operation_container):
        """Test getting available operations."""
        operation = SummaryOperation(mock_operation_container)
        ops = operation.get_operations()
        
        assert 'summary' in ops
        assert callable(ops['summary'])
    
    @pytest.mark.asyncio
    async def test_generate_summary_success(self, mock_operation_container, sample_config):
        """Test successful summary generation."""
        operation = SummaryOperation(mock_operation_container)
        
        with patch.object(operation.backbone_service, 'generate_model_summary') as mock_summary:
            mock_summary.return_value = {
                'success': True,
                'summary': {
                    'backbone_type': 'efficientnet_b4',
                    'configuration': {'pretrained': True},
                    'capabilities': {'multi_scale_detection': True}
                },
                'analysis': {
                    'inference_speed': 'Medium',
                    'accuracy': 'Excellent'
                },
                'message': 'Summary generated successfully'
            }
            
            result = await operation.generate_summary(sample_config)
            
            assert result['success'] is True
            assert 'summary' in result
            assert 'analysis' in result
    
    @pytest.mark.asyncio
    async def test_generate_summary_failure(self, mock_operation_container, sample_config):
        """Test summary generation failure."""
        operation = SummaryOperation(mock_operation_container)
        
        with patch.object(operation.backbone_service, 'generate_model_summary') as mock_summary:
            mock_summary.return_value = {
                'success': False,
                'error': 'Failed to generate summary',
                'message': 'Summary generation failed'
            }
            
            result = await operation.generate_summary(sample_config)
            
            assert result['success'] is False
            assert 'error' in result


class TestOperationCallbacks:
    """Test operation callback functionality."""
    
    @pytest.mark.asyncio
    async def test_progress_callbacks(self, mock_operation_container, sample_config):
        """Test progress callback functionality."""
        operation = ValidateOperation(mock_operation_container)
        
        # Mock the service method to capture callbacks
        async def mock_validate(config, progress_callback, log_callback):
            # Test progress callback
            await progress_callback(1, 3, "Step 1")
            await progress_callback(2, 3, "Step 2")
            await progress_callback(3, 3, "Step 3")
            return {'valid': True, 'errors': [], 'warnings': []}
        
        with patch.object(operation.backbone_service, 'validate_backbone_config', side_effect=mock_validate):
            progress_callback = AsyncMock()
            log_callback = AsyncMock()
            
            result = await operation.validate_config(
                sample_config, progress_callback, log_callback
            )
            
            assert result['valid'] is True
            assert progress_callback.call_count == 3
    
    @pytest.mark.asyncio
    async def test_log_callbacks(self, mock_operation_container, sample_config):
        """Test log callback functionality."""
        operation = LoadOperation(mock_operation_container)
        
        # Mock the service method to capture callbacks
        async def mock_load(config, progress_callback, log_callback):
            # Test log callback
            await log_callback("INFO", "Starting load")
            await log_callback("SUCCESS", "Load completed")
            return {'success': True, 'backbone': Mock(), 'info': {}}
        
        with patch.object(operation.backbone_service, 'load_backbone_model', side_effect=mock_load):
            progress_callback = AsyncMock()
            log_callback = AsyncMock()
            
            result = await operation.load_model(
                sample_config, progress_callback, log_callback
            )
            
            assert result['success'] is True
            assert log_callback.call_count == 2
    
    def test_create_progress_callback(self, mock_operation_container):
        """Test progress callback creation."""
        operation = ValidateOperation(mock_operation_container)
        callback = operation._create_progress_callback()
        
        assert callable(callback)
        
        # Test callback execution
        import asyncio
        asyncio.run(callback(50, 100, "Test message"))
        
        # Should update operation container
        mock_operation_container.update_progress.assert_called_once()
    
    def test_create_log_callback(self, mock_operation_container):
        """Test log callback creation."""
        operation = LoadOperation(mock_operation_container)
        callback = operation._create_log_callback()
        
        assert callable(callback)
        
        # Test callback execution - this should call the log method
        import asyncio
        with patch.object(operation, 'log') as mock_log:
            asyncio.run(callback("INFO", "Test message"))
            mock_log.assert_called_once_with("Test message", "info")


if __name__ == '__main__':
    pytest.main([__file__])