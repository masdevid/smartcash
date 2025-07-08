"""
File: tests/unit/ui/model/backbone/test_backbone_service.py
Description: Unit tests for backbone service
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from smartcash.ui.model.backbone.services.backbone_service import BackboneService
from smartcash.ui.model.backbone.constants import BackboneOperation, BackboneType


@pytest.fixture
def backbone_service():
    """Create a backbone service instance for testing."""
    return BackboneService()


@pytest.fixture
def sample_config():
    """Sample backbone configuration for testing."""
    return {
        'backbone_type': 'efficientnet_b4',
        'pretrained': True,
        'feature_optimization': False,
        'advanced_settings': {
            'num_classes': 17,
            'input_size': 640
        }
    }


@pytest.fixture
def progress_callback():
    """Mock progress callback."""
    return AsyncMock()


@pytest.fixture
def log_callback():
    """Mock log callback."""
    return AsyncMock()


class TestBackboneService:
    """Test cases for BackboneService."""
    
    def test_init(self, backbone_service):
        """Test service initialization."""
        assert backbone_service is not None
        assert hasattr(backbone_service, 'backbone_factory')
        assert hasattr(backbone_service, 'model_builder')
        assert hasattr(backbone_service, 'logger')
    
    @pytest.mark.asyncio
    async def test_validate_backbone_config_success(self, backbone_service, sample_config, progress_callback, log_callback):
        """Test successful backbone configuration validation."""
        with patch.object(backbone_service, '_validate_config_format') as mock_validate, \
             patch.object(backbone_service, '_check_backbone_compatibility') as mock_compat:
            
            # Setup mocks
            mock_validate.return_value = {
                'valid': True,
                'errors': [],
                'warnings': []
            }
            mock_compat.return_value = {
                'compatible': True,
                'warnings': [],
                'recommendations': []
            }
            
            result = await backbone_service.validate_backbone_config(
                sample_config, progress_callback, log_callback
            )
            
            assert result['valid'] is True
            assert 'errors' in result
            assert 'warnings' in result
            assert progress_callback.call_count >= 3
            assert log_callback.call_count >= 2
    
    @pytest.mark.asyncio
    async def test_validate_backbone_config_invalid(self, backbone_service, sample_config, progress_callback, log_callback):
        """Test backbone configuration validation with invalid config."""
        with patch.object(backbone_service, '_validate_config_format') as mock_validate:
            # Setup mock to return invalid config
            mock_validate.return_value = {
                'valid': False,
                'errors': ['Missing required field: backbone_type'],
                'warnings': []
            }
            
            result = await backbone_service.validate_backbone_config(
                sample_config, progress_callback, log_callback
            )
            
            assert result['valid'] is False
            assert len(result['errors']) > 0
    
    @pytest.mark.asyncio
    async def test_load_backbone_model_success(self, backbone_service, sample_config, progress_callback, log_callback):
        """Test successful backbone model loading."""
        with patch.object(backbone_service.backbone_factory, 'create_backbone') as mock_create, \
             patch.object(backbone_service, '_get_backbone_info') as mock_info:
            
            # Setup mocks
            mock_backbone = Mock()
            mock_create.return_value = mock_backbone
            mock_info.return_value = {
                'total_parameters': 19000000,
                'model_size_mb': 76.0,
                'architecture': 'EfficientNet-B4'
            }
            
            result = await backbone_service.load_backbone_model(
                sample_config, progress_callback, log_callback
            )
            
            assert result['success'] is True
            assert 'backbone' in result
            assert 'info' in result
            assert mock_create.called
            assert progress_callback.call_count >= 4
    
    @pytest.mark.asyncio
    async def test_load_backbone_model_failure(self, backbone_service, sample_config, progress_callback, log_callback):
        """Test backbone model loading failure."""
        with patch.object(backbone_service.backbone_factory, 'create_backbone') as mock_create:
            # Setup mock to raise exception
            mock_create.side_effect = Exception("Failed to create backbone")
            
            result = await backbone_service.load_backbone_model(
                sample_config, progress_callback, log_callback
            )
            
            assert result['success'] is False
            assert 'error' in result
            assert 'Failed to create backbone' in result['error']
    
    @pytest.mark.asyncio
    async def test_build_backbone_architecture_success(self, backbone_service, sample_config, progress_callback, log_callback):
        """Test successful backbone architecture building."""
        with patch.object(backbone_service, '_convert_to_model_config') as mock_convert, \
             patch.object(backbone_service.model_builder, 'build_model') as mock_build, \
             patch.object(backbone_service, '_configure_model_layers') as mock_layers, \
             patch.object(backbone_service, '_calculate_model_stats') as mock_stats:
            
            # Setup mocks
            mock_convert.return_value = {'backbone': 'efficientnet_b4', 'num_classes': 17}
            mock_model = Mock()
            mock_build.return_value = mock_model
            mock_layers.return_value = {'total_layers': 10, 'backbone_layers': 8, 'head_layers': 2}
            mock_stats.return_value = {'total_parameters': 19000000, 'model_size_mb': 76.0}
            
            result = await backbone_service.build_backbone_architecture(
                sample_config, progress_callback, log_callback
            )
            
            assert result['success'] is True
            assert 'model' in result
            assert 'layer_info' in result
            assert 'stats' in result
            assert mock_build.called
    
    @pytest.mark.asyncio
    async def test_generate_model_summary_success(self, backbone_service, sample_config, progress_callback, log_callback):
        """Test successful model summary generation."""
        with patch.object(backbone_service, '_generate_summary_data') as mock_summary, \
             patch.object(backbone_service, '_analyze_model_performance') as mock_analysis:
            
            # Setup mocks
            mock_summary.return_value = {
                'backbone_type': 'efficientnet_b4',
                'configuration': {'pretrained': True},
                'capabilities': {'multi_scale_detection': True}
            }
            mock_analysis.return_value = {
                'inference_speed': 'Medium',
                'accuracy': 'Excellent'
            }
            
            result = await backbone_service.generate_model_summary(
                sample_config, progress_callback, log_callback
            )
            
            assert result['success'] is True
            assert 'summary' in result
            assert 'analysis' in result
    
    def test_get_available_backbones(self, backbone_service):
        """Test getting available backbones."""
        with patch.object(backbone_service.backbone_factory, 'list_available_backbones') as mock_list:
            mock_list.return_value = ['cspdarknet', 'efficientnet_b4']
            
            backbones = backbone_service.get_available_backbones()
            
            assert len(backbones) == 2
            assert 'cspdarknet' in backbones
            assert 'efficientnet_b4' in backbones
    
    def test_get_device_info(self, backbone_service):
        """Test getting device information."""
        with patch('smartcash.ui.model.backbone.services.backbone_service.get_device_info') as mock_device:
            mock_device.return_value = {
                'device_type': 'cuda',
                'gpu_memory_gb': 8
            }
            
            device_info = backbone_service.get_device_info()
            
            assert 'device_type' in device_info
            assert 'gpu_memory_gb' in device_info
    
    def test_validate_config_format_valid(self, backbone_service):
        """Test valid configuration format validation."""
        config = {'backbone_type': 'efficientnet_b4'}
        
        with patch.object(backbone_service, 'get_available_backbones') as mock_backbones:
            mock_backbones.return_value = ['efficientnet_b4', 'cspdarknet']
            
            result = backbone_service._validate_config_format(config)
            
            assert result['valid'] is True
            assert len(result['errors']) == 0
    
    def test_validate_config_format_missing_required(self, backbone_service):
        """Test configuration validation with missing required fields."""
        config = {}  # Missing backbone_type
        
        result = backbone_service._validate_config_format(config)
        
        assert result['valid'] is False
        assert len(result['errors']) > 0
        assert any('backbone_type' in error for error in result['errors'])
    
    def test_validate_config_format_invalid_backbone(self, backbone_service):
        """Test configuration validation with invalid backbone type."""
        config = {'backbone_type': 'invalid_backbone'}
        
        with patch.object(backbone_service, 'get_available_backbones') as mock_backbones:
            mock_backbones.return_value = ['efficientnet_b4', 'cspdarknet']
            
            result = backbone_service._validate_config_format(config)
            
            assert result['valid'] is False
            assert any('Invalid backbone type' in error for error in result['errors'])
    
    @pytest.mark.asyncio
    async def test_check_backbone_compatibility(self, backbone_service, sample_config):
        """Test backbone compatibility checking."""
        with patch.object(backbone_service, 'get_device_info') as mock_device:
            mock_device.return_value = {'gpu_memory_gb': 2}  # Low memory
            
            result = await backbone_service._check_backbone_compatibility(sample_config)
            
            assert result['compatible'] is True
            # Should have warning for EfficientNet-B4 with low memory
            assert len(result['warnings']) > 0
    
    def test_convert_to_model_config(self, backbone_service, sample_config):
        """Test configuration conversion for model builder."""
        model_config = backbone_service._convert_to_model_config(sample_config)
        
        assert model_config['backbone'] == 'efficientnet_b4'
        assert model_config['num_classes'] == 17
        assert model_config['input_size'] == 640
        assert model_config['pretrained'] is True


@pytest.mark.asyncio
class TestBackboneServiceAsync:
    """Async-specific test cases."""
    
    async def test_concurrent_operations(self, backbone_service, sample_config):
        """Test that multiple operations can be called concurrently."""
        with patch.object(backbone_service.backbone_factory, 'create_backbone'), \
             patch.object(backbone_service, '_validate_config_format') as mock_validate, \
             patch.object(backbone_service, '_check_backbone_compatibility'):
            
            mock_validate.return_value = {'valid': True, 'errors': [], 'warnings': []}
            
            # Run multiple operations concurrently
            tasks = [
                backbone_service.validate_backbone_config(sample_config),
                backbone_service.validate_backbone_config(sample_config),
                backbone_service.validate_backbone_config(sample_config)
            ]
            
            results = await asyncio.gather(*tasks)
            
            assert len(results) == 3
            assert all(result['valid'] for result in results)
    
    async def test_operation_cancellation(self, backbone_service, sample_config):
        """Test operation cancellation."""
        # This is a basic test - in practice, cancellation would be more complex
        with patch.object(backbone_service, '_validate_config_format') as mock_validate:
            async def slow_validate(*args, **kwargs):
                await asyncio.sleep(0.1)
                return {'valid': True, 'errors': [], 'warnings': []}
            
            mock_validate.side_effect = slow_validate
            
            task = asyncio.create_task(
                backbone_service.validate_backbone_config(sample_config)
            )
            
            # Cancel the task
            task.cancel()
            
            with pytest.raises(asyncio.CancelledError):
                await task


if __name__ == '__main__':
    pytest.main([__file__])