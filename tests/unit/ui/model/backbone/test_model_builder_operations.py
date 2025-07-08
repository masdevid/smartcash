#!/usr/bin/env python3
"""
File: tests/unit/ui/model/backbone/test_model_builder_operations.py
Description: Comprehensive tests for model builder operations in backbone module
"""

import pytest
import asyncio
import torch
import torch.nn as nn
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from pathlib import Path

from smartcash.ui.model.backbone.services.backbone_service import BackboneService
from smartcash.ui.model.backbone.operations.build_operation import BuildOperation
from smartcash.ui.model.backbone.constants import BackboneOperation, BackboneType


@pytest.fixture
def mock_operation_container():
    """Mock operation container for testing."""
    container = Mock()
    container.update_progress = AsyncMock()
    container.log_message = AsyncMock()
    return container


@pytest.fixture
def sample_model_config():
    """Sample model configuration for testing."""
    return {
        'backbone_type': 'efficientnet_b4',
        'pretrained': True,
        'feature_optimization': False,
        'advanced_settings': {
            'num_classes': 17,
            'input_size': 640,
            'detection_layers': ['banknote', 'nominal'],
            'layer_mode': 'multilayer'
        },
        'model_settings': {
            'backbone': 'efficientnet_b4',
            'detection_layers': ['banknote'],
            'layer_mode': 'single',
            'feature_optimization': {'enabled': False}
        }
    }


@pytest.fixture
def mock_backbone():
    """Mock backbone model for testing."""
    backbone = Mock(spec=nn.Module)
    backbone.get_output_channels = Mock(return_value=[512, 256, 128])
    backbone.parameters = Mock(return_value=[torch.zeros(10), torch.zeros(20)])
    backbone.eval = Mock()
    backbone.train = Mock()
    backbone.forward = Mock(return_value=torch.randn(1, 10))
    return backbone


@pytest.fixture
def mock_model_builder():
    """Mock model builder for testing."""
    builder = Mock()
    builder.build = Mock()
    return builder


class TestModelBuilderIntegration:
    """Test model builder integration with backbone service."""
    
    def test_backbone_service_initialization(self):
        """Test backbone service initializes model builder correctly."""
        service = BackboneService()
        
        assert service is not None
        assert hasattr(service, 'backbone_factory')
        assert hasattr(service, 'model_builder')
        assert hasattr(service, 'logger')
    
    @patch('smartcash.ui.model.backbone.services.backbone_service.BackboneFactory')
    def test_backbone_factory_integration(self, mock_factory_class):
        """Test backbone factory integration."""
        # Setup mock
        mock_factory = Mock()
        mock_factory_class.return_value = mock_factory
        mock_factory.create_backbone.return_value = Mock()
        
        service = BackboneService()
        
        # Verify factory initialization
        mock_factory_class.assert_called_once()
        assert service.backbone_factory == mock_factory
    
    @pytest.mark.asyncio
    async def test_build_operation_with_model_builder(self, mock_operation_container, sample_model_config):
        """Test build operation integrates with model builder."""
        build_op = BuildOperation(mock_operation_container)
        
        progress_updates = []
        log_messages = []
        
        async def progress_callback(step, total, message):
            progress_updates.append((step, total, message))
        
        async def log_callback(level, message):
            log_messages.append((level, message))
        
        with patch.object(build_op.backbone_service, 'build_backbone_architecture') as mock_build:
            mock_build.return_value = {
                'success': True,
                'model': Mock(spec=nn.Module),
                'architecture_info': {
                    'backbone_type': 'efficientnet_b4',
                    'total_parameters': 1000000,
                    'model_size_mb': 25.5,
                    'layers': ['backbone', 'neck', 'head']
                },
                'performance_info': {
                    'memory_usage_mb': 150.0,
                    'inference_time_ms': 45.2
                }
            }
            
            result = await build_op.build_architecture(
                config=sample_model_config,
                progress_callback=progress_callback,
                log_callback=log_callback
            )
            
            # Verify build was called with correct config
            mock_build.assert_called_once()
            call_args = mock_build.call_args
            assert call_args[1]['config'] == sample_model_config
            
            # Verify result structure
            assert result['success'] is True
            assert 'model' in result
            assert 'architecture_info' in result
            
            # Verify progress and logging
            assert len(progress_updates) > 0
            assert len(log_messages) > 0


class TestBackboneArchitectureBuilding:
    """Test backbone architecture building with model builder."""
    
    @pytest.mark.asyncio
    @patch('smartcash.model.core.model_builder.ModelBuilder')
    @patch('smartcash.model.utils.progress_bridge.ModelProgressBridge')
    async def test_build_complete_model(self, mock_progress_bridge, mock_model_builder_class, sample_model_config):
        """Test building complete model through backbone service."""
        # Setup mocks
        mock_model = Mock(spec=nn.Module)
        mock_builder = Mock()
        mock_builder.build.return_value = mock_model
        mock_model_builder_class.return_value = mock_builder
        
        service = BackboneService()
        
        progress_updates = []
        log_messages = []
        
        async def progress_callback(step, total, message):
            progress_updates.append((step, total, message))
        
        async def log_callback(level, message):
            log_messages.append((level, message))
        
        result = await service.build_backbone_architecture(
            config=sample_model_config,
            progress_callback=progress_callback,
            log_callback=log_callback
        )
        
        # Verify result
        assert result['success'] is True
        assert 'model' in result or 'backbone' in result
        assert len(progress_updates) >= 4  # Should have multiple progress steps
        assert len(log_messages) > 0
    
    @pytest.mark.asyncio
    async def test_build_architecture_with_different_backbones(self, sample_model_config):
        """Test building architecture with different backbone types."""
        service = BackboneService()
        
        backbone_types = ['cspdarknet', 'efficientnet_b4']
        
        for backbone_type in backbone_types:
            config = sample_model_config.copy()
            config['backbone_type'] = backbone_type
            
            with patch.object(service, '_build_with_backend') as mock_build:
                mock_build.return_value = {
                    'success': True,
                    'model': Mock(spec=nn.Module),
                    'architecture_info': {'backbone_type': backbone_type}
                }
                
                result = await service.build_backbone_architecture(
                    config=config,
                    progress_callback=None,
                    log_callback=None
                )
                
                assert result['success'] is True
                assert result['architecture_info']['backbone_type'] == backbone_type
    
    @pytest.mark.asyncio
    async def test_build_architecture_error_handling(self, sample_model_config):
        """Test error handling during architecture building."""
        service = BackboneService()
        
        with patch.object(service, '_build_with_backend') as mock_build:
            mock_build.side_effect = Exception("Model building failed")
            
            result = await service.build_backbone_architecture(
                config=sample_model_config,
                progress_callback=None,
                log_callback=None
            )
            
            assert result['success'] is False
            assert 'error' in result
            assert 'Model building failed' in result['error']


class TestModelParameterAnalysis:
    """Test model parameter analysis functionality."""
    
    def test_model_parameter_counting(self, mock_backbone):
        """Test counting model parameters."""
        service = BackboneService()
        
        # Mock parameter counting
        with patch('torch.sum') as mock_sum:
            mock_sum.return_value = torch.tensor(1000000)
            
            param_count = service._count_model_parameters(mock_backbone)
            
            assert isinstance(param_count, int)
            assert param_count >= 0
    
    def test_model_size_calculation(self, mock_backbone):
        """Test model size calculation in MB."""
        service = BackboneService()
        
        # Test with known parameter count
        with patch.object(service, '_count_model_parameters') as mock_count:
            mock_count.return_value = 1000000  # 1M parameters
            
            size_mb = service._calculate_model_size_mb(mock_backbone)
            
            assert isinstance(size_mb, float)
            assert size_mb > 0
    
    def test_backbone_info_extraction(self, mock_backbone):
        """Test extracting comprehensive backbone information."""
        service = BackboneService()
        
        with patch.object(service, '_count_model_parameters') as mock_count:
            with patch.object(service, '_calculate_model_size_mb') as mock_size:
                mock_count.return_value = 1500000
                mock_size.return_value = 30.5
                
                info = service._get_backbone_info(mock_backbone)
                
                assert 'parameters' in info
                assert 'size_mb' in info
                assert 'output_channels' in info
                assert info['parameters'] == 1500000
                assert info['size_mb'] == 30.5


class TestModelBuilderOperationManager:
    """Test operation manager integration with model builder."""
    
    @pytest.mark.asyncio
    async def test_operation_manager_build_workflow(self, mock_operation_container, sample_model_config):
        """Test complete build workflow through operation manager."""
        from smartcash.ui.model.backbone.handlers.operation_manager import BackboneOperationManager
        
        manager = BackboneOperationManager(mock_operation_container)
        
        progress_updates = []
        log_messages = []
        
        async def progress_callback(step, total, message):
            progress_updates.append((step, total, message))
        
        async def log_callback(level, message):
            log_messages.append((level, message))
        
        with patch.object(manager, 'build_operation') as mock_build_op:
            mock_operation = AsyncMock()
            mock_operation.build_architecture.return_value = {
                'success': True,
                'model': Mock(spec=nn.Module),
                'architecture_info': {'backbone_type': 'efficientnet_b4'}
            }
            mock_build_op.return_value = mock_operation
            
            result = await manager.execute_operation(
                operation_name='build',
                config=sample_model_config,
                progress_callback=progress_callback,
                log_callback=log_callback
            )
            
            assert result['success'] is True
            mock_operation.build_architecture.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_concurrent_model_operations(self, mock_operation_container, sample_model_config):
        """Test handling concurrent model building operations."""
        from smartcash.ui.model.backbone.handlers.operation_manager import BackboneOperationManager
        
        manager = BackboneOperationManager(mock_operation_container)
        
        # Mock successful operations
        with patch.object(manager, '_execute_build_operation') as mock_build:
            mock_build.return_value = {
                'success': True,
                'model': Mock(spec=nn.Module)
            }
            
            # Start first operation
            task1 = asyncio.create_task(
                manager.execute_operation('build', sample_model_config)
            )
            
            # Give it time to start
            await asyncio.sleep(0.1)
            
            # Try to start second operation (should be rejected)
            result2 = await manager.execute_operation('build', sample_model_config)
            
            # Wait for first to complete
            result1 = await task1
            
            # First should succeed, second should indicate busy
            assert result1['success'] is True
            assert 'busy' in result2 or 'in progress' in result2.get('message', '').lower()


class TestModelValidationOperations:
    """Test model validation operations."""
    
    @pytest.mark.asyncio
    async def test_validate_model_configuration(self, sample_model_config):
        """Test validating model configuration."""
        service = BackboneService()
        
        result = await service.validate_backbone_config(
            config=sample_model_config,
            progress_callback=None,
            log_callback=None
        )
        
        assert 'valid' in result
        assert isinstance(result['valid'], bool)
        
        if result['valid']:
            assert 'errors' not in result or len(result['errors']) == 0
        else:
            assert 'errors' in result
            assert len(result['errors']) > 0
    
    @pytest.mark.asyncio
    async def test_validate_invalid_backbone_type(self, sample_model_config):
        """Test validation with invalid backbone type."""
        service = BackboneService()
        
        # Test with invalid backbone
        config = sample_model_config.copy()
        config['backbone_type'] = 'invalid_backbone'
        
        result = await service.validate_backbone_config(
            config=config,
            progress_callback=None,
            log_callback=None
        )
        
        assert result['valid'] is False
        assert 'errors' in result
        assert len(result['errors']) > 0
    
    @pytest.mark.asyncio
    async def test_validate_model_compatibility(self, sample_model_config):
        """Test validating model compatibility."""
        service = BackboneService()
        
        # Test device compatibility
        with patch('smartcash.model.utils.device_utils.get_device_info') as mock_device:
            mock_device.return_value = {
                'device': 'cpu',
                'memory_available': 8000,  # 8GB
                'cuda_available': False
            }
            
            result = await service.validate_backbone_config(
                config=sample_model_config,
                progress_callback=None,
                log_callback=None
            )
            
            assert 'valid' in result
            assert 'device_compatibility' in result or 'warnings' in result


class TestModelPerformanceAnalysis:
    """Test model performance analysis operations."""
    
    @pytest.mark.asyncio
    async def test_analyze_model_performance(self, sample_model_config, mock_backbone):
        """Test analyzing model performance."""
        service = BackboneService()
        
        with patch.object(service, '_estimate_inference_time') as mock_inference:
            with patch.object(service, '_estimate_memory_usage') as mock_memory:
                mock_inference.return_value = 45.2
                mock_memory.return_value = 150.0
                
                result = await service.generate_backbone_summary(
                    config=sample_model_config,
                    progress_callback=None,
                    log_callback=None
                )
                
                assert result['success'] is True
                assert 'performance' in result
                assert 'inference_time_ms' in result['performance']
                assert 'memory_usage_mb' in result['performance']
    
    def test_memory_usage_estimation(self, mock_backbone):
        """Test memory usage estimation."""
        service = BackboneService()
        
        # Mock parameter count for memory calculation
        with patch.object(service, '_count_model_parameters') as mock_count:
            mock_count.return_value = 1000000  # 1M parameters
            
            memory_mb = service._estimate_memory_usage(mock_backbone, batch_size=1)
            
            assert isinstance(memory_mb, float)
            assert memory_mb > 0
    
    def test_inference_time_estimation(self, mock_backbone):
        """Test inference time estimation."""
        service = BackboneService()
        
        with patch('torch.randn') as mock_input:
            with patch('time.time') as mock_time:
                mock_input.return_value = torch.randn(1, 3, 640, 640)
                mock_time.side_effect = [0.0, 0.045]  # 45ms
                
                # Mock model forward pass
                mock_backbone.forward = Mock(return_value=torch.randn(1, 10))
                mock_backbone.eval = Mock()
                
                inference_time = service._estimate_inference_time(mock_backbone)
                
                assert isinstance(inference_time, float)
                assert inference_time >= 0


class TestErrorHandlingAndRecovery:
    """Test error handling and recovery in model operations."""
    
    @pytest.mark.asyncio
    async def test_handle_backend_unavailable(self, sample_model_config):
        """Test handling when backend is unavailable."""
        service = BackboneService()
        
        # Force backend unavailable
        service.backend_available = False
        
        result = await service.build_backbone_architecture(
            config=sample_model_config,
            progress_callback=None,
            log_callback=None
        )
        
        assert result['success'] is False
        assert 'backend not available' in result.get('message', '').lower()
    
    @pytest.mark.asyncio
    async def test_handle_memory_error(self, sample_model_config):
        """Test handling memory errors during model building."""
        service = BackboneService()
        
        with patch.object(service, '_build_with_backend') as mock_build:
            mock_build.side_effect = RuntimeError("CUDA out of memory")
            
            result = await service.build_backbone_architecture(
                config=sample_model_config,
                progress_callback=None,
                log_callback=None
            )
            
            assert result['success'] is False
            assert 'memory' in result['error'].lower()
    
    @pytest.mark.asyncio
    async def test_operation_timeout_handling(self, mock_operation_container, sample_model_config):
        """Test handling operation timeouts."""
        build_op = BuildOperation(mock_operation_container)
        
        with patch.object(build_op.backbone_service, 'build_backbone_architecture') as mock_build:
            # Simulate long-running operation
            async def slow_operation(*args, **kwargs):
                await asyncio.sleep(10)  # 10 seconds
                return {'success': True}
            
            mock_build.side_effect = slow_operation
            
            # Test with timeout
            try:
                result = await asyncio.wait_for(
                    build_op.build_architecture(sample_model_config),
                    timeout=1.0  # 1 second timeout
                )
                assert False, "Should have timed out"
            except asyncio.TimeoutError:
                # Expected timeout
                pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])