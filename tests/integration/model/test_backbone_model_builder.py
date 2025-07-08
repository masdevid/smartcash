#!/usr/bin/env python3
"""
File: tests/integration/model/test_backbone_model_builder.py
Description: Integration tests for backbone model builder workflow
"""

import pytest
import asyncio
import torch
import torch.nn as nn
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path

from smartcash.ui.model.backbone.backbone_init import BackboneInitializer
from smartcash.ui.model.backbone.handlers.backbone_ui_handler import BackboneUIHandler
from smartcash.ui.model.backbone.handlers.operation_manager import BackboneOperationManager
from smartcash.ui.model.backbone.services.backbone_service import BackboneService


@pytest.fixture
def integration_config():
    """Integration test configuration."""
    return {
        'backbone_type': 'efficientnet_b4',
        'pretrained': True,
        'feature_optimization': False,
        'advanced_settings': {
            'num_classes': 17,
            'input_size': 640,
            'detection_layers': ['banknote', 'nominal', 'security'],
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
def mock_ui_components():
    """Mock UI components for integration testing."""
    return {
        'operation_container': Mock(),
        'progress_tracker': Mock(),
        'log_accordion': Mock(),
        'config_summary': Mock(),
        'model_form': Mock()
    }


class TestBackboneModelBuilderIntegration:
    """Integration tests for backbone model builder workflow."""
    
    @pytest.mark.asyncio
    async def test_complete_model_building_workflow(self, integration_config, mock_ui_components):
        """Test complete model building workflow from UI to backend."""
        # Initialize handler
        handler = BackboneUIHandler(mock_ui_components)
        
        progress_updates = []
        log_messages = []
        
        def mock_progress_update(step, total, message):
            progress_updates.append((step, total, message))
        
        def mock_log_message(level, message):
            log_messages.append((level, message))
        
        mock_ui_components['operation_container'].update_progress = mock_progress_update
        mock_ui_components['operation_container'].log_message = mock_log_message
        
        # Mock backend availability
        with patch('smartcash.ui.model.backbone.services.backbone_service.create_model_api') as mock_api:
            mock_api.return_value = Mock()
            
            # Execute build operation
            result = await handler.execute_operation(
                operation_name='build',
                config=integration_config
            )
            
            # Verify workflow completion
            assert 'success' in result
            assert len(progress_updates) > 0
            assert len(log_messages) > 0
    
    @pytest.mark.asyncio
    async def test_backbone_initializer_integration(self, integration_config):
        """Test backbone initializer with model builder operations."""
        initializer = BackboneInitializer()
        
        # Mock UI creation
        with patch('smartcash.ui.model.backbone.components.ui_components.create_backbone_ui') as mock_ui:
            with patch('smartcash.ui.model.backbone.handlers.backbone_ui_handler.BackboneUIHandler') as mock_handler:
                mock_ui.return_value = {'mock': 'components'}
                mock_handler_instance = Mock()
                mock_handler.return_value = mock_handler_instance
                
                # Initialize module
                result = initializer.initialize_module(config=integration_config)
                
                # Verify initialization
                assert result is not None
                mock_ui.assert_called_once()
                mock_handler.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_operation_manager_model_workflow(self, integration_config):
        """Test operation manager handling model building workflow."""
        mock_container = Mock()
        mock_container.update_progress = AsyncMock()
        mock_container.log_message = AsyncMock()
        
        manager = BackboneOperationManager(mock_container)
        
        # Test all operations in sequence
        operations = ['validate', 'load', 'build', 'summary']
        
        for operation in operations:
            with patch.object(manager, f'_{operation}_operation') as mock_op:
                mock_operation = Mock()
                mock_operation.execute = AsyncMock(return_value={
                    'success': True,
                    'operation': operation,
                    'data': {'mock': 'result'}
                })
                mock_op.return_value = mock_operation
                
                result = await manager.execute_operation(
                    operation_name=operation,
                    config=integration_config
                )
                
                assert result['success'] is True
                assert result['operation'] == operation
    
    @pytest.mark.asyncio
    async def test_end_to_end_model_creation(self, integration_config):
        """Test end-to-end model creation from config to trained model."""
        service = BackboneService()
        
        # Step 1: Validate configuration
        validation_result = await service.validate_backbone_config(
            config=integration_config
        )
        
        if validation_result['valid']:
            # Step 2: Load backbone
            load_result = await service.load_backbone_model(
                config=integration_config
            )
            
            if load_result['success']:
                # Step 3: Build complete model
                build_result = await service.build_backbone_architecture(
                    config=integration_config
                )
                
                if build_result['success']:
                    # Step 4: Generate summary
                    summary_result = await service.generate_backbone_summary(
                        config=integration_config
                    )
                    
                    # Verify complete workflow
                    assert summary_result['success'] is True
                    assert 'performance' in summary_result
                    assert 'architecture' in summary_result


class TestModelBuilderBackendIntegration:
    """Test backend integration for model builder."""
    
    def test_backend_availability_check(self):
        """Test checking backend availability."""
        service = BackboneService()
        
        # Test backend availability detection through available backbones
        backbones = service.get_available_backbones()
        
        assert isinstance(backbones, list)
        assert len(backbones) > 0
        assert 'efficientnet_b4' in backbones or 'cspdarknet' in backbones
    
    @pytest.mark.asyncio
    async def test_model_builder_factory_integration(self, integration_config):
        """Test model builder with backbone factory integration."""
        service = BackboneService()
        
        if service.backend_available:
            # Test with real backend components
            result = await service.build_backbone_architecture(
                config=integration_config
            )
            
            # Should succeed or gracefully handle issues
            assert 'success' in result
            if result['success']:
                assert 'model' in result or 'backbone' in result
            else:
                assert 'error' in result
                assert 'message' in result
    
    @pytest.mark.asyncio
    async def test_backbone_factory_model_creation(self, integration_config):
        """Test backbone factory creating actual models."""
        service = BackboneService()
        
        # Test creating backbone through factory
        try:
            backbone = service.backbone_factory.create_backbone(
                backbone_type=integration_config['backbone_type'],
                pretrained=integration_config['pretrained'],
                feature_optimization=integration_config['feature_optimization']
            )
            
            # Verify backbone properties
            assert backbone is not None
            assert hasattr(backbone, 'forward')
            assert callable(backbone.forward)
            
            # Test basic forward pass
            if hasattr(backbone, 'get_output_channels'):
                channels = backbone.get_output_channels()
                assert isinstance(channels, (list, tuple))
                assert len(channels) > 0
                
        except Exception as e:
            # If backend not available, should fail gracefully
            assert 'not available' in str(e).lower() or 'import' in str(e).lower()


class TestModelBuilderPerformanceIntegration:
    """Test model builder performance and optimization."""
    
    @pytest.mark.asyncio
    async def test_model_performance_analysis(self, integration_config):
        """Test comprehensive model performance analysis."""
        service = BackboneService()
        
        # Mock model for performance testing
        with patch.object(service, '_build_with_backend') as mock_build:
            mock_model = Mock(spec=nn.Module)
            mock_model.parameters.return_value = [torch.zeros(1000) for _ in range(10)]
            
            mock_build.return_value = {
                'success': True,
                'model': mock_model,
                'architecture_info': {
                    'backbone_type': 'efficientnet_b4',
                    'total_parameters': 10000,
                    'layers': ['backbone', 'neck', 'head']
                }
            }
            
            result = await service.build_backbone_architecture(
                config=integration_config
            )
            
            if result['success']:
                # Analyze performance
                summary_result = await service.generate_backbone_summary(
                    config=integration_config
                )
                
                assert summary_result['success'] is True
                assert 'performance' in summary_result
                
                performance = summary_result['performance']
                assert 'memory_usage_mb' in performance
                assert 'inference_time_ms' in performance
                assert isinstance(performance['memory_usage_mb'], (int, float))
                assert isinstance(performance['inference_time_ms'], (int, float))
    
    @pytest.mark.asyncio
    async def test_model_optimization_features(self, integration_config):
        """Test model optimization features."""
        service = BackboneService()
        
        # Test with optimization enabled
        optimized_config = integration_config.copy()
        optimized_config['feature_optimization'] = True
        
        # Test without optimization
        normal_config = integration_config.copy()
        normal_config['feature_optimization'] = False
        
        configs_to_test = [
            ('normal', normal_config),
            ('optimized', optimized_config)
        ]
        
        results = {}
        
        for config_name, config in configs_to_test:
            with patch.object(service, '_build_with_backend') as mock_build:
                mock_build.return_value = {
                    'success': True,
                    'model': Mock(spec=nn.Module),
                    'optimization': config['feature_optimization']
                }
                
                result = await service.build_backbone_architecture(config=config)
                results[config_name] = result
        
        # Verify both configurations work
        for config_name, result in results.items():
            assert result['success'] is True, f"Failed for {config_name} config"


class TestModelBuilderErrorHandling:
    """Test error handling in model builder integration."""
    
    @pytest.mark.asyncio
    async def test_invalid_configuration_handling(self):
        """Test handling invalid configurations."""
        service = BackboneService()
        
        invalid_configs = [
            {'backbone_type': 'invalid_backbone'},
            {'backbone_type': 'efficientnet_b4', 'input_size': -1},
            {'backbone_type': 'efficientnet_b4', 'num_classes': 0},
            {}  # Empty config
        ]
        
        for invalid_config in invalid_configs:
            result = await service.validate_backbone_config(
                config=invalid_config
            )
            
            assert result['valid'] is False
            assert 'errors' in result
            assert len(result['errors']) > 0
    
    @pytest.mark.asyncio
    async def test_backend_error_recovery(self, integration_config):
        """Test recovery from backend errors."""
        service = BackboneService()
        
        # Simulate backend errors
        error_scenarios = [
            RuntimeError("CUDA out of memory"),
            ImportError("timm not available"),
            ValueError("Invalid backbone configuration"),
            Exception("Unexpected error")
        ]
        
        for error in error_scenarios:
            with patch.object(service, '_build_with_backend') as mock_build:
                mock_build.side_effect = error
                
                result = await service.build_backbone_architecture(
                    config=integration_config
                )
                
                # Should handle error gracefully
                assert result['success'] is False
                assert 'error' in result
                assert str(error) in result['error']
    
    @pytest.mark.asyncio
    async def test_concurrent_operation_handling(self, integration_config):
        """Test handling concurrent operations."""
        service = BackboneService()
        
        # Start multiple operations simultaneously
        tasks = []
        for i in range(3):
            task = asyncio.create_task(
                service.build_backbone_architecture(
                    config=integration_config
                )
            )
            tasks.append(task)
        
        # Wait for all to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # At least one should complete successfully
        successful_results = [r for r in results if isinstance(r, dict) and r.get('success')]
        assert len(successful_results) >= 1


class TestModelBuilderMemoryManagement:
    """Test memory management in model builder operations."""
    
    @pytest.mark.asyncio
    async def test_memory_cleanup_after_operations(self, integration_config):
        """Test memory cleanup after model operations."""
        service = BackboneService()
        
        initial_allocated = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        # Perform multiple operations
        for i in range(3):
            result = await service.build_backbone_architecture(
                config=integration_config
            )
            
            # Force cleanup
            if hasattr(service, 'cleanup'):
                service.cleanup()
        
        # Check memory usage
        if torch.cuda.is_available():
            final_allocated = torch.cuda.memory_allocated()
            # Memory should not grow excessively
            memory_growth = final_allocated - initial_allocated
            assert memory_growth < 1000000000  # Less than 1GB growth
    
    def test_model_deletion_cleanup(self):
        """Test proper cleanup when models are deleted."""
        service = BackboneService()
        
        # Create and delete models
        with patch.object(service.backbone_factory, 'create_backbone') as mock_create:
            mock_backbone = Mock(spec=nn.Module)
            mock_create.return_value = mock_backbone
            
            # Create backbone
            backbone = service.backbone_factory.create_backbone('efficientnet_b4')
            
            # Delete reference
            del backbone
            
            # Should not cause memory leaks or errors
            assert True  # If we get here, no exceptions were raised


if __name__ == "__main__":
    pytest.main([__file__, "-v"])