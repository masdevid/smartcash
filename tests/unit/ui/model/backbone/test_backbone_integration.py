"""
File: tests/unit/ui/model/backbone/test_backbone_integration.py
Description: Integration tests for backbone module refactoring
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from smartcash.ui.model.backbone.backbone_init import BackboneInitializer, initialize_backbone_ui
from smartcash.ui.model.backbone.handlers.backbone_ui_handler import BackboneUIHandler
from smartcash.ui.model.backbone.handlers.operation_manager import BackboneOperationManager
from smartcash.ui.model.backbone.services.backbone_service import BackboneService
from smartcash.ui.model.backbone.constants import BackboneOperation


@pytest.fixture
def mock_backbone_factory():
    """Mock backbone factory for testing."""
    factory = Mock()
    factory.create_backbone.return_value = Mock()
    factory.list_available_backbones.return_value = ['cspdarknet', 'efficientnet_b4']
    factory.validate_backbone_compatibility.return_value = True
    return factory


@pytest.fixture
def mock_model_builder():
    """Mock model builder for testing."""
    builder = Mock()
    model = Mock()
    model.parameters.return_value = [Mock()]
    builder.build_model.return_value = model
    return builder


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        'model': {
            'backbone': 'efficientnet_b4',
            'pretrained': True,
            'detection_layers': ['banknote'],
            'layer_mode': 'single',
            'feature_optimization': {
                'enabled': True,
                'use_attention': True,
                'testing_mode': False
            },
            'mixed_precision': True,
            'input_size': 640,
            'num_classes': 7
        },
        'ui': {
            'show_advanced_options': False,
            'auto_validate': True,
            'show_model_info': True
        },
        'validation': {
            'auto_validate_on_change': True,
            'show_compatibility_warnings': True
        }
    }


class TestBackboneIntegration:
    """Integration tests for backbone module components."""
    
    @patch('smartcash.ui.model.backbone.components.ui_components.create_backbone_ui_components')
    def test_backbone_initializer_integration(self, mock_create_components, sample_config):
        """Test backbone initializer integration with all components."""
        # Mock UI components
        mock_components = {
            'ui': Mock(),
            'progress_tracker': Mock(),
            'backbone_dropdown': Mock(),
            'validate_btn': Mock(),
            'load_btn': Mock(),
            'build_btn': Mock(),
            'summary_btn': Mock(),
            'save_button': Mock(),
            'reset_button': Mock()
        }
        mock_create_components.return_value = mock_components
        
        # Create initializer
        initializer = BackboneInitializer()
        
        with patch.object(initializer, 'setup_backbone_factory'):
            result = initializer._initialize_impl(sample_config)
            
            assert result['success'] is True
            assert 'ui_components' in result
            assert 'module_handler' in result
            assert result['ui_components'] == mock_components
    
    @patch('smartcash.ui.model.backbone.components.ui_components.create_backbone_ui_components')
    def test_ui_handler_integration(self, mock_create_components, sample_config):
        """Test UI handler integration with operation manager."""
        # Mock UI components
        mock_components = {
            'progress_tracker': Mock(),
            'backbone_dropdown': Mock(),
            'validate_btn': Mock(),
            'load_btn': Mock(),
            'build_btn': Mock(),
            'summary_btn': Mock(),
            'save_button': Mock(),
            'reset_button': Mock()
        }
        mock_create_components.return_value = mock_components
        
        # Create handler
        handler = BackboneUIHandler()
        handler.setup(mock_components)
        
        # Check that operation manager was created
        assert handler.operation_manager is not None
        assert isinstance(handler.operation_manager, BackboneOperationManager)
        
        # Check that UI components are set
        assert handler._ui_components == mock_components
    
    def test_service_integration_with_backend(self, mock_backbone_factory, mock_model_builder):
        """Test service integration with backend components."""
        with patch('smartcash.ui.model.backbone.services.backbone_service.BackboneFactory') as mock_factory_class, \
             patch('smartcash.ui.model.backbone.services.backbone_service.ModelBuilder') as mock_builder_class:
            
            mock_factory_class.return_value = mock_backbone_factory
            mock_builder_class.return_value = mock_model_builder
            
            service = BackboneService()
            
            # Test that backend components are properly initialized
            assert service.backbone_factory == mock_backbone_factory
            assert service.model_builder == mock_model_builder
    
    @pytest.mark.asyncio
    async def test_full_operation_flow(self, mock_backbone_factory, mock_model_builder, sample_config):
        """Test full operation flow from UI to backend."""
        with patch('smartcash.ui.model.backbone.services.backbone_service.BackboneFactory') as mock_factory_class, \
             patch('smartcash.ui.model.backbone.services.backbone_service.ModelBuilder') as mock_builder_class, \
             patch('smartcash.ui.model.backbone.services.backbone_service.get_device_info') as mock_device:
            
            # Setup mocks
            mock_factory_class.return_value = mock_backbone_factory
            mock_builder_class.return_value = mock_model_builder
            mock_device.return_value = {'device_type': 'cuda', 'gpu_memory_gb': 8}
            
            # Create operation manager
            operation_manager = BackboneOperationManager()
            
            # Test validate operation
            validate_result = await operation_manager.execute_operation(
                BackboneOperation.VALIDATE.value,
                {'backbone_type': 'efficientnet_b4', 'pretrained': True}
            )
            
            assert validate_result['valid'] is True
            
            # Test load operation
            load_result = await operation_manager.execute_operation(
                BackboneOperation.LOAD.value,
                {'backbone_type': 'efficientnet_b4', 'pretrained': True}
            )
            
            assert load_result['success'] is True
            assert mock_backbone_factory.create_backbone.called
    
    @pytest.mark.asyncio
    async def test_progress_and_logging_integration(self, sample_config):
        """Test progress tracking and logging integration."""
        # Mock operation container
        mock_container = Mock()
        mock_container.update_progress = Mock()
        mock_container.log_message = Mock()
        
        # Create operation manager with container
        operation_manager = BackboneOperationManager(mock_container)
        
        # Create async mock callbacks
        progress_callback = AsyncMock()
        log_callback = AsyncMock()
        
        with patch('smartcash.ui.model.backbone.services.backbone_service.BackboneFactory'), \
             patch('smartcash.ui.model.backbone.services.backbone_service.ModelBuilder'), \
             patch('smartcash.ui.model.backbone.services.backbone_service.get_device_info'):
            
            # Execute operation with callbacks
            result = await operation_manager.execute_operation(
                BackboneOperation.VALIDATE.value,
                {'backbone_type': 'efficientnet_b4'},
                progress_callback,
                log_callback
            )
            
            assert result['valid'] is True
            # Callbacks should have been called during operation
            assert progress_callback.call_count > 0
            assert log_callback.call_count > 0
    
    def test_error_handling_integration(self, sample_config):
        """Test error handling across all components."""
        # Test service error handling
        service = BackboneService()
        
        with patch.object(service, '_validate_config_format') as mock_validate:
            mock_validate.side_effect = Exception("Validation error")
            
            # Should handle exception gracefully
            result = asyncio.run(service.validate_backbone_config({'backbone_type': 'invalid'}))
            
            assert result['valid'] is False
            assert 'error' in result
    
    @patch('smartcash.ui.model.backbone.components.ui_components.create_backbone_ui_components')
    def test_constants_integration(self, mock_create_components):
        """Test that constants are properly used across components."""
        from smartcash.ui.model.backbone.constants import (
            BackboneOperation, BackboneType, PROGRESS_STEPS, BUTTON_CONFIG
        )
        
        # Mock UI components
        mock_components = {
            'ui': Mock(),
            'progress_tracker': Mock(),
            'validate_btn': Mock(),
            'load_btn': Mock(),
            'build_btn': Mock(),
            'summary_btn': Mock()
        }
        mock_create_components.return_value = mock_components
        
        # Test that all operation types are available
        operations = [op.value for op in BackboneOperation]
        assert 'validate' in operations
        assert 'load' in operations
        assert 'build' in operations
        assert 'summary' in operations
        
        # Test that progress steps are defined for all operations
        for operation in operations:
            assert operation in PROGRESS_STEPS
            assert len(PROGRESS_STEPS[operation]) > 0
        
        # Test that button configs are defined
        for operation in ['validate', 'load', 'build', 'summary']:
            assert operation in BUTTON_CONFIG
            assert 'text' in BUTTON_CONFIG[operation]
            assert 'style' in BUTTON_CONFIG[operation]
    
    @patch('smartcash.ui.model.backbone.components.ui_components.create_backbone_ui_components')
    def test_config_flow_integration(self, mock_create_components, sample_config):
        """Test configuration flow from defaults to UI to operations."""
        # Mock UI components with form widgets
        mock_dropdown = Mock()
        mock_dropdown.value = 'efficientnet_b4'
        
        mock_components = {
            'ui': Mock(),
            'progress_tracker': Mock(),
            'backbone_dropdown': mock_dropdown,
            'detection_layers_select': Mock(),
            'layer_mode_dropdown': Mock(),
            'feature_optimization_checkbox': Mock(),
            'mixed_precision_checkbox': Mock(),
            'validate_btn': Mock(),
            'save_button': Mock(),
            'reset_button': Mock()
        }
        mock_components['detection_layers_select'].value = ('banknote',)
        mock_components['layer_mode_dropdown'].value = 'single'
        mock_components['feature_optimization_checkbox'].value = True
        mock_components['mixed_precision_checkbox'].value = True
        
        mock_create_components.return_value = mock_components
        
        # Create handler and setup
        handler = BackboneUIHandler()
        handler.setup(mock_components)
        
        # Extract config from UI
        extracted_config = handler.extract_config_from_ui()
        
        # Check that config extraction works
        assert 'model' in extracted_config
        assert extracted_config['model']['backbone'] == 'efficientnet_b4'
        
        # Test config update to UI
        new_config = sample_config.copy()
        new_config['model']['backbone'] = 'cspdarknet'
        
        handler.update_ui_from_config(new_config)
        
        # Check that UI was updated
        assert mock_dropdown.value == 'cspdarknet'


class TestBackboneErrorRecovery:
    """Test error recovery and resilience."""
    
    @patch('smartcash.ui.model.backbone.components.ui_components.create_backbone_ui_components')
    def test_initialization_failure_recovery(self, mock_create_components):
        """Test recovery from initialization failures."""
        # Make component creation fail
        mock_create_components.side_effect = Exception("Component creation failed")
        
        initializer = BackboneInitializer()
        
        # Should handle failure gracefully
        result = initializer._initialize_impl()
        
        assert result['success'] is False
        assert 'error' in result
        assert result['ui_components'] == {}
    
    def test_operation_failure_recovery(self):
        """Test recovery from operation failures."""
        operation_manager = BackboneOperationManager()
        
        # Test that manager handles unknown operations gracefully
        result = asyncio.run(operation_manager.execute_operation('unknown', {}))
        
        assert result['success'] is False
        assert 'Unknown operation type' in result['error']
    
    @patch('smartcash.ui.model.backbone.components.ui_components.create_backbone_ui_components')
    def test_ui_handler_partial_setup(self, mock_create_components):
        """Test UI handler with partial component setup."""
        # Provide only some components
        partial_components = {
            'progress_tracker': Mock(),
            'backbone_dropdown': Mock()
            # Missing other components
        }
        mock_create_components.return_value = partial_components
        
        handler = BackboneUIHandler()
        
        # Should handle partial setup gracefully
        handler.setup(partial_components)
        
        assert handler._ui_components == partial_components
        assert handler.operation_manager is not None


if __name__ == '__main__':
    pytest.main([__file__])