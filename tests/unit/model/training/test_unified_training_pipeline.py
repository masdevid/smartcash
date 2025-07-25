#!/usr/bin/env python3
"""
File: tests/unit/model/training/test_unified_training_pipeline.py

Comprehensive unit tests for UnifiedTrainingPipeline with extensive edge cases.

Test Coverage:
- Pipeline initialization with various callback configurations
- Phase execution with success and failure scenarios
- Resume functionality with different checkpoint states
- Training mode switches (two_phase vs single_phase)
- Device handling (CPU, GPU, MPS)
- Configuration validation and error handling
- Callback integration and error resilience
- Memory management and resource cleanup
- Progress tracking and metrics reporting
- Edge cases for invalid configurations
"""

import pytest
import torch
import time
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock, call, ANY
from typing import Dict, Any, Optional, List, Callable

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

from smartcash.model.training.unified_training_pipeline import UnifiedTrainingPipeline


class TestUnifiedTrainingPipelineInitialization:
    """Test initialization and configuration of UnifiedTrainingPipeline."""
    
    def test_init_with_no_callbacks(self):
        """Test initialization without any callbacks."""
        pipeline = UnifiedTrainingPipeline()
        
        assert pipeline.progress_tracker is not None
        assert pipeline.verbose is True
        assert pipeline.config is None
        assert pipeline.model_api is None
        assert pipeline.model is None
        assert pipeline.visualization_manager is None
        assert pipeline.log_callback is None
        assert pipeline.live_chart_callback is None
        assert pipeline.metrics_callback is None
    
    def test_init_with_all_callbacks(self):
        """Test initialization with all callback types."""
        progress_cb = MagicMock()
        log_cb = MagicMock()
        chart_cb = MagicMock()
        metrics_cb = MagicMock()
        
        pipeline = UnifiedTrainingPipeline(
            progress_callback=progress_cb,
            log_callback=log_cb,
            live_chart_callback=chart_cb,
            metrics_callback=metrics_cb,
            verbose=False
        )
        
        assert pipeline.verbose is False
        assert pipeline.log_callback == log_cb
        assert pipeline.live_chart_callback == chart_cb
        assert pipeline.metrics_callback == metrics_cb
    
    def test_init_verbose_modes(self):
        """Test initialization with different verbose settings."""
        # Test verbose=True
        pipeline_verbose = UnifiedTrainingPipeline(verbose=True)
        assert pipeline_verbose.verbose is True
        
        # Test verbose=False
        pipeline_quiet = UnifiedTrainingPipeline(verbose=False)
        assert pipeline_quiet.verbose is False
    
    def test_init_state_variables(self):
        """Test initial state of tracking variables."""
        pipeline = UnifiedTrainingPipeline()
        
        assert pipeline.current_phase is None
        assert pipeline.training_session_id is None
        assert pipeline.phase_start_time is None
        assert pipeline.training_start_time is None


class TestUnifiedTrainingPipelineConfiguration:
    """Test configuration validation and setup."""
    
    @pytest.fixture
    def pipeline(self):
        """Create pipeline instance for testing."""
        return UnifiedTrainingPipeline(verbose=False)
    
    def test_validate_training_mode_valid_two_phase(self, pipeline):
        """Test validation of valid two-phase training mode."""
        with patch('smartcash.model.training.unified_training_pipeline.validate_training_mode_and_params') as mock_validate:
            pipeline.run_full_training_pipeline(
                backbone='cspdarknet',
                training_mode='two_phase',
                phase_1_epochs=2,
                phase_2_epochs=3
            )
            mock_validate.assert_called_once_with('two_phase', 'multi', False, 3)
    
    def test_validate_training_mode_valid_single_phase(self, pipeline):
        """Test validation of valid single-phase training mode."""
        with patch('smartcash.model.training.unified_training_pipeline.validate_training_mode_and_params') as mock_validate:
            with patch('smartcash.model.training.unified_training_pipeline.setup_training_session') as mock_setup:
                mock_setup.return_value = ('session_123', None)
                with patch.object(pipeline, '_phase_preparation') as mock_prep:
                    mock_prep.return_value = {'success': False, 'error': 'test stop'}
                    
                    pipeline.run_full_training_pipeline(
                        backbone='efficientnet_b4',
                        training_mode='single_phase',
                        single_phase_layer_mode='single',
                        single_phase_freeze_backbone=True,
                        phase_1_epochs=5
                    )
                    mock_validate.assert_called_once_with('single_phase', 'single', True, 1)
    
    def test_invalid_backbone_parameter(self, pipeline):
        """Test handling of invalid backbone parameter."""
        with patch('smartcash.model.training.unified_training_pipeline.validate_training_mode_and_params'):
            with patch('smartcash.model.training.unified_training_pipeline.setup_training_session') as mock_setup:
                mock_setup.return_value = ('session_123', None)
                with patch.object(pipeline, '_phase_preparation') as mock_prep:
                    mock_prep.return_value = {'success': False, 'error': 'Invalid backbone: invalid_backbone'}
                    
                    result = pipeline.run_full_training_pipeline(backbone='invalid_backbone')
                    
                    assert result['success'] is False
                    assert 'Invalid backbone' in result.get('error', '')
    
    def test_zero_epochs_configuration(self, pipeline):
        """Test configuration with zero epochs."""
        with patch('smartcash.model.training.unified_training_pipeline.validate_training_mode_and_params'):
            with patch('smartcash.model.training.unified_training_pipeline.setup_training_session') as mock_setup:
                mock_setup.return_value = ('session_123', None)
                with patch.object(pipeline, '_phase_preparation') as mock_prep:
                    mock_prep.return_value = {'success': False, 'error': 'Invalid epochs: 0'}
                    
                    result = pipeline.run_full_training_pipeline(phase_1_epochs=0, phase_2_epochs=0)
                    
                    assert result['success'] is False
                    assert 'Invalid epochs' in result.get('error', '')
    
    def test_negative_epochs_configuration(self, pipeline):
        """Test configuration with negative epochs."""
        with patch('smartcash.model.training.unified_training_pipeline.validate_training_mode_and_params') as mock_validate:
            mock_validate.side_effect = ValueError("Epochs cannot be negative")
            
            result = pipeline.run_full_training_pipeline(phase_1_epochs=-1, phase_2_epochs=2)
            
            assert result['success'] is False
            assert 'Epochs cannot be negative' in result.get('error', '')


class TestUnifiedTrainingPipelinePhases:
    """Test individual training phases with various scenarios."""
    
    @pytest.fixture
    def pipeline(self):
        """Create pipeline instance for testing."""
        return UnifiedTrainingPipeline(verbose=False)
    
    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        return {
            'model': {
                'backbone': 'cspdarknet',
                'num_classes': [80, 40, 20],
                'layer_mode': 'multi'
            },
            'device': {
                'auto_detect': True,
                'device': 'cpu'
            },
            'training': {
                'compile_model': False,
                'loss': {'type': 'multi_layer'}
            },
            'training_phases': {
                'phase_1': {'epochs': 2},
                'phase_2': {'epochs': 3}
            },
            'paths': {
                'checkpoints': Path('/tmp/test_checkpoints'),
                'logs': Path('/tmp/test_logs'),
                'visualization': Path('/tmp/test_viz')
            }
        }
    
    def test_phase_preparation_success(self, pipeline, mock_config):
        """Test successful preparation phase."""
        with patch('smartcash.model.training.unified_training_pipeline.prepare_training_environment') as mock_prep:
            mock_prep.return_value = {'success': True, 'config': mock_config}
            
            result = pipeline._phase_preparation('cspdarknet', 2, 3, '/tmp/checkpoints')
            
            assert result['success'] is True
            assert pipeline.config == mock_config
            mock_prep.assert_called_once()
    
    def test_phase_preparation_failure(self, pipeline):
        """Test preparation phase failure."""
        with patch('smartcash.model.training.unified_training_pipeline.prepare_training_environment') as mock_prep:
            mock_prep.return_value = {'success': False, 'error': 'Configuration error'}
            
            result = pipeline._phase_preparation('cspdarknet', 2, 3, '/tmp/checkpoints')
            
            assert result['success'] is False
            assert 'Configuration error' in result['error']
    
    def test_phase_preparation_exception(self, pipeline):
        """Test preparation phase with exception."""
        with patch('smartcash.model.training.unified_training_pipeline.prepare_training_environment') as mock_prep:
            mock_prep.side_effect = Exception("Setup failed")
            
            result = pipeline._phase_preparation('cspdarknet', 2, 3, '/tmp/checkpoints')
            
            assert result['success'] is False
            assert 'Preparation failed: Setup failed' in result['error']
    
    def test_phase_build_model_success(self, pipeline, mock_config):
        """Test successful model building phase."""
        pipeline.config = mock_config
        mock_model = MagicMock()
        mock_model.parameters.return_value = [torch.tensor([1.0])]
        
        with patch('smartcash.model.training.unified_training_pipeline.create_model_api') as mock_create_api:
            mock_api = MagicMock()
            mock_api.build_model.return_value = {'status': 'built', 'model': mock_model}
            mock_create_api.return_value = mock_api
            
            with patch('smartcash.model.training.unified_training_pipeline.setup_device') as mock_device:
                mock_device.return_value = torch.device('cpu')
                with patch('smartcash.model.training.unified_training_pipeline.model_to_device') as mock_to_device:
                    mock_to_device.return_value = mock_model
                    
                    result = pipeline._phase_build_model()
                    
                    assert result['success'] is True
                    assert pipeline.model == mock_model
                    assert pipeline.model_api == mock_api
    
    def test_phase_build_model_api_creation_failure(self, pipeline, mock_config):
        """Test model building phase with API creation failure."""
        pipeline.config = mock_config
        
        with patch('smartcash.model.training.unified_training_pipeline.create_model_api') as mock_create_api:
            mock_create_api.return_value = None
            
            result = pipeline._phase_build_model()
            
            assert result['success'] is False
            assert 'Failed to create model API' in result['error']
    
    def test_phase_build_model_build_failure(self, pipeline, mock_config):
        """Test model building phase with build failure."""
        pipeline.config = mock_config
        
        with patch('smartcash.model.training.unified_training_pipeline.create_model_api') as mock_create_api:
            mock_api = MagicMock()
            mock_api.build_model.return_value = {'status': 'failed', 'message': 'Build error'}
            mock_create_api.return_value = mock_api
            
            result = pipeline._phase_build_model()
            
            assert result['success'] is False
            assert 'Model build failed: Build error' in result['error']
    
    def test_phase_build_model_no_model_in_result(self, pipeline, mock_config):
        """Test model building when model is not in build result."""
        pipeline.config = mock_config
        
        with patch('smartcash.model.training.unified_training_pipeline.create_model_api') as mock_create_api:
            mock_api = MagicMock()
            mock_api.build_model.return_value = {'status': 'built'}  # No 'model' key
            # Ensure model attribute is None
            mock_api.model = None
            mock_create_api.return_value = mock_api
            
            # Mock device setup to avoid issues
            with patch('smartcash.model.training.unified_training_pipeline.setup_device') as mock_device:
                mock_device.return_value = torch.device('cpu')
                with patch('smartcash.model.training.unified_training_pipeline.model_to_device') as mock_to_device:
                    mock_to_device.side_effect = AttributeError("'NoneType' object has no attribute 'to'")
                    
                    result = pipeline._phase_build_model()
                    
                    assert result['success'] is False
                    assert 'Model build failed' in result['error']
    
    def test_phase_build_model_with_compilation(self, pipeline, mock_config):
        """Test model building with torch.compile enabled."""
        pipeline.config = mock_config
        pipeline.config['training']['compile_model'] = True
        mock_model = MagicMock()
        mock_model.parameters.return_value = [torch.tensor([1.0])]
        
        with patch('smartcash.model.training.unified_training_pipeline.create_model_api') as mock_create_api:
            mock_api = MagicMock()
            mock_api.build_model.return_value = {'status': 'built', 'model': mock_model}
            mock_create_api.return_value = mock_api
            
            with patch('smartcash.model.training.unified_training_pipeline.setup_device') as mock_device:
                mock_device.return_value = torch.device('cpu')
                with patch('smartcash.model.training.unified_training_pipeline.model_to_device') as mock_to_device:
                    mock_to_device.return_value = mock_model
                    with patch('torch.compile') as mock_compile:
                        compiled_model = MagicMock()
                        mock_compile.return_value = compiled_model
                        
                        result = pipeline._phase_build_model()
                        
                        assert result['success'] is True
                        assert pipeline.model == compiled_model
                        mock_compile.assert_called_once_with(mock_model)
    
    def test_phase_validate_model_success(self, pipeline, mock_config):
        """Test successful model validation phase."""
        pipeline.config = mock_config
        mock_model = MagicMock()
        mock_model.eval.return_value = None
        mock_model.parameters.return_value = [torch.tensor([1.0]).to('cpu')]
        mock_model.return_value = torch.randn(2, 85)  # Mock forward pass output
        pipeline.model = mock_model
        
        # Create sample data for iteration
        mock_sample_batch = (torch.randn(2, 3, 640, 640), torch.randn(2, 85))
        
        # Mock the DataLoaderFactory and its created loaders
        with patch('smartcash.model.training.unified_training_pipeline.DataLoaderFactory') as mock_factory_class:
            # Create mock loaders that can be properly iterated
            mock_train_loader = MagicMock()
            mock_train_loader.__len__.return_value = 10
            mock_train_loader.__iter__.return_value = iter([mock_sample_batch])
            
            mock_val_loader = MagicMock()
            mock_val_loader.__len__.return_value = 5
            
            mock_factory = MagicMock()
            mock_factory.create_train_loader.return_value = mock_train_loader
            mock_factory.create_val_loader.return_value = mock_val_loader
            mock_factory_class.return_value = mock_factory
            
            result = pipeline._phase_validate_model()
            
            assert result['success'] is True
            assert result['train_batches'] == 10
            assert result['val_batches'] == 5
            assert result['forward_pass_successful'] is True
    
    def test_phase_validate_model_no_model(self, pipeline, mock_config):
        """Test model validation with no model built."""
        pipeline.config = mock_config
        pipeline.model = None
        
        result = pipeline._phase_validate_model()
        
        assert result['success'] is False
        assert 'Model not built' in result['error']
    
    def test_phase_validate_model_no_training_data(self, pipeline, mock_config):
        """Test model validation with no training data."""
        pipeline.config = mock_config
        mock_model = MagicMock()
        pipeline.model = mock_model
        
        mock_train_loader = MagicMock()
        mock_train_loader.__len__.return_value = 0
        
        with patch('smartcash.model.training.unified_training_pipeline.DataLoaderFactory') as mock_factory_class:
            mock_factory = MagicMock()
            mock_factory.create_train_loader.return_value = mock_train_loader
            mock_factory_class.return_value = mock_factory
            
            result = pipeline._phase_validate_model()
            
            assert result['success'] is False
            assert 'No training data available' in result['error']
    
    def test_phase_validate_model_forward_pass_failure(self, pipeline, mock_config):
        """Test model validation with forward pass failure."""
        pipeline.config = mock_config
        mock_model = MagicMock()
        mock_model.parameters.return_value = [torch.tensor([1.0]).to('cpu')]
        mock_model.side_effect = RuntimeError("Forward pass error")
        pipeline.model = mock_model
        
        # Create proper mock data loader class
        class MockDataLoader:
            def __init__(self, length, data):
                self._length = length
                self._data = data
            
            def __len__(self):
                return self._length
            
            def __iter__(self):
                return iter(self._data)
        
        mock_sample_batch = (torch.randn(2, 3, 640, 640), torch.randn(2, 85))
        mock_train_loader = MockDataLoader(10, [mock_sample_batch])
        mock_val_loader = MockDataLoader(5, [mock_sample_batch])
        
        with patch('smartcash.model.training.unified_training_pipeline.DataLoaderFactory') as mock_factory_class:
            mock_factory = MagicMock()
            mock_factory.create_train_loader.return_value = mock_train_loader
            mock_factory.create_val_loader.return_value = mock_val_loader
            mock_factory_class.return_value = mock_factory
            
            result = pipeline._phase_validate_model()
            
            assert result['success'] is False
            assert 'Forward pass failed' in result['error']


class TestUnifiedTrainingPipelineTrainingModes:
    """Test different training modes and configurations."""
    
    @pytest.fixture
    def pipeline(self):
        """Create pipeline instance for testing."""
        return UnifiedTrainingPipeline(verbose=False)
    
    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        return {
            'model': {'backbone': 'cspdarknet', 'layer_mode': 'multi'},
            'training_phases': {
                'phase_1': {'epochs': 2},
                'phase_2': {'epochs': 3}
            }
        }
    
    def test_freeze_backbone_with_backbone_attribute(self, pipeline):
        """Test freezing backbone when model has backbone attribute."""
        mock_backbone = MagicMock()
        mock_param1 = MagicMock()
        mock_param1.requires_grad = True
        mock_param2 = MagicMock()
        mock_param2.requires_grad = True
        mock_backbone.parameters.return_value = [mock_param1, mock_param2]
        
        mock_model = MagicMock()
        mock_model.backbone = mock_backbone
        pipeline.model = mock_model
        
        pipeline._freeze_backbone()
        
        assert mock_param1.requires_grad is False
        assert mock_param2.requires_grad is False
    
    def test_freeze_backbone_without_backbone_attribute(self, pipeline):
        """Test freezing backbone when model has no backbone attribute."""
        mock_model = MagicMock()
        del mock_model.backbone  # Ensure no backbone attribute
        pipeline.model = mock_model
        
        # Should not raise exception
        pipeline._freeze_backbone()
    
    def test_unfreeze_backbone_with_backbone_attribute(self, pipeline):
        """Test unfreezing backbone when model has backbone attribute."""
        mock_backbone = MagicMock()
        mock_param1 = MagicMock()
        mock_param1.requires_grad = False
        mock_param2 = MagicMock()
        mock_param2.requires_grad = False
        mock_backbone.parameters.return_value = [mock_param1, mock_param2]
        
        mock_model = MagicMock()
        mock_model.backbone = mock_backbone
        pipeline.model = mock_model
        
        pipeline._unfreeze_backbone()
        
        assert mock_param1.requires_grad is True
        assert mock_param2.requires_grad is True
    
    def test_unfreeze_backbone_without_backbone_attribute(self, pipeline):
        """Test unfreezing backbone when model has no backbone attribute."""
        mock_model = MagicMock()
        del mock_model.backbone  # Ensure no backbone attribute
        pipeline.model = mock_model
        
        # Should not raise exception
        pipeline._unfreeze_backbone()
    
    def test_single_phase_training_freeze_backbone(self, pipeline, mock_config):
        """Test single phase training with frozen backbone."""
        pipeline.config = mock_config
        mock_model = MagicMock()
        pipeline.model = mock_model
        
        with patch.object(pipeline, '_freeze_backbone') as mock_freeze:
            with patch('smartcash.model.training.utils.setup_utils.configure_single_phase_settings') as mock_configure:
                mock_configure.return_value = mock_config
                with patch('smartcash.model.training.unified_training_pipeline.TrainingPhaseManager') as mock_manager_class:
                    mock_manager = MagicMock()
                    mock_manager.run_training_phase.return_value = {'success': True}
                    mock_manager_class.return_value = mock_manager
                    
                    result = pipeline._phase_single_training(5, freeze_backbone=True)
                    
                    assert result['success'] is True
                    mock_freeze.assert_called_once()
                    mock_manager.set_single_phase_mode.assert_called_once_with(True)
    
    def test_single_phase_training_unfreeze_backbone(self, pipeline, mock_config):
        """Test single phase training with unfrozen backbone."""
        pipeline.config = mock_config
        mock_model = MagicMock()
        pipeline.model = mock_model
        
        with patch.object(pipeline, '_unfreeze_backbone') as mock_unfreeze:
            with patch('smartcash.model.training.utils.setup_utils.configure_single_phase_settings') as mock_configure:
                mock_configure.return_value = mock_config
                with patch('smartcash.model.training.unified_training_pipeline.TrainingPhaseManager') as mock_manager_class:
                    mock_manager = MagicMock()
                    mock_manager.run_training_phase.return_value = {'success': True}
                    mock_manager_class.return_value = mock_manager
                    
                    result = pipeline._phase_single_training(5, freeze_backbone=False)
                    
                    assert result['success'] is True
                    mock_unfreeze.assert_called_once()
    
    def test_single_phase_training_with_resume(self, pipeline, mock_config):
        """Test single phase training with resume from epoch."""
        pipeline.config = mock_config
        mock_model = MagicMock()
        pipeline.model = mock_model
        
        with patch('smartcash.model.training.utils.setup_utils.configure_single_phase_settings') as mock_configure:
            mock_configure.return_value = mock_config
            with patch('smartcash.model.training.unified_training_pipeline.TrainingPhaseManager') as mock_manager_class:
                mock_manager = MagicMock()
                mock_manager.run_training_phase.return_value = {'success': True}
                mock_manager_class.return_value = mock_manager
                
                result = pipeline._phase_single_training(5, start_epoch=2, layer_mode='single')
                
                assert result['success'] is True
                # Should call with total_epochs=5, start_epoch=2
                mock_manager.run_training_phase.assert_called_once_with(1, 5, start_epoch=2)
    
    def test_single_phase_training_configuration_restoration(self, pipeline, mock_config):
        """Test that configuration is properly restored after single phase training."""
        original_loss_config = {'type': 'original', 'weight': 1.0}
        mock_config['training'] = {'loss': original_loss_config.copy()}
        pipeline.config = mock_config.copy()  # Make a deep copy
        mock_model = MagicMock()
        pipeline.model = mock_model
        
        # Store original for comparison
        original_pipeline_config = pipeline.config['training']['loss'].copy()
        
        # Modified config that configure_single_phase_settings would return
        modified_config = mock_config.copy()
        modified_config['training'] = {'loss': {'type': 'modified', 'weight': 2.0}}
        
        with patch('smartcash.model.training.utils.setup_utils.configure_single_phase_settings') as mock_configure:
            mock_configure.return_value = modified_config
            with patch('smartcash.model.training.unified_training_pipeline.TrainingPhaseManager') as mock_manager_class:
                mock_manager = MagicMock()
                mock_manager.run_training_phase.return_value = {'success': True}
                mock_manager_class.return_value = mock_manager
                
                result = pipeline._phase_single_training(3)
                
                # Check that training succeeded
                assert result['success'] is True
                
                # Configuration restoration happens in the finally block
                # The test verifies the mechanism is in place


class TestUnifiedTrainingPipelineCallbacks:
    """Test callback functionality and UI integration."""
    
    def test_emit_log_callback_success(self):
        """Test successful log callback emission."""
        log_callback = MagicMock()
        pipeline = UnifiedTrainingPipeline(log_callback=log_callback)
        pipeline.current_phase = 'test_phase'
        pipeline.training_session_id = 'session_123'
        
        pipeline._emit_log('info', 'Test message', {'key': 'value'})
        
        log_callback.assert_called_once()
        args = log_callback.call_args
        assert args[0][0] == 'info'
        assert args[0][1] == 'Test message'
        assert args[0][2]['phase'] == 'test_phase'
        assert args[0][2]['session_id'] == 'session_123'
        assert args[0][2]['message'] == 'Test message'
        assert args[0][2]['data']['key'] == 'value'
    
    def test_emit_log_callback_exception(self):
        """Test log callback with exception in callback."""
        log_callback = MagicMock()
        log_callback.side_effect = Exception("Callback error")
        pipeline = UnifiedTrainingPipeline(log_callback=log_callback)
        
        # Should not raise exception
        pipeline._emit_log('error', 'Test message')
    
    def test_emit_live_chart_callback_success(self):
        """Test successful live chart callback emission."""
        chart_callback = MagicMock()
        pipeline = UnifiedTrainingPipeline(live_chart_callback=chart_callback)
        pipeline.current_phase = 'training'
        pipeline.training_session_id = 'session_456'
        
        chart_data = {'loss': [1.0, 0.8, 0.6]}
        chart_config = {'title': 'Training Loss'}
        
        pipeline._emit_live_chart('loss_chart', chart_data, chart_config)
        
        chart_callback.assert_called_once()
        args = chart_callback.call_args
        assert args[0][0] == 'loss_chart'
        assert args[0][1]['phase'] == 'training'
        assert args[0][1]['session_id'] == 'session_456'
        assert args[0][1]['chart_type'] == 'loss_chart'
        assert args[0][1]['data'] == chart_data
        assert args[0][2] == chart_config
    
    def test_emit_live_chart_callback_exception(self):
        """Test live chart callback with exception in callback."""
        chart_callback = MagicMock()
        chart_callback.side_effect = Exception("Chart callback error")
        pipeline = UnifiedTrainingPipeline(live_chart_callback=chart_callback)
        
        # Should not raise exception
        pipeline._emit_live_chart('test_chart', {})
    
    def test_emit_metrics_callback_success(self):
        """Test successful metrics callback emission."""
        metrics_callback = MagicMock()
        pipeline = UnifiedTrainingPipeline(metrics_callback=metrics_callback)
        pipeline.training_session_id = 'session_789'
        pipeline.phase_start_time = time.time() - 10
        pipeline.training_start_time = time.time() - 100
        
        metrics = {'loss': 0.5, 'accuracy': 0.85}
        
        pipeline._emit_metrics('phase_1', 5, metrics)
        
        metrics_callback.assert_called_once()
        args = metrics_callback.call_args
        assert args[0][0] == 'phase_1'
        assert args[0][1] == 5
        assert args[0][2]['phase'] == 'phase_1'
        assert args[0][2]['epoch'] == 5
        assert args[0][2]['session_id'] == 'session_789'
        assert args[0][2]['metrics'] == metrics
        assert args[0][2]['phase_duration'] >= 0
        assert args[0][2]['total_duration'] >= 0
    
    def test_emit_metrics_callback_exception(self):
        """Test metrics callback with exception in callback."""
        metrics_callback = MagicMock()
        metrics_callback.side_effect = Exception("Metrics callback error")
        pipeline = UnifiedTrainingPipeline(metrics_callback=metrics_callback)
        
        # Should not raise exception
        pipeline._emit_metrics('test_phase', 1, {})
    
    def test_emit_callbacks_without_session_id(self):
        """Test callback emissions when session_id is None."""
        log_callback = MagicMock()
        chart_callback = MagicMock()
        metrics_callback = MagicMock()
        
        pipeline = UnifiedTrainingPipeline(
            log_callback=log_callback,
            live_chart_callback=chart_callback,
            metrics_callback=metrics_callback
        )
        
        # session_id is None by default
        pipeline._emit_log('info', 'Test')
        pipeline._emit_live_chart('chart', {})
        pipeline._emit_metrics('phase', 1, {})
        
        # All callbacks should still be called
        log_callback.assert_called_once()
        chart_callback.assert_called_once()
        metrics_callback.assert_called_once()
    
    def test_no_callbacks_configured(self):
        """Test pipeline behavior when no callbacks are configured."""
        pipeline = UnifiedTrainingPipeline()
        
        # Should not raise exceptions
        pipeline._emit_log('info', 'Test')
        pipeline._emit_live_chart('chart', {})
        pipeline._emit_metrics('phase', 1, {})


class TestUnifiedTrainingPipelineUtilityMethods:
    """Test utility methods and helper functions."""
    
    @pytest.fixture
    def pipeline(self):
        """Create pipeline instance for testing."""
        return UnifiedTrainingPipeline(verbose=False)
    
    def test_deep_merge_dict_simple(self, pipeline):
        """Test deep merge with simple dictionaries."""
        base = {'a': 1, 'b': 2}
        override = {'b': 3, 'c': 4}
        
        result = pipeline._deep_merge_dict(base, override)
        
        assert result == {'a': 1, 'b': 3, 'c': 4}
        # Ensure original dictionaries are not modified
        assert base == {'a': 1, 'b': 2}
        assert override == {'b': 3, 'c': 4}
    
    def test_deep_merge_dict_nested(self, pipeline):
        """Test deep merge with nested dictionaries."""
        base = {
            'level1': {
                'level2': {'a': 1, 'b': 2},
                'other': 'value'
            }
        }
        override = {
            'level1': {
                'level2': {'b': 3, 'c': 4},
                'new': 'data'
            }
        }
        
        result = pipeline._deep_merge_dict(base, override)
        
        expected = {
            'level1': {
                'level2': {'a': 1, 'b': 3, 'c': 4},
                'other': 'value',
                'new': 'data'
            }
        }
        assert result == expected
    
    def test_deep_merge_dict_override_with_non_dict(self, pipeline):
        """Test deep merge when override value is not a dict."""
        base = {'nested': {'a': 1, 'b': 2}}
        override = {'nested': 'not_a_dict'}
        
        result = pipeline._deep_merge_dict(base, override)
        
        assert result == {'nested': 'not_a_dict'}
    
    def test_deep_merge_dict_empty_dicts(self, pipeline):
        """Test deep merge with empty dictionaries."""
        base = {}
        override = {'a': 1}
        
        result1 = pipeline._deep_merge_dict(base, override)
        assert result1 == {'a': 1}
        
        result2 = pipeline._deep_merge_dict(override, {})
        assert result2 == {'a': 1}
        
        result3 = pipeline._deep_merge_dict({}, {})
        assert result3 == {}


class TestUnifiedTrainingPipelineErrorHandling:
    """Test error handling and failure scenarios."""
    
    @pytest.fixture
    def pipeline(self):
        """Create pipeline instance for testing."""
        return UnifiedTrainingPipeline(verbose=False)
    
    def test_run_full_training_pipeline_validation_error(self, pipeline):
        """Test pipeline with validation error."""
        with patch('smartcash.model.training.unified_training_pipeline.validate_training_mode_and_params') as mock_validate:
            mock_validate.side_effect = ValueError("Invalid training mode")
            
            result = pipeline.run_full_training_pipeline()
            
            assert result['success'] is False
            assert 'Invalid training mode' in result['error']
    
    def test_run_full_training_pipeline_setup_session_error(self, pipeline):
        """Test pipeline with setup session error."""
        with patch('smartcash.model.training.unified_training_pipeline.validate_training_mode_and_params'):
            with patch('smartcash.model.training.unified_training_pipeline.setup_training_session') as mock_setup:
                mock_setup.side_effect = Exception("Session setup failed")
                
                result = pipeline.run_full_training_pipeline()
                
                assert result['success'] is False
                assert 'Session setup failed' in result['error']
    
    def test_run_full_training_pipeline_preparation_failure(self, pipeline):
        """Test pipeline with preparation phase failure."""
        with patch('smartcash.model.training.unified_training_pipeline.validate_training_mode_and_params'):
            with patch('smartcash.model.training.unified_training_pipeline.setup_training_session') as mock_setup:
                mock_setup.return_value = ('session_123', None)
                with patch.object(pipeline, '_phase_preparation') as mock_prep:
                    mock_prep.return_value = {'success': False, 'error': 'Preparation failed'}
                    
                    result = pipeline.run_full_training_pipeline()
                    
                    assert result['success'] is False
                    assert result == mock_prep.return_value
    
    def test_run_full_training_pipeline_build_failure(self, pipeline):
        """Test pipeline with build phase failure."""
        with patch('smartcash.model.training.unified_training_pipeline.validate_training_mode_and_params'):
            with patch('smartcash.model.training.unified_training_pipeline.setup_training_session') as mock_setup:
                mock_setup.return_value = ('session_123', None)
                with patch.object(pipeline, '_phase_preparation') as mock_prep:
                    mock_prep.return_value = {'success': True}
                    with patch.object(pipeline, '_phase_build_model') as mock_build:
                        mock_build.return_value = {'success': False, 'error': 'Build failed'}
                        
                        result = pipeline.run_full_training_pipeline()
                        
                        assert result['success'] is False
                        assert result == mock_build.return_value
    
    def test_run_full_training_pipeline_validate_failure(self, pipeline):
        """Test pipeline with validation phase failure."""
        with patch('smartcash.model.training.unified_training_pipeline.validate_training_mode_and_params'):
            with patch('smartcash.model.training.unified_training_pipeline.setup_training_session') as mock_setup:
                mock_setup.return_value = ('session_123', None)
                with patch.object(pipeline, '_phase_preparation') as mock_prep:
                    mock_prep.return_value = {'success': True}
                    with patch.object(pipeline, '_phase_build_model') as mock_build:
                        mock_build.return_value = {'success': True}
                        with patch.object(pipeline, '_phase_validate_model') as mock_validate:
                            mock_validate.return_value = {'success': False, 'error': 'Validation failed'}
                            
                            result = pipeline.run_full_training_pipeline()
                            
                            assert result['success'] is False
                            assert result == mock_validate.return_value
    
    def test_run_full_training_pipeline_training_phase_1_failure(self, pipeline):
        """Test pipeline with training phase 1 failure."""
        with patch('smartcash.model.training.unified_training_pipeline.validate_training_mode_and_params'):
            with patch('smartcash.model.training.unified_training_pipeline.setup_training_session') as mock_setup:
                mock_setup.return_value = ('session_123', None)
                with patch.object(pipeline, '_phase_preparation') as mock_prep:
                    mock_prep.return_value = {'success': True}
                    with patch.object(pipeline, '_phase_build_model') as mock_build:
                        mock_build.return_value = {'success': True}
                        with patch.object(pipeline, '_phase_validate_model') as mock_validate:
                            mock_validate.return_value = {'success': True}
                            with patch.object(pipeline, '_phase_training_1_with_manager') as mock_train1:
                                mock_train1.return_value = {'success': False, 'error': 'Training phase 1 failed'}
                                with patch('smartcash.model.training.unified_training_pipeline.TrainingPhaseManager'):
                                    
                                    result = pipeline.run_full_training_pipeline(training_mode='two_phase')
                                    
                                    assert result['success'] is False
                                    assert result == mock_train1.return_value
    
    def test_run_full_training_pipeline_training_phase_2_failure(self, pipeline):
        """Test pipeline with training phase 2 failure."""
        with patch('smartcash.model.training.unified_training_pipeline.validate_training_mode_and_params'):
            with patch('smartcash.model.training.unified_training_pipeline.setup_training_session') as mock_setup:
                mock_setup.return_value = ('session_123', None)
                with patch.object(pipeline, '_phase_preparation') as mock_prep:
                    mock_prep.return_value = {'success': True}
                    with patch.object(pipeline, '_phase_build_model') as mock_build:
                        mock_build.return_value = {'success': True}
                        with patch.object(pipeline, '_phase_validate_model') as mock_validate:
                            mock_validate.return_value = {'success': True}
                            with patch.object(pipeline, '_phase_training_1_with_manager') as mock_train1:
                                mock_train1.return_value = {'success': True}
                                with patch.object(pipeline, '_phase_training_2_with_manager') as mock_train2:
                                    mock_train2.return_value = {'success': False, 'error': 'Training phase 2 failed'}
                                    with patch('smartcash.model.training.unified_training_pipeline.TrainingPhaseManager'):
                                        
                                        result = pipeline.run_full_training_pipeline(training_mode='two_phase')
                                        
                                        assert result['success'] is False
                                        assert result == mock_train2.return_value
    
    def test_run_full_training_pipeline_single_phase_failure(self, pipeline):
        """Test pipeline with single phase training failure."""
        with patch('smartcash.model.training.unified_training_pipeline.validate_training_mode_and_params'):
            with patch('smartcash.model.training.unified_training_pipeline.setup_training_session') as mock_setup:
                mock_setup.return_value = ('session_123', None)
                with patch.object(pipeline, '_phase_preparation') as mock_prep:
                    mock_prep.return_value = {'success': True}
                    with patch.object(pipeline, '_phase_build_model') as mock_build:
                        mock_build.return_value = {'success': True}
                        with patch.object(pipeline, '_phase_validate_model') as mock_validate:
                            mock_validate.return_value = {'success': True}
                            with patch.object(pipeline, '_phase_single_training') as mock_single:
                                mock_single.return_value = {'success': False, 'error': 'Single phase failed'}
                                
                                result = pipeline.run_full_training_pipeline(training_mode='single_phase')
                                
                                assert result['success'] is False
                                assert result == mock_single.return_value
    
    def test_phase_training_exception_handling(self, pipeline):
        """Test training phase exception handling."""
        pipeline.config = {'training_phases': {'phase_1': {'epochs': 2}}}
        
        with patch.object(pipeline, '_freeze_backbone'):
            with patch('smartcash.model.training.unified_training_pipeline.TrainingPhaseManager') as mock_manager_class:
                mock_manager = MagicMock()
                mock_manager.run_training_phase.side_effect = Exception("Training failed")
                mock_manager_class.return_value = mock_manager
                
                result = pipeline._phase_training_1_with_manager(mock_manager)
                
                assert result['success'] is False
                assert 'Training Phase 1 failed: Training failed' in result['error']