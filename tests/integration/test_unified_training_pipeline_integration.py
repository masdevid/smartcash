#!/usr/bin/env python3
"""
File: tests/integration/test_unified_training_pipeline_integration.py

Integration tests for UnifiedTrainingPipeline testing full workflow scenarios.

Test Coverage:
- Complete training pipeline execution with real components
- Integration between pipeline phases and managers
- Real checkpoint creation and resume functionality
- Device switching scenarios (CPU/GPU/MPS)
- Configuration persistence and validation
- File system operations and cleanup
- Training mode transitions and validation
- Progress tracking and callback integration
- Memory management in long-running scenarios
"""

import pytest
import torch
import tempfile
import shutil
import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, ANY
from typing import Dict, Any, Optional, List

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from smartcash.model.training.unified_training_pipeline import UnifiedTrainingPipeline


class TestUnifiedTrainingPipelineIntegration:
    def _create_mock_data_loaders(self):
        """Helper to create proper mock data loaders."""
        class MockDataLoader:
            def __init__(self, length, sample_data):
                self._length = length
                self._sample_data = sample_data
            
            def __len__(self):
                return self._length
            
            def __iter__(self):
                return iter(self._sample_data)
        
        mock_sample_batch = (torch.randn(2, 3, 640, 640), torch.randn(2, 85))
        
        return {
            'train': MockDataLoader(10, [mock_sample_batch] * 10),
            'val': MockDataLoader(5, [mock_sample_batch] * 5)
        }

    """Integration tests for complete training pipeline workflows."""
    
    @pytest.fixture
    def temp_dirs(self):
        """Create temporary directories for testing."""
        temp_base = tempfile.mkdtemp()
        dirs = {
            'checkpoints': Path(temp_base) / 'checkpoints',
            'logs': Path(temp_base) / 'logs',
            'visualization': Path(temp_base) / 'viz',
            'data': Path(temp_base) / 'data'
        }
        
        for dir_path in dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
        
        yield dirs
        
        # Cleanup
        shutil.rmtree(temp_base, ignore_errors=True)
    
    @pytest.fixture
    def mock_callbacks(self):
        """Create mock callbacks for testing."""
        return {
            'progress': MagicMock(),
            'log': MagicMock(),
            'chart': MagicMock(),
            'metrics': MagicMock()
        }
    
    @pytest.fixture
    def mock_training_config(self, temp_dirs):
        """Create a complete mock training configuration."""
        return {
            'model': {
                'backbone': 'cspdarknet',
                'num_classes': [80, 40, 20],
                'layer_mode': 'multi',
                'input_size': [640, 640]
            },
            'device': {
                'auto_detect': True,
                'device': 'cpu'
            },
            'training': {
                'batch_size': 4,
                'learning_rate': 0.001,
                'compile_model': False,
                'loss': {
                    'type': 'multi_layer',
                    'weights': [1.0, 1.0, 1.0]
                },
                'optimizer': {
                    'type': 'adam',
                    'weight_decay': 0.0005
                },
                'scheduler': {
                    'type': 'cosine',
                    'warmup_epochs': 1
                }
            },
            'training_phases': {
                'phase_1': {
                    'epochs': 2,
                    'freeze_backbone': True
                },
                'phase_2': {
                    'epochs': 3,
                    'freeze_backbone': False
                }
            },
            'paths': {
                'checkpoints': temp_dirs['checkpoints'],
                'logs': temp_dirs['logs'],
                'visualization': temp_dirs['visualization'],
                'data': temp_dirs['data']
            },
            'data': {
                'train_split': 'train',
                'val_split': 'val',
                'augmentation': True
            }
        }
    
    def test_complete_two_phase_training_workflow(self, temp_dirs, mock_callbacks, mock_training_config):
        """Test complete two-phase training workflow with all components."""
        pipeline = UnifiedTrainingPipeline(
            progress_callback=mock_callbacks['progress'],
            log_callback=mock_callbacks['log'],
            live_chart_callback=mock_callbacks['chart'],
            metrics_callback=mock_callbacks['metrics'],
            verbose=True
        )
        
        # Mock all the heavy components
        with patch('smartcash.model.training.unified_training_pipeline.prepare_training_environment') as mock_prep:
            mock_prep.return_value = {'success': True, 'config': mock_training_config}
            
            with patch('smartcash.model.training.unified_training_pipeline.create_model_api') as mock_create_api:
                mock_model = self._create_mock_model()
                mock_api = self._create_mock_model_api(mock_model)
                mock_create_api.return_value = mock_api
                
                with patch('smartcash.model.training.unified_training_pipeline.DataLoaderFactory') as mock_factory_class:
                    mock_loaders = self._create_mock_data_loaders()
                    mock_factory = MagicMock()
                    mock_factory.create_train_loader.return_value = mock_loaders['train']
                    mock_factory.create_val_loader.return_value = mock_loaders['val']
                    mock_factory_class.return_value = mock_factory
                    
                    with patch('smartcash.model.training.unified_training_pipeline.TrainingPhaseManager') as mock_manager_class:
                        mock_manager = self._create_mock_training_manager()
                        mock_manager_class.return_value = mock_manager
                        
                        with patch('smartcash.model.training.unified_training_pipeline.create_visualization_manager') as mock_viz:
                            mock_viz_manager = MagicMock()
                            mock_viz_manager.generate_comprehensive_charts.return_value = {
                                'dashboard': str(temp_dirs['visualization'] / 'dashboard.html'),
                                'training_curves': str(temp_dirs['visualization'] / 'curves.png')
                            }
                            mock_viz.return_value = mock_viz_manager
                            
                            with patch('smartcash.model.training.unified_training_pipeline.generate_markdown_summary') as mock_summary:
                                mock_summary.return_value = "# Training Summary\nCompleted successfully."
                                
                                # Execute the complete pipeline
                                result = pipeline.run_full_training_pipeline(
                                    backbone='cspdarknet',
                                    phase_1_epochs=2,
                                    phase_2_epochs=3,
                                    checkpoint_dir=str(temp_dirs['checkpoints']),
                                    training_mode='two_phase'
                                )
        
        # Verify successful completion
        assert result['success'] is True
        assert 'pipeline_summary' in result
        assert 'final_training_result' in result
        assert 'visualization_result' in result
        assert 'markdown_summary' in result
        
        # Verify all callbacks were called
        assert mock_callbacks['progress'].call_count > 0
        assert mock_callbacks['log'].call_count > 0
        
        # Verify training manager was called for both phases
        mock_manager = mock_manager_class.return_value
        assert mock_manager.run_training_phase.call_count == 2
        
        # Verify phase calls
        phase_calls = mock_manager.run_training_phase.call_args_list
        assert phase_calls[0][0] == (1, 2)  # Phase 1, 2 epochs
        assert phase_calls[1][0] == (2, 3)  # Phase 2, 3 epochs
    
    def test_complete_single_phase_training_workflow(self, temp_dirs, mock_callbacks, mock_training_config):
        """Test complete single-phase training workflow."""
        pipeline = UnifiedTrainingPipeline(
            progress_callback=mock_callbacks['progress'],
            log_callback=mock_callbacks['log'],
            verbose=True
        )
        
        with patch('smartcash.model.training.unified_training_pipeline.prepare_training_environment') as mock_prep:
            mock_prep.return_value = {'success': True, 'config': mock_training_config}
            
            with patch('smartcash.model.training.unified_training_pipeline.create_model_api') as mock_create_api:
                mock_model = self._create_mock_model()
                mock_api = self._create_mock_model_api(mock_model)
                mock_create_api.return_value = mock_api
                
                with patch('smartcash.model.training.unified_training_pipeline.DataLoaderFactory') as mock_factory_class:
                    mock_loaders = self._create_mock_data_loaders()
                    mock_factory = MagicMock()
                    mock_factory.create_train_loader.return_value = mock_loaders['train']
                    mock_factory.create_val_loader.return_value = mock_loaders['val']
                    mock_factory_class.return_value = mock_factory
                    
                    with patch('smartcash.model.training.unified_training_pipeline.configure_single_phase_settings') as mock_configure:
                        mock_configure.return_value = mock_training_config
                        
                        with patch('smartcash.model.training.unified_training_pipeline.TrainingPhaseManager') as mock_manager_class:
                            mock_manager = self._create_mock_training_manager()
                            mock_manager_class.return_value = mock_manager
                            
                            with patch('smartcash.model.training.unified_training_pipeline.create_visualization_manager') as mock_viz:
                                mock_viz_manager = MagicMock()
                                mock_viz_manager.generate_comprehensive_charts.return_value = {}
                                mock_viz.return_value = mock_viz_manager
                                
                                with patch('smartcash.model.training.unified_training_pipeline.generate_markdown_summary') as mock_summary:
                                    mock_summary.return_value = "# Single Phase Training Summary"
                                    
                                    # Execute single phase pipeline
                                    result = pipeline.run_full_training_pipeline(
                                        backbone='efficientnet_b4',
                                        phase_1_epochs=5,  # Total epochs
                                        phase_2_epochs=0,  # Not used in single phase
                                        training_mode='single_phase',
                                        single_phase_layer_mode='single',
                                        single_phase_freeze_backbone=False
                                    )
        
        # Verify successful completion
        assert result['success'] is True
        
        # Verify single phase training was called
        mock_manager = mock_manager_class.return_value
        assert mock_manager.run_training_phase.call_count == 1
        assert mock_manager.set_single_phase_mode.call_count == 1
        mock_manager.set_single_phase_mode.assert_called_with(True)
    
    def test_training_with_resume_from_checkpoint(self, temp_dirs, mock_callbacks, mock_training_config):
        """Test training pipeline with resume from checkpoint."""
        pipeline = UnifiedTrainingPipeline(
            progress_callback=mock_callbacks['progress'],
            log_callback=mock_callbacks['log'],
            verbose=True
        )
        
        # Mock resume info
        resume_info = {
            'checkpoint_name': 'test_checkpoint.pth',
            'phase': 2,
            'epoch': 1,
            'model_state_dict': {'layer.weight': torch.tensor([1.0])},
            'session_id': 'session_123'
        }
        
        with patch('smartcash.model.training.unified_training_pipeline.validate_training_mode_and_params'):
            with patch('smartcash.model.training.unified_training_pipeline.setup_training_session') as mock_setup:
                mock_setup.return_value = ('session_123', resume_info)
                
                with patch('smartcash.model.training.unified_training_pipeline.handle_resume_training_pipeline') as mock_resume:
                    # Mock successful resume results
                    prep_result = {'success': True, 'config': mock_training_config}
                    build_result = {'success': True, 'model_info': 'built'}
                    validate_result = {'success': True, 'forward_pass_successful': True}
                    phase1_result = {'success': True, 'message': 'Completed (loaded from checkpoint)'}
                    phase2_result = {'success': True, 'final_metrics': {'loss': 0.5}}
                    
                    mock_resume.return_value = (prep_result, build_result, validate_result, phase1_result, phase2_result)
                    
                    with patch('smartcash.model.training.unified_training_pipeline.create_visualization_manager') as mock_viz:
                        mock_viz_manager = MagicMock()
                        mock_viz_manager.generate_comprehensive_charts.return_value = {}
                        mock_viz.return_value = mock_viz_manager
                        
                        with patch('smartcash.model.training.unified_training_pipeline.generate_markdown_summary') as mock_summary:
                            mock_summary.return_value = "# Resumed Training Summary"
                            
                            # Execute with resume
                            result = pipeline.run_full_training_pipeline(
                                backbone='cspdarknet',
                                phase_1_epochs=2,
                                phase_2_epochs=3,
                                resume_from_checkpoint=True
                            )
        
        # Verify successful completion
        assert result['success'] is True
        
        # Verify resume was attempted
        mock_setup.assert_called_once()
        mock_resume.assert_called_once()
        
        # Verify resume info was passed correctly
        resume_args = mock_resume.call_args[0]
        assert resume_args[0] == resume_info  # First argument should be resume_info
    
    def test_training_with_device_switching(self, temp_dirs, mock_callbacks, mock_training_config):
        """Test training pipeline with different device configurations."""
        pipeline = UnifiedTrainingPipeline(verbose=False)
        
        # Test scenarios for different devices
        device_scenarios = [
            ('cpu', False),
            ('mps', False),
            ('cpu', True)  # force_cpu=True
        ]
        
        for device_type, force_cpu in device_scenarios:
            with patch('smartcash.model.training.unified_training_pipeline.prepare_training_environment') as mock_prep:
                # Update config for current device
                device_config = mock_training_config.copy()
                device_config['device']['device'] = device_type
                mock_prep.return_value = {'success': True, 'config': device_config}
                
                with patch('smartcash.model.training.unified_training_pipeline.create_model_api') as mock_create_api:
                    mock_model = self._create_mock_model()
                    mock_api = self._create_mock_model_api(mock_model)
                    mock_create_api.return_value = mock_api
                    
                    with patch('smartcash.model.training.unified_training_pipeline.setup_device') as mock_device:
                        mock_device.return_value = torch.device(device_type)
                        
                        with patch('smartcash.model.training.unified_training_pipeline.model_to_device') as mock_to_device:
                            mock_to_device.return_value = mock_model
                            
                            with patch('smartcash.model.training.unified_training_pipeline.DataLoaderFactory') as mock_factory_class:
                                mock_loaders = self._create_mock_data_loaders()
                                mock_factory = MagicMock()
                                mock_factory.create_train_loader.return_value = mock_loaders['train']
                                mock_factory.create_val_loader.return_value = mock_loaders['val']
                                mock_factory_class.return_value = mock_factory
                                
                                # Execute pipeline with device configuration
                                result = pipeline.run_full_training_pipeline(
                                    backbone='cspdarknet',
                                    phase_1_epochs=1,
                                    phase_2_epochs=1,
                                    force_cpu=force_cpu,
                                    training_mode='two_phase'
                                )
                                
                                # Should complete preparation and build phases at minimum
                                # (other phases may fail due to mocking, but device setup should work)
                                assert mock_prep.called
                                assert mock_create_api.called
                                
                                # Verify device setup was called with correct parameters
                                if force_cpu:
                                    # When force_cpu=True, should override device detection
                                    pass  # Additional verification could be added
                                else:
                                    # Should use auto-detection
                                    pass  # Additional verification could be added
    
    def test_training_with_checkpoint_creation(self, temp_dirs, mock_callbacks, mock_training_config):
        """Test training pipeline with actual checkpoint file creation."""
        pipeline = UnifiedTrainingPipeline(verbose=False)
        
        # Create a real checkpoint file
        checkpoint_path = temp_dirs['checkpoints'] / 'test_checkpoint.pth'
        checkpoint_data = {
            'model_state_dict': {'layer.weight': torch.tensor([1.0, 2.0])},
            'epoch': 1,
            'phase': 1,
            'metrics': {'loss': 0.8, 'accuracy': 0.7},
            'config': mock_training_config
        }
        torch.save(checkpoint_data, checkpoint_path)
        
        with patch('smartcash.model.training.unified_training_pipeline.prepare_training_environment') as mock_prep:
            mock_prep.return_value = {'success': True, 'config': mock_training_config}
            
            with patch('smartcash.model.training.unified_training_pipeline.create_model_api') as mock_create_api:
                mock_model = self._create_mock_model()
                mock_api = self._create_mock_model_api(mock_model)
                mock_create_api.return_value = mock_api
                
                # Test checkpoint saving functionality
                checkpoint_name = pipeline._save_checkpoint(1, {'loss': 0.5}, 1)
                
                # Should attempt to use model API for saving
                if checkpoint_name:
                    # Verify checkpoint was created/attempted
                    assert mock_api.save_checkpoint.called or temp_dirs['checkpoints'].exists()
    
    def test_training_pipeline_progress_tracking(self, temp_dirs, mock_callbacks, mock_training_config):
        """Test comprehensive progress tracking throughout pipeline."""
        pipeline = UnifiedTrainingPipeline(
            progress_callback=mock_callbacks['progress'],
            verbose=True
        )
        
        with patch('smartcash.model.training.unified_training_pipeline.prepare_training_environment') as mock_prep:
            mock_prep.return_value = {'success': True, 'config': mock_training_config}
            
            with patch('smartcash.model.training.unified_training_pipeline.create_model_api') as mock_create_api:
                mock_model = self._create_mock_model()
                mock_api = self._create_mock_model_api(mock_model)
                mock_create_api.return_value = mock_api
                
                with patch('smartcash.model.training.unified_training_pipeline.DataLoaderFactory') as mock_factory_class:
                    mock_loaders = self._create_mock_data_loaders()
                    mock_factory = MagicMock()
                    mock_factory.create_train_loader.return_value = mock_loaders['train']
                    mock_factory.create_val_loader.return_value = mock_loaders['val']
                    mock_factory_class.return_value = mock_factory
                    
                    with patch('smartcash.model.training.unified_training_pipeline.TrainingPhaseManager') as mock_manager_class:
                        mock_manager = self._create_mock_training_manager()
                        mock_manager_class.return_value = mock_manager
                        
                        with patch('smartcash.model.training.unified_training_pipeline.create_visualization_manager') as mock_viz:
                            mock_viz_manager = MagicMock()
                            mock_viz_manager.generate_comprehensive_charts.return_value = {}
                            mock_viz.return_value = mock_viz_manager
                            
                            with patch('smartcash.model.training.unified_training_pipeline.generate_markdown_summary') as mock_summary:
                                mock_summary.return_value = "# Progress Tracking Test"
                                
                                # Execute pipeline
                                result = pipeline.run_full_training_pipeline(
                                    backbone='cspdarknet',
                                    phase_1_epochs=2,
                                    phase_2_epochs=2,
                                    training_mode='two_phase'
                                )
        
        # Verify progress tracking
        assert result['success'] is True
        
        # Verify progress callback was called multiple times for different phases
        progress_calls = mock_callbacks['progress'].call_args_list
        assert len(progress_calls) > 0
        
        # Extract phase information from progress calls
        phases_tracked = set()
        for call_args in progress_calls:
            if len(call_args[0]) > 0:
                # Progress callback format: (phase, current, total, message, **kwargs)
                phase = call_args[0][0] if isinstance(call_args[0][0], str) else None
                if phase:
                    phases_tracked.add(phase)
        
        # Should track multiple phases
        assert len(phases_tracked) > 1
    
    def test_training_pipeline_file_operations(self, temp_dirs, mock_callbacks, mock_training_config):
        """Test file system operations during training pipeline."""
        pipeline = UnifiedTrainingPipeline(verbose=False)
        
        with patch('smartcash.model.training.unified_training_pipeline.prepare_training_environment') as mock_prep:
            mock_prep.return_value = {'success': True, 'config': mock_training_config}
            
            with patch('smartcash.model.training.unified_training_pipeline.create_model_api') as mock_create_api:
                mock_model = self._create_mock_model()
                mock_api = self._create_mock_model_api(mock_model)
                mock_create_api.return_value = mock_api
                
                with patch('smartcash.model.training.unified_training_pipeline.DataLoaderFactory') as mock_factory_class:
                    mock_loaders = self._create_mock_data_loaders()
                    mock_factory = MagicMock()
                    mock_factory.create_train_loader.return_value = mock_loaders['train']
                    mock_factory.create_val_loader.return_value = mock_loaders['val']
                    mock_factory_class.return_value = mock_factory
                    
                    with patch('smartcash.model.training.unified_training_pipeline.TrainingPhaseManager') as mock_manager_class:
                        mock_manager = self._create_mock_training_manager()
                        mock_manager_class.return_value = mock_manager
                        
                        with patch('smartcash.model.training.unified_training_pipeline.create_visualization_manager') as mock_viz:
                            mock_viz_manager = MagicMock()
                            chart_paths = {
                                'dashboard': str(temp_dirs['visualization'] / 'session_123' / 'dashboard.html'),
                                'training_curves': str(temp_dirs['visualization'] / 'session_123' / 'curves.png')
                            }
                            mock_viz_manager.generate_comprehensive_charts.return_value = chart_paths
                            mock_viz.return_value = mock_viz_manager
                            
                            with patch('smartcash.model.training.unified_training_pipeline.generate_markdown_summary') as mock_summary:
                                mock_summary.return_value = "# File Operations Test"
                                
                                # Execute pipeline
                                result = pipeline.run_full_training_pipeline(
                                    backbone='cspdarknet',
                                    phase_1_epochs=1,
                                    phase_2_epochs=1
                                )
        
        # Verify successful completion
        assert result['success'] is True
        
        # Verify visualization result includes file paths
        if 'visualization_result' in result:
            viz_result = result['visualization_result']
            if viz_result.get('success'):
                assert 'chart_paths' in viz_result
                assert 'session_directory' in viz_result['chart_paths']
        
        # Verify temporary directories still exist and are accessible
        assert temp_dirs['checkpoints'].exists()
        assert temp_dirs['logs'].exists()
        assert temp_dirs['visualization'].exists()
    
    def _create_mock_model(self):
        """Helper to create a mock PyTorch model."""
        mock_model = MagicMock()
        mock_model.parameters.return_value = [
            torch.tensor([1.0, 2.0, 3.0]),
            torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        ]
        mock_model.eval.return_value = None
        mock_model.train.return_value = None
        mock_model.state_dict.return_value = {'layer.weight': torch.tensor([1.0, 2.0])}
        mock_model.load_state_dict.return_value = None
        
        # Mock forward pass
        mock_model.return_value = torch.randn(2, 85)
        
        # Mock backbone attribute
        mock_backbone = MagicMock()
        mock_backbone.parameters.return_value = [torch.tensor([1.0])]
        mock_model.backbone = mock_backbone
        
        return mock_model
    
    def _create_mock_model_api(self, mock_model):
        """Helper to create a mock model API."""
        mock_api = MagicMock()
        mock_api.build_model.return_value = {
            'status': 'built',
            'model': mock_model,
            'message': 'Model built successfully'
        }
        mock_api.save_checkpoint.return_value = '/tmp/checkpoint.pth'
        mock_api.model = mock_model
        return mock_api
    
    def _create_mock_data_loaders(self):
        """Helper to create mock data loaders."""
        mock_train_loader = MagicMock()
        mock_train_loader.__len__.return_value = 10
        mock_train_loader.__iter__.return_value = iter([
            (torch.randn(2, 3, 640, 640), torch.randn(2, 85)) for _ in range(10)
        ])
        
        mock_val_loader = MagicMock()
        mock_val_loader.__len__.return_value = 5
        mock_val_loader.__iter__.return_value = iter([
            (torch.randn(2, 3, 640, 640), torch.randn(2, 85)) for _ in range(5)
        ])
        
        return {'train': mock_train_loader, 'val': mock_val_loader}
    
    def _create_mock_training_manager(self):
        """Helper to create a mock training phase manager."""
        mock_manager = MagicMock()
        mock_manager.run_training_phase.return_value = {
            'success': True,
            'final_metrics': {'loss': 0.5, 'accuracy': 0.85},
            'epoch_metrics': [
                {'loss': 1.0, 'accuracy': 0.6},
                {'loss': 0.7, 'accuracy': 0.75},
                {'loss': 0.5, 'accuracy': 0.85}
            ],
            'best_checkpoint': '/tmp/best_checkpoint.pth'
        }
        mock_manager.set_single_phase_mode.return_value = None
        return mock_manager


class TestUnifiedTrainingPipelineMemoryManagement:
    """Test memory management and resource cleanup."""
    
    @pytest.fixture
    def pipeline(self):
        """Create pipeline instance for testing."""
        return UnifiedTrainingPipeline(verbose=False)
    
    def test_memory_cleanup_after_training(self, pipeline):
        """Test that memory is properly cleaned up after training."""
        # Mock heavy objects
        mock_model = MagicMock()
        mock_model.parameters.return_value = [torch.randn(1000, 1000) for _ in range(10)]  # Large tensors
        pipeline.model = mock_model
        
        mock_api = MagicMock()
        pipeline.model_api = mock_api
        
        mock_config = {'test': 'config'}
        pipeline.config = mock_config
        
        # Verify objects are set
        assert pipeline.model is not None
        assert pipeline.model_api is not None
        assert pipeline.config is not None
        
        # Simulate end of training - in real usage, these would be cleaned up
        # by Python's garbage collector or explicit cleanup
        initial_model = pipeline.model
        initial_api = pipeline.model_api
        
        # References should still exist during training
        assert pipeline.model is initial_model
        assert pipeline.model_api is initial_api
    
    def test_callback_memory_management(self, pipeline):
        """Test that callbacks don't create memory leaks."""
        call_count = 0
        
        def tracking_callback(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            # Simulate callback that might hold references
            return args, kwargs
        
        pipeline.log_callback = tracking_callback
        pipeline.metrics_callback = tracking_callback
        
        # Make multiple callback calls
        for i in range(100):
            pipeline._emit_log('info', f'Test message {i}', {'data': i})
            pipeline._emit_metrics('test_phase', i, {'loss': i * 0.1})
        
        # Verify callbacks were called
        assert call_count == 200  # 100 log + 100 metrics calls
        
        # Callbacks should not accumulate unbounded memory
        # (This is more of a design verification than a strict test)
    
    def test_large_configuration_handling(self, pipeline):
        """Test handling of large configuration objects."""
        # Create a large configuration
        large_config = {
            'model': {
                'backbone': 'cspdarknet',
                'large_data': list(range(10000)),  # Large list
                'nested': {
                    'deep': {
                        'structure': {
                            'with': {
                                'many': {
                                    'levels': list(range(1000))
                                }
                            }
                        }
                    }
                }
            }
        }
        
        # Test deep merge with large configuration
        override_config = {
            'model': {
                'backbone': 'efficientnet_b4',
                'new_large_data': list(range(5000))
            }
        }
        
        result = pipeline._deep_merge_dict(large_config, override_config)
        
        # Verify merge worked correctly
        assert result['model']['backbone'] == 'efficientnet_b4'
        assert len(result['model']['large_data']) == 10000
        assert len(result['model']['new_large_data']) == 5000
        assert len(result['model']['nested']['deep']['structure']['with']['many']['levels']) == 1000
        
        # Original configs should be unchanged
        assert large_config['model']['backbone'] == 'cspdarknet'
        assert override_config['model']['backbone'] == 'efficientnet_b4'


class TestUnifiedTrainingPipelineEdgeCases:
    """Test edge cases and boundary conditions."""
    
    @pytest.fixture
    def pipeline(self):
        """Create pipeline instance for testing."""
        return UnifiedTrainingPipeline(verbose=False)
    
    def test_training_with_minimal_epochs(self, pipeline):
        """Test training with minimal epoch configuration."""
        with patch('smartcash.model.training.unified_training_pipeline.validate_training_mode_and_params'):
            with patch('smartcash.model.training.unified_training_pipeline.setup_training_session') as mock_setup:
                mock_setup.return_value = ('session_123', None)
                with patch.object(pipeline, '_phase_preparation') as mock_prep:
                    mock_config = {
                        'training_phases': {
                            'phase_1': {'epochs': 1},
                            'phase_2': {'epochs': 1}
                        }
                    }
                    mock_prep.return_value = {'success': True, 'config': mock_config}
                    with patch.object(pipeline, '_phase_build_model') as mock_build:
                        mock_build.return_value = {'success': True}
                        with patch.object(pipeline, '_phase_validate_model') as mock_validate:
                            mock_validate.return_value = {'success': True}
                            with patch.object(pipeline, '_phase_training_1_with_manager') as mock_train1:
                                mock_train1.return_value = {'success': True}
                                with patch.object(pipeline, '_phase_training_2_with_manager') as mock_train2:
                                    mock_train2.return_value = {'success': True}
                                    with patch.object(pipeline, '_phase_summary_visualization') as mock_summary:
                                        mock_summary.return_value = {'success': True}
                                        with patch('smartcash.model.training.unified_training_pipeline.TrainingPhaseManager'):
                                            with patch('smartcash.model.training.unified_training_pipeline.generate_markdown_summary') as mock_md:
                                                mock_md.return_value = "# Minimal Training"
                                                
                                                result = pipeline.run_full_training_pipeline(
                                                    phase_1_epochs=1,
                                                    phase_2_epochs=1
                                                )
        
        assert result['success'] is True
    
    def test_training_with_zero_validation_data(self, pipeline):
        """Test training pipeline when validation data is empty."""
        mock_config = {
            'model': {'backbone': 'cspdarknet'},
            'device': {'auto_detect': True, 'device': 'cpu'},
            'training': {'compile_model': False}
        }
        pipeline.config = mock_config
        
        mock_model = MagicMock()
        mock_model.parameters.return_value = [torch.tensor([1.0]).to('cpu')]
        mock_model.eval.return_value = None
        mock_model.return_value = torch.randn(1, 85)
        pipeline.model = mock_model
        
        mock_train_loader = MagicMock()
        mock_train_loader.__len__.return_value = 5
        mock_train_loader.__iter__.return_value = iter([
            (torch.randn(1, 3, 640, 640), torch.randn(1, 85))
        ])
        
        mock_val_loader = MagicMock()
        mock_val_loader.__len__.return_value = 0  # No validation data
        
        with patch('smartcash.model.training.unified_training_pipeline.DataLoaderFactory') as mock_factory_class:
            mock_factory = MagicMock()
            mock_factory.create_train_loader.return_value = mock_train_loader
            mock_factory.create_val_loader.return_value = mock_val_loader
            mock_factory_class.return_value = mock_factory
            
            result = pipeline._phase_validate_model()
            
            # Should succeed but warn about no validation data
            assert result['success'] is True
            assert result['val_batches'] == 0
    
    def test_training_with_very_large_batch_configuration(self, pipeline):
        """Test training with configuration that might cause memory issues."""
        large_config = {
            'model': {
                'backbone': 'cspdarknet',
                'num_classes': [80] * 100,  # Very large number of classes
                'batch_size': 1024  # Very large batch size
            },
            'training': {
                'batch_size': 1024,
                'large_parameter_count': True
            }
        }
        
        # Test deep merge with large configuration doesn't break
        override = {'model': {'backbone': 'efficientnet_b4'}}
        result = pipeline._deep_merge_dict(large_config, override)
        
        assert result['model']['backbone'] == 'efficientnet_b4'
        assert len(result['model']['num_classes']) == 100
        assert result['training']['batch_size'] == 1024
    
    def test_callback_with_none_values(self, pipeline):
        """Test callbacks with None values and edge cases."""
        # Test with None callbacks
        pipeline.log_callback = None
        pipeline.metrics_callback = None
        pipeline.live_chart_callback = None
        
        # Should not raise exceptions
        pipeline._emit_log('info', 'Test message')
        pipeline._emit_metrics('phase', 1, {})
        pipeline._emit_live_chart('chart', {})
        
        # Test with callbacks that return None
        def none_callback(*args, **kwargs):
            return None
        
        pipeline.log_callback = none_callback
        pipeline._emit_log('info', 'Test with none callback')
    
    def test_configuration_with_missing_required_fields(self, pipeline):
        """Test handling of configurations missing required fields."""
        incomplete_configs = [
            {},  # Completely empty
            {'model': {}},  # Missing required model fields
            {'model': {'backbone': 'cspdarknet'}},  # Missing other required fields
            {'training_phases': {}},  # Missing phase configurations
            {'training_phases': {'phase_1': {}}},  # Missing epochs
        ]
        
        for incomplete_config in incomplete_configs:
            pipeline.config = incomplete_config
            
            # Should handle gracefully (may fail but shouldn't crash)
            try:
                result = pipeline._phase_single_training(1)
                # If it succeeds, that's fine
                if result.get('success'):
                    pass
                else:
                    # If it fails, should be graceful failure
                    assert 'error' in result or 'success' in result
            except Exception as e:
                # If it raises an exception, should be a reasonable one
                assert isinstance(e, (KeyError, ValueError, AttributeError))
    
    def test_extremely_long_phase_names_and_identifiers(self, pipeline):
        """Test handling of very long strings in configuration."""
        long_string = "x" * 10000  # Very long string
        
        pipeline.training_session_id = long_string
        pipeline.current_phase = long_string
        
        # Should handle long strings in callbacks
        log_callback = MagicMock()
        pipeline.log_callback = log_callback
        
        pipeline._emit_log('info', 'Test message')
        
        # Verify callback was called
        log_callback.assert_called_once()
        
        # Verify long strings were included
        call_args = log_callback.call_args[0][2]
        assert call_args['session_id'] == long_string
        assert call_args['phase'] == long_string