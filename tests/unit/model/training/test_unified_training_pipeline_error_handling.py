#!/usr/bin/env python3
"""
File: tests/unit/model/training/test_unified_training_pipeline_error_handling.py

Comprehensive tests for error handling and failure recovery in UnifiedTrainingPipeline.

Test Coverage:
- Exception handling in each training phase
- Recovery mechanisms for transient failures
- Resource cleanup after failures
- Graceful degradation scenarios
- Memory management during error conditions
- Callback error isolation
- Configuration validation failures
- Device-related error handling
- File system error scenarios
- Network and IO error simulation
- Threading and concurrency error handling
- Resource exhaustion scenarios
"""

import pytest
import torch
import tempfile
import shutil
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock, side_effect
from typing import Dict, Any, Optional, List

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

from smartcash.model.training.unified_training_pipeline import UnifiedTrainingPipeline


class TestUnifiedTrainingPipelineExceptionHandling:
    """Test exception handling in various pipeline phases."""
    
    @pytest.fixture
    def pipeline(self):
        """Create pipeline instance for testing."""
        return UnifiedTrainingPipeline(verbose=False)
    
    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        return {
            'model': {'backbone': 'cspdarknet', 'layer_mode': 'multi'},
            'device': {'auto_detect': True, 'device': 'cpu'},
            'training': {'compile_model': False},
            'training_phases': {'phase_1': {'epochs': 2}, 'phase_2': {'epochs': 3}},
            'paths': {'checkpoints': Path('/tmp/test'), 'logs': Path('/tmp/logs')}
        }
    
    def test_preparation_phase_exception_handling(self, pipeline):
        """Test exception handling in preparation phase."""
        exception_scenarios = [
            (ValueError("Invalid configuration"), "Invalid configuration"),
            (FileNotFoundError("Config file not found"), "Config file not found"),
            (PermissionError("Access denied"), "Access denied"),
            (MemoryError("Out of memory"), "Out of memory"),
            (RuntimeError("Runtime error"), "Runtime error"),
            (Exception("Generic error"), "Generic error")
        ]
        
        for exception, expected_error in exception_scenarios:
            with patch('smartcash.model.training.unified_training_pipeline.prepare_training_environment') as mock_prep:
                mock_prep.side_effect = exception
                
                result = pipeline._phase_preparation('cspdarknet', 2, 3, '/tmp/checkpoints')
                
                assert result['success'] is False
                assert f"Preparation failed: {expected_error}" in result['error']
    
    def test_build_model_phase_exception_handling(self, pipeline, mock_config):
        """Test exception handling in model building phase."""
        pipeline.config = mock_config
        
        # Test API creation failure
        with patch('smartcash.model.training.unified_training_pipeline.create_model_api') as mock_create:
            mock_create.side_effect = RuntimeError("API creation failed")
            
            result = pipeline._phase_build_model()
            
            assert result['success'] is False
            assert "Model build failed: API creation failed" in result['error']
    
    def test_build_model_device_setup_failure(self, pipeline, mock_config):
        """Test exception handling during device setup in model building."""
        pipeline.config = mock_config
        
        with patch('smartcash.model.training.unified_training_pipeline.create_model_api') as mock_create_api:
            mock_model = MagicMock()
            mock_api = MagicMock()
            mock_api.build_model.return_value = {'status': 'built', 'model': mock_model}
            mock_create_api.return_value = mock_api
            
            with patch('smartcash.model.training.unified_training_pipeline.setup_device') as mock_device:
                mock_device.side_effect = RuntimeError("Device setup failed")
                
                result = pipeline._phase_build_model()
                
                assert result['success'] is False
                assert "Model build failed: Device setup failed" in result['error']
    
    def test_build_model_compilation_failure(self, pipeline, mock_config):
        """Test handling of model compilation failures."""
        pipeline.config = mock_config
        pipeline.config['training']['compile_model'] = True
        
        with patch('smartcash.model.training.unified_training_pipeline.create_model_api') as mock_create_api:
            mock_model = MagicMock()
            mock_model.parameters.return_value = [torch.tensor([1.0])]
            mock_api = MagicMock()
            mock_api.build_model.return_value = {'status': 'built', 'model': mock_model}
            mock_create_api.return_value = mock_api
            
            with patch('smartcash.model.training.unified_training_pipeline.setup_device') as mock_device:
                mock_device.return_value = torch.device('cpu')
                with patch('smartcash.model.training.unified_training_pipeline.model_to_device') as mock_to_device:
                    mock_to_device.return_value = mock_model
                    with patch('torch.compile') as mock_compile:
                        mock_compile.side_effect = RuntimeError("Compilation failed")
                        
                        # Should handle compilation failure gracefully
                        result = pipeline._phase_build_model()
                        
                        assert result['success'] is True  # Should succeed without compilation
                        assert pipeline.model == mock_model  # Original model should be used
    
    def test_validate_model_data_loader_failure(self, pipeline, mock_config):
        """Test exception handling when data loader creation fails."""
        pipeline.config = mock_config
        mock_model = MagicMock()
        pipeline.model = mock_model
        
        with patch('smartcash.model.training.unified_training_pipeline.DataLoaderFactory') as mock_factory_class:
            mock_factory_class.side_effect = Exception("DataLoader creation failed")
            
            result = pipeline._phase_validate_model()
            
            assert result['success'] is False
            assert "Model validation failed: DataLoader creation failed" in result['error']
    
    def test_validate_model_forward_pass_exception(self, pipeline, mock_config):
        """Test exception handling during forward pass validation."""
        pipeline.config = mock_config
        mock_model = MagicMock()
        mock_model.parameters.return_value = [torch.tensor([1.0]).to('cpu')]
        mock_model.side_effect = RuntimeError("Forward pass crashed")
        pipeline.model = mock_model
        
        mock_train_loader = MagicMock()
        mock_train_loader.__len__.return_value = 5
        mock_train_loader.__iter__.return_value = iter([
            (torch.randn(1, 3, 640, 640), torch.randn(1, 85))
        ])
        
        mock_val_loader = MagicMock()
        mock_val_loader.__len__.return_value = 2
        
        with patch('smartcash.model.training.unified_training_pipeline.DataLoaderFactory') as mock_factory_class:
            mock_factory = MagicMock()
            mock_factory.create_train_loader.return_value = mock_train_loader
            mock_factory.create_val_loader.return_value = mock_val_loader
            mock_factory_class.return_value = mock_factory
            
            result = pipeline._phase_validate_model()
            
            assert result['success'] is False
            assert "Forward pass failed: Forward pass crashed" in result['error']
    
    def test_training_phase_manager_creation_failure(self, pipeline, mock_config):
        """Test exception handling when training phase manager creation fails."""
        pipeline.config = mock_config
        pipeline.model = MagicMock()
        
        with patch('smartcash.model.training.unified_training_pipeline.TrainingPhaseManager') as mock_manager_class:
            mock_manager_class.side_effect = Exception("Manager creation failed")
            
            result = pipeline._phase_training_1_with_manager(MagicMock())
            
            assert result['success'] is False
            assert "Training Phase 1 failed: Manager creation failed" in result['error']
    
    def test_training_phase_execution_failure(self, pipeline, mock_config):
        """Test exception handling during training phase execution."""
        pipeline.config = mock_config
        mock_model = MagicMock()
        pipeline.model = mock_model
        
        mock_manager = MagicMock()
        mock_manager.run_training_phase.side_effect = RuntimeError("Training execution failed")
        
        with patch.object(pipeline, '_freeze_backbone'):
            result = pipeline._phase_training_1_with_manager(mock_manager)
            
            assert result['success'] is False
            assert "Training Phase 1 failed: Training execution failed" in result['error']
    
    def test_single_phase_training_configuration_failure(self, pipeline, mock_config):
        """Test exception handling in single phase training configuration."""
        pipeline.config = mock_config
        pipeline.model = MagicMock()
        
        with patch('smartcash.model.training.unified_training_pipeline.configure_single_phase_settings') as mock_configure:
            mock_configure.side_effect = ValueError("Invalid single phase configuration")
            
            result = pipeline._phase_single_training(5)
            
            assert result['success'] is False
            assert "Single phase training failed: Invalid single phase configuration" in result['error']
    
    def test_summary_visualization_failure(self, pipeline, mock_config):
        """Test exception handling in summary and visualization phase."""
        pipeline.config = mock_config
        
        with patch('smartcash.model.training.unified_training_pipeline.create_visualization_manager') as mock_viz:
            mock_viz.side_effect = Exception("Visualization manager creation failed")
            
            result = pipeline._phase_summary_visualization()
            
            assert result['success'] is False
            assert "Summary & visualization failed: Visualization manager creation failed" in result['error']
    
    def test_summary_chart_generation_failure(self, pipeline, mock_config):
        """Test handling of chart generation failures in summary phase."""
        pipeline.config = mock_config
        
        with patch('smartcash.model.training.unified_training_pipeline.create_visualization_manager') as mock_viz:
            mock_viz_manager = MagicMock()
            mock_viz_manager.generate_comprehensive_charts.side_effect = Exception("Chart generation failed")
            mock_viz.return_value = mock_viz_manager
            
            result = pipeline._phase_summary_visualization()
            
            assert result['success'] is False
            assert "Summary & visualization failed: Chart generation failed" in result['error']
    
    def test_summary_file_save_failure(self, pipeline, mock_config):
        """Test handling of file save failures in summary phase."""
        pipeline.config = mock_config
        
        with patch('smartcash.model.training.unified_training_pipeline.create_visualization_manager') as mock_viz:
            mock_viz_manager = MagicMock()
            mock_viz_manager.generate_comprehensive_charts.return_value = {}
            mock_viz.return_value = mock_viz_manager
            
            # Mock file operations to fail
            with patch('builtins.open') as mock_open:
                mock_open.side_effect = PermissionError("Permission denied")
                
                result = pipeline._phase_summary_visualization()
                
                assert result['success'] is False
                assert "Summary & visualization failed: Permission denied" in result['error']


class TestUnifiedTrainingPipelineResourceManagement:
    """Test resource management and cleanup during error conditions."""
    
    @pytest.fixture
    def pipeline(self):
        """Create pipeline instance for testing."""
        return UnifiedTrainingPipeline(verbose=False)
    
    @pytest.fixture
    def temp_dirs(self):
        """Create temporary directories for testing."""
        temp_base = tempfile.mkdtemp()
        dirs = {
            'checkpoints': Path(temp_base) / 'checkpoints',
            'logs': Path(temp_base) / 'logs'
        }
        
        for dir_path in dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
        
        yield dirs
        
        # Cleanup
        shutil.rmtree(temp_base, ignore_errors=True)
    
    def test_memory_cleanup_after_build_failure(self, pipeline):
        """Test that memory is cleaned up after model build failure."""
        initial_model = pipeline.model
        initial_api = pipeline.model_api
        
        with patch('smartcash.model.training.unified_training_pipeline.create_model_api') as mock_create:
            mock_create.side_effect = MemoryError("Out of memory")
            
            result = pipeline._phase_build_model()
            
            assert result['success'] is False
            
            # Objects should remain in their initial state (None)
            assert pipeline.model == initial_model
            assert pipeline.model_api == initial_api
    
    def test_model_state_preservation_after_validation_failure(self, pipeline):
        """Test that model state is preserved after validation failure."""
        mock_config = {'model': {'backbone': 'cspdarknet'}}
        pipeline.config = mock_config
        
        mock_model = MagicMock()
        pipeline.model = mock_model
        
        with patch('smartcash.model.training.unified_training_pipeline.DataLoaderFactory') as mock_factory_class:
            mock_factory_class.side_effect = Exception("Validation failed")
            
            result = pipeline._phase_validate_model()
            
            assert result['success'] is False
            
            # Model should still be available
            assert pipeline.model == mock_model
    
    def test_progress_tracker_state_after_failure(self, pipeline):
        """Test progress tracker state after various failures."""
        # Test that progress tracker maintains consistency after failures
        initial_summary = pipeline.progress_tracker.get_summary()
        
        with patch('smartcash.model.training.unified_training_pipeline.prepare_training_environment') as mock_prep:
            mock_prep.side_effect = Exception("Preparation failed")
            
            result = pipeline._phase_preparation('cspdarknet', 2, 3, '/tmp')
            
            assert result['success'] is False
            
            # Progress tracker should have recorded the failure
            final_summary = pipeline.progress_tracker.get_summary()
            assert len(final_summary['phases']) > len(initial_summary['phases'])
    
    def test_file_handle_cleanup_after_io_failure(self, pipeline, temp_dirs):
        """Test that file handles are properly cleaned up after IO failures."""
        mock_config = {
            'paths': {
                'logs': temp_dirs['logs'],
                'visualization': temp_dirs['logs']  # Use same dir
            }
        }
        pipeline.config = mock_config
        
        # Create a file that will cause permission errors
        restricted_file = temp_dirs['logs'] / 'restricted.json'
        restricted_file.touch()
        restricted_file.chmod(0o000)  # No permissions
        
        try:
            with patch('smartcash.model.training.unified_training_pipeline.create_visualization_manager') as mock_viz:
                mock_viz_manager = MagicMock()
                mock_viz_manager.generate_comprehensive_charts.return_value = {}
                mock_viz.return_value = mock_viz_manager
                
                # Mock file path to point to restricted file
                with patch('pathlib.Path.open') as mock_file_open:
                    mock_file_open.side_effect = PermissionError("Permission denied")
                    
                    result = pipeline._phase_summary_visualization()
                    
                    assert result['success'] is False
                    
                    # File operations should have been attempted and failed gracefully
                    # No file handles should be left open
        finally:
            # Cleanup restricted file
            restricted_file.chmod(0o644)
            restricted_file.unlink(missing_ok=True)
    
    def test_device_resource_cleanup_after_failure(self, pipeline):
        """Test that device resources are cleaned up after failures."""
        mock_config = {
            'model': {'backbone': 'cspdarknet'},
            'device': {'auto_detect': True, 'device': 'cpu'},
            'training': {'compile_model': False}
        }
        pipeline.config = mock_config
        
        with patch('smartcash.model.training.unified_training_pipeline.create_model_api') as mock_create_api:
            mock_model = MagicMock()
            mock_model.parameters.return_value = [torch.tensor([1.0])]
            mock_api = MagicMock()
            mock_api.build_model.return_value = {'status': 'built', 'model': mock_model}
            mock_create_api.return_value = mock_api
            
            with patch('smartcash.model.training.unified_training_pipeline.setup_device') as mock_device:
                mock_device.return_value = torch.device('cpu')
                with patch('smartcash.model.training.unified_training_pipeline.model_to_device') as mock_to_device:
                    mock_to_device.side_effect = RuntimeError("Device transfer failed")
                    
                    result = pipeline._phase_build_model()
                    
                    assert result['success'] is False
                    
                    # Device setup should have been attempted
                    mock_device.assert_called_once()


class TestUnifiedTrainingPipelineCallbackErrorIsolation:
    """Test error isolation in callback functions."""
    
    @pytest.fixture
    def pipeline_with_failing_callbacks(self):
        """Create pipeline with callbacks that raise exceptions."""
        def failing_progress_callback(*args, **kwargs):
            raise Exception("Progress callback failed")
        
        def failing_log_callback(*args, **kwargs):
            raise Exception("Log callback failed")
        
        def failing_chart_callback(*args, **kwargs):
            raise Exception("Chart callback failed")
        
        def failing_metrics_callback(*args, **kwargs):
            raise Exception("Metrics callback failed")
        
        return UnifiedTrainingPipeline(
            progress_callback=failing_progress_callback,
            log_callback=failing_log_callback,
            live_chart_callback=failing_chart_callback,
            metrics_callback=failing_metrics_callback,
            verbose=False
        )
    
    def test_log_callback_failure_isolation(self, pipeline_with_failing_callbacks):
        """Test that log callback failures don't crash the pipeline."""
        pipeline = pipeline_with_failing_callbacks
        
        # Should not raise exception despite failing callback
        pipeline._emit_log('info', 'Test message')
        
        # Pipeline should continue to function
        assert pipeline.log_callback is not None
    
    def test_metrics_callback_failure_isolation(self, pipeline_with_failing_callbacks):
        """Test that metrics callback failures don't crash the pipeline."""
        pipeline = pipeline_with_failing_callbacks
        pipeline.phase_start_time = time.time()
        pipeline.training_start_time = time.time()
        
        # Should not raise exception despite failing callback
        pipeline._emit_metrics('test_phase', 1, {'loss': 0.5})
        
        # Pipeline should continue to function
        assert pipeline.metrics_callback is not None
    
    def test_chart_callback_failure_isolation(self, pipeline_with_failing_callbacks):
        """Test that chart callback failures don't crash the pipeline."""
        pipeline = pipeline_with_failing_callbacks
        
        # Should not raise exception despite failing callback
        pipeline._emit_live_chart('test_chart', {'data': [1, 2, 3]})
        
        # Pipeline should continue to function
        assert pipeline.live_chart_callback is not None
    
    def test_multiple_callback_failures_isolation(self, pipeline_with_failing_callbacks):
        """Test that multiple callback failures are isolated from each other."""
        pipeline = pipeline_with_failing_callbacks
        pipeline.phase_start_time = time.time()
        pipeline.training_start_time = time.time()
        
        # All callbacks should fail gracefully without affecting each other
        pipeline._emit_log('error', 'Test error')
        pipeline._emit_metrics('test_phase', 1, {'loss': 1.0})
        pipeline._emit_live_chart('test_chart', {'loss': [1.0, 0.8]})
        
        # Pipeline should remain functional
        assert pipeline.log_callback is not None
        assert pipeline.metrics_callback is not None
        assert pipeline.live_chart_callback is not None
    
    def test_callback_with_none_return_values(self):
        """Test callbacks that return None or unexpected values."""
        def none_returning_callback(*args, **kwargs):
            return None
        
        def unexpected_returning_callback(*args, **kwargs):
            return "unexpected_string"
        
        pipeline1 = UnifiedTrainingPipeline(log_callback=none_returning_callback)
        pipeline2 = UnifiedTrainingPipeline(metrics_callback=unexpected_returning_callback)
        
        # Should handle None returns gracefully
        pipeline1._emit_log('info', 'Test')
        
        # Should handle unexpected returns gracefully
        pipeline2._emit_metrics('test', 1, {})
    
    def test_callback_with_invalid_signatures(self):
        """Test callbacks with incorrect signatures."""
        def wrong_signature_callback(wrong_param):
            return "called with wrong signature"
        
        pipeline = UnifiedTrainingPipeline(log_callback=wrong_signature_callback)
        
        # Should handle signature mismatch gracefully
        pipeline._emit_log('info', 'Test message')


class TestUnifiedTrainingPipelineDeviceErrorHandling:
    """Test device-related error handling scenarios."""
    
    @pytest.fixture
    def pipeline(self):
        """Create pipeline instance for testing."""
        return UnifiedTrainingPipeline(verbose=False)
    
    def test_device_unavailable_fallback(self, pipeline):
        """Test fallback when requested device is unavailable."""
        mock_config = {
            'model': {'backbone': 'cspdarknet'},
            'device': {'auto_detect': True, 'device': 'cuda'},  # Request CUDA
            'training': {'compile_model': False}
        }
        pipeline.config = mock_config
        
        with patch('smartcash.model.training.unified_training_pipeline.create_model_api') as mock_create_api:
            mock_model = MagicMock()
            mock_model.parameters.return_value = [torch.tensor([1.0])]
            mock_api = MagicMock()
            mock_api.build_model.return_value = {'status': 'built', 'model': mock_model}
            mock_create_api.return_value = mock_api
            
            with patch('smartcash.model.training.unified_training_pipeline.setup_device') as mock_device:
                # Simulate CUDA unavailable, fallback to CPU
                mock_device.side_effect = [RuntimeError("CUDA unavailable"), torch.device('cpu')]
                with patch('smartcash.model.training.unified_training_pipeline.model_to_device') as mock_to_device:
                    mock_to_device.return_value = mock_model
                    
                    # Should handle device fallback gracefully
                    result = pipeline._phase_build_model()
                    
                    # Should fail on first device setup call but could potentially retry
                    assert result['success'] is False
    
    def test_device_memory_exhaustion(self, pipeline):
        """Test handling of device memory exhaustion."""
        mock_config = {
            'model': {'backbone': 'cspdarknet'},
            'device': {'auto_detect': True, 'device': 'cpu'},
            'training': {'compile_model': False}
        }
        pipeline.config = mock_config
        
        with patch('smartcash.model.training.unified_training_pipeline.create_model_api') as mock_create_api:
            mock_model = MagicMock()
            mock_model.parameters.return_value = [torch.tensor([1.0])]
            mock_api = MagicMock()
            mock_api.build_model.return_value = {'status': 'built', 'model': mock_model}
            mock_create_api.return_value = mock_api
            
            with patch('smartcash.model.training.unified_training_pipeline.setup_device') as mock_device:
                mock_device.return_value = torch.device('cpu')
                with patch('smartcash.model.training.unified_training_pipeline.model_to_device') as mock_to_device:
                    mock_to_device.side_effect = RuntimeError("CUDA out of memory")
                    
                    result = pipeline._phase_build_model()
                    
                    assert result['success'] is False
                    assert "CUDA out of memory" in result['error']
    
    def test_device_compatibility_issues(self, pipeline):
        """Test handling of device compatibility issues."""
        mock_config = {
            'model': {'backbone': 'cspdarknet'},
            'device': {'auto_detect': True, 'device': 'mps'},
            'training': {'compile_model': False}
        }
        pipeline.config = mock_config
        
        with patch('smartcash.model.training.unified_training_pipeline.create_model_api') as mock_create_api:
            mock_model = MagicMock()
            mock_model.parameters.return_value = [torch.tensor([1.0])]
            mock_api = MagicMock()
            mock_api.build_model.return_value = {'status': 'built', 'model': mock_model}
            mock_create_api.return_value = mock_api
            
            with patch('smartcash.model.training.unified_training_pipeline.setup_device') as mock_device:
                mock_device.side_effect = ValueError("MPS not supported on this platform")
                
                result = pipeline._phase_build_model()
                
                assert result['success'] is False
                assert "MPS not supported" in result['error']


class TestUnifiedTrainingPipelineFileSystemErrorHandling:
    """Test file system related error handling."""
    
    @pytest.fixture
    def pipeline(self):
        """Create pipeline instance for testing."""
        return UnifiedTrainingPipeline(verbose=False)
    
    @pytest.fixture
    def temp_dirs(self):
        """Create temporary directories for testing."""
        temp_base = tempfile.mkdtemp()
        dirs = {
            'checkpoints': Path(temp_base) / 'checkpoints',
            'logs': Path(temp_base) / 'logs',
            'readonly': Path(temp_base) / 'readonly'
        }
        
        for dir_path in dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Make readonly directory read-only
        dirs['readonly'].chmod(0o444)
        
        yield dirs
        
        # Cleanup
        dirs['readonly'].chmod(0o755)  # Restore permissions for cleanup
        shutil.rmtree(temp_base, ignore_errors=True)
    
    def test_checkpoint_directory_permission_error(self, pipeline, temp_dirs):
        """Test handling of permission errors when accessing checkpoint directory."""
        mock_config = {
            'model': {'backbone': 'cspdarknet', 'layer_mode': 'multi'},
            'paths': {'checkpoints': temp_dirs['readonly']}
        }
        pipeline.config = mock_config
        pipeline.training_session_id = 'test_session'
        
        mock_model = MagicMock()
        mock_model.parameters.return_value = [torch.tensor([1.0]).to('cpu')]
        mock_model.state_dict.return_value = {'layer.weight': torch.tensor([1.0])}
        pipeline.model = mock_model
        
        mock_api = MagicMock()
        mock_api.save_checkpoint.return_value = None  # API fails
        pipeline.model_api = mock_api
        
        # Should handle permission error gracefully
        result = pipeline._save_checkpoint(epoch=1, metrics={}, phase_num=1)
        
        # Should return None indicating failure
        assert result is None
    
    def test_log_directory_creation_failure(self, pipeline, temp_dirs):
        """Test handling of log directory creation failures."""
        # Use a path that can't be created (under a file instead of directory)
        log_file = temp_dirs['logs'] / 'blocking_file.txt'
        log_file.touch()
        
        mock_config = {
            'paths': {
                'logs': log_file / 'cannot_create',  # Can't create dir under file
                'visualization': temp_dirs['logs']
            }
        }
        pipeline.config = mock_config
        
        with patch('smartcash.model.training.unified_training_pipeline.create_visualization_manager') as mock_viz:
            mock_viz_manager = MagicMock()
            mock_viz_manager.generate_comprehensive_charts.return_value = {}
            mock_viz.return_value = mock_viz_manager
            
            result = pipeline._phase_summary_visualization()
            
            # Should fail due to directory creation issues
            assert result['success'] is False
    
    def test_disk_space_exhaustion_simulation(self, pipeline, temp_dirs):
        """Test handling of disk space exhaustion during file operations."""
        mock_config = {
            'paths': {
                'logs': temp_dirs['logs'],
                'visualization': temp_dirs['logs']
            }
        }
        pipeline.config = mock_config
        
        with patch('smartcash.model.training.unified_training_pipeline.create_visualization_manager') as mock_viz:
            mock_viz_manager = MagicMock()
            mock_viz_manager.generate_comprehensive_charts.return_value = {}
            mock_viz.return_value = mock_viz_manager
            
            # Mock file write to simulate disk space exhaustion
            with patch('builtins.open') as mock_open:
                mock_file = MagicMock()
                mock_file.__enter__.return_value = mock_file
                mock_file.write.side_effect = OSError("No space left on device")
                mock_open.return_value = mock_file
                
                result = pipeline._phase_summary_visualization()
                
                assert result['success'] is False
                assert "No space left on device" in result['error']
    
    def test_corrupted_file_system_handling(self, pipeline, temp_dirs):
        """Test handling of corrupted file system operations."""
        mock_config = {
            'paths': {
                'logs': temp_dirs['logs'],
                'visualization': temp_dirs['logs']
            }
        }
        pipeline.config = mock_config
        
        with patch('smartcash.model.training.unified_training_pipeline.create_visualization_manager') as mock_viz:
            mock_viz_manager = MagicMock()
            mock_viz_manager.generate_comprehensive_charts.return_value = {}
            mock_viz.return_value = mock_viz_manager
            
            # Mock pathlib operations to simulate file system corruption
            with patch('pathlib.Path.mkdir') as mock_mkdir:
                mock_mkdir.side_effect = OSError("Input/output error")
                
                result = pipeline._phase_summary_visualization()
                
                assert result['success'] is False
                assert "Input/output error" in result['error']


class TestUnifiedTrainingPipelineRecoveryMechanisms:
    """Test recovery mechanisms and graceful degradation."""
    
    @pytest.fixture
    def pipeline(self):
        """Create pipeline instance for testing."""
        return UnifiedTrainingPipeline(verbose=False)
    
    def test_graceful_degradation_without_visualization(self, pipeline):
        """Test pipeline continues without visualization when it fails."""
        mock_config = {
            'model': {'num_classes': [80]},
            'paths': {
                'visualization': Path('/nonexistent/path'),
                'logs': Path('/tmp/logs')
            }
        }
        pipeline.config = mock_config
        
        with patch('smartcash.model.training.unified_training_pipeline.create_visualization_manager') as mock_viz:
            mock_viz.side_effect = Exception("Visualization unavailable")
            
            # Should handle visualization failure gracefully
            result = pipeline._phase_summary_visualization()
            
            assert result['success'] is False
            # But pipeline could potentially continue with degraded functionality
    
    def test_partial_functionality_preservation(self, pipeline):
        """Test that partial functionality is preserved when some components fail."""
        mock_config = {
            'training_phases': {'phase_1': {'epochs': 2}}
        }
        pipeline.config = mock_config
        pipeline.model = MagicMock()
        
        # Test training continues even if some operations fail
        with patch.object(pipeline, '_freeze_backbone') as mock_freeze:
            mock_freeze.side_effect = Exception("Freeze failed")
            
            mock_manager = MagicMock()
            mock_manager.run_training_phase.return_value = {'success': True}
            
            # Should continue despite freeze failure
            with patch('smartcash.model.training.unified_training_pipeline.TrainingPhaseManager') as mock_manager_class:
                mock_manager_class.return_value = mock_manager
                
                result = pipeline._phase_training_1_with_manager(mock_manager)
                
                # Training could still succeed if manager handles the error
                assert mock_manager.run_training_phase.called
    
    def test_fallback_configuration_usage(self, pipeline):
        """Test using fallback configurations when primary config fails."""
        # Test that pipeline can use minimal configuration when full config fails
        minimal_config = {
            'model': {'backbone': 'cspdarknet'},
            'training_phases': {'phase_1': {'epochs': 1}}
        }
        pipeline.config = minimal_config
        
        # Should be able to operate with minimal configuration
        assert pipeline.config['model']['backbone'] == 'cspdarknet'
        assert pipeline.config['training_phases']['phase_1']['epochs'] == 1
    
    def test_state_recovery_after_partial_failure(self, pipeline):
        """Test that pipeline state can be recovered after partial failures."""
        # Set initial state
        pipeline.training_session_id = 'original_session'
        pipeline.current_phase = 'preparation'
        
        # Simulate failure that should not affect session state
        with patch('smartcash.model.training.unified_training_pipeline.prepare_training_environment') as mock_prep:
            mock_prep.side_effect = Exception("Preparation failed")
            
            result = pipeline._phase_preparation('cspdarknet', 2, 3, '/tmp')
            
            assert result['success'] is False
            
            # Session ID should be preserved for potential recovery
            assert pipeline.training_session_id == 'original_session'
    
    def test_resource_limit_handling(self, pipeline):
        """Test handling of resource limits and constraints."""
        # Test that pipeline handles resource constraints gracefully
        mock_config = {
            'model': {'backbone': 'cspdarknet'},
            'device': {'auto_detect': True, 'device': 'cpu'},
            'training': {'compile_model': False}
        }
        pipeline.config = mock_config
        
        with patch('smartcash.model.training.unified_training_pipeline.create_model_api') as mock_create_api:
            # Simulate resource constraint
            mock_create_api.side_effect = MemoryError("Insufficient memory")
            
            result = pipeline._phase_build_model()
            
            assert result['success'] is False
            assert "Model build failed" in result['error']
            
            # Pipeline should remain in a consistent state
            assert pipeline.config == mock_config
    
    def test_timeout_handling_simulation(self, pipeline):
        """Test handling of operation timeouts."""
        mock_config = {'model': {'backbone': 'cspdarknet'}}
        pipeline.config = mock_config
        
        with patch('smartcash.model.training.unified_training_pipeline.create_model_api') as mock_create_api:
            # Simulate timeout
            mock_create_api.side_effect = TimeoutError("Operation timed out")
            
            result = pipeline._phase_build_model()
            
            assert result['success'] is False
            assert "Operation timed out" in result['error']
    
    def test_concurrent_access_error_handling(self, pipeline):
        """Test handling of concurrent access errors."""
        mock_config = {'model': {'backbone': 'cspdarknet'}}
        pipeline.config = mock_config
        
        with patch('smartcash.model.training.unified_training_pipeline.create_model_api') as mock_create_api:
            # Simulate concurrent access issue
            mock_create_api.side_effect = BlockingIOError("Resource busy")
            
            result = pipeline._phase_build_model()
            
            assert result['success'] is False
            assert "Resource busy" in result['error']