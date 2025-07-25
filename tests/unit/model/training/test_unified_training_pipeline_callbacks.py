#!/usr/bin/env python3
"""
File: tests/unit/model/training/test_unified_training_pipeline_callbacks.py

Comprehensive tests for callback functionality and UI integration in UnifiedTrainingPipeline.

Test Coverage:
- Progress callback integration and data format validation
- Log callback functionality with different log levels
- Live chart callback with various chart types and data formats
- Metrics callback with comprehensive metrics data
- Callback timing and sequence verification
- Callback data persistence and consistency
- Multi-callback coordination and synchronization
- Callback performance under high-frequency calls
- UI integration patterns and data flow
- Real-time updates and responsiveness testing
- Callback error isolation and recovery
- Memory management in callback operations
"""

import pytest
import torch
import time
import json
from unittest.mock import MagicMock, patch, call, ANY
from typing import Dict, Any, Optional, List, Callable

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

from smartcash.model.training.unified_training_pipeline import UnifiedTrainingPipeline


class TestUnifiedTrainingPipelineProgressCallbacks:
    """Test progress callback functionality and integration."""
    
    @pytest.fixture
    def mock_progress_callback(self):
        """Create mock progress callback for testing."""
        return MagicMock()
    
    @pytest.fixture
    def pipeline_with_progress(self, mock_progress_callback):
        """Create pipeline with progress callback."""
        return UnifiedTrainingPipeline(
            progress_callback=mock_progress_callback,
            verbose=False
        )
    
    def test_progress_callback_basic_functionality(self, pipeline_with_progress, mock_progress_callback):
        """Test basic progress callback functionality."""
        pipeline = pipeline_with_progress
        
        # Directly test the progress tracker which calls the callback
        progress_tracker = pipeline.progress_tracker
        
        # Trigger progress updates manually to test callback
        progress_tracker.start_phase('test_phase', 100, 'Testing progress')
        progress_tracker.update_phase(50, 100, 'Halfway done')
        progress_tracker.complete_phase({'success': True})
        
        # Verify progress callback was called
        assert mock_progress_callback.call_count >= 3  # start, update, complete
    
    def test_progress_callback_data_format(self, pipeline_with_progress):
        """Test progress callback data format and structure."""
        pipeline = pipeline_with_progress
        
        # Access the progress tracker to verify callback integration
        progress_tracker = pipeline.progress_tracker
        
        # Manually trigger progress updates to test format
        progress_tracker.start_phase('test_phase', 100, 'Testing progress format')
        progress_tracker.update_phase(50, 100, 'Halfway through')
        progress_tracker.complete_phase({'success': True})
        
        # Verify progress callback was called with correct format
        callback = pipeline.progress_tracker.progress_callback
        if callback:
            assert callback.call_count >= 3  # start, update, complete
    
    def test_progress_callback_phase_transitions(self, pipeline_with_progress, mock_progress_callback):
        """Test progress callback during phase transitions."""
        pipeline = pipeline_with_progress
        progress_tracker = pipeline.progress_tracker
        
        # Simulate multiple phase transitions
        phases = ['preparation', 'build_model', 'validate_model', 'training_phase_1', 'training_phase_2']
        
        for phase in phases:
            progress_tracker.start_phase(phase, 100, f"Running {phase}")
            progress_tracker.update_phase(50, 100, f"Progress in {phase}")
            progress_tracker.complete_phase({'success': True})
        
        # Verify progress was called for multiple phases
        assert mock_progress_callback.call_count >= len(phases) * 3  # start, update, complete for each
        
        # Verify phase transitions were tracked
        call_args_list = mock_progress_callback.call_args_list
        phase_names = []
        for call_args in call_args_list:
            if len(call_args[0]) > 0 and isinstance(call_args[0][0], str):
                phase_names.append(call_args[0][0])
        
        # Should have multiple distinct phases
        unique_phases = set(phase_names)
        assert len(unique_phases) >= len(phases)
    
    def test_progress_callback_error_handling(self, mock_progress_callback):
        """Test progress callback error handling doesn't affect pipeline."""
        # Make callback raise exception
        mock_progress_callback.side_effect = Exception("Progress callback failed")
        
        pipeline = UnifiedTrainingPipeline(
            progress_callback=mock_progress_callback,
            verbose=False
        )
        
        # Pipeline should continue despite callback failure
        with patch('smartcash.model.training.unified_training_pipeline.validate_training_mode_and_params'):
            with patch('smartcash.model.training.unified_training_pipeline.setup_training_session') as mock_setup:
                mock_setup.return_value = ('session_123', None)
                with patch.object(pipeline, '_phase_preparation') as mock_prep:
                    mock_prep.return_value = {'success': True, 'config': {}}
                    
                    # Should not raise exception despite failing callback
                    result = pipeline.run_full_training_pipeline()
        
        # Verify callback was attempted
        assert mock_progress_callback.called
    
    def test_progress_callback_high_frequency_updates(self, mock_progress_callback):
        """Test progress callback performance with high-frequency updates."""
        pipeline = UnifiedTrainingPipeline(
            progress_callback=mock_progress_callback,
            verbose=False
        )
        
        # Simulate high-frequency progress updates
        progress_tracker = pipeline.progress_tracker
        progress_tracker.start_phase('high_frequency_test', 1000, 'Testing high frequency')
        
        # Rapid updates
        for i in range(100):
            progress_tracker.update_phase(i, 1000, f'Update {i}')
        
        progress_tracker.complete_phase({'success': True})
        
        # Verify all updates were processed
        assert mock_progress_callback.call_count >= 100


class TestUnifiedTrainingPipelineLogCallbacks:
    """Test log callback functionality and integration."""
    
    @pytest.fixture
    def mock_log_callback(self):
        """Create mock log callback for testing."""
        return MagicMock()
    
    @pytest.fixture
    def pipeline_with_logging(self, mock_log_callback):
        """Create pipeline with log callback."""
        return UnifiedTrainingPipeline(
            log_callback=mock_log_callback,
            verbose=False
        )
    
    def test_log_callback_different_levels(self, pipeline_with_logging, mock_log_callback):
        """Test log callback with different log levels."""
        pipeline = pipeline_with_logging
        pipeline.current_phase = 'test_phase'
        pipeline.training_session_id = 'session_123'
        
        # Test different log levels
        log_levels = ['debug', 'info', 'warning', 'error', 'critical']
        
        for level in log_levels:
            pipeline._emit_log(level, f'Test {level} message', {'test_data': level})
        
        # Verify all log levels were called
        assert mock_log_callback.call_count == len(log_levels)
        
        # Verify log levels in call arguments
        called_levels = [call[0][0] for call in mock_log_callback.call_args_list]
        assert set(called_levels) == set(log_levels)
    
    def test_log_callback_data_structure(self, pipeline_with_logging, mock_log_callback):
        """Test log callback data structure and format."""
        pipeline = pipeline_with_logging
        pipeline.current_phase = 'validation'
        pipeline.training_session_id = 'session_456'
        
        test_data = {
            'metrics': {'loss': 0.5, 'accuracy': 0.85},
            'epoch': 5,
            'batch_size': 32
        }
        
        pipeline._emit_log('info', 'Training progress update', test_data)
        
        # Verify callback was called once
        mock_log_callback.assert_called_once()
        
        # Verify data structure
        call_args = mock_log_callback.call_args
        level, message, log_data = call_args[0]
        
        assert level == 'info'
        assert message == 'Training progress update'
        assert log_data['phase'] == 'validation'
        assert log_data['session_id'] == 'session_456'
        assert log_data['message'] == 'Training progress update'
        assert log_data['data'] == test_data
        assert 'timestamp' in log_data
    
    def test_log_callback_with_none_data(self, pipeline_with_logging, mock_log_callback):
        """Test log callback with None data parameter."""
        pipeline = pipeline_with_logging
        
        pipeline._emit_log('info', 'Message without data')
        
        mock_log_callback.assert_called_once()
        
        # Verify data defaults to empty dict
        call_args = mock_log_callback.call_args[0]
        log_data = call_args[2]
        assert log_data['data'] == {}
    
    def test_log_callback_timing_consistency(self, pipeline_with_logging, mock_log_callback):
        """Test log callback timing information consistency."""
        pipeline = pipeline_with_logging
        
        start_time = time.time()
        pipeline._emit_log('info', 'First message')
        time.sleep(0.1)  # Small delay
        pipeline._emit_log('info', 'Second message')
        end_time = time.time()
        
        # Verify both calls were made
        assert mock_log_callback.call_count == 2
        
        # Verify timestamps are reasonable
        call_args_list = mock_log_callback.call_args_list
        first_timestamp = call_args_list[0][0][2]['timestamp']
        second_timestamp = call_args_list[1][0][2]['timestamp']
        
        assert start_time <= first_timestamp <= end_time
        assert start_time <= second_timestamp <= end_time
        assert first_timestamp <= second_timestamp
    
    def test_log_callback_large_data_handling(self, pipeline_with_logging, mock_log_callback):
        """Test log callback with large data payloads."""
        pipeline = pipeline_with_logging
        
        # Create large data payload
        large_data = {
            'large_list': list(range(10000)),
            'large_dict': {f'key_{i}': f'value_{i}' for i in range(1000)},
            'nested_structure': {
                'level1': {
                    'level2': {
                        'level3': list(range(1000))
                    }
                }
            }
        }
        
        pipeline._emit_log('info', 'Large data test', large_data)
        
        mock_log_callback.assert_called_once()
        
        # Verify large data was passed correctly
        call_args = mock_log_callback.call_args[0]
        log_data = call_args[2]
        assert len(log_data['data']['large_list']) == 10000
        assert len(log_data['data']['large_dict']) == 1000


class TestUnifiedTrainingPipelineLiveChartCallbacks:
    """Test live chart callback functionality."""
    
    @pytest.fixture
    def mock_chart_callback(self):
        """Create mock chart callback for testing."""
        return MagicMock()
    
    @pytest.fixture
    def pipeline_with_charts(self, mock_chart_callback):
        """Create pipeline with chart callback."""
        return UnifiedTrainingPipeline(
            live_chart_callback=mock_chart_callback,
            verbose=False
        )
    
    def test_chart_callback_basic_functionality(self, pipeline_with_charts, mock_chart_callback):
        """Test basic chart callback functionality."""
        pipeline = pipeline_with_charts
        pipeline.current_phase = 'training'
        pipeline.training_session_id = 'session_789'
        
        chart_data = {
            'epochs': [1, 2, 3, 4, 5],
            'loss': [1.0, 0.8, 0.6, 0.5, 0.4],
            'accuracy': [0.6, 0.7, 0.8, 0.85, 0.9]
        }
        
        chart_config = {
            'title': 'Training Progress',
            'x_label': 'Epoch',
            'y_label': 'Metrics'
        }
        
        pipeline._emit_live_chart('training_curves', chart_data, chart_config)
        
        mock_chart_callback.assert_called_once()
        
        # Verify call arguments
        call_args = mock_chart_callback.call_args
        chart_type, chart_info, config = call_args[0]
        
        assert chart_type == 'training_curves'
        assert chart_info['phase'] == 'training'
        assert chart_info['session_id'] == 'session_789'
        assert chart_info['chart_type'] == 'training_curves'
        assert chart_info['data'] == chart_data
        assert config == chart_config
    
    def test_chart_callback_different_chart_types(self, pipeline_with_charts, mock_chart_callback):
        """Test chart callback with different chart types."""
        pipeline = pipeline_with_charts
        
        chart_types = [
            ('loss_curve', {'loss': [1.0, 0.5, 0.3]}),
            ('accuracy_curve', {'accuracy': [0.6, 0.8, 0.9]}),
            ('confusion_matrix', {'matrix': [[100, 10], [5, 85]]}),
            ('learning_rate_schedule', {'lr': [0.001, 0.0008, 0.0005]}),
            ('gradient_norms', {'norms': [2.5, 1.8, 1.2]})
        ]
        
        for chart_type, data in chart_types:
            pipeline._emit_live_chart(chart_type, data)
        
        # Verify all chart types were called
        assert mock_chart_callback.call_count == len(chart_types)
        
        # Verify chart types in call arguments
        called_types = [call[0][0] for call in mock_chart_callback.call_args_list]
        expected_types = [chart_type for chart_type, _ in chart_types]
        assert called_types == expected_types
    
    def test_chart_callback_real_time_updates(self, pipeline_with_charts, mock_chart_callback):
        """Test chart callback with real-time updates."""
        pipeline = pipeline_with_charts
        
        # Simulate real-time training updates
        epochs = 10
        for epoch in range(epochs):
            chart_data = {
                'epoch': epoch,
                'loss': 1.0 - (epoch * 0.1),
                'accuracy': 0.5 + (epoch * 0.05)
            }
            
            pipeline._emit_live_chart('real_time_metrics', chart_data)
        
        # Verify all updates were sent
        assert mock_chart_callback.call_count == epochs
        
        # Verify data progression
        call_args_list = mock_chart_callback.call_args_list
        for i, call_args in enumerate(call_args_list):
            chart_info = call_args[0][1]
            assert chart_info['data']['epoch'] == i
            assert abs(chart_info['data']['loss'] - (1.0 - i * 0.1)) < 0.001
    
    def test_chart_callback_complex_data_structures(self, pipeline_with_charts, mock_chart_callback):
        """Test chart callback with complex data structures."""
        pipeline = pipeline_with_charts
        
        complex_data = {
            'multi_layer_metrics': {
                'layer_1': {'loss': [0.8, 0.6], 'accuracy': [0.7, 0.8]},
                'layer_2': {'loss': [0.9, 0.7], 'accuracy': [0.6, 0.75]},
                'layer_3': {'loss': [0.85, 0.65], 'accuracy': [0.65, 0.78]}
            },
            'batch_metrics': [
                {'batch_id': 1, 'loss': 0.9, 'time': 0.1},
                {'batch_id': 2, 'loss': 0.8, 'time': 0.12},
                {'batch_id': 3, 'loss': 0.7, 'time': 0.11}
            ],
            'metadata': {
                'total_batches': 100,
                'current_epoch': 5,
                'learning_rate': 0.001
            }
        }
        
        pipeline._emit_live_chart('complex_metrics_dashboard', complex_data)
        
        mock_chart_callback.assert_called_once()
        
        # Verify complex data was preserved
        call_args = mock_chart_callback.call_args[0]
        chart_info = call_args[1]
        received_data = chart_info['data']
        
        assert len(received_data['multi_layer_metrics']) == 3
        assert len(received_data['batch_metrics']) == 3
        assert received_data['metadata']['total_batches'] == 100
    
    def test_chart_callback_config_variations(self, pipeline_with_charts, mock_chart_callback):
        """Test chart callback with various configuration options."""
        pipeline = pipeline_with_charts
        
        config_variations = [
            None,  # No config
            {},  # Empty config
            {'title': 'Simple Chart'},  # Minimal config
            {  # Full config
                'title': 'Comprehensive Chart',
                'x_label': 'X Axis',
                'y_label': 'Y Axis',
                'colors': ['blue', 'red', 'green'],
                'style': 'line',
                'grid': True,
                'legend': True
            }
        ]
        
        for i, config in enumerate(config_variations):
            pipeline._emit_live_chart(f'chart_{i}', {'data': i}, config)
        
        # Verify all configs were handled
        assert mock_chart_callback.call_count == len(config_variations)
        
        # Verify configs were passed correctly
        call_args_list = mock_chart_callback.call_args_list
        for i, call_args in enumerate(call_args_list):
            passed_config = call_args[0][2]
            expected_config = config_variations[i] if config_variations[i] is not None else {}
            assert passed_config == expected_config or (expected_config == {} and passed_config is None)


class TestUnifiedTrainingPipelineMetricsCallbacks:
    """Test metrics callback functionality."""
    
    @pytest.fixture
    def mock_metrics_callback(self):
        """Create mock metrics callback for testing."""
        return MagicMock()
    
    @pytest.fixture
    def pipeline_with_metrics(self, mock_metrics_callback):
        """Create pipeline with metrics callback."""
        return UnifiedTrainingPipeline(
            metrics_callback=mock_metrics_callback,
            verbose=False
        )
    
    def test_metrics_callback_comprehensive_data(self, pipeline_with_metrics, mock_metrics_callback):
        """Test metrics callback with comprehensive metrics data."""
        pipeline = pipeline_with_metrics
        pipeline.training_session_id = 'metrics_session'
        pipeline.phase_start_time = time.time() - 100
        pipeline.training_start_time = time.time() - 1000
        
        comprehensive_metrics = {
            'loss': 0.45,
            'accuracy': 0.87,
            'precision': 0.89,
            'recall': 0.85,
            'f1_score': 0.87,
            'learning_rate': 0.001,
            'batch_time': 0.125,
            'data_time': 0.025,
            'memory_usage': 1024.5,
            'gradient_norm': 2.3,
            'per_class_accuracy': [0.9, 0.85, 0.88, 0.82],
            'confusion_matrix': [[45, 5], [3, 47]]
        }
        
        pipeline._emit_metrics('phase_1', 10, comprehensive_metrics)
        
        mock_metrics_callback.assert_called_once()
        
        # Verify metrics data structure
        call_args = mock_metrics_callback.call_args[0]
        phase, epoch, metrics_data = call_args
        
        assert phase == 'phase_1'
        assert epoch == 10
        assert metrics_data['phase'] == 'phase_1'
        assert metrics_data['epoch'] == 10
        assert metrics_data['session_id'] == 'metrics_session'
        assert metrics_data['metrics'] == comprehensive_metrics
        assert metrics_data['phase_duration'] >= 0
        assert metrics_data['total_duration'] >= 0
    
    def test_metrics_callback_timing_accuracy(self, pipeline_with_metrics, mock_metrics_callback):
        """Test metrics callback timing accuracy."""
        pipeline = pipeline_with_metrics
        
        # Set precise timing
        phase_start = time.time()
        training_start = time.time() - 500
        pipeline.phase_start_time = phase_start
        pipeline.training_start_time = training_start
        
        # Wait a small amount to measure duration
        time.sleep(0.1)
        
        pipeline._emit_metrics('timing_test', 1, {'loss': 0.5})
        
        mock_metrics_callback.assert_called_once()
        
        # Verify timing calculations
        call_args = mock_metrics_callback.call_args[0]
        metrics_data = call_args[2]
        
        phase_duration = metrics_data['phase_duration']
        total_duration = metrics_data['total_duration']
        
        assert phase_duration >= 0.1  # At least the sleep time
        assert total_duration >= 500  # At least the training start offset
        assert total_duration > phase_duration  # Total should be longer
    
    def test_metrics_callback_epoch_progression(self, pipeline_with_metrics, mock_metrics_callback):
        """Test metrics callback with epoch progression."""
        pipeline = pipeline_with_metrics
        pipeline.phase_start_time = time.time()
        pipeline.training_start_time = time.time()
        
        # Simulate epoch progression
        epochs = 5
        phases = ['phase_1', 'phase_2']
        
        for phase in phases:
            for epoch in range(epochs):
                metrics = {
                    'loss': 1.0 - (epoch * 0.1),
                    'accuracy': 0.5 + (epoch * 0.08),
                    'epoch_progress': epoch / epochs
                }
                
                pipeline._emit_metrics(phase, epoch, metrics)
        
        # Verify all epochs were reported
        expected_calls = len(phases) * epochs
        assert mock_metrics_callback.call_count == expected_calls
        
        # Verify progression in metrics
        call_args_list = mock_metrics_callback.call_args_list
        for i, call_args in enumerate(call_args_list):
            phase, epoch, metrics_data = call_args[0]
            expected_epoch = i % epochs
            assert epoch == expected_epoch
    
    def test_metrics_callback_memory_efficiency(self, pipeline_with_metrics, mock_metrics_callback):
        """Test metrics callback memory efficiency with large datasets."""
        pipeline = pipeline_with_metrics
        pipeline.phase_start_time = time.time()
        pipeline.training_start_time = time.time()
        
        # Create large metrics payload
        large_metrics = {
            'loss_history': [0.9 - i * 0.001 for i in range(10000)],
            'accuracy_history': [0.5 + i * 0.00005 for i in range(10000)],
            'batch_losses': [0.8 - i * 0.0001 for i in range(50000)],
            'gradient_norms_per_layer': {
                f'layer_{i}': [2.0 - j * 0.01 for j in range(1000)]
                for i in range(100)
            }
        }
        
        pipeline._emit_metrics('memory_test', 1, large_metrics)
        
        mock_metrics_callback.assert_called_once()
        
        # Verify large data was handled
        call_args = mock_metrics_callback.call_args[0]
        metrics_data = call_args[2]
        received_metrics = metrics_data['metrics']
        
        assert len(received_metrics['loss_history']) == 10000
        assert len(received_metrics['batch_losses']) == 50000
        assert len(received_metrics['gradient_norms_per_layer']) == 100
    
    def test_metrics_callback_without_timing_info(self, pipeline_with_metrics, mock_metrics_callback):
        """Test metrics callback when timing info is not available."""
        pipeline = pipeline_with_metrics
        # Don't set timing attributes
        pipeline.phase_start_time = None
        pipeline.training_start_time = None
        
        pipeline._emit_metrics('no_timing_test', 1, {'loss': 0.5})
        
        mock_metrics_callback.assert_called_once()
        
        # Verify timing defaults
        call_args = mock_metrics_callback.call_args[0]
        metrics_data = call_args[2]
        
        assert metrics_data['phase_duration'] == 0
        assert metrics_data['total_duration'] == 0


class TestUnifiedTrainingPipelineMultiCallbackCoordination:
    """Test coordination between multiple callbacks."""
    
    @pytest.fixture
    def all_callbacks(self):
        """Create all callback types for testing."""
        return {
            'progress': MagicMock(),
            'log': MagicMock(),
            'chart': MagicMock(),
            'metrics': MagicMock()
        }
    
    @pytest.fixture
    def pipeline_with_all_callbacks(self, all_callbacks):
        """Create pipeline with all callback types."""
        return UnifiedTrainingPipeline(
            progress_callback=all_callbacks['progress'],
            log_callback=all_callbacks['log'],
            live_chart_callback=all_callbacks['chart'],
            metrics_callback=all_callbacks['metrics'],
            verbose=False
        )
    
    def test_multi_callback_coordination(self, pipeline_with_all_callbacks, all_callbacks):
        """Test coordination between all callback types."""
        pipeline = pipeline_with_all_callbacks
        pipeline.current_phase = 'coordination_test'
        pipeline.training_session_id = 'coord_session'
        pipeline.phase_start_time = time.time()
        pipeline.training_start_time = time.time()
        
        # Trigger all callback types
        pipeline._emit_log('info', 'Coordination test message', {'test': 'data'})
        pipeline._emit_live_chart('test_chart', {'data': [1, 2, 3]}, {'title': 'Test'})
        pipeline._emit_metrics('coordination_phase', 5, {'loss': 0.3, 'accuracy': 0.9})
        
        # Verify all callbacks were called
        for callback_name, callback in all_callbacks.items():
            if callback_name != 'progress':  # Progress is handled differently
                assert callback.call_count >= 1, f"{callback_name} callback was not called"
    
    def test_callback_timing_synchronization(self, pipeline_with_all_callbacks, all_callbacks):
        """Test that callbacks maintain timing synchronization."""
        pipeline = pipeline_with_all_callbacks
        pipeline.current_phase = 'timing_sync'
        pipeline.training_session_id = 'sync_session'
        pipeline.phase_start_time = time.time()
        pipeline.training_start_time = time.time()
        
        # Record start time
        start_time = time.time()
        
        # Trigger callbacks in sequence
        pipeline._emit_log('info', 'First event')
        time.sleep(0.05)
        pipeline._emit_metrics('sync_phase', 1, {'loss': 0.5})
        time.sleep(0.05)
        pipeline._emit_live_chart('sync_chart', {'data': [1]})
        
        end_time = time.time()
        
        # Verify timing consistency across callbacks
        log_timestamp = all_callbacks['log'].call_args[0][2]['timestamp']
        metrics_timestamp = all_callbacks['metrics'].call_args[0][2]['timestamp']
        chart_timestamp = all_callbacks['chart'].call_args[0][1]['timestamp']
        
        assert start_time <= log_timestamp <= end_time
        assert start_time <= metrics_timestamp <= end_time
        assert start_time <= chart_timestamp <= end_time
        assert log_timestamp <= metrics_timestamp <= chart_timestamp
    
    def test_callback_data_consistency(self, pipeline_with_all_callbacks, all_callbacks):
        """Test data consistency across different callbacks."""
        pipeline = pipeline_with_all_callbacks
        session_id = 'consistency_session'
        phase = 'consistency_test'
        
        pipeline.current_phase = phase
        pipeline.training_session_id = session_id
        pipeline.phase_start_time = time.time()
        pipeline.training_start_time = time.time()
        
        # Trigger callbacks with consistent session info
        pipeline._emit_log('info', 'Consistency test')
        pipeline._emit_metrics(phase, 1, {'loss': 0.4})
        pipeline._emit_live_chart('consistency_chart', {'data': [1, 2]})
        
        # Verify session ID consistency
        log_session = all_callbacks['log'].call_args[0][2]['session_id']
        metrics_session = all_callbacks['metrics'].call_args[0][2]['session_id']
        chart_session = all_callbacks['chart'].call_args[0][1]['session_id']
        
        assert log_session == session_id
        assert metrics_session == session_id
        assert chart_session == session_id
        
        # Verify phase consistency
        log_phase = all_callbacks['log'].call_args[0][2]['phase']
        metrics_phase = all_callbacks['metrics'].call_args[0][2]['phase']
        chart_phase = all_callbacks['chart'].call_args[0][1]['phase']
        
        assert log_phase == phase
        assert metrics_phase == phase
        assert chart_phase == phase
    
    def test_partial_callback_failure_isolation(self, all_callbacks):
        """Test that failure of one callback doesn't affect others."""
        # Make one callback fail
        all_callbacks['log'].side_effect = Exception("Log callback failed")
        
        pipeline = UnifiedTrainingPipeline(
            progress_callback=all_callbacks['progress'],
            log_callback=all_callbacks['log'],
            live_chart_callback=all_callbacks['chart'],
            metrics_callback=all_callbacks['metrics'],
            verbose=False
        )
        
        pipeline.current_phase = 'isolation_test'
        pipeline.training_session_id = 'isolation_session'
        pipeline.phase_start_time = time.time()
        pipeline.training_start_time = time.time()
        
        # Trigger all callbacks
        pipeline._emit_log('error', 'This should fail')
        pipeline._emit_metrics('isolation_phase', 1, {'loss': 0.2})
        pipeline._emit_live_chart('isolation_chart', {'data': [5]})
        
        # Verify failed callback was attempted
        assert all_callbacks['log'].called
        
        # Verify other callbacks still worked
        assert all_callbacks['metrics'].called
        assert all_callbacks['chart'].called
    
    def test_high_frequency_multi_callback_performance(self, pipeline_with_all_callbacks, all_callbacks):
        """Test performance with high-frequency multi-callback updates."""
        pipeline = pipeline_with_all_callbacks
        pipeline.current_phase = 'performance_test'
        pipeline.training_session_id = 'perf_session'
        pipeline.phase_start_time = time.time()
        pipeline.training_start_time = time.time()
        
        # High-frequency updates
        iterations = 100
        start_time = time.time()
        
        for i in range(iterations):
            pipeline._emit_log('debug', f'Update {i}', {'iteration': i})
            pipeline._emit_metrics('perf_phase', i, {'loss': 1.0 - i * 0.01})
            pipeline._emit_live_chart('perf_chart', {'iteration': i, 'value': i * 2})
        
        end_time = time.time()
        
        # Verify all callbacks were called the expected number of times
        assert all_callbacks['log'].call_count == iterations
        assert all_callbacks['metrics'].call_count == iterations
        assert all_callbacks['chart'].call_count == iterations
        
        # Verify reasonable performance (should complete in reasonable time)
        total_time = end_time - start_time
        assert total_time < 10.0  # Should complete within 10 seconds


class TestUnifiedTrainingPipelineCallbackIntegrationPatterns:
    """Test real-world callback integration patterns."""
    
    def test_training_progress_integration_pattern(self):
        """Test realistic training progress integration pattern."""
        # Track all callback data for analysis
        callback_data = {
            'progress_updates': [],
            'log_entries': [],
            'chart_updates': [],
            'metrics_reports': []
        }
        
        def progress_tracker(*args, **kwargs):
            callback_data['progress_updates'].append((args, kwargs))
        
        def log_tracker(level, message, data):
            callback_data['log_entries'].append((level, message, data))
        
        def chart_tracker(chart_type, data, config):
            callback_data['chart_updates'].append((chart_type, data, config))
        
        def metrics_tracker(phase, epoch, data):
            callback_data['metrics_reports'].append((phase, epoch, data))
        
        pipeline = UnifiedTrainingPipeline(
            progress_callback=progress_tracker,
            log_callback=log_tracker,
            live_chart_callback=chart_tracker,
            metrics_callback=metrics_tracker,
            verbose=False
        )
        
        # Simulate realistic callback usage directly
        pipeline.current_phase = 'integration_test'
        pipeline.training_session_id = 'test_session'
        pipeline.phase_start_time = time.time()
        pipeline.training_start_time = time.time()
        
        # Trigger various callbacks
        pipeline._emit_log('info', 'Training pipeline started', {'session': 'test_session'})
        pipeline._emit_metrics('phase_1', 1, {'loss': 0.5, 'accuracy': 0.8})
        pipeline._emit_live_chart('training_progress', {'epoch': 1, 'loss': 0.5}, {'title': 'Progress'})
        
        # Trigger progress updates via progress tracker
        pipeline.progress_tracker.start_phase('test_phase', 100, 'Testing integration')
        pipeline.progress_tracker.update_phase(50, 100, 'Halfway')
        pipeline.progress_tracker.complete_phase({'success': True})
        
        # Analyze callback integration
        assert len(callback_data['progress_updates']) > 0
        assert len(callback_data['log_entries']) > 0
        assert len(callback_data['chart_updates']) > 0
        assert len(callback_data['metrics_reports']) > 0
        
        # Verify realistic data flow patterns
        log_entries = callback_data['log_entries']
        assert any('Training pipeline started' in entry[1] for entry in log_entries)
    
    def test_real_time_monitoring_pattern(self):
        """Test real-time monitoring integration pattern."""
        monitoring_state = {
            'current_metrics': {},
            'alert_conditions': [],
            'performance_history': []
        }
        
        def monitoring_log_callback(level, message, data):
            if level == 'error':
                monitoring_state['alert_conditions'].append((message, data))
        
        def monitoring_metrics_callback(phase, epoch, data):
            monitoring_state['current_metrics'] = data['metrics']
            monitoring_state['performance_history'].append({
                'phase': phase,
                'epoch': epoch,
                'timestamp': data['timestamp'],
                'metrics': data['metrics']
            })
        
        def monitoring_chart_callback(chart_type, data, config):
            # Real-time chart updates for monitoring dashboard
            if 'loss' in data['data']:
                current_loss = data['data']['loss']
                if isinstance(current_loss, (int, float)) and current_loss > 2.0:
                    monitoring_state['alert_conditions'].append(
                        (f'High loss detected in {chart_type}', {'loss': current_loss})
                    )
        
        pipeline = UnifiedTrainingPipeline(
            log_callback=monitoring_log_callback,
            metrics_callback=monitoring_metrics_callback,
            live_chart_callback=monitoring_chart_callback,
            verbose=False
        )
        
        # Simulate monitoring scenarios
        pipeline.phase_start_time = time.time()
        pipeline.training_start_time = time.time()
        pipeline.current_phase = 'monitoring_test'
        pipeline.training_session_id = 'monitor_session'
        
        # Normal metrics
        pipeline._emit_metrics('phase_1', 1, {'loss': 0.5, 'accuracy': 0.8})
        pipeline._emit_live_chart('loss_chart', {'loss': 0.5}, {})
        
        # Alert condition
        pipeline._emit_metrics('phase_1', 2, {'loss': 2.5, 'accuracy': 0.6})
        pipeline._emit_live_chart('loss_chart', {'loss': 2.5}, {})
        pipeline._emit_log('error', 'Training instability detected', {'loss': 2.5})
        
        # Verify monitoring state
        assert len(monitoring_state['performance_history']) == 2
        assert len(monitoring_state['alert_conditions']) >= 2  # High loss + error log
        assert monitoring_state['current_metrics']['loss'] == 2.5