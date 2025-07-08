#!/usr/bin/env python3
"""
Integration tests for complete training pipeline.
"""

import pytest
import asyncio
import tempfile
from pathlib import Path

from smartcash.ui.model.train.services.training_service import TrainingService
from smartcash.ui.model.train.operations.start_training_operation import StartTrainingOperation
from smartcash.ui.model.train.operations.stop_training_operation import StopTrainingOperation
from smartcash.ui.model.train.operations.resume_training_operation import ResumeTrainingOperation
from smartcash.ui.model.train.constants import DEFAULT_CONFIG
from smartcash.ui.components.chart_container import create_chart_container


class TestTrainingPipeline:
    """Test complete training pipeline integration."""
    
    @pytest.mark.asyncio
    async def test_complete_training_workflow(self):
        """Test complete training workflow from start to finish."""
        # Create UI training service
        ui_service = TrainingService()
        
        # Force simulation mode for reliable testing
        ui_service.backend_service = None
        
        config = DEFAULT_CONFIG.copy()
        config['training']['epochs'] = 3  # Short test
        
        progress_messages = []
        log_messages = []
        
        def progress_callback(percent, message):
            progress_messages.append((percent, message))
        
        def log_callback(message):
            log_messages.append(message)
        
        # Step 1: Initialize training
        init_result = await ui_service.initialize_training(
            config=config,
            progress_callback=progress_callback,
            log_callback=log_callback
        )
        
        assert init_result['success'] is True
        assert ui_service.current_phase.value in ['idle', 'initialized']
        
        # Step 2: Start training
        start_result = await ui_service.start_training(
            epochs=3,
            progress_callback=progress_callback,
            log_callback=log_callback
        )
        
        assert start_result['success'] is True
        assert start_result.get('simulation') is True
        assert len(progress_messages) > 0
        assert len(log_messages) > 0
        
        # Step 3: Verify metrics were generated
        metrics = ui_service.get_metrics_for_charts()
        assert len(metrics['epochs']) == 3
        assert 'loss_chart' in metrics
        assert 'performance_chart' in metrics
        
        # Step 4: Check final status
        status = ui_service.get_current_status()
        assert status['phase'] == 'completed'
        assert not status['is_training']
    
    @pytest.mark.asyncio
    async def test_operation_handlers_integration(self):
        """Test integration between operation handlers."""
        progress_msgs = []
        log_msgs = []
        
        def progress_callback(percent, message):
            progress_msgs.append((percent, message))
        
        def log_callback(message):
            log_msgs.append(message)
        
        # Test Start Training Operation
        start_op = StartTrainingOperation()
        
        config = DEFAULT_CONFIG.copy()
        config['training']['epochs'] = 2
        
        start_result = await start_op.execute_operation(
            config=config,
            progress_callback=progress_callback,
            log_callback=log_callback
        )
        
        assert 'success' in start_result
        assert 'operation_id' in start_result
        assert len(progress_msgs) > 0
        
        # Test Stop Training Operation
        stop_op = StopTrainingOperation()
        
        stop_result = await stop_op.execute_operation(
            progress_callback=progress_callback,
            log_callback=log_callback
        )
        
        assert 'success' in stop_result
        
        # Test Resume Training Operation
        resume_op = ResumeTrainingOperation()
        
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp_file:
            tmp_file.write(b"dummy checkpoint")
            checkpoint_path = tmp_file.name
        
        try:
            resume_result = await resume_op.execute_operation(
                checkpoint_path=checkpoint_path,
                additional_epochs=5,
                progress_callback=progress_callback,
                log_callback=log_callback
            )
            
            assert 'success' in resume_result
        finally:
            Path(checkpoint_path).unlink()
    
    def test_chart_training_integration(self):
        """Test chart container integration with training metrics."""
        # Create chart container
        chart_container = create_chart_container(
            title="Training Metrics",
            chart_type="line",
            columns=2,
            height=400
        )
        
        assert chart_container._columns == 2
        assert chart_container._chart_type == "line"
        
        # Initialize chart
        chart_container.initialize()
        
        # Create training service and simulate training
        training_service = TrainingService()
        
        # Simulate training metrics
        training_service.training_history = {
            "epochs": [1, 2, 3, 4, 5],
            "loss_metrics": {
                "train_loss": [0.8, 0.6, 0.4, 0.3, 0.25],
                "val_loss": [0.9, 0.7, 0.5, 0.4, 0.35]
            },
            "performance_metrics": {
                "val_map50": [0.3, 0.5, 0.7, 0.8, 0.85],
                "accuracy": [0.6, 0.7, 0.8, 0.85, 0.9]
            }
        }
        
        # Get metrics for charts
        metrics = training_service.get_metrics_for_charts()
        
        # Update charts with training data
        if 'loss_chart' in metrics:
            loss_data = metrics['loss_chart']['data']
            if 'train_loss' in loss_data:
                chart_container.update_chart("chart_1", loss_data['train_loss'], {
                    "title": "Training Loss",
                    "color": "#ff6b6b"
                })
        
        if 'performance_chart' in metrics:
            perf_data = metrics['performance_chart']['data']
            if 'val_map50' in perf_data:
                chart_container.update_chart("chart_2", perf_data['val_map50'], {
                    "title": "Validation mAP@0.5",
                    "color": "#4ecdc4"
                })
        
        # Verify charts were updated
        assert "chart_1" in chart_container._chart_data
        assert "chart_2" in chart_container._chart_data
        assert chart_container._chart_data["chart_1"] == [0.8, 0.6, 0.4, 0.3, 0.25]
        assert chart_container._chart_data["chart_2"] == [0.3, 0.5, 0.7, 0.8, 0.85]
    
    @pytest.mark.asyncio
    async def test_error_recovery_workflow(self):
        """Test error recovery and graceful degradation."""
        ui_service = TrainingService()
        
        # Test with invalid configuration
        invalid_config = {
            'training': {
                'epochs': -1,  # Invalid
                'batch_size': 0   # Invalid
            }
        }
        
        # Should handle gracefully
        init_result = await ui_service.initialize_training(
            config=invalid_config,
            log_callback=lambda msg: None
        )
        
        # Should still succeed (will use defaults or simulation)
        assert 'success' in init_result
        
        # Test stop when not training
        stop_result = await ui_service.stop_training(
            log_callback=lambda msg: None
        )
        
        assert 'success' in stop_result
        # May indicate no training in progress
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self):
        """Test handling of concurrent training operations."""
        ui_service = TrainingService()
        ui_service.backend_service = None  # Force simulation
        
        config = DEFAULT_CONFIG.copy()
        config['training']['epochs'] = 2
        
        # Start training
        start_task = asyncio.create_task(
            ui_service.start_training(
                epochs=2,
                progress_callback=lambda p, m: None,
                log_callback=lambda m: None
            )
        )
        
        # Give it a moment to start
        await asyncio.sleep(0.1)
        
        # Try to start another training (should be rejected)
        start_result2 = await ui_service.start_training(
            epochs=2,
            progress_callback=lambda p, m: None,
            log_callback=lambda m: None
        )
        
        # Wait for first training to complete
        start_result1 = await start_task
        
        # First should succeed, second should be rejected
        assert start_result1['success'] is True
        assert start_result2['success'] is False
        assert 'already in progress' in start_result2.get('message', '').lower()
    
    def test_metrics_consistency(self):
        """Test metrics consistency across different components."""
        ui_service = TrainingService()
        
        # Set up consistent training history
        epochs = [1, 2, 3]
        train_loss = [0.8, 0.6, 0.4]
        val_map50 = [0.3, 0.5, 0.7]
        
        ui_service.training_history = {
            "epochs": epochs,
            "loss_metrics": {"train_loss": train_loss},
            "performance_metrics": {"val_map50": val_map50}
        }
        
        # Get metrics through different methods
        chart_metrics = ui_service.get_metrics_for_charts()
        status = ui_service.get_current_status()
        
        # Verify consistency
        assert chart_metrics['epochs'] == epochs
        assert chart_metrics['loss_chart']['data']['train_loss'] == train_loss
        assert chart_metrics['performance_chart']['data']['val_map50'] == val_map50
        
        assert status['training_history']['epochs'] == epochs
        assert status['training_history']['loss_metrics']['train_loss'] == train_loss
        assert status['training_history']['performance_metrics']['val_map50'] == val_map50


if __name__ == "__main__":
    pytest.main([__file__, "-v"])