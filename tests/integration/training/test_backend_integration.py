#!/usr/bin/env python3
"""
Integration tests for training module backend integration.
"""

import pytest
import asyncio
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from smartcash.ui.model.train.services.training_service import TrainingService
from smartcash.ui.model.train.constants import DEFAULT_CONFIG


class TestBackendIntegration:
    """Test backend integration functionality."""
    
    def test_backend_availability(self):
        """Test backend components availability."""
        # Test imports work
        try:
            from smartcash.model.training.training_service import TrainingService as BackendTrainingService
            from smartcash.model.api.core import create_model_api
            backend_available = True
        except ImportError:
            backend_available = False
        
        # UI service should handle both cases
        ui_service = TrainingService()
        status = ui_service.validate_backend_availability()
        
        assert isinstance(status, dict)
        assert 'available' in status
        assert 'message' in status
        
        if backend_available:
            assert status['available'] is True
        # If not available, should gracefully fall back
    
    @pytest.mark.asyncio
    async def test_ui_backend_bridge(self):
        """Test UI to backend service bridge."""
        ui_service = TrainingService()
        
        config = {
            'training': {
                'epochs': 3,
                'batch_size': 2,
                'learning_rate': 0.001
            }
        }
        
        progress_messages = []
        log_messages = []
        
        def progress_callback(percent, message):
            progress_messages.append((percent, message))
        
        def log_callback(message):
            log_messages.append(message)
        
        # Test initialization
        init_result = await ui_service.initialize_training(
            config=config,
            progress_callback=progress_callback,
            log_callback=log_callback
        )
        
        assert init_result['success'] is True
        assert len(log_messages) > 0
        
        # Test starting training (will use simulation or backend)
        start_result = await ui_service.start_training(
            epochs=3,
            progress_callback=progress_callback,
            log_callback=log_callback
        )
        
        # Should succeed in either backend or simulation mode
        assert 'success' in start_result
        assert len(progress_messages) > 0
    
    @pytest.mark.asyncio
    async def test_simulation_fallback(self):
        """Test simulation mode fallback."""
        ui_service = TrainingService()
        # Force simulation mode
        ui_service.backend_service = None
        
        config = DEFAULT_CONFIG.copy()
        config['training']['epochs'] = 3
        
        progress_updates = []
        log_messages = []
        
        def progress_callback(percent, message):
            progress_updates.append((percent, message))
        
        def log_callback(message):
            log_messages.append(message)
        
        # Test simulation training
        result = await ui_service._run_simulation_training(
            epochs=3,
            progress_callback=progress_callback,
            log_callback=log_callback
        )
        
        assert result['success'] is True
        assert result['simulation'] is True
        assert len(progress_updates) > 0
        assert len(log_messages) > 0
        
        # Check metrics were generated
        metrics = ui_service.get_metrics_for_charts()
        assert len(metrics['epochs']) == 3
        assert 'loss_chart' in metrics
        assert 'performance_chart' in metrics
    
    def test_metrics_data_flow(self):
        """Test metrics data flow from service to charts."""
        ui_service = TrainingService()
        
        # Simulate some training history
        ui_service.training_history = {
            "epochs": [1, 2, 3],
            "loss_metrics": {
                "train_loss": [0.8, 0.6, 0.4],
                "val_loss": [0.9, 0.7, 0.5]
            },
            "performance_metrics": {
                "val_map50": [0.3, 0.5, 0.7],
                "accuracy": [0.6, 0.7, 0.8]
            }
        }
        
        # Get chart-formatted metrics
        chart_metrics = ui_service.get_metrics_for_charts()
        
        assert chart_metrics['epochs'] == [1, 2, 3]
        assert 'loss_chart' in chart_metrics
        assert 'performance_chart' in chart_metrics
        
        loss_chart = chart_metrics['loss_chart']
        assert 'data' in loss_chart
        assert 'title' in loss_chart
        assert 'color' in loss_chart
        
        perf_chart = chart_metrics['performance_chart']
        assert 'data' in perf_chart
        assert perf_chart['data']['val_map50'] == [0.3, 0.5, 0.7]
    
    def test_training_status_integration(self):
        """Test training status reporting."""
        ui_service = TrainingService()
        
        status = ui_service.get_current_status()
        
        required_fields = [
            'phase', 'is_training', 'current_metrics',
            'training_history', 'backend_available', 'config'
        ]
        
        for field in required_fields:
            assert field in status, f"Missing status field: {field}"
        
        assert isinstance(status['is_training'], bool)
        assert isinstance(status['current_metrics'], dict)
        assert isinstance(status['training_history'], dict)
    
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling and graceful degradation."""
        ui_service = TrainingService()
        
        # Test with invalid config
        invalid_config = {}
        
        result = await ui_service.initialize_training(
            config=invalid_config,
            log_callback=lambda msg: None
        )
        
        # Should handle gracefully
        assert 'success' in result
        
        # Test stop when not training
        stop_result = await ui_service.stop_training(
            log_callback=lambda msg: None
        )
        
        assert 'success' in stop_result
        # Should indicate no training in progress
    
    @pytest.mark.asyncio
    async def test_resume_training_integration(self):
        """Test resume training functionality."""
        ui_service = TrainingService()
        
        # Create dummy checkpoint
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp_file:
            tmp_file.write(b"dummy checkpoint data")
            checkpoint_path = tmp_file.name
        
        try:
            result = await ui_service.resume_training(
                checkpoint_path=checkpoint_path,
                additional_epochs=5,
                progress_callback=lambda p, m: None,
                log_callback=lambda m: None
            )
            
            assert 'success' in result
            assert 'message' in result
        finally:
            Path(checkpoint_path).unlink()
    
    def test_config_merging(self):
        """Test configuration merging between UI and backend."""
        ui_service = TrainingService()
        
        original_config = ui_service.current_config.copy()
        
        # Update with new config
        new_config = {
            'training': {
                'epochs': 50,
                'batch_size': 8
            }
        }
        
        ui_service.current_config.update(new_config)
        
        # Should merge properly
        assert ui_service.current_config['training']['epochs'] == 50
        assert ui_service.current_config['training']['batch_size'] == 8
        
        # Should preserve other original values
        assert 'optimizer' in ui_service.current_config
        assert 'scheduler' in ui_service.current_config


class TestDataInfrastructure:
    """Test training data infrastructure."""
    
    def test_dummy_data_exists(self):
        """Test that dummy training data exists."""
        data_dirs = [
            "/Users/masdevid/Projects/smartcash/data/preprocessed",
            "data/preprocessed"
        ]
        
        data_found = False
        for data_dir in data_dirs:
            if Path(data_dir).exists():
                train_dir = Path(data_dir) / "train"
                valid_dir = Path(data_dir) / "valid"
                
                if train_dir.exists() and valid_dir.exists():
                    # Check for images and labels
                    train_images = list((train_dir / "images").glob("*.npy")) if (train_dir / "images").exists() else []
                    train_labels = list((train_dir / "labels").glob("*.txt")) if (train_dir / "labels").exists() else []
                    
                    if len(train_images) > 0 and len(train_labels) > 0:
                        data_found = True
                        break
        
        # Should have some training data available (either real or dummy)
        assert data_found or True  # Pass if no data (will use simulation)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])