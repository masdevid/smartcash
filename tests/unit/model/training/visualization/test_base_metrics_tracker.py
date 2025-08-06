"""
Unit tests for base_metrics_tracker.py
"""
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock


class TestBaseMetricsTracker:
    """Test cases for BaseMetricsTracker class."""
    
    def test_initialization(self, sample_metrics_tracker):
        """Test that the tracker initializes with correct attributes."""
        tracker = sample_metrics_tracker
        
        assert tracker.num_classes_per_layer == {
            'layer_1': 7, 'layer_2': 7, 'layer_3': 3
        }
        assert tracker.class_names['layer_1'] == ['001', '002', '005', '010', '020', '050', '100']
        assert tracker.save_dir == Path('/tmp/test_visualization')
        assert tracker.verbose is True
        assert len(tracker.epoch_metrics) == 0
        
    def test_update_metrics(self, sample_metrics_tracker, sample_metrics):
        """Test updating metrics for an epoch."""
        tracker = sample_metrics_tracker
        
        # First update
        tracker.update_metrics(
            epoch=0,
            phase='training',
            metrics=sample_metrics,
            phase_num=1
        )
        
        assert len(tracker.epoch_metrics) == 1
        assert tracker.epoch_metrics[0]['epoch'] == 0
        assert tracker.epoch_metrics[0]['phase'] == 'training'
        assert tracker.epoch_metrics[0]['train_loss'] == 0.5
        
        # Check layer metrics
        assert len(tracker.layer_metrics['layer_1']['accuracy']) == 1
        assert tracker.layer_metrics['layer_1']['accuracy'][0] == 0.9
        
    def test_calculate_layer_confusion_matrix(self, sample_metrics_tracker, sample_predictions, sample_ground_truth):
        """Test confusion matrix calculation."""
        tracker = sample_metrics_tracker
        layer = 'layer_1'
        epoch = 1
        
        # Get test data from fixtures
        predictions = sample_predictions[layer]
        ground_truth = sample_ground_truth[layer]
        num_classes = tracker.num_classes_per_layer[layer]
        
        # Call the method
        tracker._calculate_layer_confusion_matrix(layer, predictions, ground_truth, epoch)
        
        # Check that confusion matrix was calculated and stored
        assert layer in tracker.confusion_matrices
        assert len(tracker.confusion_matrices[layer]) == 1
        cm_data = tracker.confusion_matrices[layer][0]
        
        # Verify the results
        assert cm_data['epoch'] == epoch
        assert isinstance(cm_data['matrix'], np.ndarray)
        assert cm_data['matrix'].shape == (num_classes, num_classes)
        assert 0 <= cm_data['accuracy'] <= 1.0
        
        # Verify metrics were updated
        for metric in ['precision', 'recall', 'f1_score', 'accuracy']:
            assert len(tracker.layer_metrics[layer][metric]) == 1
            assert 0 <= tracker.layer_metrics[layer][metric][0] <= 1.0
        
    def test_phase_transition(self, sample_metrics_tracker, sample_metrics):
        """Test phase transition handling."""
        tracker = sample_metrics_tracker
        
        # Initial phase
        tracker.update_metrics(
            epoch=0,
            phase='training',
            metrics=sample_metrics,
            phase_num=1
        )
        
        # Phase transition
        tracker.update_metrics(
            epoch=1,
            phase='training',
            metrics=sample_metrics,
            phase_num=2
        )
        
        # Check if phase transitions are tracked in epoch_metrics
        assert len(tracker.epoch_metrics) == 2
        assert tracker.epoch_metrics[0]['phase_num'] == 1
        assert tracker.epoch_metrics[1]['phase_num'] == 2
        
    def test_cleanup(self, sample_metrics_tracker):
        """Test cleanup of resources."""
        tracker = sample_metrics_tracker
        
        # Mock the executor
        mock_executor = MagicMock()
        tracker.io_executor = mock_executor
        
        # Call cleanup
        tracker.cleanup()
        
        # Should call shutdown on executor
        mock_executor.shutdown.assert_called_once_with(wait=True)
        
    def test_update_layer_metrics(self, sample_metrics_tracker, sample_metrics):
        """Test updating layer metrics."""
        tracker = sample_metrics_tracker
        
        # Update metrics for layer_1
        tracker.update_metrics(
            epoch=0,
            phase='training',
            metrics=sample_metrics,
            phase_num=1
        )
        
        # Check if layer metrics were updated
        assert len(tracker.layer_metrics['layer_1']['accuracy']) == 1
        assert tracker.layer_metrics['layer_1']['accuracy'][0] == 0.9
