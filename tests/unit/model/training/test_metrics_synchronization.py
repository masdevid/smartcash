#!/usr/bin/env python3
"""
Test script for metrics synchronization between validation executor and callbacks.
"""

import unittest
from unittest.mock import Mock, patch
import torch

from smartcash.model.training.core.validation_executor import ValidationExecutor
from smartcash.model.training.utils.ui_metrics_callback import create_ui_metrics_callback
from smartcash.model.training.utils.metric_color_utils import ColorScheme


class TestMetricsSynchronization(unittest.TestCase):
    """Test metrics synchronization between components."""

    def setUp(self):
        """Set up test components."""
        self.mock_model = Mock()
        self.mock_model.eval = Mock()
        
        mock_param = Mock()
        mock_param.device = torch.device('cpu')
        self.mock_model.parameters.return_value = [mock_param]
        
        self.config = {
            'training': {'batch_size': 4},
            'model': {'num_classes': 7}
        }
        
        self.mock_progress_tracker = Mock()
        
        self.validation_executor = ValidationExecutor(
            self.mock_model, self.config, self.mock_progress_tracker
        )
        
        self.metrics_received = []
        def capture_metrics(phase, epoch, metrics, colored_metrics):
            self.metrics_received.append({
                'phase': phase,
                'epoch': epoch,
                'metrics': metrics.copy(),
                'colored_metrics': colored_metrics
            })
        
        self.metrics_callback = create_ui_metrics_callback(
            verbose=False,
            console_scheme=ColorScheme.EMOJI
        )
        self.metrics_callback.ui_callback = capture_metrics

    def test_validation_metrics_structure(self):
        """Test that validation metrics have the expected structure."""
        running_val_loss = 2.5
        num_batches = 10
        all_predictions = {}
        all_targets = {}
        
        mock_map_metrics = {
            'map50': 0.45,
            'map50_95': 0.32,
            'precision': 0.88,
            'recall': 0.89,
            'f1': 0.885
        }

        # Patch the dependencies of compute_final_metrics
        with patch.object(self.validation_executor.metrics_computer.map_calculator, 'compute_map', return_value=mock_map_metrics):
            with patch.object(self.validation_executor.metrics_computer, '_compute_classification_metrics', return_value={
                'layer_1_accuracy': 0.85, 'layer_1_precision': 0.78,
                'layer_1_recall': 0.82, 'layer_1_f1': 0.80
            }):
                metrics = self.validation_executor.metrics_computer.compute_final_metrics(
                    running_val_loss, num_batches, all_predictions, all_targets, phase_num=1
                )
        
        expected_keys = [
            'val_loss', 'val_map50', 'val_map50_95', 
            'val_precision', 'val_recall', 'val_f1', 'val_accuracy'
        ]
        
        for key in expected_keys:
            self.assertIn(key, metrics, f"Missing expected metric: {key}")
            self.assertIsInstance(metrics[key], (int, float), f"Metric {key} should be numeric")
        
        self.assertEqual(metrics['val_map50'], 0.45)
        self.assertEqual(metrics['val_map50_95'], 0.32)
        self.assertEqual(metrics['val_loss'], running_val_loss / num_batches)
        self.assertEqual(metrics['val_accuracy'], 0.85) # From layer_1_accuracy in phase 1

    def test_metrics_callback_compatibility(self):
        """Test that validation metrics are compatible with the UI metrics callback."""
        sample_metrics = {
            'val_loss': 1.234, 'val_accuracy': 0.80, 'val_layer_1_accuracy': 0.85
        }
        
        result = self.metrics_callback('training_phase_1', 5, sample_metrics, max_epochs=10)
        
        self.assertIn('original_metrics', result)
        self.assertEqual(result['original_metrics']['val_accuracy'], 0.80)

    def test_phase_aware_layer_filtering(self):
        """Test that metrics callback properly filters layers based on training phase."""
        # Phase 1: only layer_1 is shown
        phase1_metrics = {'val_layer_1_accuracy': 0.85, 'val_layer_2_accuracy': 0.72}
        self.metrics_callback('training_phase_1', 3, phase1_metrics)
        summary = self.metrics_callback.get_metric_summary_for_ui()
        self.assertEqual(summary['phase_info']['active_layers'], ['layer_1'])
        self.assertTrue(summary['phase_info']['filter_zeros'])
        
        # Phase 2: active layers are shown
        phase2_metrics = {
            'val_layer_1_accuracy': 0.85, 
            'val_layer_2_accuracy': 0.72,
            'val_layer_3_accuracy': 0.9 # Active layer
        }
        self.metrics_callback('training_phase_2', 3, phase2_metrics)
        summary = self.metrics_callback.get_metric_summary_for_ui()
        self.assertEqual(summary['phase_info']['active_layers'], ['layer_1', 'layer_2', 'layer_3'])
        self.assertFalse(summary['phase_info']['filter_zeros'])

        # Phase 2: only layer 1 and 3 are active
        phase2_metrics = {
            'val_layer_1_accuracy': 0.85, 
            'val_layer_2_accuracy': 0.0,
            'val_layer_3_accuracy': 0.9 # Active layer
        }
        self.metrics_callback('training_phase_2', 3, phase2_metrics)
        summary = self.metrics_callback.get_metric_summary_for_ui()
        self.assertEqual(summary['phase_info']['active_layers'], ['layer_1', 'layer_3'])
        self.assertFalse(summary['phase_info']['filter_zeros'])

    def test_callback_only_training_example_compatibility(self):
        """Test compatibility with the callback_only_training_example.py structure."""
        collected = []
        def example_callback(phase, epoch, metrics, colored_metrics):
            collected.append(metrics)
        
        ui_callback = create_ui_metrics_callback(verbose=False)
        ui_callback.set_ui_callback(example_callback)
        
        metrics = {'val_loss': 1.23, 'val_accuracy': 0.78}
        ui_callback('training_phase_1', 5, metrics, max_epochs=10)
        
        self.assertEqual(len(collected), 1)
        self.assertEqual(collected[0]['val_accuracy'], 0.78)

if __name__ == '__main__':
    unittest.main(verbosity=2)
