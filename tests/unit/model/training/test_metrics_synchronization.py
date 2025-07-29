#!/usr/bin/env python3
"""
Test script for metrics synchronization between validation executor and callbacks.

This script ensures that computed validation metrics are properly synchronized
with the metrics callback used in callback_only_training_example.py.
"""

import unittest
from unittest.mock import Mock, patch
import torch

from smartcash.model.training.core.validation_executor import ValidationExecutor
from smartcash.model.training.core.parallel_map_calculator import ParallelMAPCalculator
from smartcash.model.training.utils.ui_metrics_callback import create_ui_metrics_callback
from smartcash.model.training.utils.metric_color_utils import ColorScheme


class TestMetricsSynchronization(unittest.TestCase):
    """Test metrics synchronization between components."""
    
    def setUp(self):
        """Set up test components."""
        # Mock model
        self.mock_model = Mock()
        self.mock_model.eval = Mock()
        
        # Mock device
        mock_param = Mock()
        mock_param.device = torch.device('cpu')
        self.mock_model.parameters.return_value = [mock_param]
        
        # Mock config
        self.config = {
            'training': {'batch_size': 4},
            'model': {'num_classes': 7}
        }
        
        # Mock progress tracker
        self.mock_progress_tracker = Mock()
        
        # Create validation executor
        self.validation_executor = ValidationExecutor(
            self.mock_model, self.config, self.mock_progress_tracker
        )
        
        # Create metrics callback
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
        # Create mock validation data
        running_val_loss = 2.5
        num_batches = 10
        all_predictions = {}
        all_targets = {}
        
        # Mock mAP calculator
        mock_map_metrics = {
            'val_map50': 0.45,
            'val_map50_95': 0.32
        }
        
        with patch.object(self.validation_executor.map_calculator, 'compute_final_map', 
                         return_value=mock_map_metrics):
            with patch.object(self.validation_executor, '_compute_classification_metrics',
                             return_value={
                                 'val_layer_1_accuracy': 0.85,
                                 'val_layer_1_precision': 0.78,
                                 'val_layer_1_recall': 0.82,
                                 'val_layer_1_f1': 0.80
                             }):
                
                metrics = self.validation_executor._compute_final_metrics(
                    running_val_loss, num_batches, all_predictions, all_targets
                )
        
        # Verify essential metrics are present
        expected_keys = [
            'val_loss', 'val_map50', 'val_map50_95', 
            'val_precision', 'val_recall', 'val_f1', 'val_accuracy'
        ]
        
        for key in expected_keys:
            self.assertIn(key, metrics, f"Missing expected metric: {key}")
            self.assertIsInstance(metrics[key], (int, float), f"Metric {key} should be numeric")
        
        # Verify mAP metrics are properly set
        self.assertEqual(metrics['val_map50'], 0.45)
        self.assertEqual(metrics['val_map50_95'], 0.32)
        
        # Verify loss calculation
        expected_loss = running_val_loss / num_batches
        self.assertEqual(metrics['val_loss'], expected_loss)
    
    def test_metrics_callback_compatibility(self):
        """Test that validation metrics are compatible with the UI metrics callback."""
        # Simulate metrics that would come from validation
        sample_validation_metrics = {
            'val_loss': 1.234,
            'val_map50': 0.567,
            'val_map50_95': 0.389,
            'val_precision': 0.75,
            'val_recall': 0.82,
            'val_f1': 0.78,
            'val_accuracy': 0.80,
            'val_layer_1_accuracy': 0.85,
            'val_layer_1_precision': 0.78,
            'val_layer_1_recall': 0.82,
            'val_layer_1_f1': 0.80,
            'val_layer_2_accuracy': 0.72,
            'val_layer_2_precision': 0.70,
            'val_layer_2_recall': 0.75,
            'val_layer_2_f1': 0.72
        }
        
        # Test callback processing
        result = self.metrics_callback('training_phase_1', 5, sample_validation_metrics, max_epochs=10)
        
        # Verify callback processed the metrics
        self.assertIn('original_metrics', result)
        self.assertIn('colored_metrics', result)
        self.assertEqual(result['phase'], 'training_phase_1')
        self.assertEqual(result['epoch'], 5)
        
        # Verify all validation metrics are preserved
        original_metrics = result['original_metrics']
        for key, value in sample_validation_metrics.items():
            self.assertIn(key, original_metrics)
            self.assertEqual(original_metrics[key], value)
        
        # Verify colored metrics are generated for numeric metrics
        colored_metrics = result['colored_metrics']
        for key, value in sample_validation_metrics.items():
            if isinstance(value, (int, float)):
                self.assertIn(key, colored_metrics)
                self.assertIn('status', colored_metrics[key])
                self.assertIn('colors', colored_metrics[key])
        
        # Verify UI callback was triggered
        self.assertEqual(len(self.metrics_received), 1)
        received = self.metrics_received[0]
        self.assertEqual(received['phase'], 'training_phase_1')
        self.assertEqual(received['epoch'], 5)
        self.assertEqual(received['metrics'], sample_validation_metrics)
    
    def test_parallel_map_calculator_metrics_format(self):
        """Test that parallel mAP calculator returns metrics in expected format."""
        # Create a ParallelMAPCalculator instance
        map_calc = ParallelMAPCalculator(max_workers=2)
        
        # Mock the internal compute_final_map method
        expected_metrics = {
            'val_map50': 0.678,
            'val_map50_95': 0.456
        }
        
        with patch.object(map_calc.ap_calculator, 'compute_map', 
                         return_value=(0.678, 0.456, {})):
            metrics = map_calc.compute_final_map()
        
        # Verify the format matches what validation executor expects
        self.assertIn('val_map50', metrics)
        self.assertIn('val_map50_95', metrics)
        self.assertIsInstance(metrics['val_map50'], float)
        self.assertIsInstance(metrics['val_map50_95'], float)
        self.assertEqual(metrics['val_map50'], 0.678)
        self.assertEqual(metrics['val_map50_95'], 0.456)
    
    def test_phase_aware_layer_filtering(self):
        """Test that metrics callback properly filters layers based on training phase."""
        # Test Phase 1 (should only show layer_1)
        phase1_metrics = {
            'val_loss': 1.0,
            'val_layer_1_accuracy': 0.85,
            'val_layer_2_accuracy': 0.72,
            'val_layer_3_accuracy': 0.68
        }
        
        result = self.metrics_callback('training_phase_1', 3, phase1_metrics)
        summary = self.metrics_callback.get_metric_summary_for_ui()
        
        # In phase 1, should prioritize layer_1 and filter zeros
        phase_info = summary['phase_info']
        self.assertEqual(phase_info['active_layers'], ['layer_1'])
        self.assertTrue(phase_info['filter_zeros'])
        
        # Test Phase 2 (should show all layers)
        phase2_metrics = phase1_metrics.copy()
        result = self.metrics_callback('training_phase_2', 3, phase2_metrics)
        summary = self.metrics_callback.get_metric_summary_for_ui()
        
        phase_info = summary['phase_info']
        self.assertEqual(phase_info['active_layers'], ['layer_1', 'layer_2', 'layer_3'])
        self.assertFalse(phase_info['filter_zeros'])
    
    def test_callback_only_training_example_compatibility(self):
        """Test compatibility with the callback_only_training_example.py structure."""
        # Simulate the exact callback structure used in the example
        collected_metrics = []
        
        def example_metrics_callback(phase: str, epoch: int, metrics: dict, colored_metrics: dict):
            """Simulate the metrics callback from the example."""
            collected_metrics.append({
                'phase': phase,
                'epoch': epoch,
                'metrics': metrics,
                'colored_metrics': colored_metrics
            })
            
            # Verify essential validation metrics are present
            if 'val_' in str(metrics.keys()):
                expected_val_metrics = ['val_loss', 'val_map50', 'val_accuracy']
                for metric in expected_val_metrics:
                    if metric in metrics:
                        self.assertIsInstance(metrics[metric], (int, float))
        
        # Use the callback
        ui_callback = create_ui_metrics_callback(verbose=False)
        ui_callback.set_ui_callback(example_metrics_callback)
        
        # Test with realistic validation metrics
        validation_metrics = {
            'train_loss': 0.95,
            'val_loss': 1.234,
            'val_map50': 0.456,
            'val_map50_95': 0.321,
            'val_accuracy': 0.78,
            'val_precision': 0.75,
            'val_recall': 0.82,
            'layer_1_accuracy': 0.85
        }
        
        # Process metrics
        result = ui_callback('training_phase_1', 5, validation_metrics, max_epochs=10)
        
        # Verify callback was triggered and received correct data
        self.assertEqual(len(collected_metrics), 1)
        callback_data = collected_metrics[0]
        
        self.assertEqual(callback_data['phase'], 'training_phase_1')
        self.assertEqual(callback_data['epoch'], 5)
        
        # Verify all validation metrics were passed through
        received_metrics = callback_data['metrics']
        for key, value in validation_metrics.items():
            self.assertEqual(received_metrics[key], value)


def run_tests():
    """Run all synchronization tests."""
    unittest.main(verbosity=2)


if __name__ == '__main__':
    run_tests()