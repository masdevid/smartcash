#!/usr/bin/env python3
"""
Test for the metrics fix in Phase 1 training.
"""

import unittest
from smartcash.model.training.utils.research_metrics import ResearchMetricsManager


class TestMetricsFix(unittest.TestCase):
    """Test the metrics standardization fix."""

    def setUp(self):
        """Set up test fixtures."""
        self.metrics_manager = ResearchMetricsManager()

    def test_phase_1_metrics_standardization(self):
        """Test that Phase 1 metrics are correctly standardized."""
        # Simulate raw metrics from Phase 1
        raw_metrics = {
            'loss': 0.5,
            'layer_1_accuracy': 0.0286,
            'layer_1_precision': 0.03,
            'layer_1_recall': 0.025,
            'layer_1_f1': 0.027,
            'map50': 0.4752,
            'map50_95': 0.3,
        }

        # Standardize metrics for Phase 1 validation
        standardized = self.metrics_manager.standardize_metric_names(
            raw_metrics, phase_num=1, is_validation=True
        )

        # Check that val_* metrics come from layer_1_* in Phase 1
        self.assertEqual(standardized['val_accuracy'], 0.0286)
        self.assertEqual(standardized['val_precision'], 0.03)
        self.assertEqual(standardized['val_recall'], 0.025)
        self.assertEqual(standardized['val_f1'], 0.027)
        self.assertEqual(standardized['val_loss'], 0.5)
        self.assertEqual(standardized['val_map50'], 0.4752)

    def test_phase_2_metrics_standardization(self):
        """Test that Phase 2 metrics are correctly standardized."""
        # Simulate raw metrics from Phase 2
        raw_metrics = {
            'loss': 0.3,
            'accuracy': 0.85,
            'precision': 0.82,
            'recall': 0.88,
            'f1': 0.85,
            'layer_1_accuracy': 0.9,
            'layer_1_precision': 0.88,
            'layer_1_recall': 0.92,
            'layer_1_f1': 0.9,
            'map50': 0.87,
            'map50_95': 0.7,
        }

        # Standardize metrics for Phase 2 validation
        standardized = self.metrics_manager.standardize_metric_names(
            raw_metrics, phase_num=2, is_validation=True
        )

        # Check that val_* metrics come from top-level metrics in Phase 2
        self.assertEqual(standardized['val_accuracy'], 0.85)
        self.assertEqual(standardized['val_precision'], 0.82)
        self.assertEqual(standardized['val_recall'], 0.88)
        self.assertEqual(standardized['val_f1'], 0.85)
        self.assertEqual(standardized['val_loss'], 0.3)
        self.assertEqual(standardized['val_map50'], 0.87)


if __name__ == '__main__':
    unittest.main()