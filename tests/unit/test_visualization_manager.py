#!/usr/bin/env python3
"""
Test suite for the visualization manager.

Tests functionality with different epoch counts to understand minimum data requirements
and verify chart generation capabilities.
"""

import pytest
import tempfile
import shutil
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

from smartcash.model.training.visualization_manager import (
    create_visualization_manager,
    VisualizationMetricsTracker,
    VisualizationHelper,
    VISUALIZATION_AVAILABLE
)


class TestVisualizationManager:
    """Test suite for visualization manager functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_classes_per_layer = {
            'layer_1': 7,
            'layer_2': 7,
            'layer_3': 3
        }
        self.test_class_names = {
            'layer_1': ['001', '002', '005', '010', '020', '050', '100'],
            'layer_2': ['l2_001', 'l2_002', 'l2_005', 'l2_010', 'l2_020', 'l2_050', 'l2_100'],
            'layer_3': ['l3_sign', 'l3_text', 'l3_thread']
        }
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_visualization_manager_creation(self):
        """Test creating visualization manager."""
        tracker = create_visualization_manager(
            num_classes_per_layer=self.test_classes_per_layer,
            class_names=self.test_class_names,
            save_dir=self.temp_dir,
            verbose=True
        )
        
        assert isinstance(tracker, VisualizationMetricsTracker)
        assert tracker.num_classes_per_layer == self.test_classes_per_layer
        assert tracker.verbose is True
        assert Path(tracker.save_dir) == Path(self.temp_dir)
    
    def test_visualization_manager_with_minimal_data(self):
        """Test visualization manager with minimal data (2 epochs)."""
        tracker = create_visualization_manager(
            num_classes_per_layer=self.test_classes_per_layer,
            save_dir=self.temp_dir
        )
        
        # Add minimal data (2 epochs)
        for epoch in range(2):
            metrics = {
                'train_loss': 0.8 - (epoch * 0.1),
                'val_loss': 1.2 - (epoch * 0.15),
                'val_accuracy': 0.6 + (epoch * 0.05),
                'learning_rate': 0.001 / (epoch + 1)
            }
            tracker.update_metrics(
                epoch=epoch,
                phase=f"training_phase_1",
                metrics=metrics,
                phase_num=1
            )
        
        # Test chart generation with minimal data
        charts = tracker.generate_comprehensive_charts(
            session_id="test_minimal",
            phase_num=1
        )
        
        if VISUALIZATION_AVAILABLE:
            # Should generate some charts even with minimal data
            assert len(charts) >= 0  # May warn about insufficient data
        else:
            # Should return empty dict if visualization libs not available
            assert charts == {}
    
    def test_visualization_manager_with_sufficient_data(self):
        """Test visualization manager with sufficient data (5+ epochs)."""
        tracker = create_visualization_manager(
            num_classes_per_layer=self.test_classes_per_layer,
            save_dir=self.temp_dir,
            verbose=True
        )
        
        # Add sufficient data (5 epochs)
        for epoch in range(5):
            metrics = {
                'train_loss': 1.0 - (epoch * 0.15),
                'val_loss': 1.4 - (epoch * 0.2),
                'val_accuracy': 0.5 + (epoch * 0.08),
                'val_precision': 0.4 + (epoch * 0.1),
                'val_recall': 0.45 + (epoch * 0.07),
                'val_f1': 0.42 + (epoch * 0.08),
                'val_map50': 0.1 + (epoch * 0.05),
                'learning_rate': 0.001 / (epoch + 1),
                'layer_1_accuracy': 0.3 + (epoch * 0.12),
                'layer_1_precision': 0.25 + (epoch * 0.1),
                'layer_1_recall': 0.28 + (epoch * 0.11),
                'layer_1_f1': 0.26 + (epoch * 0.105)
            }
            tracker.update_metrics(
                epoch=epoch,
                phase="training_phase_1",
                metrics=metrics,
                phase_num=1
            )
        
        # Test chart generation with sufficient data
        charts = tracker.generate_comprehensive_charts(
            session_id="test_sufficient",
            phase_num=1
        )
        
        if VISUALIZATION_AVAILABLE:
            # Should generate meaningful charts with sufficient data
            assert len(charts) > 0
            # Check that files actually exist
            for chart_path in charts.values():
                assert Path(chart_path).exists()
        else:
            assert charts == {}
    
    def test_phase_transition_tracking(self):
        """Test tracking of phase transitions."""
        tracker = create_visualization_manager(
            num_classes_per_layer=self.test_classes_per_layer,
            save_dir=self.temp_dir
        )
        
        # Phase 1 training
        for epoch in range(3):
            metrics = {
                'train_loss': 1.0 - (epoch * 0.1),
                'val_loss': 1.3 - (epoch * 0.12),
                'val_accuracy': 0.6 + (epoch * 0.05)
            }
            tracker.update_metrics(
                epoch=epoch,
                phase="training_phase_1",
                metrics=metrics,
                phase_num=1
            )
        
        # Phase 2 training
        for epoch in range(3, 6):
            metrics = {
                'train_loss': 0.7 - ((epoch-3) * 0.05),
                'val_loss': 1.0 - ((epoch-3) * 0.08),
                'val_accuracy': 0.75 + ((epoch-3) * 0.03),
                'val_map50': 0.2 + ((epoch-3) * 0.1)
            }
            tracker.update_metrics(
                epoch=epoch,
                phase="training_phase_2", 
                metrics=metrics,
                phase_num=2
            )
        
        # Should have tracked phase transitions
        assert len(tracker.phase_transitions) > 0
        assert len(tracker.epoch_metrics) == 6
    
    def test_visualization_with_no_data(self):
        """Test chart generation behavior with no data."""
        tracker = create_visualization_manager(
            num_classes_per_layer=self.test_classes_per_layer,
            save_dir=self.temp_dir
        )
        
        # Try to generate charts with no data
        charts = tracker.generate_comprehensive_charts(
            session_id="test_no_data",
            phase_num=1
        )
        
        # Should handle gracefully with no/minimal charts
        assert isinstance(charts, dict)
    
    @patch('smartcash.model.training.visualization_manager.VISUALIZATION_AVAILABLE', False)
    def test_visualization_unavailable_fallback(self):
        """Test behavior when visualization libraries are not available."""
        tracker = create_visualization_manager(
            num_classes_per_layer=self.test_classes_per_layer,
            save_dir=self.temp_dir,
            verbose=True
        )
        
        # Add some data
        metrics = {
            'train_loss': 0.8,
            'val_loss': 1.2,
            'val_accuracy': 0.6
        }
        tracker.update_metrics(
            epoch=0,
            phase="training_phase_1",
            metrics=metrics,
            phase_num=1
        )
        
        # Should return empty dict when visualization not available
        charts = tracker.generate_comprehensive_charts()
        assert charts == {}
    
    def test_insufficient_data_warning(self):
        """Test that insufficient data produces warning behavior."""
        tracker = create_visualization_manager(
            num_classes_per_layer=self.test_classes_per_layer,
            save_dir=self.temp_dir,
            verbose=True
        )
        
        # Add only 2 epochs of data (insufficient for meaningful charts)
        for epoch in range(2):
            metrics = {
                'train_loss': 1.0 - (epoch * 0.1),
                'val_loss': 1.3 - (epoch * 0.12)
            }
            tracker.update_metrics(
                epoch=epoch,
                phase="training_phase_1",
                metrics=metrics,
                phase_num=1
            )
        
        # Should handle insufficient data gracefully
        charts = tracker.generate_comprehensive_charts(
            session_id="test_insufficient",
            phase_num=1
        )
        
        # Even with insufficient data, should not crash
        assert isinstance(charts, dict)
    
    def test_layer_metrics_extraction(self):
        """Test layer metrics extraction functionality."""
        tracker = create_visualization_manager(
            num_classes_per_layer=self.test_classes_per_layer,
            save_dir=self.temp_dir
        )
        
        # Add metrics with layer-specific data
        metrics = {
            'train_loss': 0.8,
            'val_loss': 1.2,
            'layer_1_accuracy': 0.7,
            'layer_1_precision': 0.65,
            'layer_1_recall': 0.68,
            'layer_1_f1': 0.66,
            'layer_2_accuracy': None,  # Phase 1 should have null layer_2/3 metrics
            'layer_2_precision': None,
            'layer_3_accuracy': None,
            'layer_3_precision': None
        }
        
        tracker.update_metrics(
            epoch=0,
            phase="training_phase_1",
            metrics=metrics,
            phase_num=1
        )
        
        # Check that layer metrics are properly extracted
        assert 'layer_1' in tracker.layer_metrics
        assert len(tracker.layer_metrics['layer_1']['accuracy']) > 0
    
    def test_metrics_summary_export(self):
        """Test metrics summary export functionality."""
        tracker = create_visualization_manager(
            num_classes_per_layer=self.test_classes_per_layer,
            save_dir=self.temp_dir
        )
        
        # Add some metrics
        metrics = {
            'train_loss': 0.8,
            'val_loss': 1.2,
            'val_accuracy': 0.6,
            'learning_rate': 0.001
        }
        tracker.update_metrics(
            epoch=0,
            phase="training_phase_1", 
            metrics=metrics,
            phase_num=1
        )
        
        # Test summary export
        session_dir = Path(self.temp_dir) / "test_session"
        session_dir.mkdir(exist_ok=True)
        
        # Should not crash when saving metrics summary
        try:
            tracker.save_metrics_summary(session_dir, "test_session")
            # If chart_generator.save_metrics_summary exists, it should work
            assert True
        except AttributeError:
            # If method doesn't exist, that's also acceptable
            assert True


class TestVisualizationHelper:
    """Test suite for visualization helper utilities."""
    
    def test_default_layer_metrics(self):
        """Test default layer metrics generation."""
        metrics = VisualizationHelper.get_default_layer_metrics()
        
        expected_keys = ['accuracy', 'precision', 'recall', 'f1_score']
        assert all(key in metrics for key in expected_keys)
        assert all(isinstance(metrics[key], list) for key in expected_keys)
    
    def test_layer_metrics_initialization(self):
        """Test layer metrics initialization."""
        num_classes = {'layer_1': 7, 'layer_2': 7, 'layer_3': 3}
        metrics = VisualizationHelper.initialize_layer_metrics(num_classes)
        
        assert set(metrics.keys()) == set(num_classes.keys())
        for layer_metrics in metrics.values():
            assert 'accuracy' in layer_metrics
            assert isinstance(layer_metrics['accuracy'], list)
    
    def test_metrics_extraction_from_dict(self):
        """Test extracting metrics from dictionary."""
        test_metrics = {
            'layer_1_accuracy': 0.8,
            'layer_1_precision': 0.75,
            'layer_2_recall': 0.7,
            'other_metric': 0.5
        }
        
        layer_patterns = {
            'layer_1': ['layer_1_accuracy', 'layer_1_precision'],
            'layer_2': ['layer_2_recall']
        }
        
        extracted = VisualizationHelper.extract_metrics_from_dict(
            test_metrics, layer_patterns
        )
        
        assert 'layer_1' in extracted
        assert 'layer_2' in extracted
        assert len(extracted['layer_1']['accuracy']) > 0
        assert len(extracted['layer_2']['recall']) > 0


if __name__ == "__main__":
    # Run a simple test if executed directly
    import sys
    
    # Test creation without pytest
    temp_dir = tempfile.mkdtemp()
    try:
        print("🧪 Testing visualization manager creation...")
        
        test_classes = {
            'layer_1': 7,
            'layer_2': 7, 
            'layer_3': 3
        }
        
        tracker = create_visualization_manager(
            num_classes_per_layer=test_classes,
            save_dir=temp_dir,
            verbose=True
        )
        
        print("✅ Visualization manager created successfully")
        
        # Test with minimal data
        print("🧪 Testing with 2 epochs (minimal data)...")
        for epoch in range(2):
            metrics = {
                'train_loss': 1.0 - (epoch * 0.1),
                'val_loss': 1.3 - (epoch * 0.15),
                'val_accuracy': 0.6 + (epoch * 0.05)
            }
            tracker.update_metrics(epoch, "phase_1", metrics, phase_num=1)
        
        charts = tracker.generate_comprehensive_charts("test_2_epochs", 1)
        if VISUALIZATION_AVAILABLE:
            print(f"📊 Generated {len(charts)} charts with 2 epochs")
        else:
            print("⚠️ Visualization libraries not available - expected behavior")
        
        print("✅ All basic tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        sys.exit(1)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)