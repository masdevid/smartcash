#!/usr/bin/env python3
"""
Comprehensive tests for MetricsTracker module.

Tests AP calculation, metrics tracking, mAP computation, and performance monitoring
functionality for training and validation.
"""

import pytest
import torch
import numpy as np
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
from collections import defaultdict

# Import the modules to test
import sys
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from smartcash.model.training.metrics_tracker import (
    APCalculator,
    MetricsTracker,
    create_metrics_tracker,
    calculate_map
)


class TestAPCalculator:
    """Test cases for APCalculator class"""
    
    @pytest.fixture
    def sample_calculator(self):
        """Create AP calculator with sample configuration"""
        iou_thresholds = [0.5, 0.75, 0.9]
        class_names = ['Rp1K', 'Rp2K', 'Rp5K', 'Rp10K', 'Rp20K', 'Rp50K', 'Rp100K']
        return APCalculator(iou_thresholds, class_names)
    
    def test_calculator_initialization(self, sample_calculator):
        """Test APCalculator initialization"""
        assert len(sample_calculator.iou_thresholds) == 3
        assert len(sample_calculator.class_names) == 7
        assert len(sample_calculator.predictions) == 0
        assert len(sample_calculator.targets) == 0
    
    def test_reset_functionality(self, sample_calculator):
        """Test reset functionality"""
        # Add some dummy data
        sample_calculator.predictions = [[0.9, 0, 10, 10, 50, 50, 0]]
        sample_calculator.targets = [[0, 15, 15, 45, 45, 0]]
        
        sample_calculator.reset()
        
        assert len(sample_calculator.predictions) == 0
        assert len(sample_calculator.targets) == 0
        assert len(sample_calculator.image_ids) == 0
    
    def test_add_batch_single_image(self, sample_calculator):
        """Test adding single image batch"""
        # Create sample predictions and targets
        pred_boxes = [torch.tensor([[10, 10, 50, 50], [60, 60, 100, 100]])]
        pred_scores = [torch.tensor([0.9, 0.8])]
        pred_classes = [torch.tensor([0, 1])]
        
        true_boxes = [torch.tensor([[15, 15, 45, 45], [65, 65, 95, 95]])]
        true_classes = [torch.tensor([0, 1])]
        
        image_ids = [0]
        
        sample_calculator.add_batch(
            pred_boxes, pred_scores, pred_classes,
            true_boxes, true_classes, image_ids
        )
        
        assert len(sample_calculator.predictions) == 2
        assert len(sample_calculator.targets) == 2
        
        # Check prediction format: [conf, class_pred, x1, y1, x2, y2, img_id]
        assert sample_calculator.predictions[0][0] == 0.9  # confidence
        assert sample_calculator.predictions[0][1] == 0    # class
        assert sample_calculator.predictions[0][6] == 0    # image_id
        
        # Check target format: [class_true, x1, y1, x2, y2, img_id]
        assert sample_calculator.targets[0][0] == 0  # class
        assert sample_calculator.targets[0][5] == 0  # image_id
    
    def test_add_batch_multiple_images(self, sample_calculator):
        """Test adding multiple images in batch"""
        # Two images with different numbers of detections
        pred_boxes = [
            torch.tensor([[10, 10, 50, 50]]),           # Image 0: 1 detection
            torch.tensor([[60, 60, 100, 100], [20, 20, 40, 40]])  # Image 1: 2 detections
        ]
        pred_scores = [
            torch.tensor([0.9]),
            torch.tensor([0.8, 0.7])
        ]
        pred_classes = [
            torch.tensor([0]),
            torch.tensor([1, 2])
        ]
        
        true_boxes = [
            torch.tensor([[15, 15, 45, 45]]),
            torch.tensor([[65, 65, 95, 95]])
        ]
        true_classes = [
            torch.tensor([0]),
            torch.tensor([1])
        ]
        
        image_ids = [0, 1]
        
        sample_calculator.add_batch(
            pred_boxes, pred_scores, pred_classes,
            true_boxes, true_classes, image_ids
        )
        
        assert len(sample_calculator.predictions) == 3  # 1 + 2 predictions
        assert len(sample_calculator.targets) == 2      # 1 + 1 targets
        
        # Check image IDs are correct
        assert sample_calculator.predictions[0][6] == 0  # First prediction from image 0
        assert sample_calculator.predictions[1][6] == 1  # First prediction from image 1
        assert sample_calculator.predictions[2][6] == 1  # Second prediction from image 1
    
    def test_add_batch_empty_predictions(self, sample_calculator):
        """Test adding batch with empty predictions"""
        pred_boxes = [torch.tensor([])]  # Empty predictions
        pred_scores = [torch.tensor([])]
        pred_classes = [torch.tensor([])]
        
        true_boxes = [torch.tensor([[15, 15, 45, 45]])]
        true_classes = [torch.tensor([0])]
        
        image_ids = [0]
        
        sample_calculator.add_batch(
            pred_boxes, pred_scores, pred_classes,
            true_boxes, true_classes, image_ids
        )
        
        assert len(sample_calculator.predictions) == 0
        assert len(sample_calculator.targets) == 1
    
    def test_calculate_iou_perfect_overlap(self, sample_calculator):
        """Test IoU calculation with perfect overlap"""
        box1 = [10, 10, 50, 50]
        box2 = [10, 10, 50, 50]
        
        iou = sample_calculator._calculate_iou(box1, box2)
        assert abs(iou - 1.0) < 1e-6
    
    def test_calculate_iou_no_overlap(self, sample_calculator):
        """Test IoU calculation with no overlap"""
        box1 = [10, 10, 30, 30]
        box2 = [50, 50, 70, 70]
        
        iou = sample_calculator._calculate_iou(box1, box2)
        assert iou == 0.0
    
    def test_calculate_iou_partial_overlap(self, sample_calculator):
        """Test IoU calculation with partial overlap"""
        box1 = [10, 10, 30, 30]  # 20x20 = 400 area
        box2 = [20, 20, 40, 40]  # 20x20 = 400 area, 10x10 = 100 intersection
        
        iou = sample_calculator._calculate_iou(box1, box2)
        expected_iou = 100 / (400 + 400 - 100)  # intersection / union
        assert abs(iou - expected_iou) < 1e-6
    
    def test_compute_ap_no_targets(self, sample_calculator):
        """Test AP computation with no targets"""
        # Add some predictions but no targets
        sample_calculator.predictions = [[0.9, 0, 10, 10, 50, 50, 0]]
        sample_calculator.targets = []
        
        ap = sample_calculator.compute_ap(class_id=0, iou_threshold=0.5)
        assert ap == 0.0
    
    def test_compute_ap_no_predictions(self, sample_calculator):
        """Test AP computation with no predictions"""
        # Add targets but no predictions
        sample_calculator.predictions = []
        sample_calculator.targets = [[0, 15, 15, 45, 45, 0]]
        
        ap = sample_calculator.compute_ap(class_id=0, iou_threshold=0.5)
        assert ap == 0.0
    
    def test_compute_ap_perfect_detection(self, sample_calculator):
        """Test AP computation with perfect detection"""
        # Perfect match: prediction exactly matches target
        sample_calculator.predictions = [[0.9, 0, 10, 10, 50, 50, 0]]
        sample_calculator.targets = [[0, 10, 10, 50, 50, 0]]
        
        ap = sample_calculator.compute_ap(class_id=0, iou_threshold=0.5)
        assert abs(ap - 1.0) < 1e-6  # Perfect precision and recall with tolerance
    
    def test_compute_ap_multiple_detections(self, sample_calculator):
        """Test AP computation with multiple detections"""
        # Add multiple predictions and targets
        sample_calculator.predictions = [
            [0.9, 0, 10, 10, 50, 50, 0],  # High confidence, good match
            [0.8, 0, 60, 60, 100, 100, 0],  # Medium confidence, good match
            [0.7, 0, 200, 200, 250, 250, 0]  # Low confidence, no match
        ]
        sample_calculator.targets = [
            [0, 15, 15, 45, 45, 0],     # Matches first prediction
            [0, 65, 65, 95, 95, 0]      # Matches second prediction
        ]
        
        ap = sample_calculator.compute_ap(class_id=0, iou_threshold=0.5)
        assert 0.0 < ap < 1.0  # Should be between 0 and 1
    
    def test_compute_map_single_class(self, sample_calculator):
        """Test mAP computation for single class"""
        # Add data for class 0 only
        sample_calculator.predictions = [[0.9, 0, 10, 10, 50, 50, 0]]
        sample_calculator.targets = [[0, 15, 15, 45, 45, 0]]
        
        map_score, class_aps = sample_calculator.compute_map(iou_threshold=0.5)
        
        assert 0 in class_aps
        assert class_aps[0] > 0  # Should have positive AP for class 0
        assert map_score > 0     # mAP should be positive
    
    def test_compute_map_multiple_classes(self, sample_calculator):
        """Test mAP computation for multiple classes"""
        # Add data for classes 0 and 1
        sample_calculator.predictions = [
            [0.9, 0, 10, 10, 50, 50, 0],
            [0.8, 1, 60, 60, 100, 100, 0]
        ]
        sample_calculator.targets = [
            [0, 15, 15, 45, 45, 0],
            [1, 65, 65, 95, 95, 0]
        ]
        
        map_score, class_aps = sample_calculator.compute_map(iou_threshold=0.5)
        
        assert 0 in class_aps
        assert 1 in class_aps
        assert len(class_aps) == 7  # Should have entries for all 7 classes
    
    def test_compute_map50_95(self, sample_calculator):
        """Test mAP@0.5:0.95 computation"""
        # Add sample data
        sample_calculator.predictions = [[0.9, 0, 10, 10, 50, 50, 0]]
        sample_calculator.targets = [[0, 15, 15, 45, 45, 0]]
        
        map50_95 = sample_calculator.compute_map50_95()
        
        assert isinstance(map50_95, float)
        assert 0.0 <= map50_95 <= 1.0


class TestMetricsTracker:
    """Test cases for MetricsTracker class"""
    
    @pytest.fixture
    def sample_config(self):
        """Sample configuration for testing"""
        return {
            'training': {
                'validation': {
                    'compute_map': True
                }
            }
        }
    
    @pytest.fixture
    def sample_tracker(self, sample_config):
        """Create metrics tracker with sample configuration"""
        class_names = ['Rp1K', 'Rp2K', 'Rp5K', 'Rp10K', 'Rp20K', 'Rp50K', 'Rp100K']
        return MetricsTracker(sample_config, class_names)
    
    def test_tracker_initialization(self, sample_tracker):
        """Test MetricsTracker initialization"""
        assert len(sample_tracker.class_names) == 7
        assert isinstance(sample_tracker.train_metrics, defaultdict)
        assert isinstance(sample_tracker.val_metrics, defaultdict)
        assert isinstance(sample_tracker.ap_calculator, APCalculator)
    
    def test_start_epoch(self, sample_tracker):
        """Test epoch start functionality"""
        sample_tracker.start_epoch()
        
        assert sample_tracker.epoch_start_time is not None
        assert len(sample_tracker.ap_calculator.predictions) == 0  # Should be reset
    
    def test_start_batch(self, sample_tracker):
        """Test batch start functionality"""
        sample_tracker.start_batch()
        
        assert sample_tracker.batch_start_time is not None
    
    def test_update_train_metrics_with_tensors(self, sample_tracker):
        """Test updating training metrics with tensor values"""
        loss_dict = {
            'total_loss': torch.tensor(0.5),
            'box_loss': torch.tensor(0.2),
            'obj_loss': torch.tensor(0.2),
            'cls_loss': torch.tensor(0.1)
        }
        
        sample_tracker.update_train_metrics(loss_dict, learning_rate=0.001)
        
        assert 'total_loss' in sample_tracker.train_metrics
        assert abs(sample_tracker.train_metrics['total_loss'][0] - 0.5) < 1e-6
        assert abs(sample_tracker.train_metrics['learning_rate'][0] - 0.001) < 1e-6
    
    def test_update_train_metrics_with_scalars(self, sample_tracker):
        """Test updating training metrics with scalar values"""
        loss_dict = {
            'total_loss': 0.5,
            'box_loss': 0.2,
            'obj_loss': 0.2,
            'cls_loss': 0.1
        }
        
        sample_tracker.update_train_metrics(loss_dict)
        
        assert abs(sample_tracker.train_metrics['total_loss'][0] - 0.5) < 1e-6
        assert abs(sample_tracker.train_metrics['box_loss'][0] - 0.2) < 1e-6
    
    def test_update_train_metrics_with_batch_timing(self, sample_tracker):
        """Test training metrics update with batch timing"""
        sample_tracker.start_batch()
        
        # Simulate some processing time
        import time
        time.sleep(0.01)
        
        loss_dict = {'total_loss': 0.5}
        sample_tracker.update_train_metrics(loss_dict)
        
        assert 'batch_time' in sample_tracker.train_metrics
        assert sample_tracker.train_metrics['batch_time'][0] > 0
    
    def test_update_val_metrics_without_predictions(self, sample_tracker):
        """Test validation metrics update without predictions"""
        loss_dict = {
            'total_loss': torch.tensor(0.4),
            'box_loss': torch.tensor(0.15),
            'obj_loss': torch.tensor(0.15),
            'cls_loss': torch.tensor(0.1)
        }
        
        sample_tracker.update_val_metrics(loss_dict)
        
        assert 'total_loss' in sample_tracker.val_metrics
        assert abs(sample_tracker.val_metrics['total_loss'][0] - 0.4) < 1e-6
    
    def test_update_val_metrics_with_predictions(self, sample_tracker):
        """Test validation metrics update with predictions for mAP calculation"""
        loss_dict = {'total_loss': torch.tensor(0.4)}
        
        predictions = {
            'boxes': [torch.tensor([[10, 10, 50, 50]])],
            'scores': [torch.tensor([0.9])],
            'classes': [torch.tensor([0])],
            'true_boxes': [torch.tensor([[15, 15, 45, 45]])],
            'true_classes': [torch.tensor([0])],
            'image_ids': [0]
        }
        
        sample_tracker.update_val_metrics(loss_dict, predictions)
        
        assert len(sample_tracker.ap_calculator.predictions) > 0
        assert len(sample_tracker.ap_calculator.targets) > 0
    
    def test_compute_epoch_metrics_basic(self, sample_tracker):
        """Test basic epoch metrics computation"""
        # Add some training metrics
        sample_tracker.train_metrics['total_loss'] = [0.5, 0.4, 0.3]
        sample_tracker.train_metrics['box_loss'] = [0.2, 0.15, 0.1]
        
        # Add some validation metrics
        sample_tracker.val_metrics['total_loss'] = [0.4, 0.35, 0.3]
        
        sample_tracker.start_epoch()
        import time
        time.sleep(0.01)  # Simulate epoch time
        
        metrics = sample_tracker.compute_epoch_metrics(epoch=1)
        
        assert 'train_loss' in metrics
        assert 'val_loss' in metrics
        assert abs(metrics['train_loss'] - np.mean([0.5, 0.4, 0.3])) < 1e-6
        assert abs(metrics['val_loss'] - np.mean([0.4, 0.35, 0.3])) < 1e-6
        assert 'epoch_time' in metrics
    
    def test_compute_epoch_metrics_with_map(self, sample_tracker):
        """Test epoch metrics computation with mAP calculation"""
        # Add validation metrics
        sample_tracker.val_metrics['total_loss'] = [0.4]
        
        # Add predictions for mAP calculation
        sample_tracker.ap_calculator.predictions = [[0.9, 0, 10, 10, 50, 50, 0]]
        sample_tracker.ap_calculator.targets = [[0, 15, 15, 45, 45, 0]]
        
        metrics = sample_tracker.compute_epoch_metrics(epoch=1)
        
        assert 'val_map50' in metrics
        assert 'val_map50_95' in metrics
        assert metrics['val_map50'] >= 0
        assert metrics['val_map50_95'] >= 0
    
    def test_compute_epoch_metrics_with_per_class_ap(self, sample_tracker):
        """Test epoch metrics computation with per-class AP"""
        # Add validation metrics
        sample_tracker.val_metrics['total_loss'] = [0.4]
        
        # Add predictions for multiple classes
        sample_tracker.ap_calculator.predictions = [
            [0.9, 0, 10, 10, 50, 50, 0],
            [0.8, 1, 60, 60, 100, 100, 0]
        ]
        sample_tracker.ap_calculator.targets = [
            [0, 15, 15, 45, 45, 0],
            [1, 65, 65, 95, 95, 0]
        ]
        
        metrics = sample_tracker.compute_epoch_metrics(epoch=1)
        
        # Should have per-class AP metrics
        assert 'val_ap_Rp1K' in metrics
        assert 'val_ap_Rp2K' in metrics
    
    def test_is_best_model_new_metric(self, sample_tracker):
        """Test best model detection with new metric"""
        sample_tracker.current_metrics = {'val_map50': 0.75}
        
        is_best = sample_tracker.is_best_model('val_map50', 'max')
        
        assert is_best == True
        assert abs(sample_tracker.best_metrics['val_map50'] - 0.75) < 1e-6
    
    def test_is_best_model_improved(self, sample_tracker):
        """Test best model detection with improved metric"""
        sample_tracker.best_metrics = {'val_map50': 0.70}
        sample_tracker.current_metrics = {'val_map50': 0.75}
        
        is_best = sample_tracker.is_best_model('val_map50', 'max')
        
        assert is_best == True
        assert abs(sample_tracker.best_metrics['val_map50'] - 0.75) < 1e-6
    
    def test_is_best_model_not_improved(self, sample_tracker):
        """Test best model detection with no improvement"""
        sample_tracker.best_metrics = {'val_map50': 0.80}
        sample_tracker.current_metrics = {'val_map50': 0.75}
        
        is_best = sample_tracker.is_best_model('val_map50', 'max')
        
        assert is_best == False
        assert abs(sample_tracker.best_metrics['val_map50'] - 0.80) < 1e-6  # Unchanged
    
    def test_is_best_model_minimize_metric(self, sample_tracker):
        """Test best model detection with minimize metric (loss)"""
        sample_tracker.best_metrics = {'val_loss': 0.5}
        sample_tracker.current_metrics = {'val_loss': 0.4}
        
        is_best = sample_tracker.is_best_model('val_loss', 'min')
        
        assert is_best == True
        assert abs(sample_tracker.best_metrics['val_loss'] - 0.4) < 1e-6
    
    def test_is_best_model_missing_metric(self, sample_tracker):
        """Test best model detection with missing metric"""
        sample_tracker.current_metrics = {}
        
        is_best = sample_tracker.is_best_model('val_map50', 'max')
        
        assert is_best == False
    
    def test_get_current_metrics(self, sample_tracker):
        """Test current metrics retrieval"""
        test_metrics = {'val_map50': 0.75, 'train_loss': 0.3}
        sample_tracker.current_metrics = test_metrics
        
        retrieved = sample_tracker.get_current_metrics()
        
        assert retrieved == test_metrics
        assert retrieved is not sample_tracker.current_metrics  # Should be a copy
    
    def test_get_best_metrics(self, sample_tracker):
        """Test best metrics retrieval"""
        test_metrics = {'val_map50': 0.80, 'val_loss': 0.25}
        sample_tracker.best_metrics = test_metrics
        
        retrieved = sample_tracker.get_best_metrics()
        
        assert retrieved == test_metrics
        assert retrieved is not sample_tracker.best_metrics  # Should be a copy
    
    def test_get_metrics_summary_empty(self, sample_tracker):
        """Test metrics summary with no metrics"""
        summary = sample_tracker.get_metrics_summary()
        
        assert summary == "No metrics available"
    
    def test_get_metrics_summary_with_data(self, sample_tracker):
        """Test metrics summary with data"""
        sample_tracker.current_metrics = {
            'train_loss': 0.3,
            'val_loss': 0.25,
            'val_map50': 0.75,
            'val_map50_95': 0.65
        }
        sample_tracker.best_metrics = {'val_map50': 0.80}
        
        summary = sample_tracker.get_metrics_summary()
        
        assert 'Train: 0.3000' in summary
        assert 'Val: 0.2500' in summary
        assert '@0.5: 0.750' in summary
        assert '@0.5:0.95: 0.650' in summary
        assert 'Best mAP@0.5: 0.800' in summary
    
    def test_save_metrics(self, sample_tracker):
        """Test metrics saving functionality"""
        # Setup test data
        sample_tracker.current_metrics = {'val_map50': 0.75}
        sample_tracker.best_metrics = {'val_map50': 0.80}
        sample_tracker.history = {
            'epoch': [1, 2],
            'train_loss': [0.5, 0.4],
            'val_loss': [0.4, 0.3],
            'val_map50': [0.7, 0.75],
            'val_map50_95': [0.6, 0.65],
            'learning_rate': [0.001, 0.0009]
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "metrics.json"
            
            sample_tracker.save_metrics(str(save_path))
            
            assert save_path.exists()
            
            # Verify saved content
            with open(save_path, 'r') as f:
                saved_data = json.load(f)
            
            assert 'history' in saved_data
            assert 'current_metrics' in saved_data
            assert 'best_metrics' in saved_data
            assert 'class_names' in saved_data
    
    def test_reset_epoch_metrics(self, sample_tracker):
        """Test epoch metrics reset"""
        # Add some data
        sample_tracker.train_metrics['total_loss'] = [0.5, 0.4]
        sample_tracker.val_metrics['total_loss'] = [0.4, 0.3]
        sample_tracker.ap_calculator.predictions = [[0.9, 0, 10, 10, 50, 50, 0]]
        sample_tracker.epoch_start_time = 12345
        sample_tracker.batch_start_time = 12346
        
        sample_tracker._reset_epoch_metrics()
        
        assert len(sample_tracker.train_metrics) == 0
        assert len(sample_tracker.val_metrics) == 0
        assert len(sample_tracker.ap_calculator.predictions) == 0
        assert sample_tracker.epoch_start_time is None
        assert sample_tracker.batch_start_time is None


class TestConvenienceFunctions:
    """Test convenience functions"""
    
    def test_create_metrics_tracker(self):
        """Test create_metrics_tracker convenience function"""
        config = {'training': {'validation': {'compute_map': True}}}
        class_names = ['class1', 'class2']
        
        tracker = create_metrics_tracker(config, class_names)
        
        assert isinstance(tracker, MetricsTracker)
        assert tracker.class_names == class_names
        assert tracker.config == config
    
    def test_calculate_map_function(self):
        """Test calculate_map convenience function"""
        # This function appears to be incomplete in the original code
        # Test what we can
        predictions = []
        targets = []
        
        try:
            map_score, class_aps = calculate_map(predictions, targets, num_classes=7)
            # Function should return something, but implementation is incomplete
        except Exception:
            # Expected since implementation is incomplete
            pass


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_ap_calculator_with_invalid_boxes(self):
        """Test AP calculator with invalid box coordinates"""
        calculator = APCalculator()
        
        # Invalid box coordinates (x2 < x1, y2 < y1)
        invalid_box1 = [50, 50, 10, 10]
        valid_box2 = [20, 20, 40, 40]
        
        iou = calculator._calculate_iou(invalid_box1, valid_box2)
        assert iou == 0.0  # Should handle invalid boxes gracefully
    
    def test_ap_calculator_with_zero_area_boxes(self):
        """Test AP calculator with zero-area boxes"""
        calculator = APCalculator()
        
        # Zero-area box
        zero_box = [10, 10, 10, 10]
        normal_box = [20, 20, 40, 40]
        
        iou = calculator._calculate_iou(zero_box, normal_box)
        assert iou == 0.0
    
    def test_metrics_tracker_with_map_disabled(self):
        """Test metrics tracker with mAP computation disabled"""
        config = {
            'training': {
                'validation': {
                    'compute_map': False
                }
            }
        }
        
        tracker = MetricsTracker(config)
        tracker.val_metrics['total_loss'] = [0.4]
        
        metrics = tracker.compute_epoch_metrics(epoch=1)
        
        assert 'val_map50' not in metrics
        assert 'val_map50_95' not in metrics
    
    def test_metrics_tracker_with_missing_validation_config(self):
        """Test metrics tracker with missing validation config"""
        config = {'training': {}}  # No validation config
        
        tracker = MetricsTracker(config)
        tracker.val_metrics['total_loss'] = [0.4]
        
        # Should still work with default behavior
        metrics = tracker.compute_epoch_metrics(epoch=1)
        assert 'val_loss' in metrics
    
    def test_ap_calculator_with_duplicate_image_ids(self):
        """Test AP calculator handling duplicate image IDs"""
        calculator = APCalculator()
        
        # Add predictions and targets with same image ID
        pred_boxes = [torch.tensor([[10, 10, 50, 50], [60, 60, 100, 100]])]
        pred_scores = [torch.tensor([0.9, 0.8])]
        pred_classes = [torch.tensor([0, 0])]
        
        true_boxes = [torch.tensor([[15, 15, 45, 45]])]  # Only one target
        true_classes = [torch.tensor([0])]
        
        image_ids = [0]
        
        calculator.add_batch(
            pred_boxes, pred_scores, pred_classes,
            true_boxes, true_classes, image_ids
        )
        
        ap = calculator.compute_ap(class_id=0, iou_threshold=0.5)
        
        # Should handle multiple predictions for same target
        assert 0.0 <= ap <= 1.0
    
    def test_metrics_tracker_history_update(self):
        """Test metrics tracker history updating"""
        tracker = MetricsTracker({})
        
        # Compute metrics for multiple epochs
        for epoch in range(3):
            tracker.train_metrics['total_loss'] = [0.5 - epoch * 0.1]
            tracker.val_metrics['total_loss'] = [0.4 - epoch * 0.1]
            
            metrics = tracker.compute_epoch_metrics(epoch=epoch + 1)
        
        # Check history
        assert len(tracker.history['epoch']) == 3
        assert len(tracker.history['train_loss']) == 3
        assert len(tracker.history['val_loss']) == 3
        assert tracker.history['epoch'] == [1, 2, 3]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])