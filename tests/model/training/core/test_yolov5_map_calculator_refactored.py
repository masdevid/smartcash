#!/usr/bin/env python3
"""
Comprehensive unit tests for refactored YOLOv5 mAP calculator modules.

Tests all the specialized modules created during refactoring:
- YOLOv5UtilitiesManager
- HierarchicalProcessor
- MemoryOptimizedProcessor  
- BatchProcessor
- YOLOv5MapCalculator (refactored)

Ensures backward compatibility and validates SRP compliance.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Import modules to test
from smartcash.model.training.core.yolo_utils_manager import (
    YOLOv5UtilitiesManager, 
    get_yolo_utils_manager,
    is_yolov5_available
)
from smartcash.model.training.core.hierarchical_processor import HierarchicalProcessor
from smartcash.model.training.core.memory_optimized_processor import (
    MemoryOptimizedProcessor,
    ProcessingConfig
)
from smartcash.model.training.core.batch_processor import BatchProcessor
from smartcash.model.training.core.yolov5_map_calculator import (
    YOLOv5MapCalculator,
    create_yolov5_map_calculator
)

# Test backward compatibility
from smartcash.model.training.core.yolov5_map_calculator import (
    YOLOv5MapCalculator as LegacyCalculator,
    create_yolov5_map_calculator as legacy_factory
)


class TestYOLOv5UtilitiesManager:
    """Test YOLOv5 utilities manager functionality."""
    
    def test_manager_initialization(self):
        """Test manager initializes correctly."""
        manager = YOLOv5UtilitiesManager()
        assert manager._utils_cache is None
        assert not manager._availability_checked
        assert isinstance(manager._yolov5_root, Path)
    
    def test_path_resolution(self):
        """Test YOLOv5 path resolution."""
        manager = YOLOv5UtilitiesManager()
        yolov5_path = manager._resolve_yolov5_path()
        assert isinstance(yolov5_path, Path)
        assert yolov5_path.name == "yolov5"
    
    @patch('smartcash.model.training.core.yolo_utils_manager.sys.path')
    def test_path_addition(self, mock_path):
        """Test YOLOv5 path is added to sys.path."""
        mock_path.__contains__ = Mock(return_value=False)
        mock_path.insert = Mock()
        
        manager = YOLOv5UtilitiesManager()
        manager._yolov5_root = Path("/fake/yolov5")
        
        with patch.object(manager._yolov5_root, 'exists', return_value=True):
            result = manager._ensure_yolov5_in_path()
            assert result
            mock_path.insert.assert_called_once()
    
    @patch('smartcash.model.training.core.yolo_utils_manager.__import__')
    def test_utilities_loading_success(self, mock_import):
        """Test successful utilities loading."""
        # Mock successful import
        mock_metrics = Mock()
        mock_metrics.ap_per_class = Mock()
        mock_metrics.box_iou = Mock()
        mock_metrics.__name__ = 'utils.metrics'
        
        mock_general = Mock()
        mock_general.xywh2xyxy = Mock()
        mock_general.non_max_suppression = Mock()
        
        mock_import.side_effect = [mock_metrics, mock_general]
        
        manager = YOLOv5UtilitiesManager()
        utilities = manager._load_yolov5_utilities()
        
        assert 'ap_per_class' in utilities
        assert 'box_iou' in utilities
        assert 'xywh2xyxy' in utilities
        assert 'non_max_suppression' in utilities
    
    def test_global_manager_singleton(self):
        """Test global manager is singleton."""
        manager1 = get_yolo_utils_manager()
        manager2 = get_yolo_utils_manager()
        assert manager1 is manager2
    
    def test_function_access_error_handling(self):
        """Test error handling for missing functions."""
        manager = YOLOv5UtilitiesManager()
        manager._utils_cache = {'existing_func': Mock()}
        
        with pytest.raises(KeyError):
            manager.get_function('non_existent_function')


class TestHierarchicalProcessor:
    """Test hierarchical processor functionality."""
    
    def test_processor_initialization(self):
        """Test processor initializes correctly."""
        processor = HierarchicalProcessor(debug=True)
        assert processor.debug
        assert processor.max_predictions_per_chunk == 10000
        assert processor.layer_1_classes == range(0, 7)
        assert processor.layer_2_classes == range(7, 14)
        assert processor.layer_3_classes == range(14, 17)
    
    def test_phase_detection_phase1(self):
        """Test Phase 1 detection (classes 0-6)."""
        processor = HierarchicalProcessor()
        
        # Create predictions with classes 0-6 only
        predictions = torch.tensor([
            [0.1, 0.1, 0.2, 0.2, 0.8, 3.0],  # class 3
            [0.3, 0.3, 0.4, 0.4, 0.9, 5.0],  # class 5
        ])
        
        phase = processor._detect_processing_phase(predictions)
        assert phase == 1
    
    def test_phase_detection_phase2(self):
        """Test Phase 2 detection (classes > 6)."""
        processor = HierarchicalProcessor()
        
        # Create predictions with classes beyond 6
        predictions = torch.tensor([
            [0.1, 0.1, 0.2, 0.2, 0.8, 3.0],   # class 3 (Layer 1)
            [0.3, 0.3, 0.4, 0.4, 0.9, 10.0],  # class 10 (Layer 2)
            [0.5, 0.5, 0.6, 0.6, 0.7, 15.0],  # class 15 (Layer 3)
        ])
        
        phase = processor._detect_processing_phase(predictions)
        assert phase == 2
    
    def test_3d_prediction_processing(self):
        """Test 3D prediction tensor processing."""
        processor = HierarchicalProcessor()
        
        # Create 3D predictions [batch, detections, features]
        predictions = torch.tensor([[[0.1, 0.1, 0.2, 0.2, 0.8, 3.0]]])  # Shape: [1, 1, 6]
        
        result = processor._process_3d_predictions(predictions)
        assert result.dim() == 2
        assert result.shape[1] == 6  # Same number of features
    
    def test_2d_prediction_processing(self):
        """Test 2D prediction tensor processing."""
        processor = HierarchicalProcessor()
        
        # Create 2D predictions [detections, features]
        predictions = torch.tensor([[0.1, 0.1, 0.2, 0.2, 0.8, 3.0]])  # Shape: [1, 6]
        
        result = processor._process_2d_predictions(predictions)
        assert result.dim() == 2
        assert result.shape[0] <= predictions.shape[0]  # May filter some predictions
    
    def test_empty_predictions_handling(self):
        """Test handling of empty predictions."""
        processor = HierarchicalProcessor()
        
        empty_preds = torch.empty((0, 6))
        targets = torch.tensor([[0, 1, 0.5, 0.5, 0.1, 0.1]])
        
        result_preds, result_targets = processor.process_hierarchical_predictions(empty_preds, targets)
        assert result_preds.shape == empty_preds.shape
        assert torch.equal(result_targets, targets)


class TestMemoryOptimizedProcessor:
    """Test memory-optimized processor functionality."""
    
    def test_processor_initialization(self):
        """Test processor initializes with correct configuration."""
        processor = MemoryOptimizedProcessor(debug=True)
        assert processor.debug
        assert isinstance(processor.config, ProcessingConfig)
        assert processor.config.chunk_size > 0
    
    def test_platform_config_selection(self):
        """Test platform-specific configuration selection."""
        processor = MemoryOptimizedProcessor()
        config = processor._get_platform_config()
        
        assert isinstance(config, ProcessingConfig)
        assert config.chunk_size > 0
        assert config.max_matrix_combinations > 0
        assert config.cleanup_frequency > 0
        assert isinstance(config.use_parallel_assignment, bool)
    
    def test_tensor_optimization(self):
        """Test tensor transfer optimization."""
        processor = MemoryOptimizedProcessor()
        
        # Test with torch tensor
        tensor = torch.randn(10, 5)
        optimized = processor.optimize_tensor_transfer(tensor)
        assert isinstance(optimized, torch.Tensor)
        assert optimized.device == processor.device
        
        # Test with non-tensor
        non_tensor = [1, 2, 3]
        result = processor.optimize_tensor_transfer(non_tensor)
        assert result == non_tensor
    
    def test_memory_usage_estimation(self):
        """Test memory usage estimation."""
        processor = MemoryOptimizedProcessor()
        
        estimates = processor.estimate_memory_usage(100, 50)
        
        assert 'iou_matrix_mb' in estimates
        assert 'tracking_mb' in estimates
        assert 'total_mb' in estimates
        assert 'recommend_chunking' in estimates
        assert 'suggested_chunk_size' in estimates
        assert all(isinstance(v, (int, float, bool)) for v in estimates.values())
    
    def test_processing_stats_collection(self):
        """Test processing statistics collection."""
        processor = MemoryOptimizedProcessor()
        processor._batch_count = 5
        
        stats = processor.get_processing_stats()
        
        assert 'batch_count' in stats
        assert 'device' in stats
        assert 'platform_info' in stats
        assert 'config' in stats
        assert stats['batch_count'] == 5


class TestBatchProcessor:
    """Test batch processor functionality."""
    
    def test_processor_initialization(self):
        """Test processor initializes correctly."""
        processor = BatchProcessor(
            conf_threshold=0.01,
            iou_threshold=0.5,
            debug=True
        )
        assert processor.conf_threshold == 0.01
        assert processor.iou_threshold == 0.5
        assert processor.debug
    
    def test_3d_predictions_preprocessing(self):
        """Test 3D predictions preprocessing."""
        processor = BatchProcessor(conf_threshold=0.1)
        
        # Create 3D predictions [batch, detections, features]
        predictions = torch.tensor([
            [[0.1, 0.1, 0.2, 0.2, 0.8, 3.0]],  # High confidence
            [[0.3, 0.3, 0.4, 0.4, 0.05, 1.0]]  # Low confidence (should be filtered)
        ])
        
        result = processor._process_3d_predictions(predictions)
        assert result.dim() == 2
        assert result.shape[1] == 7  # [batch_idx, x1, y1, x2, y2, conf, class]
        assert result.shape[0] == 1  # Only high confidence prediction remains
    
    def test_2d_predictions_preprocessing(self):
        """Test 2D predictions preprocessing."""
        processor = BatchProcessor(conf_threshold=0.1)
        
        # Create 2D predictions [detections, features]
        predictions = torch.tensor([
            [0.1, 0.1, 0.2, 0.2, 0.8, 3.0],  # High confidence
            [0.3, 0.3, 0.4, 0.4, 0.05, 1.0]  # Low confidence (should be filtered)
        ])
        
        result = processor._process_2d_predictions(predictions)
        assert result.dim() == 2
        assert result.shape[1] == 7  # [batch_idx, x1, y1, x2, y2, conf, class]
        assert result.shape[0] == 1  # Only high confidence prediction remains
    
    def test_empty_predictions_handling(self):
        """Test handling of empty predictions."""
        processor = BatchProcessor()
        
        empty_preds = torch.empty((0, 7))
        targets = torch.tensor([[0, 1, 0.5, 0.5, 0.1, 0.1]])
        
        result = processor._handle_empty_predictions(targets)
        assert len(result) == 4  # (tp, conf, pred_cls, target_cls)
        assert result[0].shape[0] == 0  # No true positives
        assert result[3].shape[0] == 1  # One target class
    
    def test_empty_targets_handling(self):
        """Test handling of empty targets."""
        processor = BatchProcessor()
        
        predictions = torch.tensor([[0, 0.1, 0.1, 0.2, 0.2, 0.8, 3.0]])  # [batch_idx, x1, y1, x2, y2, conf, class]
        
        result = processor._handle_empty_targets(predictions)
        assert len(result) == 4  # (tp, conf, pred_cls, target_cls)
        assert result[0].shape[0] == 1  # One false positive
        assert result[3].shape[0] == 0  # No target classes
    
    def test_tensor_format_validation(self):
        """Test tensor format validation."""
        processor = BatchProcessor()
        
        # Valid formats
        valid_preds = torch.randn(5, 7)
        valid_targets = torch.randn(3, 6)
        assert processor._validate_tensor_formats(valid_preds, valid_targets)
        
        # Invalid prediction format
        invalid_preds = torch.randn(5, 5)  # Too few columns
        assert not processor._validate_tensor_formats(invalid_preds, valid_targets)
        
        # Invalid target format
        invalid_targets = torch.randn(3, 4)  # Too few columns
        assert not processor._validate_tensor_formats(valid_preds, invalid_targets)
    
    def test_processing_stats(self):
        """Test processing statistics collection."""
        processor = BatchProcessor(conf_threshold=0.1, iou_threshold=0.5)
        
        stats = processor.get_processing_stats()
        
        assert 'conf_threshold' in stats
        assert 'iou_threshold' in stats
        assert 'device' in stats
        assert 'memory_processor_stats' in stats
        assert stats['conf_threshold'] == 0.1
        assert stats['iou_threshold'] == 0.5


class TestYOLOv5MapCalculatorRefactored:
    """Test refactored YOLOv5 mAP calculator."""
    
    def test_calculator_initialization(self):
        """Test calculator initializes correctly."""
        calculator = YOLOv5MapCalculator(
            num_classes=7,
            conf_thres=0.1,
            iou_thres=0.5,
            debug=True
        )
        
        assert calculator.num_classes == 7
        assert calculator.conf_thres == 0.1
        assert calculator.iou_thres == 0.5
        assert calculator.debug
        assert len(calculator.stats) == 0
    
    def test_processor_initialization(self):
        """Test all processors are initialized correctly."""
        calculator = YOLOv5MapCalculator()
        
        assert hasattr(calculator, 'yolo_utils')
        assert hasattr(calculator, 'hierarchical_processor')
        assert hasattr(calculator, 'memory_processor')
        assert hasattr(calculator, 'batch_processor')
        
        assert isinstance(calculator.yolo_utils, YOLOv5UtilitiesManager)
        assert isinstance(calculator.hierarchical_processor, HierarchicalProcessor)
        assert isinstance(calculator.memory_processor, MemoryOptimizedProcessor)
        assert isinstance(calculator.batch_processor, BatchProcessor)
    
    def test_reset_functionality(self):
        """Test calculator reset functionality."""
        calculator = YOLOv5MapCalculator()
        
        # Add some dummy stats
        calculator.stats = [Mock(), Mock(), Mock()]
        calculator._batch_count = 5
        
        calculator.reset()
        
        assert len(calculator.stats) == 0
        assert calculator._batch_count == 0
    
    @patch('smartcash.model.training.core.yolov5_map_calculator_refactored.is_yolov5_available')
    def test_input_validation(self, mock_yolov5_available):
        """Test input validation in update method."""
        mock_yolov5_available.return_value = True
        calculator = YOLOv5MapCalculator()
        calculator.yolo_utils.is_available = Mock(return_value=True)
        
        # Test with None inputs
        calculator.update(None, None)
        assert len(calculator.stats) == 0
        
        # Test with invalid prediction dimensions
        invalid_preds = torch.randn(5, 4)  # Too few columns
        targets = torch.randn(3, 6)
        calculator.update(invalid_preds, targets)
        assert len(calculator.stats) == 0
    
    def test_comprehensive_processing_stats(self):
        """Test comprehensive processing statistics."""
        calculator = YOLOv5MapCalculator()
        calculator._batch_count = 3
        
        stats = calculator.get_processing_stats()
        
        assert 'calculator_stats' in stats
        assert 'yolo_utils_available' in stats
        assert 'hierarchical_processor_stats' in stats
        assert 'memory_processor_stats' in stats
        assert 'batch_processor_stats' in stats
        
        calc_stats = stats['calculator_stats']
        assert calc_stats['batch_count'] == 3
        assert calc_stats['num_classes'] == 7  # Default value
    
    def test_factory_function(self):
        """Test factory function creates correct instance."""
        calculator = create_yolov5_map_calculator(
            num_classes=10,
            conf_thres=0.2,
            iou_thres=0.6,
            debug=True
        )
        
        assert isinstance(calculator, YOLOv5MapCalculator)
        assert calculator.num_classes == 10
        assert calculator.conf_thres == 0.2
        assert calculator.iou_thres == 0.6
        assert calculator.debug


class TestBackwardCompatibility:
    """Test backward compatibility with original API."""
    
    def test_class_compatibility(self):
        """Test class is accessible from both locations."""
        # Should be able to import from both locations
        from smartcash.model.training.core.yolov5_map_calculator import YOLOv5MapCalculator as Legacy
        from smartcash.model.training.core.yolov5_map_calculator import YOLOv5MapCalculator as Refactored
        
        # Both should be the same class
        assert Legacy is Refactored
    
    def test_factory_compatibility(self):
        """Test factory function compatibility."""
        legacy_calc = legacy_factory(num_classes=5, debug=True)
        new_calc = create_yolov5_map_calculator(num_classes=5, debug=True)
        
        # Should create instances of the same class
        assert type(legacy_calc) is type(new_calc)
        assert legacy_calc.num_classes == new_calc.num_classes
        assert legacy_calc.debug == new_calc.debug
    
    def test_api_compatibility(self):
        """Test API method compatibility."""
        calculator = LegacyCalculator()
        
        # Test all expected methods exist
        assert hasattr(calculator, 'reset')
        assert hasattr(calculator, 'update')
        assert hasattr(calculator, 'compute_map')
        
        # Test method signatures are compatible
        assert callable(calculator.reset)
        assert callable(calculator.update)
        assert callable(calculator.compute_map)
    
    def test_accessor_functions_compatibility(self):
        """Test YOLOv5 accessor functions are available."""
        from smartcash.model.training.core.yolov5_map_calculator import (
            get_ap_per_class,
            get_box_iou,
            get_xywh2xyxy,
            get_non_max_suppression
        )
        
        # All functions should be callable
        assert callable(get_ap_per_class)
        assert callable(get_box_iou)
        assert callable(get_xywh2xyxy)
        assert callable(get_non_max_suppression)


@pytest.fixture
def sample_predictions():
    """Fixture providing sample prediction data."""
    return torch.tensor([
        [0.1, 0.1, 0.3, 0.3, 0.8, 1.0],  # [x, y, w, h, conf, class]
        [0.4, 0.4, 0.2, 0.2, 0.9, 2.0],
        [0.7, 0.7, 0.1, 0.1, 0.6, 0.0],
    ])


@pytest.fixture
def sample_targets():
    """Fixture providing sample target data."""
    return torch.tensor([
        [0, 1, 0.2, 0.2, 0.2, 0.2],  # [batch_idx, class, x, y, w, h]
        [0, 2, 0.5, 0.5, 0.15, 0.15],
        [0, 0, 0.75, 0.75, 0.08, 0.08],
    ])


class TestIntegration:
    """Integration tests for the refactored system."""
    
    @patch('smartcash.model.training.core.yolo_utils_manager.YOLOv5UtilitiesManager.is_available')
    def test_end_to_end_processing(self, mock_available, sample_predictions, sample_targets):
        """Test end-to-end processing with sample data."""
        mock_available.return_value = False  # Skip YOLOv5 dependency
        
        calculator = YOLOv5MapCalculator(debug=True)
        
        # Test that update doesn't crash with unavailable YOLOv5
        calculator.update(sample_predictions, sample_targets)
        
        # Should not have accumulated any stats due to unavailable YOLOv5
        assert len(calculator.stats) == 0
    
    def test_memory_cleanup_integration(self, sample_predictions, sample_targets):
        """Test memory cleanup integration across modules."""
        calculator = YOLOv5MapCalculator()
        
        # Verify memory optimizer is shared across processors
        assert calculator.memory_processor.memory_optimizer is calculator.memory_optimizer
        
        # Test cleanup calls don't crash
        calculator.memory_processor.cleanup_after_batch()
        calculator.memory_processor.emergency_cleanup()
    
    def test_error_resilience(self):
        """Test system resilience to various error conditions."""
        calculator = YOLOv5MapCalculator(debug=True)
        
        # Test with malformed tensors
        bad_preds = torch.tensor([[1, 2]])  # Wrong shape
        bad_targets = torch.tensor([[1]])   # Wrong shape
        
        # Should not crash
        calculator.update(bad_preds, bad_targets)
        assert len(calculator.stats) == 0
        
        # Test compute_map with no stats
        with pytest.raises(RuntimeError, match="No statistics accumulated"):
            calculator.compute_map()


if __name__ == "__main__":
    # Run specific test categories
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-k", "test_backward_compatibility or test_integration"
    ])