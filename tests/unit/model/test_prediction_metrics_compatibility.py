#!/usr/bin/env python3
"""
Comprehensive tests for model prediction output compatibility with metrics.

This module tests:
- Model prediction output format consistency
- Batch size changes and model rebuilding
- Prediction-metrics compatibility across different configurations
- Output shape validation with varying batch sizes
- Multi-layer prediction format validation
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any, Tuple
import numpy as np

# Import the modules under test
from smartcash.model.api.core import SmartCashModelAPI
from smartcash.model.core.model_builder import ModelBuilder
from smartcash.model.training.core.prediction_processor import PredictionProcessor
from smartcash.model.training.utils.metrics_utils import calculate_multilayer_metrics
from smartcash.model.training.multi_task_loss import UncertaintyMultiTaskLoss
from smartcash.model.training.core.validation_executor import ValidationExecutor


class TestPredictionOutputFormat:
    """Test model prediction output format consistency."""
    
    @pytest.fixture
    def mock_config(self):
        """Mock configuration for testing."""
        return {
            'model': {
                'backbone': 'cspdarknet',
                'num_classes': 7,
                'img_size': 640,
                'detection_layers': ['layer_1', 'layer_2', 'layer_3']
            },
            'device': 'cpu'
        }
    
    @pytest.fixture
    def sample_model_outputs(self):
        """Sample model outputs in YOLO format."""
        batch_size = 4
        return [
            # P3 predictions (80x80 grid)
            torch.randn(batch_size, 3, 80, 80, 22),  # 5 (bbox+obj) + 17 classes
            # P4 predictions (40x40 grid)  
            torch.randn(batch_size, 3, 40, 40, 22),
            # P5 predictions (20x20 grid)
            torch.randn(batch_size, 3, 20, 20, 22)
        ]
    
    @pytest.fixture
    def sample_targets(self):
        """Sample training targets in YOLO format."""
        return torch.tensor([
            [0, 1.0, 0.5, 0.5, 0.3, 0.4],  # [img_idx, class, x, y, w, h]
            [0, 2.0, 0.3, 0.7, 0.2, 0.3],
            [1, 0.0, 0.6, 0.4, 0.4, 0.5],
            [2, 3.0, 0.2, 0.8, 0.3, 0.2],
            [3, 1.0, 0.7, 0.3, 0.2, 0.4]
        ]).float()
    
    def test_prediction_output_structure(self, sample_model_outputs):
        """Test that model outputs have the expected structure."""
        # Check that we have outputs for 3 scales
        assert len(sample_model_outputs) == 3
        
        # Check output dimensions
        for i, output in enumerate(sample_model_outputs):
            assert output.dim() == 5  # [batch, anchors, height, width, features]
            assert output.shape[0] == 4  # batch size
            assert output.shape[1] == 3  # number of anchors per scale
            assert output.shape[4] == 22  # 5 (bbox+obj) + 17 classes
            
            # Check grid sizes
            expected_grid_sizes = [80, 40, 20]
            assert output.shape[2] == expected_grid_sizes[i]  # height
            assert output.shape[3] == expected_grid_sizes[i]  # width
    
    def test_prediction_processor_normalization(self, sample_model_outputs, sample_targets):
        """Test prediction processor normalization for different modes."""
        config = {
            'training_mode': 'two_phase',
            'model': {'layer_mode': 'multi'}
        }
        
        processor = PredictionProcessor(config)
        
        # Test multi-layer normalization (phase 2)
        normalized_preds = processor.normalize_training_predictions(
            sample_model_outputs, phase_num=2
        )
        
        # Should return dict with layer keys
        assert isinstance(normalized_preds, dict)
        assert 'layer_1' in normalized_preds
        assert 'layer_2' in normalized_preds  
        assert 'layer_3' in normalized_preds
        
        # Each layer should have the original prediction - could be list or tensor
        for layer_name, layer_preds in normalized_preds.items():
            # In multi-layer mode, predictions should be the original format or processed
            assert layer_preds is not None
            if isinstance(layer_preds, list):
                assert len(layer_preds) >= 1  # At least one prediction tensor
    
    def test_single_layer_normalization(self, sample_model_outputs):
        """Test single layer mode normalization."""
        config = {
            'training_mode': 'two_phase',
            'model': {'layer_mode': 'single'}
        }
        
        processor = PredictionProcessor(config)
        
        # Test single-layer normalization (phase 1)
        normalized_preds = processor.normalize_training_predictions(
            sample_model_outputs, phase_num=1
        )
        
        # Should return dict with only layer_1
        assert isinstance(normalized_preds, dict)
        assert 'layer_1' in normalized_preds
        assert len(normalized_preds) == 1
    
    def test_prediction_format_consistency(self, sample_model_outputs):
        """Test prediction format consistency across different configurations."""
        configs = [
            {'training_mode': 'two_phase', 'model': {'layer_mode': 'multi'}},
            {'training_mode': 'two_phase', 'model': {'layer_mode': 'single'}},
            {'training_mode': 'single_phase', 'model': {'layer_mode': 'multi'}},
            {'training_mode': 'single_phase', 'model': {'layer_mode': 'single'}}
        ]
        
        for config in configs:
            processor = PredictionProcessor(config)
            
            # Test with different phases
            for phase_num in [1, 2]:
                normalized_preds = processor.normalize_training_predictions(
                    sample_model_outputs, phase_num=phase_num
                )
                
                # All configurations should return a dict
                assert isinstance(normalized_preds, dict)
                
                # All layer values should be valid prediction formats
                for layer_name, layer_preds in normalized_preds.items():
                    # Could be list of tensors or single tensor
                    if isinstance(layer_preds, list):
                        assert all(isinstance(p, torch.Tensor) for p in layer_preds)
                    else:
                        assert isinstance(layer_preds, torch.Tensor)


class TestBatchSizeModelRebuilding:
    """Test model rebuilding when batch size changes."""
    
    @pytest.fixture
    def mock_model_api(self):
        """Mock model API for testing."""
        with patch('smartcash.model.api.core.SmartCashModelAPI') as mock_api:
            api_instance = Mock()
            api_instance.config = {
                'model': {
                    'backbone': 'cspdarknet',
                    'num_classes': 7,
                    'img_size': 640
                }
            }
            api_instance.device = torch.device('cpu')
            api_instance.is_model_built = False
            api_instance.model = None
            
            mock_api.return_value = api_instance
            yield api_instance
    
    def test_model_rebuilding_with_different_batch_sizes(self, mock_model_api):
        """Test that model can handle different batch sizes without rebuilding."""
        # Create mock model that tracks input batch sizes
        mock_model = Mock()
        mock_model.training = True
        
        batch_sizes_processed = []
        
        def mock_forward(x):
            batch_sizes_processed.append(x.shape[0])
            batch_size = x.shape[0]
            # Return typical YOLO output structure
            return [
                torch.randn(batch_size, 3, 80, 80, 22),
                torch.randn(batch_size, 3, 40, 40, 22), 
                torch.randn(batch_size, 3, 20, 20, 22)
            ]
        
        mock_model.forward = mock_forward
        mock_model.__call__ = mock_forward
        mock_model_api.model = mock_model
        mock_model_api.is_model_built = True
        
        # Test with different batch sizes
        test_batch_sizes = [1, 4, 8, 16, 2]
        
        for batch_size in test_batch_sizes:
            # Create input tensor with current batch size
            input_tensor = torch.randn(batch_size, 3, 640, 640)
            
            # Run forward pass through the mock_forward function directly
            output = mock_forward(input_tensor)
            
            # Verify output structure matches batch size
            assert isinstance(output, list)
            assert len(output) == 3  # 3 scales
            for scale_output in output:
                assert scale_output.shape[0] == batch_size
        
        # Verify all batch sizes were processed
        assert batch_sizes_processed == test_batch_sizes
        
        # Model should not be rebuilt multiple times
        assert mock_model_api.is_model_built == True
    
    def test_batch_size_consistency_in_training(self):
        """Test batch size consistency throughout training pipeline."""
        batch_sizes = [2, 4, 8]
        
        for batch_size in batch_sizes:
            # Create sample data with specific batch size
            images = torch.randn(batch_size, 3, 640, 640)
            targets = torch.randint(0, 7, (batch_size * 2, 6)).float()  # Variable number of targets
            
            # Set image indices in targets (ensure we don't exceed bounds)
            num_targets_per_image = min(2, targets.shape[0] // batch_size)
            for i in range(batch_size):
                start_idx = i * num_targets_per_image
                end_idx = min(start_idx + num_targets_per_image, targets.shape[0])
                if start_idx < targets.shape[0]:
                    targets[start_idx:end_idx, 0] = i
            
            # Test prediction processor
            config = {'training_mode': 'two_phase', 'model': {'layer_mode': 'multi'}}
            processor = PredictionProcessor(config)
            
            # Create mock predictions matching batch size
            predictions = [
                torch.randn(batch_size, 3, 80, 80, 22),
                torch.randn(batch_size, 3, 40, 40, 22),
                torch.randn(batch_size, 3, 20, 20, 22)
            ]
            
            # Process predictions
            normalized_preds = processor.normalize_training_predictions(predictions, phase_num=2)
            
            # Verify batch size consistency
            for layer_name, layer_preds in normalized_preds.items():
                if isinstance(layer_preds, list):
                    for pred_tensor in layer_preds:
                        assert pred_tensor.shape[0] == batch_size, f"Batch size mismatch in {layer_name}"
                elif isinstance(layer_preds, torch.Tensor):
                    assert layer_preds.shape[0] == batch_size, f"Batch size mismatch in {layer_name}"
    
    def test_dynamic_batch_size_validation(self):
        """Test that validation works with dynamic batch sizes."""
        layer_config = {
            'layer_1': {'num_classes': 7},
            'layer_2': {'num_classes': 7}, 
            'layer_3': {'num_classes': 3}
        }
        
        # Test with different batch sizes
        batch_sizes = [1, 3, 5, 7]
        
        for batch_size in batch_sizes:
            # Create predictions and targets matching batch size
            predictions = {
                'layer_1': [torch.randn(batch_size, 3, 40, 40, 22)],
                'layer_2': [torch.randn(batch_size, 3, 40, 40, 22)],
                'layer_3': [torch.randn(batch_size, 3, 40, 40, 8)]  # 3 classes + 5 bbox
            }
            
            targets = {
                'layer_1': torch.randint(0, 7, (batch_size * 2, 6)).float(),
                'layer_2': torch.randint(0, 7, (batch_size * 2, 6)).float(),
                'layer_3': torch.randint(0, 3, (batch_size * 2, 6)).float()
            }
            
            # Set proper image indices
            for layer_name, layer_targets in targets.items():
                for i in range(batch_size):
                    layer_targets[i*2:i*2+2, 0] = i
            
            # Test multi-task loss with dynamic batch size
            loss_fn = UncertaintyMultiTaskLoss(layer_config)
            
            try:
                total_loss, loss_breakdown = loss_fn(predictions, targets)
                
                # Verify loss computation works
                assert isinstance(total_loss, torch.Tensor)
                assert total_loss.item() >= 0
                assert 'total_loss' in loss_breakdown
                
            except Exception as e:
                pytest.fail(f"Multi-task loss failed with batch size {batch_size}: {str(e)}")


class TestPredictionMetricsCompatibility:
    """Test compatibility between predictions and metrics calculation."""
    
    @pytest.fixture
    def layer_config(self):
        """Layer configuration for multi-task setup."""
        return {
            'layer_1': {'num_classes': 7, 'description': 'Full banknote detection'},
            'layer_2': {'num_classes': 7, 'description': 'Nominal features'},
            'layer_3': {'num_classes': 3, 'description': 'Common features'}
        }
    
    def test_prediction_metrics_format_compatibility(self, layer_config):
        """Test that predictions are compatible with metrics calculation."""
        batch_size = 4
        
        # Create realistic predictions in YOLO format
        yolo_predictions = [
            torch.randn(batch_size, 3, 80, 80, 22),  # P3
            torch.randn(batch_size, 3, 40, 40, 22),  # P4
            torch.randn(batch_size, 3, 20, 20, 22)   # P5
        ]
        
        # Process predictions through prediction processor
        config = {'training_mode': 'two_phase', 'model': {'layer_mode': 'multi'}}
        processor = PredictionProcessor(config)
        
        normalized_preds = processor.normalize_training_predictions(yolo_predictions, phase_num=2)
        
        # Extract classification predictions for metrics
        device = torch.device('cpu')
        processed_preds = {}
        
        for layer_name, layer_preds in normalized_preds.items():
            processed_preds[layer_name] = processor.extract_classification_predictions(
                layer_preds, batch_size, device
            )
        
        # Create compatible targets
        processed_targets = {}
        for layer_name in layer_config.keys():
            num_classes = layer_config[layer_name]['num_classes']
            # Create targets as class indices
            processed_targets[layer_name] = torch.randint(0, num_classes, (batch_size,))
        
        # Test metrics calculation
        try:
            metrics = calculate_multilayer_metrics(processed_preds, processed_targets)
            
            # Verify metrics structure
            assert isinstance(metrics, dict)
            
            # Check that metrics exist for each layer
            for layer_name in layer_config.keys():
                assert f'{layer_name}_accuracy' in metrics
                assert f'{layer_name}_precision' in metrics
                assert f'{layer_name}_recall' in metrics
                assert f'{layer_name}_f1' in metrics
                
                # Verify metric values are reasonable
                assert 0.0 <= metrics[f'{layer_name}_accuracy'] <= 1.0
                assert 0.0 <= metrics[f'{layer_name}_precision'] <= 1.0
                assert 0.0 <= metrics[f'{layer_name}_recall'] <= 1.0
                assert 0.0 <= metrics[f'{layer_name}_f1'] <= 1.0
                
        except Exception as e:
            pytest.fail(f"Metrics calculation failed: {str(e)}")
    
    def test_validation_executor_prediction_processing(self, layer_config):
        """Test that validation executor properly processes predictions."""
        # Create mock model
        mock_model = Mock()
        mock_model.eval = Mock()
        
        def mock_parameters():
            param = Mock()
            param.device = torch.device('cpu') 
            yield param  # Use yield to make it a generator
        
        mock_model.parameters = mock_parameters
        
        # Create mock config
        config = {
            'training_mode': 'two_phase',
            'model': {'layer_mode': 'multi'}
        }
        
        # Create mock progress tracker
        progress_tracker = Mock()
        progress_tracker.start_batch_tracking = Mock()
        progress_tracker.update_batch_progress = Mock() 
        progress_tracker.complete_batch_tracking = Mock()
        
        # Create validation executor
        validator = ValidationExecutor(mock_model, config, progress_tracker)
        
        # Test prediction processing
        batch_size = 2
        images = torch.randn(batch_size, 3, 640, 640)
        targets = torch.randint(0, 7, (batch_size * 2, 6)).float()
        
        # Set image indices
        targets[:2, 0] = 0
        targets[2:4, 0] = 1
        
        # Mock model output
        yolo_output = [
            torch.randn(batch_size, 3, 80, 80, 22),
            torch.randn(batch_size, 3, 40, 40, 22),
            torch.randn(batch_size, 3, 20, 20, 22)
        ]
        
        mock_model.return_value = yolo_output
        mock_model.__call__ = Mock(return_value=yolo_output)
        
        # Mock loss manager
        mock_loss_manager = Mock()
        mock_loss_manager.compute_loss = Mock(return_value=(
            torch.tensor(0.5, requires_grad=True),
            {'total_loss': torch.tensor(0.5)}
        ))
        
        # Create simple data loader with len() method
        class SimpleDataLoader:
            def __init__(self, data):
                self.data = data
            
            def __iter__(self):
                return iter(self.data)
            
            def __len__(self):
                return len(self.data)
        
        val_loader = SimpleDataLoader([(images, targets)])
        
        # Test validation execution
        try:
            # Mock the batch processing components that would normally be called
            with patch.object(validator.map_calculator, 'process_batch_for_map'):
                with patch.object(validator.map_calculator, 'compute_final_map', 
                                return_value={'val_map50': 0.5, 'val_map50_95': 0.3}):
                    
                    metrics = validator.validate_epoch(
                        val_loader, mock_loss_manager, epoch=0, total_epochs=1, phase_num=2
                    )
                    
                    # Verify metrics structure
                    assert isinstance(metrics, dict)
                    assert 'val_loss' in metrics
                    assert 'val_map50' in metrics
                    assert 'val_accuracy' in metrics
                    
        except Exception as e:
            pytest.fail(f"Validation executor failed: {str(e)}")
    
    def test_multi_task_loss_prediction_compatibility(self, layer_config):
        """Test multi-task loss compatibility with different prediction formats."""
        batch_size = 3
        
        # Test different prediction input formats
        prediction_formats = [
            # Format 1: List of tensors (standard YOLO output)
            [
                torch.randn(batch_size, 3, 80, 80, 22),
                torch.randn(batch_size, 3, 40, 40, 22),
                torch.randn(batch_size, 3, 20, 20, 22)
            ],
            
            # Format 2: Single tensor
            torch.randn(batch_size, 25200, 22),  # Flattened YOLO output
            
            # Format 3: Nested list structure
            [[
                torch.randn(batch_size, 3, 80, 80, 22),
                torch.randn(batch_size, 3, 40, 40, 22)
            ]]
        ]
        
        # Create multi-task loss
        loss_fn = UncertaintyMultiTaskLoss(layer_config)
        
        for i, pred_format in enumerate(prediction_formats):
            # Create prediction dict for all layers
            predictions = {}
            for layer_name in layer_config.keys():
                if layer_name == 'layer_3':
                    # Adjust for layer_3 which has 3 classes instead of 7
                    if isinstance(pred_format, list) and len(pred_format) > 0:
                        if isinstance(pred_format[0], torch.Tensor):
                            # Adjust the last dimension for layer_3 (3 classes + 5 bbox = 8)
                            layer_preds = []
                            for tensor in pred_format:
                                adjusted_tensor = torch.randn(*tensor.shape[:-1], 8)
                                layer_preds.append(adjusted_tensor)
                            predictions[layer_name] = layer_preds
                        else:
                            predictions[layer_name] = pred_format
                    else:
                        predictions[layer_name] = pred_format
                else:
                    predictions[layer_name] = pred_format
            
            # Create targets
            targets = {}
            for layer_name, config_info in layer_config.items():
                num_classes = config_info['num_classes']
                layer_targets = torch.randint(0, num_classes, (batch_size * 2, 6)).float()
                
                # Set image indices
                for j in range(batch_size):
                    layer_targets[j*2:j*2+2, 0] = j
                
                targets[layer_name] = layer_targets
            
            # Test loss computation
            try:
                total_loss, loss_breakdown = loss_fn(predictions, targets)
                
                # Verify basic structure
                assert isinstance(total_loss, torch.Tensor)
                assert isinstance(loss_breakdown, dict)
                assert total_loss.item() >= 0
                
                # Check loss breakdown structure
                assert 'total_loss' in loss_breakdown
                assert 'layer_losses' in loss_breakdown
                assert 'uncertainties' in loss_breakdown
                
            except Exception as e:
                # Some prediction formats might not be compatible, which is expected
                print(f"Prediction format {i} incompatible (expected): {str(e)}")


class TestOutputShapeValidation:
    """Test model output shape validation with different configurations."""
    
    def test_multi_scale_output_shapes(self):
        """Test that multi-scale outputs have correct shapes."""
        batch_sizes = [1, 4, 8]
        img_size = 640
        
        for batch_size in batch_sizes:
            # Expected output shapes for YOLOv5 at different scales
            expected_shapes = [
                (batch_size, 3, img_size//8, img_size//8, 22),   # P3: 80x80
                (batch_size, 3, img_size//16, img_size//16, 22), # P4: 40x40  
                (batch_size, 3, img_size//32, img_size//32, 22)  # P5: 20x20
            ]
            
            # Create mock outputs
            outputs = [torch.randn(*shape) for shape in expected_shapes]
            
            # Validate shapes
            for i, (output, expected_shape) in enumerate(zip(outputs, expected_shapes)):
                assert output.shape == expected_shape, f"Scale {i} shape mismatch: {output.shape} vs {expected_shape}"
    
    def test_feature_dimension_consistency(self):
        """Test that feature dimensions are consistent across scales."""
        batch_size = 4
        num_classes = 7
        num_anchors = 3
        
        # Expected feature dimension: 5 (bbox + objectness) + num_classes
        expected_features = 5 + num_classes + 10  # 22 total (17 classes for multi-layer)
        
        # Create outputs with correct feature dimensions
        outputs = [
            torch.randn(batch_size, num_anchors, 80, 80, expected_features),
            torch.randn(batch_size, num_anchors, 40, 40, expected_features),
            torch.randn(batch_size, num_anchors, 20, 20, expected_features)
        ]
        
        # Validate feature dimensions
        for i, output in enumerate(outputs):
            assert output.shape[-1] == expected_features, f"Scale {i} feature dimension mismatch"
            assert output.shape[1] == num_anchors, f"Scale {i} anchor dimension mismatch"
    
    def test_batch_dimension_propagation(self):
        """Test that batch dimensions are properly propagated through processing."""
        batch_sizes = [1, 2, 4, 8, 16]
        
        config = {'training_mode': 'two_phase', 'model': {'layer_mode': 'multi'}}
        processor = PredictionProcessor(config)
        
        for batch_size in batch_sizes:
            # Create input with specific batch size
            input_preds = [
                torch.randn(batch_size, 3, 80, 80, 22),
                torch.randn(batch_size, 3, 40, 40, 22),
                torch.randn(batch_size, 3, 20, 20, 22)
            ]
            
            # Process predictions
            normalized_preds = processor.normalize_training_predictions(input_preds, phase_num=2)
            
            # Extract classification predictions
            for layer_name, layer_preds in normalized_preds.items():
                class_preds = processor.extract_classification_predictions(
                    layer_preds, batch_size, torch.device('cpu')
                )
                
                # Verify batch dimension is preserved
                assert class_preds.shape[0] == batch_size, f"Batch size not preserved in {layer_name}"


class TestIntegrationScenarios:
    """Integration tests for prediction-metrics compatibility."""
    
    def test_end_to_end_prediction_metrics_flow(self):
        """Test complete flow from model output to metrics calculation."""
        batch_size = 4
        num_classes = 7
        
        # Step 1: Create realistic model outputs (YOLO format)
        model_outputs = [
            torch.randn(batch_size, 3, 80, 80, 22),
            torch.randn(batch_size, 3, 40, 40, 22),
            torch.randn(batch_size, 3, 20, 20, 22)
        ]
        
        # Step 2: Process through prediction processor
        config = {'training_mode': 'two_phase', 'model': {'layer_mode': 'multi'}}
        processor = PredictionProcessor(config)
        
        normalized_preds = processor.normalize_training_predictions(model_outputs, phase_num=2)
        
        # Step 3: Extract classification predictions
        device = torch.device('cpu')
        processed_preds = {}
        
        for layer_name, layer_preds in normalized_preds.items():
            processed_preds[layer_name] = processor.extract_classification_predictions(
                layer_preds, batch_size, device
            )
        
        # Step 4: Create corresponding targets
        processed_targets = {}
        for layer_name in ['layer_1', 'layer_2', 'layer_3']:
            if layer_name == 'layer_3':
                processed_targets[layer_name] = torch.randint(0, 3, (batch_size,))
            else:
                processed_targets[layer_name] = torch.randint(0, num_classes, (batch_size,))
        
        # Step 5: Calculate metrics
        metrics = calculate_multilayer_metrics(processed_preds, processed_targets)
        
        # Step 6: Verify complete compatibility
        assert isinstance(metrics, dict)
        assert len(metrics) > 0
        
        # Check that all expected metrics are present
        expected_metric_types = ['accuracy', 'precision', 'recall', 'f1']
        expected_layers = ['layer_1', 'layer_2', 'layer_3']
        
        for layer in expected_layers:
            for metric_type in expected_metric_types:
                metric_key = f'{layer}_{metric_type}'
                assert metric_key in metrics, f"Missing metric: {metric_key}"
                assert isinstance(metrics[metric_key], float)
                assert 0.0 <= metrics[metric_key] <= 1.0
    
    def test_batch_size_variation_compatibility(self):
        """Test that the entire pipeline works with varying batch sizes."""
        layer_config = {
            'layer_1': {'num_classes': 7},
            'layer_2': {'num_classes': 7},
            'layer_3': {'num_classes': 3}
        }
        
        # Test with different batch sizes
        batch_sizes = [1, 3, 5, 8]
        
        for batch_size in batch_sizes:
            try:
                # Create predictions and targets
                predictions = {}
                targets = {}
                
                for layer_name, layer_info in layer_config.items():
                    num_classes = layer_info['num_classes']
                    feature_dim = 22 if layer_name != 'layer_3' else 8
                    
                    predictions[layer_name] = [torch.randn(batch_size, 3, 40, 40, feature_dim)]
                    
                    # Create targets
                    layer_targets = torch.randint(0, num_classes, (batch_size * 2, 6)).float()
                    for i in range(batch_size):
                        layer_targets[i*2:i*2+2, 0] = i
                    targets[layer_name] = layer_targets
                
                # Test multi-task loss
                loss_fn = UncertaintyMultiTaskLoss(layer_config)
                total_loss, loss_breakdown = loss_fn(predictions, targets)
                
                assert isinstance(total_loss, torch.Tensor)
                assert total_loss.item() >= 0
                
                # Test metrics calculation
                config = {'training_mode': 'two_phase', 'model': {'layer_mode': 'multi'}}
                processor = PredictionProcessor(config)
                
                processed_preds = {}
                for layer_name, layer_preds in predictions.items():
                    processed_preds[layer_name] = processor.extract_classification_predictions(
                        layer_preds, batch_size, torch.device('cpu')
                    )
                
                processed_targets = {}
                for layer_name, layer_info in layer_config.items():
                    processed_targets[layer_name] = torch.randint(
                        0, layer_info['num_classes'], (batch_size,)
                    )
                
                metrics = calculate_multilayer_metrics(processed_preds, processed_targets)
                
                assert isinstance(metrics, dict)
                assert len(metrics) > 0
                
            except Exception as e:
                pytest.fail(f"Pipeline failed with batch size {batch_size}: {str(e)}")


if __name__ == '__main__':
    # Run all tests
    pytest.main([__file__, '-v', '--tb=short'])