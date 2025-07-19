"""
File: tests/model/test_model_building.py
Description: Unit tests for enhanced model building components with dummy data
"""

import pytest
import torch
import torch.nn as nn
from typing import Dict, Any, List
from unittest.mock import Mock, patch

# Import components to test
from smartcash.model.architectures.backbones.efficientnet import EfficientNetBackbone
from smartcash.model.architectures.backbones.cspdarknet import CSPDarknet
from smartcash.model.architectures.heads.multi_layer_head import MultiLayerHead, create_banknote_detection_head
from smartcash.model.training.multi_task_loss import UncertaintyMultiTaskLoss, create_banknote_multi_task_loss
from smartcash.model.core.yolo_model_builder import YOLOModelBuilder, build_banknote_detection_model
from smartcash.common.logger import SmartCashLogger


class TestEfficientNetBackbone:
    """Test cases for enhanced EfficientNet backbone"""
    
    @pytest.fixture
    def dummy_logger(self):
        """Create mock logger for testing"""
        logger = Mock()
        logger.info = Mock()
        logger.error = Mock()
        logger.warning = Mock()
        logger.debug = Mock()
        # Add missing success method
        logger.success = Mock()
        return logger
    
    def test_efficientnet_backbone_creation(self, dummy_logger):
        """Test EfficientNet backbone creation with testing mode"""
        backbone = EfficientNetBackbone(
            model_name='efficientnet_b4',
            testing_mode=True,
            multi_layer_heads=True,
            logger=dummy_logger
        )
        
        assert backbone.model_name == 'efficientnet_b4'
        assert backbone.testing_mode is True
        assert backbone.multi_layer_heads is True
        assert len(backbone.out_channels) == 3
        
        # Test info retrieval
        info = backbone.get_info()
        assert info['type'] == 'EfficientNet'
        assert info['multi_layer_heads'] is True
        assert info['fpn_compatible'] is True
    
    def test_efficientnet_forward_pass(self, dummy_logger):
        """Test EfficientNet forward pass with dummy data"""
        backbone = EfficientNetBackbone(
            model_name='efficientnet_b4',
            testing_mode=True,
            logger=dummy_logger
        )
        
        # Create dummy input tensor
        dummy_input = torch.randn(2, 3, 640, 640)
        
        # Forward pass
        features = backbone(dummy_input)
        
        # Validate outputs
        assert len(features) == 3  # P3, P4, P5
        assert all(isinstance(f, torch.Tensor) for f in features)
        assert features[0].shape[1] in backbone.out_channels  # Check channel count
    
    def test_efficientnet_build_for_yolo(self, dummy_logger):
        """Test EfficientNet YOLO model building"""
        backbone = EfficientNetBackbone(
            model_name='efficientnet_b4',
            testing_mode=True,
            multi_layer_heads=True,
            logger=dummy_logger
        )
        
        build_result = backbone.build_for_yolo()
        
        assert build_result['success'] is True
        assert 'layer_specifications' in build_result
        assert len(build_result['layer_specifications']) == 3  # layer_1, layer_2, layer_3
        assert 'recommended_neck' in build_result
        assert build_result['recommended_neck'] == 'FPN-PAN'
    
    def test_efficientnet_training_preparation(self, dummy_logger):
        """Test EfficientNet training preparation"""
        backbone = EfficientNetBackbone(
            model_name='efficientnet_b4',
            testing_mode=True,
            logger=dummy_logger
        )
        
        # Test freezing
        freeze_result = backbone.prepare_for_training(freeze_backbone=True)
        assert freeze_result['success'] is True
        assert freeze_result['frozen'] is True
        assert freeze_result['phase'] == 'phase_1'
        
        # Test unfreezing
        unfreeze_result = backbone.prepare_for_training(freeze_backbone=False)
        assert unfreeze_result['success'] is True
        assert unfreeze_result['frozen'] is False
        assert unfreeze_result['phase'] == 'phase_2'


class TestCSPDarknetBackbone:
    """Test cases for enhanced CSPDarknet backbone"""
    
    @pytest.fixture
    def dummy_logger(self):
        """Create mock logger for testing"""
        logger = Mock()
        logger.info = Mock()
        logger.error = Mock()
        logger.warning = Mock()
        logger.debug = Mock()
        # Add missing success method
        logger.success = Mock()
        return logger
    
    def test_cspdarknet_backbone_creation(self, dummy_logger):
        """Test CSPDarknet backbone creation with testing mode"""
        backbone = CSPDarknet(
            model_size='yolov5s',
            testing_mode=True,
            multi_layer_heads=True,
            logger=dummy_logger
        )
        
        assert backbone.model_size == 'yolov5s'
        assert backbone.testing_mode is True
        assert backbone.multi_layer_heads is True
        
        # Test info retrieval
        info = backbone.get_info()
        assert info['type'] == 'CSPDarknet'
        assert info['multi_layer_heads'] is True
        assert info['fpn_compatible'] is True
    
    def test_cspdarknet_forward_pass(self, dummy_logger):
        """Test CSPDarknet forward pass with dummy data"""
        backbone = CSPDarknet(
            model_size='yolov5s',
            testing_mode=True,
            logger=dummy_logger
        )
        
        # Create dummy input tensor
        dummy_input = torch.randn(2, 3, 640, 640)
        
        # Forward pass
        features = backbone(dummy_input)
        
        # Validate outputs
        assert len(features) == 3  # P3, P4, P5
        assert all(isinstance(f, torch.Tensor) for f in features)
    
    def test_cspdarknet_build_for_yolo(self, dummy_logger):
        """Test CSPDarknet YOLO model building"""
        backbone = CSPDarknet(
            model_size='yolov5s',
            testing_mode=True,
            multi_layer_heads=True,
            logger=dummy_logger
        )
        
        build_result = backbone.build_for_yolo()
        
        assert build_result['success'] is True
        assert 'layer_specifications' in build_result
        assert len(build_result['layer_specifications']) == 3


class TestMultiLayerHead:
    """Test cases for multi-layer detection head"""
    
    @pytest.fixture
    def dummy_logger(self):
        """Create mock logger for testing"""
        logger = Mock()
        logger.info = Mock()
        logger.error = Mock()
        logger.warning = Mock()
        logger.debug = Mock()
        # Add missing success method
        logger.success = Mock()
        return logger
    
    @pytest.fixture
    def sample_layer_specs(self):
        """Sample layer specifications for testing"""
        return {
            'layer_1': {
                'description': 'Full banknote detection',
                'classes': ['001', '002', '005', '010', '020', '050', '100'],
                'num_classes': 7
            },
            'layer_2': {
                'description': 'Nominal-defining features',
                'classes': ['l2_001', 'l2_002', 'l2_005', 'l2_010', 'l2_020', 'l2_050', 'l2_100'],
                'num_classes': 7
            },
            'layer_3': {
                'description': 'Common features',
                'classes': ['l3_sign', 'l3_text', 'l3_thread'],
                'num_classes': 3
            }
        }
    
    def test_multi_layer_head_creation(self, sample_layer_specs, dummy_logger):
        """Test multi-layer head creation"""
        in_channels = [256, 512, 1024]  # P3, P4, P5
        
        head = MultiLayerHead(
            in_channels=in_channels,
            layer_specs=sample_layer_specs,
            use_attention=True,
            logger=dummy_logger
        )
        
        assert head.layer_names == ['layer_1', 'layer_2', 'layer_3']
        assert head.use_attention is True
        assert len(head.heads) == 3  # One for each layer
        
        # Test layer info
        layer_info = head.get_layer_info()
        assert layer_info['layer_count'] == 3
        assert layer_info['total_classes'] == 17  # 7 + 7 + 3
        assert layer_info['use_attention'] is True
    
    def test_multi_layer_head_forward_pass(self, sample_layer_specs, dummy_logger):
        """Test multi-layer head forward pass"""
        in_channels = [256, 512, 1024]
        
        head = MultiLayerHead(
            in_channels=in_channels,
            layer_specs=sample_layer_specs,
            logger=dummy_logger
        )
        
        # Create dummy feature maps
        dummy_features = [
            torch.randn(2, 256, 80, 80),   # P3
            torch.randn(2, 512, 40, 40),   # P4
            torch.randn(2, 1024, 20, 20)   # P5
        ]
        
        # Forward pass
        predictions = head(dummy_features)
        
        # Validate outputs
        assert len(predictions) == 3  # Three layers
        assert 'layer_1' in predictions
        assert 'layer_2' in predictions
        assert 'layer_3' in predictions
        
        # Check each layer has 3 scales
        for layer_name in predictions:
            assert len(predictions[layer_name]) == 3  # P3, P4, P5
            for pred in predictions[layer_name]:
                assert pred.dim() == 5  # [B, anchors, H, W, 5+classes]
    
    def test_banknote_detection_head_factory(self, dummy_logger):
        """Test banknote detection head factory function"""
        in_channels = [256, 512, 1024]
        
        head = create_banknote_detection_head(
            in_channels=in_channels,
            logger=dummy_logger
        )
        
        assert isinstance(head, MultiLayerHead)
        assert len(head.layer_names) == 3
        
        # Test output shapes
        output_shapes = head.get_output_shapes()
        assert len(output_shapes) == 3  # Three layers
        assert 'layer_1' in output_shapes


class TestUncertaintyMultiTaskLoss:
    """Test cases for uncertainty-based multi-task loss"""
    
    @pytest.fixture
    def dummy_logger(self):
        """Create mock logger for testing"""
        logger = Mock()
        logger.info = Mock()
        logger.error = Mock()
        logger.warning = Mock()
        logger.debug = Mock()
        # Add missing success method
        logger.success = Mock()
        return logger
    
    @pytest.fixture
    def sample_layer_config(self):
        """Sample layer configuration for loss testing"""
        return {
            'layer_1': {
                'description': 'Full banknote detection',
                'num_classes': 7
            },
            'layer_2': {
                'description': 'Nominal-defining features',
                'num_classes': 7
            },
            'layer_3': {
                'description': 'Common features',
                'num_classes': 3
            }
        }
    
    def test_uncertainty_loss_creation(self, sample_layer_config, dummy_logger):
        """Test uncertainty-based multi-task loss creation"""
        loss_function = UncertaintyMultiTaskLoss(
            layer_config=sample_layer_config,
            logger=dummy_logger
        )
        
        assert loss_function.num_layers == 3
        assert loss_function.layer_names == ['layer_1', 'layer_2', 'layer_3']
        assert 'layer_1' in loss_function.log_vars
        assert 'layer_2' in loss_function.log_vars
        assert 'layer_3' in loss_function.log_vars
    
    def test_uncertainty_weights_retrieval(self, sample_layer_config, dummy_logger):
        """Test uncertainty weights and values retrieval"""
        loss_function = UncertaintyMultiTaskLoss(
            layer_config=sample_layer_config,
            logger=dummy_logger
        )
        
        # Test weights
        weights = loss_function.get_uncertainty_weights()
        assert len(weights) == 3
        assert all(isinstance(w, float) for w in weights.values())
        
        # Test uncertainties
        uncertainties = loss_function.get_uncertainty_values()
        assert len(uncertainties) == 3
        assert all(isinstance(u, float) for u in uncertainties.values())
    
    def test_banknote_multi_task_loss_factory(self, dummy_logger):
        """Test banknote multi-task loss factory function"""
        loss_function = create_banknote_multi_task_loss(
            use_adaptive=False,
            logger=dummy_logger
        )
        
        assert isinstance(loss_function, UncertaintyMultiTaskLoss)
        assert loss_function.num_layers == 3


class TestYOLOModelBuilder:
    """Test cases for YOLO model builder"""
    
    @pytest.fixture
    def dummy_logger(self):
        """Create mock logger for testing"""
        logger = Mock()
        logger.info = Mock()
        logger.error = Mock()
        logger.warning = Mock()
        logger.debug = Mock()
        # Add missing success method
        logger.success = Mock()
        return logger
    
    @pytest.fixture
    def sample_config(self):
        """Sample configuration for model building"""
        return {
            'backbone': {
                'type': 'efficientnet_b4',
                'pretrained': False  # Use False for testing
            },
            'neck': {
                'type': 'fpn_pan'
            },
            'head': {
                'multi_layer': True,
                'use_attention': True,
                'num_anchors': 3
            },
            'model': {
                'img_size': 640
            },
            'loss': {
                'box_weight': 0.05,
                'obj_weight': 1.0,
                'cls_weight': 0.5,
                'dynamic_weighting': True
            }
        }
    
    def test_yolo_model_builder_creation(self, sample_config, dummy_logger):
        """Test YOLO model builder creation"""
        builder = YOLOModelBuilder(config=sample_config, logger=dummy_logger)
        
        assert builder.config == sample_config
        assert builder.backbone_config == sample_config['backbone']
        assert builder.head_config == sample_config['head']
    
    def test_build_complete_model(self, sample_config, dummy_logger):
        """Test complete model building with testing mode"""
        builder = YOLOModelBuilder(config=sample_config, logger=dummy_logger)
        
        # Build model in testing mode
        build_result = builder.build_model(testing_mode=True)
        
        assert build_result['success'] is True
        assert 'model' in build_result
        assert 'loss_function' in build_result
        assert 'build_info' in build_result
        
        # Test model structure
        model = build_result['model']
        assert hasattr(model, 'backbone')
        assert hasattr(model, 'neck')
        assert hasattr(model, 'head')
        
        # Test forward pass
        dummy_input = torch.randn(1, 3, 640, 640)
        predictions = model(dummy_input)
        assert isinstance(predictions, dict)
        assert len(predictions) == 3  # Three layers
    
    def test_build_banknote_detection_model_factory(self, dummy_logger):
        """Test banknote detection model factory function"""
        # Test EfficientNet model
        efficientnet_result = build_banknote_detection_model(
            backbone_type='efficientnet_b4',
            multi_layer=True,
            testing_mode=True
        )
        
        assert efficientnet_result['success'] is True
        assert 'model' in efficientnet_result
        
        # Test CSPDarknet model
        cspdarknet_result = build_banknote_detection_model(
            backbone_type='cspdarknet',
            multi_layer=True,
            testing_mode=True
        )
        
        assert cspdarknet_result['success'] is True
        assert 'model' in cspdarknet_result
    
    def test_model_parameter_counting(self, sample_config, dummy_logger):
        """Test model parameter counting"""
        builder = YOLOModelBuilder(config=sample_config, logger=dummy_logger)
        build_result = builder.build_model(testing_mode=True)
        
        model = build_result['model']
        
        # Test parameter counting
        total_params = model.count_parameters()
        trainable_params = model.count_trainable_parameters()
        
        assert isinstance(total_params, int)
        assert isinstance(trainable_params, int)
        assert total_params > 0
        assert trainable_params > 0
        assert trainable_params <= total_params
    
    def test_model_training_preparation(self, sample_config, dummy_logger):
        """Test model training preparation"""
        builder = YOLOModelBuilder(config=sample_config, logger=dummy_logger)
        build_result = builder.build_model(testing_mode=True)
        
        model = build_result['model']
        
        # Test phase 1 preparation (frozen backbone)
        phase1_result = model.prepare_for_training(freeze_backbone=True)
        assert phase1_result['phase'] == 'phase_1'
        assert phase1_result['frozen_backbone'] is True
        
        # Test phase 2 preparation (unfrozen)
        phase2_result = model.prepare_for_training(freeze_backbone=False)
        assert phase2_result['phase'] == 'phase_2'
        assert phase2_result['frozen_backbone'] is False


# Integration tests
class TestModelBuildingIntegration:
    """Integration tests for complete model building workflow"""
    
    def test_end_to_end_efficientnet_workflow(self):
        """Test complete EfficientNet model building workflow"""
        # Build model
        result = build_banknote_detection_model(
            backbone_type='efficientnet_b4',
            multi_layer=True,
            testing_mode=True
        )
        
        assert result['success'] is True
        
        model = result['model']
        loss_function = result['loss_function']
        
        # Test forward pass
        dummy_input = torch.randn(2, 3, 640, 640)
        predictions = model(dummy_input)
        
        # Test loss calculation
        dummy_targets = {
            'layer_1': torch.zeros(5, 6),  # 5 dummy targets, 6 columns [batch_idx, class, x, y, w, h]
            'layer_2': torch.zeros(3, 6),  # 3 dummy targets
            'layer_3': torch.zeros(2, 6),  # 2 dummy targets
        }
        
        total_loss, loss_breakdown = loss_function(predictions, dummy_targets)
        
        assert isinstance(total_loss, torch.Tensor)
        assert 'total_loss' in loss_breakdown
        assert 'uncertainties' in loss_breakdown
    
    def test_end_to_end_cspdarknet_workflow(self):
        """Test complete CSPDarknet model building workflow"""
        # Build model
        result = build_banknote_detection_model(
            backbone_type='cspdarknet',
            multi_layer=True,
            testing_mode=True
        )
        
        assert result['success'] is True
        
        model = result['model']
        
        # Test model info
        model_info = model.get_model_info()
        assert 'architecture' in model_info
        assert 'total_parameters' in model_info
        assert 'trainable_parameters' in model_info
        
        # Test training preparation phases
        phase1 = model.prepare_for_training(freeze_backbone=True)
        phase2 = model.prepare_for_training(freeze_backbone=False)
        
        assert phase1['trainable_parameters'] < phase2['trainable_parameters']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])