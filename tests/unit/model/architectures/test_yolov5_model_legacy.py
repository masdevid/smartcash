#!/usr/bin/env python3
"""
Legacy tests for YOLOv5 Model module.

This module tests the legacy SmartCashYOLOv5 class to ensure functionality
before cleanup. These tests verify that the legacy implementation works
correctly and can be safely replaced by the new integration module.
"""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
import sys
from unittest.mock import patch, MagicMock

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from smartcash.model.architectures.yolov5_model import (
    SmartCashYOLOv5,
    SmartCashMultiDetect,
    create_smartcash_yolov5_model
)


class TestSmartCashYOLOv5Legacy:
    """Test cases for legacy SmartCashYOLOv5 class"""
    
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Setup test environment"""
        self.test_config = {
            "nc": 7,
            "depth_multiple": 0.33,
            "width_multiple": 0.50,
            "anchors": [
                [10, 13, 16, 30, 33, 23],
                [30, 61, 62, 45, 59, 119],
                [116, 90, 156, 198, 373, 326]
            ]
        }
    
    @patch('smartcash.model.architectures.yolov5_model.DetectionModel')
    def test_model_initialization_with_dict_config(self, mock_detection_model):
        """Test model initialization with dictionary configuration"""
        mock_instance = MagicMock()
        mock_detection_model.return_value = mock_instance
        
        model = SmartCashYOLOv5(
            cfg=self.test_config,
            ch=3,
            nc=7,
            backbone_type="cspdarknet"
        )
        
        assert model.backbone_type == "cspdarknet"
        assert model.current_phase == 1
        assert model.force_single_layer is False
    
    @patch('smartcash.model.architectures.yolov5_model.DetectionModel')
    def test_model_initialization_with_yaml_config(self, mock_detection_model):
        """Test model initialization with YAML file configuration"""
        mock_instance = MagicMock()
        mock_detection_model.return_value = mock_instance
        
        # Test with non-existent YAML file (should use default config)
        model = SmartCashYOLOv5(
            cfg="nonexistent.yaml",
            ch=3,
            nc=7,
            backbone_type="cspdarknet"
        )
        
        assert model.backbone_type == "cspdarknet"
    
    def test_get_default_config_cspdarknet(self):
        """Test default configuration generation for CSPDarknet"""
        with patch('smartcash.model.architectures.yolov5_model.DetectionModel'):
            model = SmartCashYOLOv5.__new__(SmartCashYOLOv5)
            config = model._get_default_config("cspdarknet")
            
            assert config["nc"] == 7
            assert "anchors" in config
            assert "backbone" in config
            assert "head" in config
            assert len(config["anchors"]) == 3
    
    def test_get_default_config_efficientnet(self):
        """Test default configuration generation for EfficientNet"""
        with patch('smartcash.model.architectures.yolov5_model.DetectionModel'):
            model = SmartCashYOLOv5.__new__(SmartCashYOLOv5)
            config = model._get_default_config("efficientnet_b4")
            
            assert config["nc"] == 7
            assert "anchors" in config
    
    def test_unsupported_backbone_type(self):
        """Test error handling for unsupported backbone"""
        with patch('smartcash.model.architectures.yolov5_model.DetectionModel'):
            model = SmartCashYOLOv5.__new__(SmartCashYOLOv5)
            
            with pytest.raises(ValueError, match="Unsupported backbone type"):
                model._get_default_config("unsupported_backbone")
    
    @patch('smartcash.model.architectures.yolov5_model.DetectionModel')
    def test_apply_custom_anchors(self, mock_detection_model):
        """Test custom anchor application"""
        mock_instance = MagicMock()
        mock_detection_model.return_value = mock_instance
        
        custom_anchors = [
            [10, 13, 16, 30, 33, 23],
            [30, 61, 62, 45, 59, 119],
            [116, 90, 156, 198, 373, 326]
        ]
        
        model = SmartCashYOLOv5(
            cfg=self.test_config,
            ch=3,
            nc=7,
            anchors=custom_anchors,
            backbone_type="cspdarknet"
        )
        
        assert model.custom_anchors == custom_anchors
    
    @patch('smartcash.model.architectures.yolov5_model.DetectionModel')
    @patch('smartcash.model.architectures.yolov5_model.CSPDarknet')
    def test_replace_backbone_cspdarknet(self, mock_cspdarknet, mock_detection_model):
        """Test backbone replacement with CSPDarknet"""
        # Mock the backbone
        mock_backbone_instance = MagicMock()
        mock_backbone_instance.backbone = [MagicMock(), MagicMock()]
        mock_backbone_instance.feature_indices = [0, 1]
        mock_cspdarknet.return_value = mock_backbone_instance
        
        # Mock the model
        mock_model_instance = MagicMock()
        mock_model_instance.model = [MagicMock(), MagicMock()]
        mock_detection_model.return_value = mock_model_instance
        
        model = SmartCashYOLOv5(
            cfg=self.test_config,
            ch=3,
            nc=7,
            backbone_type="cspdarknet"
        )
        
        mock_cspdarknet.assert_called_once()
    
    @patch('smartcash.model.architectures.yolov5_model.DetectionModel')
    @patch('smartcash.model.architectures.yolov5_model.EfficientNetB4')
    def test_replace_backbone_efficientnet(self, mock_efficientnet, mock_detection_model):
        """Test backbone replacement with EfficientNet"""
        # Mock the backbone
        mock_backbone_instance = MagicMock()
        mock_backbone_instance.backbone = [MagicMock(), MagicMock()]
        mock_backbone_instance.feature_indices = [0, 1]
        mock_efficientnet.return_value = mock_backbone_instance
        
        # Mock the model
        mock_model_instance = MagicMock()
        mock_model_instance.model = [MagicMock(), MagicMock()]
        mock_detection_model.return_value = mock_model_instance
        
        model = SmartCashYOLOv5(
            cfg=self.test_config,
            ch=3,
            nc=7,
            backbone_type="efficientnet_b4"
        )
        
        mock_efficientnet.assert_called_once()


class TestSmartCashMultiDetectLegacy:
    """Test cases for legacy SmartCashMultiDetect class"""
    
    @pytest.fixture
    def sample_layer_specs(self):
        """Sample layer specifications for testing"""
        return {
            'layer_1': {'nc': 7, 'description': 'Full banknote detection'},
            'layer_2': {'nc': 7, 'description': 'Nominal-defining features'},
            'layer_3': {'nc': 3, 'description': 'Common features'}
        }
    
    def test_multidetect_initialization(self, sample_layer_specs):
        """Test SmartCashMultiDetect initialization"""
        anchors = [
            [10, 13, 16, 30, 33, 23],
            [30, 61, 62, 45, 59, 119],
            [116, 90, 156, 198, 373, 326]
        ]
        ch = [256, 512, 1024]  # Input channels from backbone
        
        detect = SmartCashMultiDetect(
            nc=7,
            anchors=anchors,
            ch=ch,
            layer_specs=sample_layer_specs
        )
        
        assert detect.nl == 3  # 3 detection scales
        assert detect.na == 3  # 3 anchors per scale
        assert len(detect.layer_specs) == 3
        assert len(detect.detection_heads) == 3
    
    def test_multidetect_default_layer_specs(self):
        """Test SmartCashMultiDetect with default layer specifications"""
        anchors = [
            [10, 13, 16, 30, 33, 23],
            [30, 61, 62, 45, 59, 119],
            [116, 90, 156, 198, 373, 326]
        ]
        ch = [256, 512, 1024]
        
        detect = SmartCashMultiDetect(nc=7, anchors=anchors, ch=ch)
        
        assert len(detect.layer_specs) == 3
        assert 'layer_1' in detect.layer_specs
        assert 'layer_2' in detect.layer_specs
        assert 'layer_3' in detect.layer_specs
    
    def test_multidetect_forward_training_phase1(self, sample_layer_specs):
        """Test forward pass in training mode, phase 1"""
        anchors = [
            [10, 13, 16, 30, 33, 23],
            [30, 61, 62, 45, 59, 119],
            [116, 90, 156, 198, 373, 326]
        ]
        ch = [256, 512, 1024]
        
        detect = SmartCashMultiDetect(
            nc=7,
            anchors=anchors,
            ch=ch,
            layer_specs=sample_layer_specs
        )
        detect.training = True
        detect.current_phase = 1
        
        # Create sample input features
        x = [
            torch.randn(1, 256, 80, 80),   # P3
            torch.randn(1, 512, 40, 40),   # P4
            torch.randn(1, 1024, 20, 20)   # P5
        ]
        
        output = detect(x)
        
        # In phase 1, should return only layer_1 results
        assert isinstance(output, list)
        assert len(output) == 3  # 3 scales
    
    def test_multidetect_forward_training_phase2(self, sample_layer_specs):
        """Test forward pass in training mode, phase 2"""
        anchors = [
            [10, 13, 16, 30, 33, 23],
            [30, 61, 62, 45, 59, 119],
            [116, 90, 156, 198, 373, 326]
        ]
        ch = [256, 512, 1024]
        
        detect = SmartCashMultiDetect(
            nc=7,
            anchors=anchors,
            ch=ch,
            layer_specs=sample_layer_specs
        )
        detect.training = True
        detect.current_phase = 2
        
        # Create sample input features
        x = [
            torch.randn(1, 256, 80, 80),   # P3
            torch.randn(1, 512, 40, 40),   # P4
            torch.randn(1, 1024, 20, 20)   # P5
        ]
        
        output = detect(x)
        
        # In phase 2, should return all layers
        assert isinstance(output, dict)
        assert 'layer_1' in output
        assert 'layer_2' in output
        assert 'layer_3' in output
    
    def test_multidetect_forward_inference(self, sample_layer_specs):
        """Test forward pass in inference mode"""
        anchors = [
            [10, 13, 16, 30, 33, 23],
            [30, 61, 62, 45, 59, 119],
            [116, 90, 156, 198, 373, 326]
        ]
        ch = [256, 512, 1024]
        
        detect = SmartCashMultiDetect(
            nc=7,
            anchors=anchors,
            ch=ch,
            layer_specs=sample_layer_specs
        )
        detect.training = False
        detect.stride = torch.tensor([8., 16., 32.])  # Set strides
        
        # Create sample input features
        x = [
            torch.randn(1, 256, 80, 80),   # P3
            torch.randn(1, 512, 40, 40),   # P4
            torch.randn(1, 1024, 20, 20)   # P5
        ]
        
        output = detect(x)
        
        # In inference, should return all layers
        assert isinstance(output, dict)
        assert 'layer_1' in output
        assert 'layer_2' in output
        assert 'layer_3' in output
    
    def test_multidetect_make_grid(self, sample_layer_specs):
        """Test grid generation for anchor boxes"""
        anchors = [
            [10, 13, 16, 30, 33, 23],
            [30, 61, 62, 45, 59, 119],
            [116, 90, 156, 198, 373, 326]
        ]
        ch = [256, 512, 1024]
        
        detect = SmartCashMultiDetect(
            nc=7,
            anchors=anchors,
            ch=ch,
            layer_specs=sample_layer_specs
        )
        detect.stride = torch.tensor([8., 16., 32.])
        
        grid, anchor_grid = detect._make_grid(nx=80, ny=80, i=0)
        
        assert grid.shape == (1, 3, 80, 80, 2)
        assert anchor_grid.shape == (1, 3, 80, 80, 2)


class TestCreateSmartCashYOLOv5ModelLegacy:
    """Test the legacy factory function"""
    
    @patch('smartcash.model.architectures.yolov5_model.SmartCashYOLOv5')
    def test_factory_function_cspdarknet(self, mock_model_class):
        """Test factory function with CSPDarknet"""
        mock_instance = MagicMock()
        mock_model_class.return_value = mock_instance
        
        model = create_smartcash_yolov5_model(
            backbone_type="cspdarknet",
            model_size="s",
            pretrained=True,
            num_classes=7
        )
        
        assert model == mock_instance
        mock_model_class.assert_called_once()
    
    @patch('smartcash.model.architectures.yolov5_model.SmartCashYOLOv5')
    def test_factory_function_efficientnet(self, mock_model_class):
        """Test factory function with EfficientNet"""
        mock_instance = MagicMock()
        mock_model_class.return_value = mock_instance
        
        model = create_smartcash_yolov5_model(
            backbone_type="efficientnet_b4",
            model_size="s",
            pretrained=True,
            num_classes=7
        )
        
        assert model == mock_instance
        mock_model_class.assert_called_once()


class TestLegacyEdgeCases:
    """Test edge cases for legacy implementation"""
    
    @patch('smartcash.model.architectures.yolov5_model.DetectionModel')
    def test_model_with_no_anchors(self, mock_detection_model):
        """Test model initialization without custom anchors"""
        mock_instance = MagicMock()
        mock_detection_model.return_value = mock_instance
        
        model = SmartCashYOLOv5(
            cfg={"nc": 7},
            ch=3,
            nc=7,
            anchors=None,
            backbone_type="cspdarknet"
        )
        
        assert model.custom_anchors is None
    
    @patch('smartcash.model.architectures.yolov5_model.DetectionModel')
    def test_model_with_flat_anchors(self, mock_detection_model):
        """Test model with flat anchor list"""
        mock_instance = MagicMock()
        mock_instance.model = [MagicMock()]
        mock_detection_model.return_value = mock_instance
        
        flat_anchors = [10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119]
        
        model = SmartCashYOLOv5(
            cfg={"nc": 7},
            ch=3,
            nc=7,
            anchors=flat_anchors,
            backbone_type="cspdarknet"
        )
        
        assert model.custom_anchors == flat_anchors
    
    def test_multidetect_force_single_layer(self):
        """Test force single layer mode"""
        anchors = [[10, 13], [16, 30], [33, 23]]
        ch = [256, 512, 1024]
        
        detect = SmartCashMultiDetect(nc=7, anchors=anchors, ch=ch)
        detect.training = True
        detect.force_single_layer = True
        
        x = [
            torch.randn(1, 256, 80, 80),
            torch.randn(1, 512, 40, 40),
            torch.randn(1, 1024, 20, 20)
        ]
        
        output = detect(x)
        
        # Should return single layer output
        assert isinstance(output, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])