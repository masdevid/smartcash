"""
Tests for the YOLOv5 backbone implementation.
"""

import os
import pytest
import torch
from unittest.mock import MagicMock, patch

from smartcash.model.architectures.backbones import YOLOv5Backbone, BackboneFactory, BackboneConfig
from smartcash.common.logger import SmartCashLogger


def test_yolov5_backbone_initialization():
    """Test YOLOv5Backbone initialization with default parameters."""
    # Create a mock logger
    logger = MagicMock(spec=SmartCashLogger)
    
    # Initialize backbone
    backbone = YOLOv5Backbone(
        model_size='s',
        pretrained=False,
        freeze=False,
        logger=logger
    )
    
    # Verify the model was created with the correct configuration
    assert backbone.model_size == 's'
    assert not backbone.pretrained
    assert not backbone.frozen
    assert len(backbone.model) > 0  # Should have layers
    assert backbone.out_channels == [128, 256, 512]  # For YOLOv5s

def test_yolov5_backbone_forward():
    """Test YOLOv5Backbone forward pass."""
    # Initialize backbone
    backbone = YOLOv5Backbone(model_size='s', pretrained=False)
    
    # Create a dummy input tensor
    x = torch.randn(1, 3, 640, 640)  # Batch size 1, 3 channels, 640x640 image
    
    # Forward pass
    features = backbone(x)
    
    # Verify output shapes
    assert len(features) == 3  # Should return 3 feature maps
    assert features[0].shape == (1, 128, 80, 80)   # P3/8
    assert features[1].shape == (1, 256, 40, 40)   # P4/16
    assert features[2].shape == (1, 512, 20, 20)   # P5/32

def test_backbone_factory_create_yolov5():
    """Test BackboneFactory with YOLOv5 backbone."""
    # Create a config
    config = BackboneConfig(
        backbone_type='yolov5',
        model_size='s',
        pretrained=False,
        freeze=False
    )
    
    # Create backbone using factory
    backbone = BackboneFactory.create_backbone(config)
    
    # Verify the correct type was created
    assert isinstance(backbone, YOLOv5Backbone)
    assert backbone.model_size == 's'
    assert not backbone.pretrained
    assert not backbone.frozen

@patch('smartcash.model.architectures.backbones.yolov5_backbone_impl.YOLOv5Backbone')
def test_yolov5_backbone_factory_mock(mock_backbone):
    """Test YOLOv5 backbone creation through factory with mock."""
    # Setup mock
    mock_instance = MagicMock()
    mock_backbone.return_value = mock_instance
    
    # Create config
    config = BackboneConfig(
        backbone_type='yolov5',
        model_size='m',
        pretrained=True,
        freeze=True
    )
    
    # Create backbone
    backbone = BackboneFactory.create_backbone(config)
    
    # Verify factory called the constructor correctly
    mock_backbone.assert_called_once_with(
        model_size='m',
        pretrained=True,
        freeze=True,
        logger=None
    )
    assert backbone == mock_instance

def test_yolov5_backbone_get_info():
    """Test YOLOv5Backbone get_info method."""
    backbone = YOLOv5Backbone(model_size='s', pretrained=False)
    info = backbone.get_info()
    
    assert info['type'] == 'YOLOv5'
    assert info['size'] == 'S'
    assert not info['pretrained']
    assert not info['frozen']
    assert info['out_channels'] == [128, 256, 512]  # For YOLOv5s

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
