"""
Isolated tests for backbone implementations.
These tests only test the backbone functionality without external dependencies.
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch, ANY

# Mock YOLOv5 imports
class MockDetect(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.args = args
        self.kwargs = kwargs
    def forward(self, x):
        return [torch.rand(1, 3, 80, 80, 6)]  # Mock detection output

# Mock YOLOv5 model
class MockYOLOv5Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1),  # /2
            nn.Conv2d(32, 64, 3, 2, 1), # /4
            nn.Conv2d(64, 128, 3, 2, 1), # /8
            nn.Conv2d(128, 256, 3, 2, 1), # /16
            nn.Conv2d(256, 512, 3, 2, 1), # /32
        )
    def forward(self, x):
        return [self.model(x)]

# Patch the imports before importing our modules
with patch.dict('sys.modules', {
    'models.yolo': MagicMock(Detect=MockDetect),
    'models.yolo.Detect': MockDetect,
    'models.common': MagicMock(),
    'models.experimental': MagicMock(),
    'yolov5': MagicMock(),
    'yolov5.models': MagicMock(),
    'yolov5.models.common': MagicMock(),
    'yolov5.models.experimental': MagicMock(),
}):
    # Now import our modules
    from smartcash.model.architectures.backbones.backbone import YOLOv5Backbone
    from smartcash.model.architectures.backbones.backbone_factory import BackboneFactory, BackboneConfig


def test_yolov5_backbone_initialization():
    """Test YOLOv5Backbone initialization with default parameters."""
    backbone = YOLOv5Backbone(
        model_size='s',
        pretrained=False,
        freeze=False
    )
    
    assert backbone.model_size == 's'
    assert not backbone.pretrained
    assert not backbone.frozen
    assert len(backbone.model) > 0


def test_yolov5_backbone_forward():
    """Test YOLOv5Backbone forward pass."""
    backbone = YOLOv5Backbone(model_size='s', pretrained=False)
    x = torch.randn(1, 3, 640, 640)
    
    with patch('torch.jit.trace', return_value=MockYOLOv5Model()):
        features = backbone(x)
    
    assert isinstance(features, list)
    assert len(features) == 1  # Mock model returns single feature map
    assert features[0].shape == (1, 512, 20, 20)  # 640/32 = 20


def test_backbone_factory_create_yolov5():
    """Test BackboneFactory with YOLOv5 backbone."""
    config = BackboneConfig(
        backbone_type='yolov5',
        model_size='s',
        pretrained=False,
        freeze=False
    )
    
    with patch('torch.jit.trace', return_value=MockYOLOv5Model()):
        backbone = BackboneFactory.create_backbone(config)
    
    assert backbone.model_size == 's'
    assert not backbone.pretrained
    assert not backbone.frozen


def test_backbone_factory_invalid_type():
    """Test BackboneFactory with invalid backbone type."""
    config = BackboneConfig(backbone_type='invalid')
    
    with pytest.raises(ValueError):
        BackboneFactory.create_backbone(config)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
