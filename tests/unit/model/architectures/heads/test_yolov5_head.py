"""
Tests for YOLOv5 Multi-Layer Detection Head
"""
import unittest
import torch
import torch.nn as nn
from smartcash.model.architectures.heads.yolov5_head import (
    YOLOv5MultiLayerDetect, 
    YOLOv5HeadAdapter
)
from smartcash.common.logger import SmartCashLogger

class TestYOLOv5MultiLayerDetect(unittest.TestCase):
    """Test cases for YOLOv5MultiLayerDetect"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.logger = SmartCashLogger(__name__)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Test configuration
        self.nc = 7  # Number of classes
        self.anchors = [
            [10, 13, 16, 30, 33, 23],  # P3/8
            [30, 61, 62, 45, 59, 119],  # P4/16
            [116, 90, 156, 198, 373, 326]  # P5/32
        ]
        self.channels = [256, 512, 1024]  # Example channel sizes for P3, P4, P5
        
        # Create test input tensors
        self.batch_size = 2
        self.input_shapes = [
            (self.batch_size, ch, 80 // (2**i), 80 // (2**i))  # Decreasing spatial dimensions
            for i, ch in enumerate(self.channels)
        ]
        self.test_inputs = [
            torch.randn(shape, device=self.device) for shape in self.input_shapes
        ]
    
    def test_initialization(self):
        """Test initialization of YOLOv5MultiLayerDetect"""
        # Create detection head
        detect = YOLOv5MultiLayerDetect(
            nc=self.nc,
            anchors=self.anchors,
            ch=self.channels,
            logger=self.logger
        ).to(self.device)
        
        # Check if all required attributes are set
        self.assertTrue(hasattr(detect, 'layer_specs'))
        self.assertTrue(hasattr(detect, 'multi_heads'))
        self.assertTrue(hasattr(detect, 'stride'))
        self.assertTrue(hasattr(detect, 'export'))
        self.assertTrue(hasattr(detect, 'training'))
        
        # Check if primary detection layers are created
        self.assertTrue(hasattr(detect, 'primary_detection'))
        self.assertEqual(len(detect.primary_detection), len(self.channels))
        
        # Check if multi-heads are created
        self.assertEqual(len(detect.multi_heads), 2)  # layer_2 and layer_3
        
    def test_forward_pass(self):
        """Test forward pass through the detection head"""
        # Create detection head
        detect = YOLOv5MultiLayerDetect(
            nc=self.nc,
            anchors=self.anchors,
            ch=self.channels,
            logger=self.logger
        ).to(self.device)
        
        # Set to eval mode
        detect.eval()
        
        # Forward pass
        with torch.no_grad():
            outputs = detect(self.test_inputs)
        
        # Check output structure
        self.assertIsInstance(outputs, dict)
        self.assertIn('layer_1', outputs)
        self.assertIn('layer_2', outputs)
        self.assertIn('layer_3', outputs)
        
        # Check output shapes for each layer
        for layer_name, layer_outputs in outputs.items():
            self.assertIsInstance(layer_outputs, list)
            self.assertEqual(len(layer_outputs), len(self.test_inputs))
            
            # Check each output tensor
            for i, out in enumerate(layer_outputs):
                self.assertIsInstance(out, torch.Tensor)
                self.assertEqual(out.device, self.test_inputs[0].device)
                
                # Check batch size and spatial dimensions
                self.assertEqual(out.shape[0], self.batch_size)
                self.assertEqual(out.shape[2], self.test_inputs[i].shape[2])  # Height
                self.assertEqual(out.shape[3], self.test_inputs[i].shape[3])  # Width
    
    def test_training_mode(self):
        """Test behavior in training mode"""
        # Create detection head
        detect = YOLOv5MultiLayerDetect(
            nc=self.nc,
            anchors=self.anchors,
            ch=self.channels,
            logger=self.logger
        ).to(self.device)
        
        # Set to training mode
        detect.train()
        
        # Forward pass
        outputs = detect(self.test_inputs)
        
        # Check output structure
        self.assertIsInstance(outputs, dict)
        self.assertIn('layer_1', outputs)
        self.assertIn('layer_2', outputs)
        self.assertIn('layer_3', outputs)
    
    def test_yolov5_head_adapter(self):
        """Test YOLOv5HeadAdapter utility functions"""
        # Test create_multi_layer_head
        head = YOLOv5HeadAdapter.create_multi_layer_head(
            ch=self.channels,
            nc=self.nc,
            anchors=self.anchors,
            logger=self.logger
        )
        self.assertIsInstance(head, YOLOv5MultiLayerDetect)
        
        # Test create_banknote_head
        banknote_head = YOLOv5HeadAdapter.create_banknote_head(
            ch=self.channels,
            anchors=self.anchors,
            logger=self.logger
        )
        self.assertIsInstance(banknote_head, YOLOv5MultiLayerDetect)
        self.assertEqual(banknote_head.layer_specs['layer_1']['nc'], 7)  # Default banknote classes


if __name__ == "__main__":
    unittest.main()
