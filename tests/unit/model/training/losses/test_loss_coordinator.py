"""
Tests for the LossCoordinator class.
"""
import unittest
import torch
import numpy as np
from smartcash.model.training.losses.loss_coordinator import LossCoordinator

class TestLossCoordinator(unittest.TestCase):
    """Test cases for the LossCoordinator class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'training': {
                'loss': {
                    'box_weight': 0.05,
                    'obj_weight': 1.0,
                    'cls_weight': 0.5,
                    'focal_loss': False,
                    'label_smoothing': 0.0,
                    'dynamic_weighting': False
                }
            },
            'current_phase': 1,
            'model': {
                'layer_mode': 'single'
            }
        }
        self.loss_coordinator = LossCoordinator(self.config)
        
        # Set up test data
        self.batch_size = 2
        self.num_anchors = 3
        self.grid_size = 13
        self.num_classes = 7
        self.pred_shape = (self.batch_size, self.num_anchors, self.grid_size, self.grid_size, 5 + self.num_classes)
        
        # Create dummy predictions
        self.predictions = {
            'layer_1': [
                torch.randn(self.pred_shape, requires_grad=True)
            ]
        }
        
        # Create dummy targets [batch_idx, class_id, x, y, w, h]
        self.targets = torch.tensor([
            [0, 1, 0.5, 0.5, 0.2, 0.3],  # Object in first image
            [1, 2, 0.3, 0.7, 0.1, 0.2]   # Object in second image
        ], dtype=torch.float32)

    def test_loss_computation(self):
        """Test that loss computation returns non-zero values."""
        # Use CPU for consistent testing
        device = torch.device('cpu')
        
        # Move tensors to CPU
        predictions = {k: [p.to(device) for p in v] for k, v in self.predictions.items()}
        targets = self.targets.to(device)
        
        print("\n=== Test Loss Computation ===")
        print(f"Predictions shape: {predictions['layer_1'][0].shape}")
        print(f"Targets: {targets}")
        
        # Compute loss
        loss, loss_breakdown = self.loss_coordinator.compute_loss(predictions, targets)
        
        print(f"Loss: {loss.item()}")
        print(f"Loss breakdown: {loss_breakdown}")
        
        # Check that loss is a scalar tensor with requires_grad=True
        self.assertTrue(isinstance(loss, torch.Tensor))
        self.assertTrue(loss.requires_grad)
        self.assertEqual(loss.dim(), 0)  # Scalar
        
        # Check that loss is non-zero
        self.assertGreater(loss.item(), 0, f"Loss should be greater than zero, got {loss.item()}")
        
        # Check individual loss components
        for key in ['box_loss', 'obj_loss', 'cls_loss']:
            self.assertGreater(loss_breakdown[key], 0, f"{key} should be greater than zero, got {loss_breakdown[key]}")
        
        # Check loss breakdown structure
        for key in ['box_loss', 'obj_loss', 'cls_loss']:
            self.assertIn(key, loss_breakdown)
            self.assertIsInstance(loss_breakdown[key], float)
            self.assertGreaterEqual(loss_breakdown[key], 0, f"{key} should be non-negative")
        
        # Check metrics
        self.assertIn('metrics', loss_breakdown)
        metrics = loss_breakdown['metrics']
        for metric in ['box_ciou', 'obj_accuracy', 'cls_accuracy']:
            self.assertIn(metric, metrics)
            self.assertIsInstance(metrics[metric], float)
    
    def test_backward_pass(self):
        """Test that backward pass works correctly."""
        # Use CPU for consistent testing
        device = torch.device('cpu')
        
        # Create a fresh copy of predictions for this test
        # We'll track both the original tensors and their clones
        original_tensors = {}
        predictions = {}
        
        for k, v in self.predictions.items():
            original_tensors[k] = []
            predictions[k] = []
            
            for i, p in enumerate(v):
                # Create a new tensor with requires_grad=True and the same data
                p_clone = p.detach().clone()
                p_clone.requires_grad_(True)
                
                # Store both the original and the clone
                original_tensors[k].append(p_clone)
                predictions[k].append(p_clone)
        
        targets = self.targets.to(device)
        
        print("\n=== Test Backward Pass ===")
        print(f"Predictions shape: {predictions['layer_1'][0].shape}")
        print(f"Targets: {targets}")
        
        # Forward pass
        loss, _ = self.loss_coordinator.compute_loss(predictions, targets)
        print(f"Loss before backward: {loss.item()}")
        
        # Backward pass
        loss.backward()
        
        # Check gradients on the original tensors
        for i, pred in enumerate(original_tensors['layer_1']):
            # The gradient should be in the .grad attribute of the original tensor
            self.assertIsNotNone(pred.grad, f"Gradient for prediction {i} is None")
            grad_sum = torch.sum(torch.abs(pred.grad)).item()
            print(f"Gradient sum for prediction {i}: {grad_sum}")
            self.assertNotEqual(grad_sum, 0, f"Gradients should not be zero, but got {grad_sum}")
    
    def test_empty_targets(self):
        """Test that loss computation handles empty targets."""
        # Use CPU for consistent testing
        device = torch.device('cpu')
        
        # Move tensors to CPU
        predictions = {k: [p.to(device) for p in v] for k, v in self.predictions.items()}
        empty_targets = torch.zeros((0, 6), device=device)
        
        print("\n=== Test Empty Targets ===")
        print(f"Predictions shape: {predictions['layer_1'][0].shape}")
        print(f"Empty targets shape: {empty_targets.shape}")
        
        # Compute loss with empty targets
        loss, loss_breakdown = self.loss_coordinator.compute_loss(predictions, empty_targets)
        
        print(f"Loss with empty targets: {loss.item()}")
        print(f"Loss breakdown: {loss_breakdown}")
        
        # Loss should be non-zero due to objectness loss (background)
        self.assertGreater(loss.item(), 0, 
                         f"Loss should be greater than zero even with empty targets, got {loss.item()}")
        self.assertGreater(loss_breakdown['obj_loss'], 0, 
                         f"Objectness loss should be positive, got {loss_breakdown['obj_loss']}")

if __name__ == '__main__':
    unittest.main()
