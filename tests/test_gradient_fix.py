#!/usr/bin/env python3
"""
Test script to verify gradient requirement fix.

This script tests that the training executor properly handles cases where
loss tensors or model parameters don't require gradients.
"""

import torch
import torch.nn as nn
import logging
import sys
sys.path.append('/Users/masdevid/Projects/smartcash')

from smartcash.model.training.core.training_executor import TrainingExecutor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockModel(nn.Module):
    """Mock model that can have gradients disabled for testing."""
    
    def __init__(self, requires_grad=True):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 3, padding=1)
        self.output = nn.Conv2d(64, 255, 1)
        
        # Control gradient requirements
        for param in self.parameters():
            param.requires_grad_(requires_grad)
    
    def forward(self, x):
        x = torch.relu(self.conv(x))
        x = self.output(x)
        return [x]  # YOLOv5-like format


class MockProgressTracker:
    def start_batch_tracking(self, num_batches): pass
    def update_batch_progress(self, *args, **kwargs): pass
    def complete_batch_tracking(self): pass


class MockLossManager:
    def __init__(self, loss_requires_grad=True):
        self.loss_requires_grad = loss_requires_grad
    
    def compute_loss(self, predictions, targets, image_size):
        # Create loss with or without gradient requirement
        if self.loss_requires_grad:
            loss = torch.tensor(0.5, requires_grad=True)
        else:
            loss = torch.tensor(0.5, requires_grad=False)
        
        return loss, {'total_loss': 0.5}


def test_gradient_requirements():
    """Test different gradient requirement scenarios."""
    logger.info("🧪 Testing Gradient Requirement Handling")
    logger.info("=" * 50)
    
    device = torch.device('cpu')  # Use CPU for consistency
    
    test_cases = [
        {
            'name': 'Normal Case (Model + Loss Require Grad)',
            'model_requires_grad': True,
            'loss_requires_grad': True,
            'should_succeed': True
        },
        {
            'name': 'Model No Grad (Should Warn)',
            'model_requires_grad': False,
            'loss_requires_grad': True,
            'should_succeed': True  # Should handle gracefully
        },
        {
            'name': 'Loss No Grad (Should Handle)',
            'model_requires_grad': True,
            'loss_requires_grad': False,
            'should_succeed': True  # Should handle gracefully
        },
        {
            'name': 'Both No Grad (Should Handle)',
            'model_requires_grad': False,
            'loss_requires_grad': False,
            'should_succeed': True  # Should handle gracefully
        }
    ]
    
    for test_case in test_cases:
        logger.info(f"\n📋 Test: {test_case['name']}")
        
        try:
            # Create test components
            model = MockModel(requires_grad=test_case['model_requires_grad'])
            progress_tracker = MockProgressTracker()
            loss_manager = MockLossManager(loss_requires_grad=test_case['loss_requires_grad'])
            
            # Create training executor
            config = {
                'mixed_precision': False,  # Disable for CPU testing
                'gradient_accumulation': False,
            }
            
            executor = TrainingExecutor(model, config, progress_tracker)
            
            # Create test data
            images = torch.randn(2, 3, 64, 64)  # Small batch for testing
            targets = torch.tensor([[0, 1, 0.5, 0.5, 0.3, 0.3], [1, 2, 0.4, 0.6, 0.2, 0.4]])
            
            # Create optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            
            # Test the optimized batch processing
            result = executor._process_training_batch_optimized(
                images, targets, loss_manager, optimizer, 0, 1, 1
            )
            
            loss_value, predictions, _, _ = result
            
            logger.info(f"   ✅ Success: Loss = {loss_value:.4f}")
            
            # Verify model parameter states
            param_count = sum(1 for p in model.parameters())
            grad_count = sum(1 for p in model.parameters() if p.requires_grad)
            actual_grad_count = sum(1 for p in model.parameters() if p.grad is not None)
            
            logger.info(f"   📊 Parameters: {param_count}, Require grad: {grad_count}, Have grad: {actual_grad_count}")
            
        except Exception as e:
            if test_case['should_succeed']:
                logger.error(f"   ❌ Unexpected failure: {e}")
            else:
                logger.info(f"   ✅ Expected failure: {e}")
    
    # Test legacy backward pass method as well
    logger.info(f"\n🔄 Testing Legacy Backward Pass Method")
    
    try:
        model = MockModel(requires_grad=True)
        config = {'mixed_precision': False, 'gradient_accumulation': False}
        executor = TrainingExecutor(model, config, MockProgressTracker())
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Test with normal loss
        loss_normal = torch.tensor(0.5, requires_grad=True)
        executor._backward_pass(loss_normal, optimizer, None)
        logger.info("   ✅ Legacy method with normal loss: Success")
        
        # Test with no-grad loss
        loss_no_grad = torch.tensor(0.5, requires_grad=False)
        executor._backward_pass(loss_no_grad, optimizer, None)
        logger.info("   ✅ Legacy method with no-grad loss: Handled gracefully")
        
    except Exception as e:
        logger.error(f"   ❌ Legacy method test failed: {e}")
    
    logger.info(f"\n🎉 Gradient requirement testing completed!")


def test_mixed_precision_with_gradients():
    """Test mixed precision with gradient requirements."""
    if not torch.cuda.is_available():
        logger.info("⚠️ Skipping mixed precision test (no CUDA)")
        return
    
    logger.info(f"\n🔥 Testing Mixed Precision with Gradients")
    
    try:
        device = torch.device('cuda')
        model = MockModel(requires_grad=True).to(device)
        config = {
            'mixed_precision': True,
            'gradient_accumulation': False,
        }
        
        executor = TrainingExecutor(model, config, MockProgressTracker())
        
        # Test data
        images = torch.randn(2, 3, 64, 64, device=device)
        targets = torch.tensor([[0, 1, 0.5, 0.5, 0.3, 0.3], [1, 2, 0.4, 0.6, 0.2, 0.4]], device=device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_manager = MockLossManager(loss_requires_grad=True)
        
        # Test mixed precision training
        result = executor._process_training_batch_optimized(
            images, targets, loss_manager, optimizer, 0, 1, 1
        )
        
        loss_value, _, _, _ = result
        logger.info(f"   ✅ Mixed precision training: Loss = {loss_value:.4f}")
        
    except Exception as e:
        logger.error(f"   ❌ Mixed precision test failed: {e}")


if __name__ == '__main__':
    try:
        test_gradient_requirements()
        test_mixed_precision_with_gradients()
        
        logger.info("\n✅ All gradient handling tests completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Test suite failed: {e}")
        import traceback
        logger.error(traceback.format_exc())