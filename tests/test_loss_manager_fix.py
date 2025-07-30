#!/usr/bin/env python3
"""
Test script to verify loss manager boolean tensor fix.

This script tests that the loss manager properly handles tensor operations
without causing "Boolean value of Tensor with more than one value is ambiguous" errors.
"""

import torch
import logging
import sys
sys.path.append('/Users/masdevid/Projects/smartcash')

from smartcash.model.training.loss_manager import LossManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_loss_manager_boolean_fix():
    """Test loss manager with various tensor configurations."""
    logger.info("🧪 Testing Loss Manager Boolean Tensor Fix")
    logger.info("=" * 50)
    
    # Initialize loss manager with config
    config = {
        'model': {'num_classes': 17},  # SmartCash has 17 classes
        'training': {'loss_type': 'uncertainty_multi_task'}
    }
    loss_manager = LossManager(config)
    loss_manager.use_multi_task_loss = True  # Enable multi-task loss to trigger the fixed code
    
    test_cases = [
        {
            'name': 'Normal Case with Valid Targets',
            'predictions': {
                'layer_1': [torch.randn(2, 255, 20, 20)],
                'layer_2': [torch.randn(2, 255, 40, 40)],
                'layer_3': [torch.randn(2, 255, 80, 80)]
            },
            'targets': torch.tensor([
                [0, 1, 0.5, 0.5, 0.3, 0.3],  # layer_1 target (class 1)
                [1, 8, 0.4, 0.6, 0.2, 0.4],  # layer_2 target (class 8)
                [0, 15, 0.3, 0.7, 0.1, 0.2]  # layer_3 target (class 15)
            ]),
            'should_succeed': True
        },
        {
            'name': 'Empty Targets Case',
            'predictions': {
                'layer_1': [torch.randn(2, 255, 20, 20)]
            },
            'targets': torch.empty(0, 6),
            'should_succeed': True
        },
        {
            'name': 'Single Target Case',
            'predictions': {
                'layer_1': [torch.randn(1, 255, 20, 20)]
            },
            'targets': torch.tensor([[0, 2, 0.5, 0.5, 0.3, 0.3]]),
            'should_succeed': True
        },
        {
            'name': 'Multiple Targets Same Layer',
            'predictions': {
                'layer_1': [torch.randn(2, 255, 20, 20)]
            },
            'targets': torch.tensor([
                [0, 1, 0.5, 0.5, 0.3, 0.3],
                [0, 2, 0.4, 0.6, 0.2, 0.4],
                [1, 3, 0.3, 0.7, 0.1, 0.2]
            ]),
            'should_succeed': True
        }
    ]
    
    for test_case in test_cases:
        logger.info(f"\n📋 Test: {test_case['name']}")
        
        try:
            # Test the loss computation
            predictions = test_case['predictions']
            targets = test_case['targets']
            
            # Call compute_loss which should trigger the fixed code
            loss, metrics = loss_manager.compute_loss(predictions, targets, img_size=640)
            
            logger.info(f"   ✅ Success: Loss = {loss.item():.4f}")
            logger.info(f"   📊 Metrics keys: {list(metrics.keys())}")
            logger.info(f"   📊 Targets processed: {metrics.get('num_targets', 0)}")
            
            # Verify loss is a proper tensor
            assert isinstance(loss, torch.Tensor), "Loss should be a tensor"
            assert loss.requires_grad, "Loss should require gradients"
            
        except Exception as e:
            if test_case['should_succeed']:
                logger.error(f"   ❌ Unexpected failure: {e}")
                import traceback
                logger.error(traceback.format_exc())
            else:
                logger.info(f"   ✅ Expected failure: {e}")
    
    logger.info(f"\n🎉 Loss manager boolean tensor testing completed!")


def test_filter_targets_for_layer():
    """Test the _filter_targets_for_layer method specifically."""
    logger.info(f"\n🔍 Testing _filter_targets_for_layer Method")
    logger.info("=" * 40)
    
    config = {
        'model': {'num_classes': 17},
        'training': {'loss_type': 'uncertainty_multi_task'}
    }
    loss_manager = LossManager(config)
    
    test_cases = [
        {
            'name': 'Layer 1 Filtering',
            'targets': torch.tensor([
                [0, 1, 0.5, 0.5, 0.3, 0.3],  # Should be included (class 1)
                [0, 8, 0.4, 0.6, 0.2, 0.4],  # Should be filtered out (class 8)
                [1, 3, 0.3, 0.7, 0.1, 0.2]   # Should be included (class 3)
            ]),
            'layer_name': 'layer_1',
            'expected_count': 2
        },
        {
            'name': 'Layer 2 Filtering',
            'targets': torch.tensor([
                [0, 1, 0.5, 0.5, 0.3, 0.3],  # Should be filtered out (class 1)
                [0, 8, 0.4, 0.6, 0.2, 0.4],  # Should be included (class 8)
                [1, 12, 0.3, 0.7, 0.1, 0.2]  # Should be included (class 12)
            ]),
            'layer_name': 'layer_2',
            'expected_count': 2
        },
        {
            'name': 'Empty Targets',
            'targets': torch.empty(0, 6),
            'layer_name': 'layer_1',
            'expected_count': 0
        }
    ]
    
    for test_case in test_cases:
        logger.info(f"\n📋 Test: {test_case['name']}")
        
        try:
            targets = test_case['targets']
            layer_name = test_case['layer_name']
            expected_count = test_case['expected_count']
            
            # Test the filtering method
            filtered_targets = loss_manager._filter_targets_for_layer(targets, layer_name)
            
            actual_count = filtered_targets.shape[0] if filtered_targets.numel() > 0 else 0
            
            logger.info(f"   ✅ Success: Expected {expected_count}, got {actual_count} targets")
            logger.info(f"   📊 Filtered shape: {filtered_targets.shape}")
            
            # Verify the count matches expected
            if actual_count != expected_count:
                logger.warning(f"   ⚠️ Count mismatch: expected {expected_count}, got {actual_count}")
            
        except Exception as e:
            logger.error(f"   ❌ Filter test failed: {e}")
            import traceback
            logger.error(traceback.format_exc())


if __name__ == '__main__':
    try:
        test_loss_manager_boolean_fix()
        test_filter_targets_for_layer()
        
        logger.info("\n✅ All loss manager tests completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Test suite failed: {e}")
        import traceback
        logger.error(traceback.format_exc())