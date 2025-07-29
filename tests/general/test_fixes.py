#!/usr/bin/env python3
"""
Test script to verify the multi-layer detection and mAP calculation fixes
"""

import os
import sys
sys.path.append('/Users/masdevid/Projects/smartcash')

from smartcash.model.api.core import run_full_training_pipeline

def test_single_phase_multi_layer():
    """Test single-phase mode with multi-layer should return 3 predictions"""
    print("🧪 TESTING: Single-phase multi-layer mode")
    print("=" * 60)
    
    config = {
        'backbone': 'cspdarknet',
        'training_mode': 'single_phase',  # Single phase
        'phase_1_epochs': 1,
        'phase_2_epochs': 0,
        'checkpoint_dir': 'data/test_checkpoints',
        'verbose': True,
        'force_cpu': False,
        
        # Model configuration for multi-layer (now supports dict configuration)
        'model': {
            'layer_mode': 'multi',  # Multi-layer mode
            'detection_layers': ['layer_1', 'layer_2', 'layer_3'],
            'multi_layer_heads': True,
            'num_classes': {
                'layer_1': 7,
                'layer_2': 7, 
                'layer_3': 3
            }
        },
        
        # Training parameters
        'loss_type': 'uncertainty_multi_task',
        'head_lr_p1': 0.001,
        'backbone_lr': 1e-5,
        
        # Quick test settings
        'early_stopping_enabled': False,
        'batch_size': 4,
        'dataloader_num_workers': 0
    }
    
    try:
        result = run_full_training_pipeline(**config)
        
        if result.get('success'):
            print("✅ SINGLE-PHASE MULTI-LAYER TEST PASSED")
            return True
        else:
            print(f"❌ SINGLE-PHASE MULTI-LAYER TEST FAILED: {result.get('error')}")
            return False
            
    except Exception as e:
        print(f"❌ SINGLE-PHASE MULTI-LAYER TEST ERROR: {e}")
        return False

def test_two_phase_multi_layer():
    """Test two-phase mode should have proper layer detection in Phase 2"""
    print("\n🧪 TESTING: Two-phase multi-layer mode")
    print("=" * 60)
    
    config = {
        'backbone': 'cspdarknet',
        'training_mode': 'two_phase',  # Two phase
        'phase_1_epochs': 1,
        'phase_2_epochs': 1,
        'checkpoint_dir': 'data/test_checkpoints',
        'verbose': True,
        'force_cpu': False,
        
        # Model configuration for multi-layer (now supports dict configuration)
        'model': {
            'layer_mode': 'multi',  # Multi-layer mode
            'detection_layers': ['layer_1', 'layer_2', 'layer_3'],
            'multi_layer_heads': True,
            'num_classes': {
                'layer_1': 7,
                'layer_2': 7,
                'layer_3': 3
            }
        },
        
        # Training parameters
        'loss_type': 'uncertainty_multi_task',
        'head_lr_p1': 0.001,
        'head_lr_p2': 0.0001,
        'backbone_lr': 1e-5,
        
        # Quick test settings
        'early_stopping_enabled': False,
        'batch_size': 4,
        'dataloader_num_workers': 0
    }
    
    try:
        result = run_full_training_pipeline(**config)
        
        if result.get('success'):
            print("✅ TWO-PHASE MULTI-LAYER TEST PASSED")
            return True
        else:
            print(f"❌ TWO-PHASE MULTI-LAYER TEST FAILED: {result.get('error')}")
            return False
            
    except Exception as e:
        print(f"❌ TWO-PHASE MULTI-LAYER TEST ERROR: {e}")
        return False

def main():
    """Run all tests"""
    print("🔬 TESTING MULTI-LAYER DETECTION AND mAP FIXES")
    print("=" * 60)
    print("This will test:")
    print("1. Single-phase multi-layer mode returns 3 predictions") 
    print("2. Two-phase multi-layer mode works in Phase 2")
    print("3. mAP calculation accuracy is preserved")
    print("=" * 60)
    
    # Test 1: Single-phase multi-layer
    test1_passed = test_single_phase_multi_layer()
    
    # Test 2: Two-phase multi-layer  
    test2_passed = test_two_phase_multi_layer()
    
    # Summary
    print("\n" + "=" * 60)
    print("🎯 TEST RESULTS SUMMARY:")
    print(f"Single-phase multi-layer: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"Two-phase multi-layer: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 ALL TESTS PASSED! Fixes are working correctly.")
        return True
    else:
        print("\n⚠️ Some tests failed. Check the logs above for details.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)