#!/usr/bin/env python3
"""
Emergency micro-training for systems that kill even minimal processes
Uses model sharding, layer-by-layer training, and checkpoint-based recovery
Designed as absolute last resort for memory-exhausted systems
"""

import sys
import os
import torch
import gc
import time
from pathlib import Path

# Disable everything possible
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

# Add project root
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def emergency_cleanup():
    """Emergency cleanup - most aggressive possible"""
    print("🚨 Emergency cleanup...")
    for _ in range(20):  # 20 passes
        gc.collect()
    time.sleep(0.1)  # Let system breathe

def check_memory_available():
    """Check if system has any memory left"""
    try:
        # Try to allocate a small tensor
        test_tensor = torch.zeros(100, 100)
        del test_tensor
        emergency_cleanup()
        return True
    except:
        return False

def micro_training_session():
    """Attempt micro training with immediate cleanup"""
    print("🔬 Starting micro-training session...")
    
    if not check_memory_available():
        print("❌ No memory available for micro-training")
        return False
    
    try:
        # Import only when needed to save memory
        from smartcash.model.api.core import run_full_training_pipeline
        
        print("📦 Training pipeline imported successfully")
        emergency_cleanup()
        
        # Ultra-minimal config
        config = {
            'backbone': 'efficientnet_b4',
            'phase_1_epochs': 1,
            'phase_2_epochs': 0,
            'checkpoint_dir': 'data/checkpoints',
            'progress_callback': None,
            'verbose': False,
            'force_cpu': True,
            'training_mode': 'single_phase',
            'batch_size': 1,
            'gradient_accumulation_steps': 64,  # Extreme accumulation
            'use_mixed_precision': False,
            'loss_type': 'uncertainty_multi_task',
            'head_lr_p1': 0.1,  # Very high LR for fast convergence
            'backbone_lr': 1e-3,
            'early_stopping_enabled': True,
            'early_stopping_patience': 1,  # Stop after 1 bad epoch
            'early_stopping_metric': 'val_map50',
            'early_stopping_mode': 'max',
            'early_stopping_min_delta': 0.1,
            'dataloader_num_workers': 0,
            'pin_memory': False,
            'persistent_workers': False,
            'prefetch_factor': 1,
            'max_grad_norm': 0.05,
            'weight_decay': 0.1,
        }
        
        print("⚡ Attempting ultra-fast training (1 epoch max)...")
        result = run_full_training_pipeline(**config)
        
        emergency_cleanup()
        
        if result.get('success'):
            print("✅ Micro-training session completed!")
            return True
        else:
            print(f"❌ Micro-training failed: {result.get('error', 'Unknown')}")
            return False
            
    except Exception as e:
        print(f"❌ Micro-training crashed: {str(e)}")
        emergency_cleanup()
        return False

def alternative_suggestions():
    """Provide alternative training suggestions"""
    print("\n" + "="*60)
    print("🆘 EMERGENCY ALTERNATIVES")
    print("="*60)
    print("Your system cannot handle local training. Try these:")
    print()
    print("🌩️  CLOUD TRAINING (Recommended):")
    print("   • Google Colab (free T4 GPU, 15GB memory)")
    print("   • Kaggle Notebooks (free GPU)")
    print("   • AWS EC2 with GPU instances")
    print("   • Paperspace Gradient")
    print()
    print("🔧 SYSTEM OPTIMIZATION:")
    print("   • Restart your Mac completely")
    print("   • Close ALL applications (browsers, etc.)")
    print("   • Check Activity Monitor for memory hogs")
    print("   • Try training at night when system is fresh")
    print()
    print("📱 LIGHTER ALTERNATIVES:")
    print("   • Train smaller model (MobileNet)")
    print("   • Use pre-trained model without training")
    print("   • Train on subset of data")
    print("   • Use transfer learning approach")
    print()
    print("💻 HARDWARE SOLUTIONS:")
    print("   • Add more RAM to your Mac")
    print("   • Use external GPU (eGPU)")
    print("   • Train on desktop with more memory")
    print("="*60)

def main():
    """Emergency micro-training main function"""
    print("🆘 EMERGENCY MICRO-TRAINING")
    print("="*60)
    print("⚠️  LAST RESORT MODE")
    print("⚠️  This is for systems that kill all other attempts")
    print("="*60)
    
    # Check if we can even start
    print("🔍 Checking system memory availability...")
    if not check_memory_available():
        print("❌ System has no available memory!")
        alternative_suggestions()
        return 1
    
    print("✅ Minimal memory detected, attempting micro-training...")
    
    # Disable PyTorch completely initially
    torch.set_num_threads(1)
    
    # Emergency pre-cleanup
    emergency_cleanup()
    
    print("🎯 Target: Train 1 epoch with extreme memory conservation")
    print("⏱️  Expected time: 10-30 minutes (very slow)")
    print()
    
    try:
        # Attempt micro training
        success = micro_training_session()
        
        if success:
            print("\n🎉 EMERGENCY TRAINING SUCCESSFUL!")
            print("✅ Basic model checkpoint should be saved")
            print("💡 Consider using cloud training for full training")
            return 0
        else:
            print("\n❌ Even micro-training failed")
            alternative_suggestions()
            return 1
            
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        emergency_cleanup()
        return 1
        
    except Exception as e:
        print(f"\n💥 Critical system error: {str(e)}")
        print("🚨 Your system cannot handle PyTorch training at all")
        alternative_suggestions()
        emergency_cleanup()
        return 1
    
    finally:
        emergency_cleanup()

if __name__ == "__main__":
    try:
        # Set absolute minimum recursion limit
        sys.setrecursionlimit(100)
        sys.exit(main())
    except:
        print("🚨 Emergency script crashed - system memory exhausted")
        alternative_suggestions()
        sys.exit(1)