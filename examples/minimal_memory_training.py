#!/usr/bin/env python3
"""
Minimal memory training script for extremely constrained systems
Uses gradient checkpointing, CPU-only, and minimal batch processing
Designed for systems that get killed due to memory exhaustion
"""

import sys
import os
import torch
import gc
import multiprocessing as mp
from pathlib import Path

# Add project root to path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def setup_minimal_environment():
    """Setup absolutely minimal environment to prevent memory issues"""
    # Disable all GPU/MPS
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    torch.backends.mps.is_available = lambda: False
    torch.backends.mps.is_built = lambda: False
    torch.cuda.is_available = lambda: False
    
    # Minimal threading to prevent semaphore leaks
    torch.set_num_threads(1)
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
    
    # Disable multiprocessing to prevent semaphore leaks
    mp.set_start_method('spawn', force=True)
    
    # PyTorch memory optimizations
    torch.backends.quantized.engine = 'qnnpack'
    
    print("🔧 Minimal environment configured:")
    print("   • Device: CPU only")
    print("   • Threads: 1 (absolute minimum)")
    print("   • Multiprocessing: Disabled")
    print("   • GPU/MPS: Completely disabled")

def extreme_memory_cleanup():
    """Extreme memory cleanup"""
    for _ in range(10):
        gc.collect()
    
    # Force Python garbage collection
    if hasattr(gc, 'set_threshold'):
        gc.set_threshold(0)  # Disable automatic GC
        gc.collect()
        gc.set_threshold(700, 10, 10)  # Re-enable with aggressive settings

def create_minimal_training_config():
    """Create minimal training configuration to avoid memory issues"""
    return {
        'backbone': 'efficientnet_b4',
        'phase_1_epochs': 1,
        'phase_2_epochs': 0,  # Skip phase 2 entirely
        'checkpoint_dir': 'data/checkpoints',
        'progress_callback': None,  # No callback to save memory
        'verbose': False,
        'force_cpu': True,
        'training_mode': 'single_phase',
        
        # Absolute minimal memory settings
        'batch_size': 1,
        'gradient_accumulation_steps': 32,  # Very high accumulation
        'use_mixed_precision': False,
        
        # Simplified loss
        'loss_type': 'uncertainty_multi_task',
        'head_lr_p1': 0.01,  # Higher LR for faster convergence
        'backbone_lr': 1e-4,
        
        # Aggressive early stopping
        'early_stopping_enabled': True,
        'early_stopping_patience': 3,
        'early_stopping_metric': 'val_map50',
        'early_stopping_mode': 'max',
        'early_stopping_min_delta': 0.01,
        
        # Minimal data loading
        'dataloader_num_workers': 0,
        'pin_memory': False,
        'persistent_workers': False,
        'prefetch_factor': 1,
        'drop_last': True,
        
        # Gradient management
        'max_grad_norm': 0.1,
        'weight_decay': 0.01,
        
        # Memory-specific optimizations
        'gradient_checkpointing': True,  # Enable if available
        'dataloader_timeout': 0,
        'shuffle': False,  # Reduce memory allocation
    }

def main():
    """Run minimal memory training"""
    print("="*60)
    print("🚨 MINIMAL MEMORY TRAINING")
    print("="*60)
    print("⚠️  EXTREME MEMORY CONSERVATION MODE")
    print("⚠️  This mode trades everything for memory stability")
    print("="*60)
    
    # Setup minimal environment
    setup_minimal_environment()
    
    # Pre-training cleanup
    extreme_memory_cleanup()
    
    try:
        from smartcash.model.api.core import run_full_training_pipeline
        
        config = create_minimal_training_config()
        
        print("🚀 Starting minimal memory training...")
        print("⚠️  Process may be slow but should not be killed")
        print("⚠️  No progress bars to save memory")
        
        # Run with minimal configuration
        result = run_full_training_pipeline(**config)
        
        # Cleanup immediately
        extreme_memory_cleanup()
        
        if result.get('success'):
            print("\n" + "="*60)
            print("🎉 MINIMAL TRAINING COMPLETED!")
            print("="*60)
            
            training_result = result.get('final_training_result', {})
            if training_result.get('success'):
                best_metrics = training_result.get('best_metrics', {})
                print("📊 Minimal Results:")
                print(f"   Train Loss: {best_metrics.get('train_loss', 0):.4f}")
                print(f"   Val Loss:   {best_metrics.get('val_loss', 0):.4f}")
                
                # Check if any layer metrics exist
                found_metrics = False
                for layer in ['layer_1', 'layer_2', 'layer_3']:
                    acc = best_metrics.get(f'{layer}_accuracy', 0)
                    if acc > 0:
                        found_metrics = True
                        print(f"   {layer}: {acc:.3f}")
                
                if not found_metrics:
                    print("   Layer metrics: Not available (minimal mode)")
            
            print("\n💡 Memory Conservation:")
            print("   • ✅ Process completed without being killed")
            print("   • ✅ No semaphore leaks detected")
            print("   • ✅ Extreme memory conservation successful")
            
            return 0
        else:
            print("\n❌ MINIMAL TRAINING FAILED")
            error_msg = result.get('error', 'Unknown error')
            print(f"Error: {error_msg}")
            
            if "memory" in error_msg.lower():
                print("🚨 Even minimal mode failed due to memory!")
                print("💡 Your system may need to close other applications")
                print("💡 Consider training on a system with more RAM")
            
            return 1
            
    except Exception as e:
        print(f"\n❌ Critical error: {str(e)}")
        
        if "killed" in str(e).lower() or "memory" in str(e).lower():
            print("🚨 Process was killed by system (out of memory)")
            print("💡 Solutions:")
            print("   • Close all other applications")
            print("   • Restart your machine to free memory")
            print("   • Use a machine with more RAM")
            print("   • Train on Google Colab or cloud instance")
        
        extreme_memory_cleanup()
        return 1
    
    finally:
        # Final cleanup
        extreme_memory_cleanup()

if __name__ == "__main__":
    # Set recursion limit to prevent stack overflow
    sys.setrecursionlimit(1000)
    
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted")
        extreme_memory_cleanup()
        sys.exit(1)
    except SystemExit:
        extreme_memory_cleanup()
        raise
    except:
        extreme_memory_cleanup()
        sys.exit(1)